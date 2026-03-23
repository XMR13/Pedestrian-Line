from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Mapping, Optional

from .event_uploader import iter_spool_runs


@dataclass
class RetentionRunInfo:
    run_dir: Path
    run_uid: Optional[str]
    size_bytes: int
    status: str
    reason: str
    ended_at_utc: Optional[str] = None
    age_days: Optional[float] = None
    state_path: Optional[Path] = None
    eligible_by_age: bool = False
    selected_for_deletion: bool = False
    deletion_basis: Optional[str] = None


@dataclass
class RetentionSummary:
    root_dir: Path
    now_utc: str
    max_age_days: int
    max_total_bytes: Optional[int]
    min_free_bytes: Optional[int]
    state_filename: str
    dry_run: bool
    scanned_runs: int = 0
    deleted_runs: int = 0
    protected_runs: int = 0
    retained_recent_runs: int = 0
    eligible_runs: int = 0
    bytes_reclaimable: int = 0
    bytes_deleted: int = 0
    total_runs_bytes: int = 0
    disk_total_bytes: Optional[int] = None
    disk_used_bytes: Optional[int] = None
    disk_free_bytes_before: Optional[int] = None
    projected_runs_bytes_after: int = 0
    projected_disk_free_bytes_after: Optional[int] = None
    pressure_bytes_target: int = 0
    pressure_bytes_remaining_after: int = 0
    runs: List[RetentionRunInfo] = field(default_factory=list)


def apply_retention_policy(
    root_dir: Path,
    *,
    max_age_days: int = 90,
    max_total_bytes: Optional[int] = None,
    min_free_bytes: Optional[int] = None,
    state_filename: str = ".portal_upload_state.json",
    protect_incomplete_runs: bool = True,
    dry_run: bool = True,
    now: Optional[datetime] = None,
) -> RetentionSummary:
    root = Path(root_dir)
    now_dt = _normalize_now(now)
    summary = RetentionSummary(
        root_dir=root,
        now_utc=_iso_utc(now_dt),
        max_age_days=max(int(max_age_days), 0),
        max_total_bytes=_normalize_optional_non_negative_int(max_total_bytes),
        min_free_bytes=_normalize_optional_non_negative_int(min_free_bytes),
        state_filename=str(state_filename),
        dry_run=bool(dry_run),
    )

    if not root.exists():
        summary.projected_runs_bytes_after = 0
        return summary

    for run_dir in iter_spool_runs(root):
        info = inspect_run(
            run_dir,
            now=now_dt,
            max_age_days=summary.max_age_days,
            state_filename=summary.state_filename,
            protect_incomplete_runs=bool(protect_incomplete_runs),
        )
        summary.scanned_runs += 1
        summary.runs.append(info)
        summary.total_runs_bytes += int(info.size_bytes)

    disk_usage = _disk_usage_or_none(root)
    if disk_usage is not None:
        summary.disk_total_bytes = int(disk_usage.total)
        summary.disk_used_bytes = int(disk_usage.used)
        summary.disk_free_bytes_before = int(disk_usage.free)

    _plan_retention(summary)

    for info in sorted(_selected_candidates(summary.runs), key=_delete_sort_key):
        if summary.dry_run:
            continue
        _delete_run_dir(info.run_dir)
        summary.deleted_runs += 1
        summary.bytes_deleted += int(info.size_bytes)

    return summary


def inspect_run(
    run_dir: Path,
    *,
    now: datetime,
    max_age_days: int,
    state_filename: str,
    protect_incomplete_runs: bool,
) -> RetentionRunInfo:
    run_dir = Path(run_dir)
    size_bytes = _dir_size_bytes(run_dir)
    run_meta = _load_json_dict(run_dir / "run.json")
    if run_meta is None:
        return RetentionRunInfo(
            run_dir=run_dir,
            run_uid=None,
            size_bytes=size_bytes,
            status="protected_ambiguous",
            reason="run.json missing or invalid",
        )

    run_uid = _text(run_meta.get("run_uid"))
    ended_at_utc = _resolve_run_ended_at(run_meta)
    if ended_at_utc is None:
        status = "protected_incomplete" if protect_incomplete_runs else "protected_ambiguous"
        reason = "run is not closed"
        return RetentionRunInfo(
            run_dir=run_dir,
            run_uid=run_uid,
            size_bytes=size_bytes,
            status=status,
            reason=reason,
        )

    ended_at_dt = _parse_utc(ended_at_utc)
    if ended_at_dt is None:
        return RetentionRunInfo(
            run_dir=run_dir,
            run_uid=run_uid,
            size_bytes=size_bytes,
            status="protected_ambiguous",
            reason="ended_at_utc is invalid",
            ended_at_utc=ended_at_utc,
        )

    state_path = run_dir / str(state_filename)
    state = _load_json_dict(state_path)
    if state is None:
        return RetentionRunInfo(
            run_dir=run_dir,
            run_uid=run_uid,
            size_bytes=size_bytes,
            status="protected_ambiguous",
            reason="upload state missing or invalid",
            ended_at_utc=ended_at_utc,
            age_days=_age_days(now, ended_at_dt),
            state_path=state_path,
        )

    state_run_uid = _text(state.get("run_uid"))
    if state_run_uid and run_uid and state_run_uid != run_uid:
        return RetentionRunInfo(
            run_dir=run_dir,
            run_uid=run_uid,
            size_bytes=size_bytes,
            status="protected_ambiguous",
            reason="upload state run_uid mismatch",
            ended_at_utc=ended_at_utc,
            age_days=_age_days(now, ended_at_dt),
            state_path=state_path,
        )

    completed_at_utc = _text(state.get("completed_at_utc"))
    if completed_at_utc is None:
        status = "protected_incomplete" if protect_incomplete_runs else "protected_ambiguous"
        return RetentionRunInfo(
            run_dir=run_dir,
            run_uid=run_uid,
            size_bytes=size_bytes,
            status=status,
            reason="delivery not completed",
            ended_at_utc=ended_at_utc,
            age_days=_age_days(now, ended_at_dt),
            state_path=state_path,
        )

    age_days = _age_days(now, ended_at_dt)
    if age_days < float(max(max_age_days, 0)):
        return RetentionRunInfo(
            run_dir=run_dir,
            run_uid=run_uid,
            size_bytes=size_bytes,
            status="retained_recent",
            reason=f"completed but younger than {max(max_age_days, 0)} days",
            ended_at_utc=ended_at_utc,
            age_days=age_days,
            state_path=state_path,
        )

    return RetentionRunInfo(
        run_dir=run_dir,
        run_uid=run_uid,
        size_bytes=size_bytes,
        status="delete_eligible",
        reason=f"completed and older than {max(max_age_days, 0)} days",
        ended_at_utc=ended_at_utc,
        age_days=age_days,
        state_path=state_path,
        eligible_by_age=True,
    )


def format_retention_summary(summary: RetentionSummary) -> List[str]:
    lines = [
        (
            f"[retention] scanned={summary.scanned_runs} eligible={summary.eligible_runs} "
            f"deleted={summary.deleted_runs} protected={summary.protected_runs} "
            f"recent={summary.retained_recent_runs} reclaimable={summary.bytes_reclaimable}B "
            f"deleted_bytes={summary.bytes_deleted}B total_runs={summary.total_runs_bytes}B "
            f"pressure_target={summary.pressure_bytes_target}B "
            f"pressure_remaining={summary.pressure_bytes_remaining_after}B dry_run={summary.dry_run}"
        )
    ]
    for info in sorted(summary.runs, key=_display_sort_key):
        run_label = info.run_uid or info.run_dir.name
        age_part = ""
        if info.age_days is not None:
            age_part = f" age_days={info.age_days:.1f}"
        basis_part = ""
        if info.deletion_basis is not None:
            basis_part = f" basis={info.deletion_basis}"
        lines.append(
            f"[retention] {info.status} run_uid={run_label} size={info.size_bytes}B{age_part}{basis_part} reason={info.reason}"
        )
    return lines


def _plan_retention(summary: RetentionSummary) -> None:
    age_candidates: List[RetentionRunInfo] = []
    pressure_candidates: List[RetentionRunInfo] = []

    for info in summary.runs:
        if info.eligible_by_age:
            age_candidates.append(info)
            pressure_candidates.append(info)
        elif info.status == "retained_recent":
            pressure_candidates.append(info)

    for info in sorted(age_candidates, key=_delete_sort_key):
        _mark_selected_for_deletion(info, basis="age")

    summary.pressure_bytes_target = _pressure_bytes_target(
        total_runs_bytes=summary.total_runs_bytes,
        max_total_bytes=summary.max_total_bytes,
        disk_free_bytes=summary.disk_free_bytes_before,
        min_free_bytes=summary.min_free_bytes,
    )

    planned_bytes = sum(int(info.size_bytes) for info in _selected_candidates(summary.runs))
    remaining_pressure = max(summary.pressure_bytes_target - planned_bytes, 0)
    if remaining_pressure > 0:
        for info in sorted(pressure_candidates, key=_delete_sort_key):
            if info.selected_for_deletion:
                continue
            _mark_selected_for_deletion(
                info,
                basis="pressure",
                reason=_pressure_reason(summary.max_total_bytes, summary.min_free_bytes),
            )
            planned_bytes += int(info.size_bytes)
            remaining_pressure = max(summary.pressure_bytes_target - planned_bytes, 0)
            if remaining_pressure <= 0:
                break

    summary.bytes_reclaimable = sum(int(info.size_bytes) for info in _selected_candidates(summary.runs))
    summary.eligible_runs = sum(1 for info in summary.runs if info.selected_for_deletion)
    summary.retained_recent_runs = sum(1 for info in summary.runs if info.status == "retained_recent")
    summary.protected_runs = sum(1 for info in summary.runs if info.status.startswith("protected_"))
    summary.projected_runs_bytes_after = max(summary.total_runs_bytes - summary.bytes_reclaimable, 0)
    if summary.disk_free_bytes_before is not None:
        summary.projected_disk_free_bytes_after = int(summary.disk_free_bytes_before) + int(summary.bytes_reclaimable)
    summary.pressure_bytes_remaining_after = max(summary.pressure_bytes_target - summary.bytes_reclaimable, 0)


def _selected_candidates(items: List[RetentionRunInfo]) -> List[RetentionRunInfo]:
    return [info for info in items if info.selected_for_deletion]


def _mark_selected_for_deletion(
    info: RetentionRunInfo,
    *,
    basis: str,
    reason: Optional[str] = None,
) -> None:
    info.status = "delete_eligible"
    info.selected_for_deletion = True
    info.deletion_basis = str(basis)
    if reason is not None:
        info.reason = reason


def _pressure_bytes_target(
    *,
    total_runs_bytes: int,
    max_total_bytes: Optional[int],
    disk_free_bytes: Optional[int],
    min_free_bytes: Optional[int],
) -> int:
    required = 0
    if max_total_bytes is not None:
        required = max(required, max(int(total_runs_bytes) - int(max_total_bytes), 0))
    if min_free_bytes is not None and disk_free_bytes is not None:
        required = max(required, max(int(min_free_bytes) - int(disk_free_bytes), 0))
    return required


def _pressure_reason(max_total_bytes: Optional[int], min_free_bytes: Optional[int]) -> str:
    reasons: List[str] = []
    if max_total_bytes is not None:
        reasons.append(f"spool exceeds max_total_bytes={int(max_total_bytes)}")
    if min_free_bytes is not None:
        reasons.append(f"disk free space is below min_free_bytes={int(min_free_bytes)}")
    if not reasons:
        return "completed run selected under retention pressure policy"
    return "completed run selected to satisfy " + " and ".join(reasons)


def _load_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _resolve_run_ended_at(run_meta: Mapping[str, Any]) -> Optional[str]:
    ended_at = _text(run_meta.get("ended_at_utc"))
    if ended_at is not None:
        return ended_at
    health_summary = run_meta.get("health_summary")
    if isinstance(health_summary, Mapping):
        return _text(health_summary.get("ended_at_utc"))
    return None


def _normalize_now(now: Optional[datetime]) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _parse_utc(value: str) -> Optional[datetime]:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _age_days(now: datetime, then: datetime) -> float:
    delta = now - then
    return max(delta.total_seconds(), 0.0) / 86400.0


def _delete_run_dir(run_dir: Path) -> None:
    shutil.rmtree(run_dir)
    parent = run_dir.parent
    if parent.exists():
        try:
            next(parent.iterdir())
        except StopIteration:
            parent.rmdir()


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += int(path.stat().st_size)
        except FileNotFoundError:
            continue
    return total


def _disk_usage_or_none(root: Path):
    try:
        return shutil.disk_usage(root)
    except Exception:
        return None


def _normalize_optional_non_negative_int(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(int(value), 0)


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _delete_sort_key(info: RetentionRunInfo) -> tuple[float, str]:
    age = info.age_days if info.age_days is not None else -1.0
    return (-age, str(info.run_dir))


def _display_sort_key(info: RetentionRunInfo) -> tuple[str, str]:
    return (info.status, str(info.run_dir))


__all__ = [
    "RetentionRunInfo",
    "RetentionSummary",
    "apply_retention_policy",
    "format_retention_summary",
    "inspect_run",
]
