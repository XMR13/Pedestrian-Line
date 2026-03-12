from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from .config import ROOT_DIR
from .event_uploader import (
    DeliveryApiClient,
    UploaderConfig,
    iter_spool_runs,
    process_pending_runs,
    process_single_run,
)


DEFAULT_STATE_FILENAME = ".portal_upload_state.json"
MAX_RUNS_LIMIT = 200
MAX_EVENTS_LIMIT = 500


class SyncRequest(BaseModel):
    force: bool = False
    dry_run: bool = False
    max_runs: Optional[int] = Field(default=None, ge=1)


class SingleRunSyncRequest(BaseModel):
    force: bool = False
    dry_run: bool = False


@dataclass
class EdgeApiRuntime:
    spool_dir: Path
    uploader_cfg: Optional[UploaderConfig] = None
    service_started_at_utc: str = field(default_factory=lambda: _utcnow_iso())

    def health_payload(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "service_started_at_utc": self.service_started_at_utc,
            "spool_dir": str(self.spool_dir),
            "spool_exists": self.spool_dir.exists(),
            "uploader_enabled": self.uploader_cfg is not None,
        }

    def status_payload(self) -> Dict[str, Any]:
        runs = self.list_recent_runs(limit=1)
        counts = self._collect_delivery_counts()
        return {
            "ok": True,
            "service_started_at_utc": self.service_started_at_utc,
            "spool_dir": str(self.spool_dir),
            "spool_exists": self.spool_dir.exists(),
            "uploader_enabled": self.uploader_cfg is not None,
            "runs_total": counts["runs_total"],
            "delivery_state_counts": {
                "completed": counts["completed"],
                "pending": counts["pending"],
                "failed": counts["failed"],
                "in_progress": counts["in_progress"],
                "unknown": counts["unknown"],
            },
            "latest_run": runs[0] if runs else None,
        }

    def list_recent_runs(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        state_filename = self._state_filename()
        for run_dir in iter_spool_runs(self.spool_dir):
            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue
            status_meta = _load_json_dict(run_dir / "status.json")
            state_meta = _load_json_dict(run_dir / state_filename)
            rows.append(_build_run_summary(run_dir, run_meta, status_meta, state_meta))

        rows.sort(key=_run_sort_key, reverse=True)
        return rows[: _clamp_limit(limit, MAX_RUNS_LIMIT)]

    def list_recent_events(
        self,
        *,
        limit: int = 50,
        run_uid: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        target_run_uid = str(run_uid).strip() if run_uid else None

        for run_dir in iter_spool_runs(self.spool_dir):
            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue
            current_run_uid = _text(run_meta.get("run_uid"))
            if target_run_uid is not None and current_run_uid != target_run_uid:
                continue
            for event in _iter_jsonl_records(run_dir / "events.jsonl"):
                items.append(_build_event_summary(run_dir, run_meta, event))

        items.sort(key=_event_sort_key, reverse=True)
        return items[: _clamp_limit(limit, MAX_EVENTS_LIMIT)]

    def retry_pending_runs(
        self,
        *,
        force: bool = False,
        dry_run: bool = False,
        max_runs: Optional[int] = None,
    ) -> Dict[str, Any]:
        cfg = self._require_uploader_cfg()
        summary = process_pending_runs(
            cfg,
            force=bool(force),
            dry_run=bool(dry_run),
            max_runs=max_runs,
        )
        return {
            "ok": True,
            "mode": "pending_runs",
            "force": bool(force),
            "dry_run": bool(dry_run),
            "max_runs": max_runs,
            "summary": {
                "discovered_runs": int(summary.discovered_runs),
                "completed_runs": int(summary.completed_runs),
                "skipped_runs": int(summary.skipped_runs),
                "failed_runs": int(summary.failed_runs),
            },
        }

    def retry_single_run(
        self,
        run_uid: str,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        cfg = self._require_uploader_cfg()
        target_uid = str(run_uid).strip()
        run_dir = self._find_run_dir(target_uid)
        if run_dir is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"run_uid not found: {target_uid}")

        client = DeliveryApiClient(
            base_url=cfg.api_base_url,
            api_key=cfg.api_key,
            timeout_s=cfg.timeout_s,
        )
        run_status = process_single_run(
            run_dir,
            cfg=cfg,
            client=client,
            force=bool(force),
            dry_run=bool(dry_run),
        )
        return {
            "ok": True,
            "mode": "single_run",
            "run_uid": target_uid,
            "run_dir": str(run_dir),
            "force": bool(force),
            "dry_run": bool(dry_run),
            "status": str(run_status),
        }

    def _require_uploader_cfg(self) -> UploaderConfig:
        if self.uploader_cfg is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Uploader is not configured for this service instance.",
            )
        return self.uploader_cfg

    def _find_run_dir(self, run_uid: str) -> Optional[Path]:
        for run_dir in iter_spool_runs(self.spool_dir):
            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue
            if _text(run_meta.get("run_uid")) == run_uid:
                return run_dir
        return None

    def _state_filename(self) -> str:
        if self.uploader_cfg is not None and str(self.uploader_cfg.state_filename).strip():
            return str(self.uploader_cfg.state_filename).strip()
        return DEFAULT_STATE_FILENAME

    def _collect_delivery_counts(self) -> Dict[str, int]:
        counts = {
            "runs_total": 0,
            "completed": 0,
            "pending": 0,
            "failed": 0,
            "in_progress": 0,
            "unknown": 0,
        }
        state_filename = self._state_filename()
        for run_dir in iter_spool_runs(self.spool_dir):
            counts["runs_total"] += 1
            state_meta = _load_json_dict(run_dir / state_filename)
            state_name = _delivery_state_name(state_meta)
            if state_name in counts:
                counts[state_name] += 1
            else:
                counts["unknown"] += 1
        return counts


def create_app(
    *,
    spool_dir: Path,
    uploader_cfg: Optional[UploaderConfig] = None,
    title: str = "Pedestrian Line Edge Service",
) -> FastAPI:
    app = FastAPI(title=title, version="0.1.0")
    app.state.runtime = EdgeApiRuntime(spool_dir=Path(spool_dir), uploader_cfg=uploader_cfg)

    router = APIRouter()

    @router.get("/healthz", tags=["health"])
    def healthz(runtime: EdgeApiRuntime = Depends(_get_runtime)) -> Dict[str, Any]:
        return runtime.health_payload()

    @router.get("/status", tags=["health"])
    def status_view(runtime: EdgeApiRuntime = Depends(_get_runtime)) -> Dict[str, Any]:
        return runtime.status_payload()

    @router.get("/runs/recent", tags=["runs"])
    def recent_runs(
        limit: int = Query(default=20, ge=1, le=MAX_RUNS_LIMIT),
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> Dict[str, Any]:
        items = runtime.list_recent_runs(limit=limit)
        return {
            "items": items,
            "count": len(items),
            "limit": int(limit),
        }

    @router.get("/events/recent", tags=["events"])
    def recent_events(
        limit: int = Query(default=50, ge=1, le=MAX_EVENTS_LIMIT),
        run_uid: Optional[str] = Query(default=None),
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> Dict[str, Any]:
        items = runtime.list_recent_events(limit=limit, run_uid=run_uid)
        return {
            "items": items,
            "count": len(items),
            "limit": int(limit),
            "run_uid": run_uid,
        }

    @router.post("/sync/retry", tags=["sync"])
    def sync_retry(
        payload: SyncRequest,
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> Dict[str, Any]:
        return runtime.retry_pending_runs(
            force=bool(payload.force),
            dry_run=bool(payload.dry_run),
            max_runs=payload.max_runs,
        )

    @router.post("/sync/run/{run_uid}", tags=["sync"])
    def sync_single_run(
        run_uid: str,
        payload: SingleRunSyncRequest,
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> Dict[str, Any]:
        return runtime.retry_single_run(
            run_uid,
            force=bool(payload.force),
            dry_run=bool(payload.dry_run),
        )

    app.include_router(router)
    return app


def _get_runtime(request: Request) -> EdgeApiRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, EdgeApiRuntime):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Service runtime is not initialized.")
    return runtime


def _load_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _iter_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            try:
                obj = json.loads(row)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    break
                continue
            if isinstance(obj, dict):
                items.append(obj)
    return items


def _build_run_summary(
    run_dir: Path,
    run_meta: Mapping[str, Any],
    status_meta: Optional[Mapping[str, Any]],
    state_meta: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    health_summary = status_meta.get("health_summary") if isinstance(status_meta, Mapping) else None
    if not isinstance(health_summary, Mapping):
        health_summary = run_meta.get("health_summary") if isinstance(run_meta.get("health_summary"), Mapping) else None

    return {
        "run_uid": _text(run_meta.get("run_uid")),
        "run_dir": str(run_dir),
        "site_id": _text(run_meta.get("site_id")),
        "camera_id": _text(run_meta.get("camera_id")),
        "started_at_utc": _text(run_meta.get("started_at_utc")),
        "updated_at_utc": _coalesce_text(
            (status_meta or {}).get("updated_at_utc") if isinstance(status_meta, Mapping) else None,
            run_meta.get("updated_at_utc"),
        ),
        "ended_at_utc": _coalesce_text(
            run_meta.get("ended_at_utc"),
            (health_summary or {}).get("ended_at_utc") if isinstance(health_summary, Mapping) else None,
        ),
        "source_type": _coalesce_text(run_meta.get("source_type"), _mapping_get_text(run_meta.get("source"), "type")),
        "source_value": _coalesce_text(run_meta.get("source_value"), _mapping_get_text(run_meta.get("source"), "value")),
        "line_mode": _text(run_meta.get("line_mode")),
        "delivery_state": _delivery_state_name(state_meta),
        "delivery_completed_at_utc": _mapping_get_text(state_meta, "completed_at_utc"),
        "delivery_last_error": _mapping_get_text(state_meta, "last_error"),
        "report_csv_relpath": _text(run_meta.get("report_csv_relpath")),
        "lifecycle_status": _mapping_get_text(health_summary, "lifecycle_status"),
        "frames_total": _mapping_get_int(health_summary, "frames_total"),
        "frames_processed": _mapping_get_int(health_summary, "frames_processed"),
        "events_emitted_total": _mapping_get_int(health_summary, "events_emitted_total"),
        "count_a_to_b": _mapping_get_int(health_summary, "count_a_to_b"),
        "count_b_to_a": _mapping_get_int(health_summary, "count_b_to_a"),
        "effective_fps": _mapping_get_float(health_summary, "effective_fps"),
        "processed_fps": _mapping_get_float(health_summary, "processed_fps"),
    }


def _build_event_summary(
    run_dir: Path,
    run_meta: Mapping[str, Any],
    event: Mapping[str, Any],
) -> Dict[str, Any]:
    thumb_rel = _text(event.get("thumb_relpath"))
    scene_rel = _text(event.get("scene_relpath"))
    return {
        "event_uid": _text(event.get("event_uid")),
        "run_uid": _coalesce_text(event.get("run_uid"), run_meta.get("run_uid")),
        "site_id": _coalesce_text(event.get("site_id"), run_meta.get("site_id")),
        "camera_id": _coalesce_text(event.get("camera_id"), run_meta.get("camera_id")),
        "occurred_at_utc": _text(event.get("occurred_at_utc")),
        "direction": _text(event.get("direction")),
        "track_id": _mapping_get_int(event, "track_id"),
        "class_id": _mapping_get_int(event, "class_id"),
        "class_name": _text(event.get("class_name")),
        "confidence": _mapping_get_float(event, "confidence"),
        "frame_index": _mapping_get_int(event, "frame_index"),
        "video_time_s": _mapping_get_float(event, "video_time_s"),
        "thumb_relpath": thumb_rel,
        "scene_relpath": scene_rel,
        "thumb_path": str(run_dir / thumb_rel) if thumb_rel else None,
        "scene_path": str(run_dir / scene_rel) if scene_rel else None,
    }


def _delivery_state_name(state_meta: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(state_meta, Mapping):
        return "pending"
    if _text(state_meta.get("completed_at_utc")):
        return "completed"
    if _text(state_meta.get("last_error")):
        return "failed"
    if _mapping_get_int(state_meta, "events_uploaded_count") is not None:
        return "in_progress"
    return "pending"


def _run_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _coalesce_text(row.get("updated_at_utc"), row.get("started_at_utc"), row.get("run_uid")) or "",
        _text(row.get("run_uid")) or "",
    )


def _event_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _coalesce_text(row.get("occurred_at_utc"), row.get("event_uid")) or "",
        _text(row.get("event_uid")) or "",
    )


def _mapping_get_text(mapping: Any, key: str) -> Optional[str]:
    if not isinstance(mapping, Mapping):
        return None
    return _text(mapping.get(key))


def _mapping_get_int(mapping: Any, key: str) -> Optional[int]:
    if not isinstance(mapping, Mapping):
        return None
    value = mapping.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _mapping_get_float(mapping: Any, key: str) -> Optional[float]:
    if not isinstance(mapping, Mapping):
        return None
    value = mapping.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coalesce_text(*values: Any) -> Optional[str]:
    for value in values:
        out = _text(value)
        if out is not None:
            return out
    return None


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _clamp_limit(value: int, max_value: int) -> int:
    return max(1, min(int(value), int(max_value)))


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_spool_dir() -> Path:
    return ROOT_DIR / "spool"


app = create_app(spool_dir=_default_spool_dir())


__all__ = [
    "DEFAULT_STATE_FILENAME",
    "EdgeApiRuntime",
    "SingleRunSyncRequest",
    "SyncRequest",
    "app",
    "create_app",
]
