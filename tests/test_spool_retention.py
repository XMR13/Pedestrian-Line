from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pedestrian_line_counter.main as main_module
import pedestrian_line_counter.spool_retention as retention_module
from pedestrian_line_counter.spool_retention import apply_retention_policy


def _write_run(
    root: Path,
    *,
    day: str,
    run_uid: str,
    ended_at: datetime | None,
    state: dict[str, object] | None,
    extra_bytes: bytes = b"jpeg-data",
) -> Path:
    run_dir = root / day / run_uid
    (run_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    (run_dir / "scene").mkdir(parents=True, exist_ok=True)
    run_json = {
        "run_uid": run_uid,
        "site_id": "site_a",
        "camera_id": "cam_01",
        "started_at_utc": "2026-01-01T00:00:00Z",
        "ended_at_utc": ended_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if ended_at is not None
        else None,
        "source": {"type": "video", "value": "media/input.mp4"},
        "fps": 30.0,
        "frame_size": {"width": 1280, "height": 720},
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event_uid": f"{run_uid}_e1",
                "run_uid": run_uid,
                "site_id": "site_a",
                "camera_id": "cam_01",
                "occurred_at_utc": "2026-01-01T00:00:03Z",
                "frame_index": 90,
                "direction": "A_TO_B",
                "track_id": 1,
                "thumb_relpath": "thumbs/e1.jpg",
                "scene_relpath": "scene/e1.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "thumbs" / "e1.jpg").write_bytes(extra_bytes)
    (run_dir / "scene" / "e1.jpg").write_bytes(extra_bytes)
    if state is not None:
        (run_dir / ".portal_upload_state.json").write_text(json.dumps(state), encoding="utf-8")
    return run_dir


def test_retention_dry_run_keeps_old_completed_run(tmp_path) -> None:
    now = datetime(2026, 3, 10, tzinfo=timezone.utc)
    run_dir = _write_run(
        tmp_path,
        day="2025-11-01",
        run_uid="run_old",
        ended_at=now - timedelta(days=100),
        state={"run_uid": "run_old", "completed_at_utc": "2025-11-02T00:00:00Z"},
    )

    summary = apply_retention_policy(tmp_path, dry_run=True, now=now)

    assert summary.scanned_runs == 1
    assert summary.eligible_runs == 1
    assert summary.deleted_runs == 0
    assert summary.bytes_reclaimable > 0
    assert run_dir.exists()
    assert summary.runs[0].status == "delete_eligible"


def test_retention_deletes_only_old_completed_runs(tmp_path) -> None:
    now = datetime(2026, 3, 10, tzinfo=timezone.utc)
    old_run = _write_run(
        tmp_path,
        day="2025-11-01",
        run_uid="run_old",
        ended_at=now - timedelta(days=100),
        state={"run_uid": "run_old", "completed_at_utc": "2025-11-02T00:00:00Z"},
    )
    recent_run = _write_run(
        tmp_path,
        day="2026-02-20",
        run_uid="run_recent",
        ended_at=now - timedelta(days=10),
        state={"run_uid": "run_recent", "completed_at_utc": "2026-02-21T00:00:00Z"},
    )

    summary = apply_retention_policy(tmp_path, dry_run=False, now=now)

    assert summary.scanned_runs == 2
    assert summary.deleted_runs == 1
    assert summary.bytes_deleted > 0
    assert not old_run.exists()
    assert recent_run.exists()
    statuses = {info.run_uid: info.status for info in summary.runs}
    assert statuses["run_old"] == "delete_eligible"
    assert statuses["run_recent"] == "retained_recent"


def test_retention_protects_missing_or_incomplete_state(tmp_path) -> None:
    now = datetime(2026, 3, 10, tzinfo=timezone.utc)
    missing_state = _write_run(
        tmp_path,
        day="2025-10-01",
        run_uid="run_missing_state",
        ended_at=now - timedelta(days=120),
        state=None,
    )
    incomplete = _write_run(
        tmp_path,
        day="2025-10-02",
        run_uid="run_incomplete",
        ended_at=now - timedelta(days=120),
        state={"run_uid": "run_incomplete", "events_uploaded_count": 1},
    )

    summary = apply_retention_policy(tmp_path, dry_run=False, now=now)

    assert summary.deleted_runs == 0
    assert missing_state.exists()
    assert incomplete.exists()
    reasons = {info.run_uid: info.reason for info in summary.runs}
    statuses = {info.run_uid: info.status for info in summary.runs}
    assert statuses["run_missing_state"] == "protected_ambiguous"
    assert statuses["run_incomplete"] == "protected_incomplete"
    assert "upload state" in reasons["run_missing_state"]
    assert "delivery not completed" == reasons["run_incomplete"]


def test_retention_pressure_budget_selects_oldest_completed_run(tmp_path) -> None:
    now = datetime(2026, 3, 10, tzinfo=timezone.utc)
    old_run = _write_run(
        tmp_path,
        day="2026-03-01",
        run_uid="run_old",
        ended_at=now - timedelta(days=9),
        state={"run_uid": "run_old", "completed_at_utc": "2026-03-01T00:00:00Z"},
        extra_bytes=b"x" * 64,
    )
    recent_run = _write_run(
        tmp_path,
        day="2026-03-09",
        run_uid="run_recent",
        ended_at=now - timedelta(days=1),
        state={"run_uid": "run_recent", "completed_at_utc": "2026-03-09T00:00:00Z"},
        extra_bytes=b"y" * 64,
    )

    baseline = apply_retention_policy(tmp_path, max_age_days=365, dry_run=True, now=now)
    summary = apply_retention_policy(
        tmp_path,
        max_age_days=365,
        max_total_bytes=int(baseline.total_runs_bytes) - 1,
        dry_run=False,
        now=now,
    )

    assert summary.deleted_runs == 1
    by_uid = {info.run_uid: info for info in summary.runs}
    assert by_uid["run_old"].selected_for_deletion is True
    assert by_uid["run_old"].deletion_basis == "pressure"
    assert by_uid["run_recent"].selected_for_deletion is False
    assert not old_run.exists()
    assert recent_run.exists()


def test_retention_pressure_free_space_uses_disk_free_threshold(monkeypatch, tmp_path) -> None:
    now = datetime(2026, 3, 10, tzinfo=timezone.utc)
    _write_run(
        tmp_path,
        day="2026-03-08",
        run_uid="run_recent",
        ended_at=now - timedelta(days=2),
        state={"run_uid": "run_recent", "completed_at_utc": "2026-03-08T00:00:00Z"},
        extra_bytes=b"z" * 64,
    )

    class _Usage:
        total = 10_000
        used = 9_500
        free = 500

    monkeypatch.setattr(retention_module, "_disk_usage_or_none", lambda root: _Usage())

    summary = apply_retention_policy(
        tmp_path,
        max_age_days=365,
        min_free_bytes=600,
        dry_run=True,
        now=now,
    )

    assert summary.pressure_bytes_target == 100
    assert summary.eligible_runs == 1
    assert summary.runs[0].deletion_basis == "pressure"
    assert summary.projected_disk_free_bytes_after == 500 + summary.bytes_reclaimable


def test_retention_cli_runs_without_opening_video(monkeypatch, tmp_path, capsys) -> None:
    _write_run(
        tmp_path,
        day="2000-01-01",
        run_uid="run_old",
        ended_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        state={"run_uid": "run_old", "completed_at_utc": "2000-01-02T00:00:00Z"},
    )

    def _should_not_open(*_args, **_kwargs):
        raise AssertionError("video source should not be opened during retention-only command")

    monkeypatch.setattr(main_module, "_open_source_with_first_frame", _should_not_open)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--spool-dir",
            str(tmp_path),
            "--spool-retention-run",
            "--spool-retention-dry-run",
        ],
    )

    main_module.main()

    out = capsys.readouterr().out
    assert "[retention] scanned=1 eligible=1" in out
    assert (tmp_path / "2000-01-01" / "run_old").exists()


def test_retention_protects_invalid_state_json(tmp_path) -> None:
    now = datetime(2026, 3, 10, tzinfo=timezone.utc)
    run_dir = _write_run(
        tmp_path,
        day="2025-10-03",
        run_uid="run_invalid_state",
        ended_at=now - timedelta(days=120),
        state={"run_uid": "run_invalid_state", "completed_at_utc": "2025-10-04T00:00:00Z"},
    )
    (run_dir / ".portal_upload_state.json").write_text("{invalid", encoding="utf-8")

    summary = apply_retention_policy(tmp_path, dry_run=False, now=now)

    assert summary.deleted_runs == 0
    assert run_dir.exists()
    assert summary.runs[0].status == "protected_ambiguous"
    assert summary.runs[0].reason == "upload state missing or invalid"
