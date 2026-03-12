from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import pedestrian_line_counter.api as api_module
from pedestrian_line_counter.api import create_app
from pedestrian_line_counter.event_uploader import RetryConfig, SyncSummary, UploaderConfig


def _write_run(
    root: Path,
    *,
    day: str,
    run_uid: str,
    started_at_utc: str,
    occurred_at_utc: str,
    completed_at_utc: str | None = None,
    last_error: str | None = None,
    lifecycle_status: str = "stopped",
) -> Path:
    run_dir = root / day / run_uid
    (run_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    (run_dir / "scene").mkdir(parents=True, exist_ok=True)

    run_json = {
        "run_uid": run_uid,
        "site_id": "site_a",
        "camera_id": "cam_01",
        "started_at_utc": started_at_utc,
        "updated_at_utc": started_at_utc,
        "ended_at_utc": started_at_utc,
        "source": {"type": "rtsp", "value": "rtsp://camera"},
        "line_mode": "line",
        "report_csv_relpath": "report.csv",
        "health_summary": {
            "lifecycle_status": lifecycle_status,
            "frames_total": 100,
            "frames_processed": 50,
            "events_emitted_total": 1,
            "count_a_to_b": 1,
            "count_b_to_a": 0,
            "effective_fps": 12.5,
            "processed_fps": 6.25,
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_uid": run_uid,
                "updated_at_utc": occurred_at_utc,
                "health_summary": run_json["health_summary"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event_uid": f"{run_uid}_e1",
                "run_uid": run_uid,
                "site_id": "site_a",
                "camera_id": "cam_01",
                "occurred_at_utc": occurred_at_utc,
                "frame_index": 90,
                "video_time_s": 3.0,
                "direction": "A_TO_B",
                "track_id": 1,
                "class_id": 2,
                "class_name": "truck",
                "confidence": 0.91,
                "thumb_relpath": "thumbs/e1.jpg",
                "scene_relpath": "scene/e1.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "thumbs" / "e1.jpg").write_bytes(b"jpg")
    (run_dir / "scene" / "e1.jpg").write_bytes(b"jpg")

    state = {}
    if completed_at_utc is not None:
        state["completed_at_utc"] = completed_at_utc
    if last_error is not None:
        state["last_error"] = last_error
    if state:
        state["run_uid"] = run_uid
        (run_dir / ".portal_upload_state.json").write_text(json.dumps(state), encoding="utf-8")

    return run_dir


def test_healthz_and_status_expose_spool_state(tmp_path) -> None:
    _write_run(
        tmp_path,
        day="2026-03-11",
        run_uid="run_a",
        started_at_utc="2026-03-11T10:00:00Z",
        occurred_at_utc="2026-03-11T10:05:00Z",
        completed_at_utc="2026-03-11T10:06:00Z",
    )
    _write_run(
        tmp_path,
        day="2026-03-10",
        run_uid="run_b",
        started_at_utc="2026-03-10T09:00:00Z",
        occurred_at_utc="2026-03-10T09:05:00Z",
        last_error="temporary failure",
    )

    client = TestClient(create_app(spool_dir=tmp_path))

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.json()["spool_exists"] is True

    status = client.get("/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["runs_total"] == 2
    assert payload["delivery_state_counts"]["completed"] == 1
    assert payload["delivery_state_counts"]["failed"] == 1
    assert payload["latest_run"]["run_uid"] == "run_a"


def test_recent_runs_and_events_are_sorted_newest_first(tmp_path) -> None:
    _write_run(
        tmp_path,
        day="2026-03-09",
        run_uid="run_old",
        started_at_utc="2026-03-09T09:00:00Z",
        occurred_at_utc="2026-03-09T09:05:00Z",
    )
    _write_run(
        tmp_path,
        day="2026-03-11",
        run_uid="run_new",
        started_at_utc="2026-03-11T09:00:00Z",
        occurred_at_utc="2026-03-11T09:05:00Z",
    )

    client = TestClient(create_app(spool_dir=tmp_path))

    runs = client.get("/runs/recent", params={"limit": 2})
    assert runs.status_code == 200
    run_items = runs.json()["items"]
    assert [item["run_uid"] for item in run_items] == ["run_new", "run_old"]

    events = client.get("/events/recent", params={"limit": 2})
    assert events.status_code == 200
    event_items = events.json()["items"]
    assert [item["run_uid"] for item in event_items] == ["run_new", "run_old"]
    assert event_items[0]["thumb_path"].endswith("thumbs/e1.jpg")


def test_sync_endpoints_use_existing_uploader_flow(monkeypatch, tmp_path) -> None:
    _write_run(
        tmp_path,
        day="2026-03-11",
        run_uid="run_sync",
        started_at_utc="2026-03-11T10:00:00Z",
        occurred_at_utc="2026-03-11T10:05:00Z",
    )

    uploader_cfg = UploaderConfig(
        spool_dir=tmp_path,
        api_base_url="http://it.local",
        api_key="secret",
        retry=RetryConfig(max_attempts=1, initial_delay_s=0.0, max_delay_s=0.0, backoff_factor=1.0),
    )
    client = TestClient(create_app(spool_dir=tmp_path, uploader_cfg=uploader_cfg))

    recorded: dict[str, object] = {}

    def _fake_process_pending_runs(cfg, *, force: bool = False, dry_run: bool = False, max_runs=None):
        recorded["pending"] = {
            "spool_dir": str(cfg.spool_dir),
            "force": bool(force),
            "dry_run": bool(dry_run),
            "max_runs": max_runs,
        }
        return SyncSummary(discovered_runs=1, completed_runs=1, skipped_runs=0, failed_runs=0)

    class _FakeClient:
        def __init__(self, *, base_url: str, api_key: str, timeout_s: float) -> None:
            recorded["client"] = {
                "base_url": base_url,
                "api_key": api_key,
                "timeout_s": timeout_s,
            }

    def _fake_process_single_run(run_dir, *, cfg, client, force: bool = False, dry_run: bool = False):
        recorded["single"] = {
            "run_dir": str(run_dir),
            "force": bool(force),
            "dry_run": bool(dry_run),
            "api_base_url": cfg.api_base_url,
            "client_type": type(client).__name__,
        }
        return "completed"

    monkeypatch.setattr(api_module, "process_pending_runs", _fake_process_pending_runs)
    monkeypatch.setattr(api_module, "DeliveryApiClient", _FakeClient)
    monkeypatch.setattr(api_module, "process_single_run", _fake_process_single_run)

    retry_resp = client.post("/sync/retry", json={"force": True, "dry_run": True, "max_runs": 3})
    assert retry_resp.status_code == 200
    assert retry_resp.json()["summary"]["completed_runs"] == 1
    assert recorded["pending"] == {
        "spool_dir": str(tmp_path),
        "force": True,
        "dry_run": True,
        "max_runs": 3,
    }

    single_resp = client.post("/sync/run/run_sync", json={"force": False, "dry_run": True})
    assert single_resp.status_code == 200
    assert single_resp.json()["status"] == "completed"
    assert recorded["single"] == {
        "run_dir": str(tmp_path / "2026-03-11" / "run_sync"),
        "force": False,
        "dry_run": True,
        "api_base_url": "http://it.local",
        "client_type": "_FakeClient",
    }
