import json
from pathlib import Path

from pedestrian_line_counter.portal_uploader import (
    RetryConfig,
    RetryableUploadError,
    UploaderConfig,
    process_single_run,
)


class _FakeClient:
    def __init__(self) -> None:
        self.run_calls = 0
        self.events_calls = 0
        self.thumb_calls = 0

    def upsert_run(self, payload):
        self.run_calls += 1
        if self.run_calls == 1:
            raise RetryableUploadError("temporary")
        return {"ok": True, "run_uid": payload["run_uid"]}

    def upsert_events(self, payload):
        self.events_calls += 1
        return {"ok": True, "n": len(payload.get("events", []))}

    def upload_thumbnail(self, event_uid, *, filename, content, kind="object"):
        _ = event_uid
        _ = filename
        _ = content
        _ = kind
        self.thumb_calls += 1
        return {"ok": True}


def _write_run(run_dir: Path, *, with_thumb: bool = True) -> None:
    run_dir.mkdir(parents=True)
    run_json = {
        "run_uid": "run_x",
        "site_id": "site_a",
        "camera_id": "cam_01",
        "started_at_utc": "2026-02-20T00:00:00Z",
        "source": {"type": "video", "value": "media/input.mp4"},
        "frame_size": {"width": 100, "height": 80},
        "fps": 30.0,
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")

    rel_thumb = "thumbs/e1.jpg" if with_thumb else None
    event = {
        "event_uid": "e1",
        "run_uid": "run_x",
        "site_id": "site_a",
        "camera_id": "cam_01",
        "occurred_at_utc": "2026-02-20T00:00:03Z",
        "frame_index": 90,
        "video_time_s": 3.0,
        "direction": "A_TO_B",
        "track_id": 1,
        "class_id": 2,
        "class_name": "truck",
        "confidence": 0.91,
        "bbox": [10, 10, 40, 50],
        "thumb_relpath": rel_thumb,
    }
    (run_dir / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    if with_thumb:
        thumbs = run_dir / "thumbs"
        thumbs.mkdir(parents=True)
        (thumbs / "e1.jpg").write_bytes(b"fake-jpeg")


def test_process_single_run_retries_and_writes_state(tmp_path) -> None:
    run_dir = tmp_path / "2026-02-20" / "run_x"
    _write_run(run_dir, with_thumb=True)

    cfg = UploaderConfig(
        spool_dir=tmp_path,
        api_base_url="http://portal.local",
        api_key="secret",
        state_filename=".state.json",
        events_batch_size=100,
        retry=RetryConfig(max_attempts=3, initial_delay_s=0.0, max_delay_s=0.0, backoff_factor=1.0),
    )
    fake = _FakeClient()

    status = process_single_run(run_dir, cfg=cfg, client=fake, force=False, dry_run=False)

    assert status == "completed"
    assert fake.run_calls == 2
    assert fake.events_calls == 1
    assert fake.thumb_calls == 1

    state = json.loads((run_dir / ".state.json").read_text(encoding="utf-8"))
    assert state["run_uid"] == "run_x"
    assert state["events_uploaded_count"] == 1
    assert state["thumbs_uploaded_count"] == 1
    assert state["completed_at_utc"]

    status2 = process_single_run(run_dir, cfg=cfg, client=fake, force=False, dry_run=False)
    assert status2 == "skipped"
    assert fake.events_calls == 1


def test_process_single_run_skips_missing_thumbnail_file(tmp_path) -> None:
    run_dir = tmp_path / "2026-02-20" / "run_x"
    _write_run(run_dir, with_thumb=False)

    cfg = UploaderConfig(
        spool_dir=tmp_path,
        api_base_url="http://portal.local",
        api_key="secret",
        state_filename=".state.json",
        retry=RetryConfig(max_attempts=2, initial_delay_s=0.0, max_delay_s=0.0, backoff_factor=1.0),
    )
    fake = _FakeClient()

    status = process_single_run(run_dir, cfg=cfg, client=fake, force=False, dry_run=False)

    assert status == "completed"
    assert fake.thumb_calls == 0
    state = json.loads((run_dir / ".state.json").read_text(encoding="utf-8"))
    assert state["thumbs_uploaded_count"] == 0
