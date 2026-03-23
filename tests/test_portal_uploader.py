from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest

from pedestrian_line_counter import portal_uploader
from pedestrian_line_counter.event_uploader import (
    DeliveryApiClient,
    RetryConfig,
    RetryableUploadError,
    UploaderConfig,
    UploadError,
    process_single_run,
    resolve_portal_api_key,
)


class _FakeClient:
    def __init__(self) -> None:
        self.run_calls = 0
        self.events_calls = 0
        self.events_batch_sizes = []
        self.thumb_calls = 0

    def upsert_run(self, payload):
        self.run_calls += 1
        if self.run_calls == 1:
            raise RetryableUploadError("temporary")
        return {"ok": True, "run_uid": payload["run_uid"]}

    def upsert_events(self, payload):
        self.events_calls += 1
        self.events_batch_sizes.append(len(payload.get("events", [])))
        event_uids = [str(row.get("event_uid")) for row in payload.get("events", []) if row.get("event_uid")]
        return {"ok": True, "accepted_event_uids": event_uids}

    def upload_thumbnail(self, event_uid, *, filename, content, kind="object"):
        _ = filename
        _ = content
        self.thumb_calls += 1
        return {"ok": True, "event_uid": event_uid, "kind": kind}


class _MockDeliveryBackend:
    def __init__(self) -> None:
        self.run_payloads = []
        self.event_batches = []
        self.thumbnail_calls = []
        self._event_failures = {}
        self._thumbnail_failures = {}
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    def fail_event_batch_once(self, *event_uids: str) -> None:
        key = tuple(event_uids)
        self._event_failures[key] = self._event_failures.get(key, 0) + 1

    def fail_thumbnail_once(self, event_uid: str, kind: str = "object") -> None:
        key = (event_uid, kind)
        self._thumbnail_failures[key] = self._thumbnail_failures.get(key, 0) + 1

    def _make_handler(self):
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length)

                if self.path == "/api/runs/upsert":
                    payload = json.loads(body.decode("utf-8"))
                    backend.run_payloads.append(payload)
                    self._send_json(200, {"ok": True, "run_uid": payload.get("run_uid")})
                    return

                if self.path == "/api/events/upsert":
                    payload = json.loads(body.decode("utf-8"))
                    event_uids = tuple(
                        str(row.get("event_uid"))
                        for row in payload.get("events", [])
                        if row.get("event_uid") is not None
                    )
                    backend.event_batches.append(event_uids)
                    remaining = backend._event_failures.get(event_uids, 0)
                    if remaining > 0:
                        backend._event_failures[event_uids] = remaining - 1
                        self._send_json(503, {"error": "temporary event failure"})
                        return
                    self._send_json(200, {"ok": True, "accepted_event_uids": list(event_uids)})
                    return

                if self.path.startswith("/api/events/") and self.path.endswith("/thumbnail"):
                    event_uid = self.path.split("/")[3]
                    kind = str(self.headers.get("X-Evidence-Kind", "object"))
                    backend.thumbnail_calls.append((event_uid, kind, len(body)))
                    key = (event_uid, kind)
                    remaining = backend._thumbnail_failures.get(key, 0)
                    if remaining > 0:
                        backend._thumbnail_failures[key] = remaining - 1
                        self._send_json(503, {"error": "temporary thumbnail failure"})
                        return
                    self._send_json(200, {"ok": True, "event_uid": event_uid, "kind": kind})
                    return

                self._send_json(404, {"error": "not found"})

            def log_message(self, fmt, *args):
                return

            def _send_json(self, status: int, payload) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        return Handler


def _make_event(
    event_uid: str,
    *,
    frame_index: int,
    video_time_s: float,
    direction: str = "A_TO_B",
    track_id: int = 1,
    class_id: int = 2,
    class_name: str = "truck",
    confidence: float = 0.91,
    bbox: list[int] | None = None,
    thumb_relpath: str | None = None,
    scene_relpath: str | None = None,
) -> dict:
    return {
        "event_uid": event_uid,
        "run_uid": "run_x",
        "site_id": "site_a",
        "camera_id": "cam_01",
        "occurred_at_utc": f"2026-02-20T00:00:{frame_index:02d}Z",
        "frame_index": frame_index,
        "video_time_s": video_time_s,
        "direction": direction,
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox or [10, 10, 40, 50],
        "thumb_relpath": thumb_relpath,
        "scene_relpath": scene_relpath,
    }


def _write_run(
    run_dir: Path,
    *,
    with_thumb: bool = True,
    ended_at_utc: str | None = "2026-02-20T00:00:10Z",
    events: list[dict] | None = None,
) -> None:
    run_dir.mkdir(parents=True)
    run_json = {
        "run_uid": "run_x",
        "site_id": "site_a",
        "camera_id": "cam_01",
        "started_at_utc": "2026-02-20T00:00:00Z",
        "ended_at_utc": ended_at_utc,
        "source": {"type": "video", "value": "media/input.mp4"},
        "frame_size": {"width": 100, "height": 80},
        "fps": 30.0,
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")

    if events is None:
        rel_thumb = "thumbs/e1.jpg" if with_thumb else None
        events = [
            _make_event(
                "e1",
                frame_index=3,
                video_time_s=3.0,
                thumb_relpath=rel_thumb,
            )
        ]

    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    for event in events:
        for relpath in (event.get("thumb_relpath"), event.get("scene_relpath")):
            if not relpath:
                continue
            file_path = run_dir / str(relpath)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"fake-jpeg")


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


def test_process_single_run_in_progress_uploads_only_new_events(tmp_path) -> None:
    run_dir = tmp_path / "2026-02-20" / "run_x"
    _write_run(run_dir, with_thumb=False, ended_at_utc=None)

    cfg = UploaderConfig(
        spool_dir=tmp_path,
        api_base_url="http://portal.local",
        api_key="secret",
        state_filename=".state.json",
        retry=RetryConfig(max_attempts=2, initial_delay_s=0.0, max_delay_s=0.0, backoff_factor=1.0),
    )
    fake = _FakeClient()

    status_1 = process_single_run(run_dir, cfg=cfg, client=fake, force=False, dry_run=False)
    assert status_1 == "completed"
    assert fake.run_calls == 2
    assert fake.events_calls == 1
    assert fake.events_batch_sizes == [1]

    event_2 = _make_event(
        "e2",
        frame_index=5,
        video_time_s=5.0,
        direction="B_TO_A",
        track_id=2,
        confidence=0.95,
        bbox=[20, 20, 60, 70],
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(_make_event("e1", frame_index=3, video_time_s=3.0)),
                json.dumps(event_2),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status_2 = process_single_run(run_dir, cfg=cfg, client=fake, force=False, dry_run=False)
    assert status_2 == "completed"
    assert fake.run_calls == 2
    assert fake.events_calls == 2
    assert fake.events_batch_sizes == [1, 1]

    state = json.loads((run_dir / ".state.json").read_text(encoding="utf-8"))
    assert state["events_uploaded_count"] == 2
    assert "completed_at_utc" not in state
    assert sorted(state["uploaded_event_uids"]) == ["e1", "e2"]

    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_meta["updated_at_utc"] = "2026-02-20T00:00:06Z"
    run_meta["health_summary"] = {"lifecycle_status": "running", "events_emitted_total": 2}
    (run_dir / "run.json").write_text(json.dumps(run_meta), encoding="utf-8")

    status_3 = process_single_run(run_dir, cfg=cfg, client=fake, force=False, dry_run=False)
    assert status_3 == "completed"
    assert fake.run_calls == 3
    assert fake.events_calls == 2


def test_process_single_run_resumes_from_failed_event_batch_without_reuploading_completed_batches(tmp_path) -> None:
    run_dir = tmp_path / "2026-02-20" / "run_x"
    events = [
        _make_event("e1", frame_index=1, video_time_s=1.0),
        _make_event("e2", frame_index=2, video_time_s=2.0),
        _make_event("e3", frame_index=3, video_time_s=3.0),
    ]
    _write_run(run_dir, with_thumb=False, events=events)

    backend = _MockDeliveryBackend()
    backend.fail_event_batch_once("e3")
    client = DeliveryApiClient(base_url=backend.base_url, api_key="secret", timeout_s=2.0)
    cfg = UploaderConfig(
        spool_dir=tmp_path,
        api_base_url=backend.base_url,
        api_key="secret",
        state_filename=".state.json",
        events_batch_size=2,
        retry=RetryConfig(max_attempts=1, initial_delay_s=0.0, max_delay_s=0.0, backoff_factor=1.0),
    )

    try:
        with pytest.raises(UploadError):
            process_single_run(run_dir, cfg=cfg, client=client, force=False, dry_run=False)

        state_after_failure = json.loads((run_dir / ".state.json").read_text(encoding="utf-8"))
        assert state_after_failure["events_uploaded_count"] == 2
        assert sorted(state_after_failure["uploaded_event_uids"]) == ["e1", "e2"]
        assert state_after_failure["last_error"]

        status = process_single_run(run_dir, cfg=cfg, client=client, force=False, dry_run=False)
        assert status == "completed"

        state_after_success = json.loads((run_dir / ".state.json").read_text(encoding="utf-8"))
        assert state_after_success["events_uploaded_count"] == 3
        assert sorted(state_after_success["uploaded_event_uids"]) == ["e1", "e2", "e3"]
        assert state_after_success["completed_at_utc"]
        assert backend.event_batches.count(("e1", "e2")) == 1
        assert backend.event_batches.count(("e3",)) == 2
    finally:
        backend.close()


def test_process_single_run_resumes_from_failed_thumbnail_without_reuploading_completed_evidence(tmp_path) -> None:
    run_dir = tmp_path / "2026-02-20" / "run_x"
    events = [
        _make_event("e1", frame_index=1, video_time_s=1.0, thumb_relpath="thumbs/e1.jpg"),
        _make_event("e2", frame_index=2, video_time_s=2.0, thumb_relpath="thumbs/e2.jpg"),
    ]
    _write_run(run_dir, events=events)

    backend = _MockDeliveryBackend()
    backend.fail_thumbnail_once("e2", "object")
    client = DeliveryApiClient(base_url=backend.base_url, api_key="secret", timeout_s=2.0)
    cfg = UploaderConfig(
        spool_dir=tmp_path,
        api_base_url=backend.base_url,
        api_key="secret",
        state_filename=".state.json",
        retry=RetryConfig(max_attempts=1, initial_delay_s=0.0, max_delay_s=0.0, backoff_factor=1.0),
    )

    try:
        with pytest.raises(UploadError):
            process_single_run(run_dir, cfg=cfg, client=client, force=False, dry_run=False)

        state_after_failure = json.loads((run_dir / ".state.json").read_text(encoding="utf-8"))
        assert state_after_failure["thumbs_uploaded_count"] == 1
        assert state_after_failure["uploaded_thumb_markers"] == ["e1:thumbs/e1.jpg"]
        assert state_after_failure["last_error"]

        status = process_single_run(run_dir, cfg=cfg, client=client, force=False, dry_run=False)
        assert status == "completed"

        state_after_success = json.loads((run_dir / ".state.json").read_text(encoding="utf-8"))
        assert state_after_success["thumbs_uploaded_count"] == 2
        assert sorted(state_after_success["uploaded_thumb_markers"]) == ["e1:thumbs/e1.jpg", "e2:thumbs/e2.jpg"]
        assert [call[:2] for call in backend.thumbnail_calls] == [
            ("e1", "object"),
            ("e2", "object"),
            ("e2", "object"),
        ]
    finally:
        backend.close()


def test_resolve_portal_api_key_prefers_direct_value(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PORTAL_API_KEY", "env-secret")
    settings_path = tmp_path / "appsettings.Local.json"
    settings_path.write_text(json.dumps({"Portal": {"ApiKey": "file-secret"}}), encoding="utf-8")

    key = resolve_portal_api_key(
        "direct-secret",
        api_key_env="PORTAL_API_KEY",
        appsettings_local_path=str(settings_path),
    )
    assert key == "direct-secret"


def test_resolve_portal_api_key_uses_local_settings_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PORTAL_API_KEY", raising=False)
    settings_path = tmp_path / "appsettings.Local.json"
    settings_path.write_text(json.dumps({"Portal": {"ApiKey": "file-secret"}}), encoding="utf-8")

    key = resolve_portal_api_key(
        None,
        api_key_env="PORTAL_API_KEY",
        appsettings_local_path=str(settings_path),
    )
    assert key == "file-secret"


def test_portal_uploader_module_remains_compatible() -> None:
    assert portal_uploader.PortalApiClient is not None
    assert callable(portal_uploader.process_pending_runs)
    assert callable(portal_uploader.resolve_portal_api_key)
