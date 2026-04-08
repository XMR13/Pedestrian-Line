from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

import pedestrian_line_counter.main as main_module
from pedestrian_line_counter.event_uploader import SyncSummary
from pedestrian_line_counter.stream_reader import ReaderPoll


class _FakeCap:
    def __init__(self) -> None:
        self.released = False
        self.grab_calls = 0
        self.read_calls = 0

    def isOpened(self) -> bool:
        return True

    def get(self, _prop: int) -> float:
        return 30.0

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def read(self):
        self.read_calls += 1
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        return True, frame

    def grab(self) -> bool:
        self.grab_calls += 1
        return True

    def release(self) -> None:
        self.released = True


class _FakeDetector:
    def __init__(self, _cfg) -> None:
        pass

    def detect(self, _frame):
        return []


class _FakeTracker:
    def __init__(self, _cfg) -> None:
        pass

    def update(self, _detections, _frame_index):
        return []

    def clear_runtime_state(self) -> None:
        return None


class _FakeLineCounter:
    def __init__(self, **_kwargs) -> None:
        self.count_a_to_b = 0
        self.count_b_to_a = 0
        self.count_by_class_dir = {"a_to_b": {}, "b_to_a": {}}

    def update(self, _tracks, frame_index: int = 0):
        _ = frame_index
        return []

    def clear_runtime_state(self) -> None:
        return None


class _InterruptingDetector:
    def __init__(self, _cfg) -> None:
        pass

    def detect(self, _frame):
        raise KeyboardInterrupt()


class _FakeStreamReader:
    def __init__(self, cap, *, queue_size: int = 3, overflow_policy: str = "drop_oldest") -> None:
        _ = queue_size
        _ = overflow_policy
        self._cap = cap
        self.read_frames = 0
        self.dropped = 0
        self.read_failures = 0

    def start(self) -> None:
        return None

    def get_with_status(self, *, timeout_s: float = 1.0) -> ReaderPoll:
        _ = timeout_s
        return ReaderPoll(item=None, status="stopped", reason="fake_end")

    def stop(self, *, join_timeout_s: float = 2.0, reason: str = "stopped_by_caller") -> None:
        _ = join_timeout_s
        _ = reason
        self._cap.release()


def _run_main(monkeypatch, argv_tail: List[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["prog"] + argv_tail)
    main_module.main()


def _write_prior_local_run(
    root_dir: Path,
    *,
    run_uid: str,
    input_path: Path,
    camera_id: str,
    video_start: str,
    completed: bool = True,
) -> Path:
    run_dir = root_dir / "2026-03-11" / run_uid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    payload = {
        "run_uid": run_uid,
        "site_id": "site_a",
        "camera_id": camera_id,
        "started_at_utc": "2026-03-11T03:00:00Z",
        "updated_at_utc": "2026-03-11T03:02:00Z",
        "ended_at_utc": "2026-03-11T03:02:00Z" if completed else None,
        "lifecycle_status": "stopped" if completed else "running",
        "source_type": "video",
        "source_value": str(input_path),
        "video_start": video_start,
        "video_start_epoch_utc": main_module._parse_rfc3339_to_epoch_utc(video_start),
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def test_live_single_loop_integrated_upload_smoke(monkeypatch, tmp_path, capsys) -> None:
    fake_cap = _FakeCap()
    frame0 = np.zeros((16, 16, 3), dtype=np.uint8)
    upload_calls: List[dict] = []

    def _open_impl(source, **_kwargs):
        _ = source
        return fake_cap, frame0, 30.0, 16, 16, "FAKE"

    def _fake_process_pending_runs(cfg, *, force: bool = False, dry_run: bool = False, max_runs: Optional[int] = None):
        upload_calls.append(
            {
                "spool_dir": str(cfg.spool_dir),
                "force": bool(force),
                "dry_run": bool(dry_run),
                "max_runs": max_runs,
            }
        )
        return SyncSummary(discovered_runs=1, completed_runs=1, skipped_runs=0, failed_runs=0)

    monkeypatch.setattr(main_module, "Detector", _FakeDetector)
    monkeypatch.setattr(main_module, "Tracker", _FakeTracker)
    monkeypatch.setattr(main_module, "LineCounter", _FakeLineCounter)
    monkeypatch.setattr(main_module, "StreamReader", _FakeStreamReader)
    monkeypatch.setattr(main_module, "_open_source_with_first_frame", _open_impl)
    monkeypatch.setattr(main_module, "process_pending_runs", _fake_process_pending_runs)

    _run_main(
        monkeypatch,
        [
            "--backend",
            "motion",
            "--rtsp-url",
            "rtsp://unit-test",
            "--max-frames",
            "1",
            "--no-write",
            "--no-draw",
            "--no-progress",
            "--spool-dir",
            str(tmp_path),
            "--site-id",
            "site_a",
            "--camera-id",
            "cam_01",
            "--portal-upload",
            "--portal-api-base-url",
            "http://portal.local:5000",
            "--portal-api-key",
            "secret",
            "--portal-upload-interval-s",
            "0.2",
        ],
    )

    assert fake_cap.released is True
    assert len(upload_calls) >= 1
    assert all(c["dry_run"] is False for c in upload_calls)

    run_jsons = list(Path(tmp_path).glob("*/*/run.json"))
    assert len(run_jsons) == 1
    run_meta = json.loads(run_jsons[0].read_text(encoding="utf-8"))
    assert run_meta.get("ended_at_utc")
    assert isinstance(run_meta.get("health_summary"), dict)
    assert run_meta["health_summary"].get("lifecycle_status") == "stopped"

    status_json = run_jsons[0].parent / "status.json"
    assert status_json.exists()
    status_meta = json.loads(status_json.read_text(encoding="utf-8"))
    assert status_meta.get("run_uid") == run_meta.get("run_uid")
    assert status_meta.get("lifecycle_status") == "stopped"

    out = capsys.readouterr().out
    assert "[main][portal] integrated uploader enabled" in out
    assert "[main] Done." in out


def test_live_startup_fails_when_camera_has_no_first_frame(monkeypatch) -> None:
    def _open_fail(_source, **_kwargs):
        raise RuntimeError("Could not read any frame from source: rtsp://missing")

    monkeypatch.setattr(main_module, "_open_source_with_first_frame", _open_fail)

    with pytest.raises(SystemExit, match="Could not read any frame from source"):
        _run_main(
            monkeypatch,
            [
                "--backend",
                "motion",
                "--rtsp-url",
                "rtsp://missing",
                "--no-write",
                "--no-draw",
                "--no-progress",
            ],
        )


def test_main_fails_fast_when_ffmpeg_output_is_requested_but_missing(monkeypatch, tmp_path) -> None:
    fake_cap = _FakeCap()
    frame0 = np.zeros((16, 16, 3), dtype=np.uint8)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")

    def _open_impl(source, **_kwargs):
        _ = source
        return fake_cap, frame0, 30.0, 16, 16, "FAKE"

    monkeypatch.setattr(main_module, "_open_source_with_first_frame", _open_impl)
    monkeypatch.setattr(main_module, "Detector", _FakeDetector)
    monkeypatch.setattr(main_module, "Tracker", _FakeTracker)
    monkeypatch.setattr(main_module, "LineCounter", _FakeLineCounter)
    monkeypatch.setattr(main_module, "is_ffmpeg_available", lambda: False)

    with pytest.raises(SystemExit, match="ffmpeg executable is not available"):
        _run_main(
            monkeypatch,
            [
                "--backend",
                "motion",
                "--input",
                str(input_path),
                "--output",
                "media/output_test.mp4",
                "--output-encoder",
                "ffmpeg",
                "--no-draw",
                "--no-progress",
                "--max-frames",
                "1",
            ],
        )


def test_main_releases_writer_with_interrupt_flag_after_ctrl_c(monkeypatch, tmp_path) -> None:
    fake_cap = _FakeCap()
    frame0 = np.zeros((16, 16, 3), dtype=np.uint8)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")

    class _SingleFrameCap(_FakeCap):
        def __init__(self, frame):
            super().__init__()
            self._frame = frame
            self._served = False

        def read(self):
            if self._served:
                return False, None
            self._served = True
            return True, self._frame.copy()

    class _FakeWriter:
        def __init__(self) -> None:
            self.interrupted_arg = None

        def write(self, _frame) -> None:
            return None

        def release(self, *, interrupted: bool = False) -> None:
            self.interrupted_arg = interrupted

    fake_writer = _FakeWriter()

    def _open_impl(source, **_kwargs):
        _ = source
        return _SingleFrameCap(frame0), frame0, 30.0, 16, 16, "FAKE"

    monkeypatch.setattr(main_module, "_open_source_with_first_frame", _open_impl)
    monkeypatch.setattr(main_module, "Detector", _InterruptingDetector)
    monkeypatch.setattr(main_module, "Tracker", _FakeTracker)
    monkeypatch.setattr(main_module, "LineCounter", _FakeLineCounter)
    monkeypatch.setattr(main_module, "create_video_writer", lambda _cfg: fake_writer)
    monkeypatch.setattr(main_module, "is_ffmpeg_available", lambda: True)

    _run_main(
        monkeypatch,
        [
            "--backend",
            "motion",
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "out.mp4"),
            "--output-encoder",
            "ffmpeg",
            "--no-draw",
            "--no-progress",
            "--max-frames",
            "1",
        ],
    )

    assert fake_writer.interrupted_arg is True


def test_file_fast_skip_is_enabled_with_no_write(monkeypatch, tmp_path, capsys) -> None:
    fake_cap = _FakeCap()
    frame0 = np.zeros((16, 16, 3), dtype=np.uint8)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")

    def _open_impl(source, **_kwargs):
        _ = source
        return fake_cap, frame0, 30.0, 16, 16, "FAKE"

    monkeypatch.setattr(main_module, "_open_source_with_first_frame", _open_impl)
    monkeypatch.setattr(main_module, "Detector", _FakeDetector)
    monkeypatch.setattr(main_module, "Tracker", _FakeTracker)
    monkeypatch.setattr(main_module, "LineCounter", _FakeLineCounter)

    _run_main(
        monkeypatch,
        [
            "--backend",
            "motion",
            "--input",
            str(input_path),
            "--frame-stride",
            "2",
            "--fast-skip",
            "--no-write",
            "--no-draw",
            "--no-progress",
            "--max-frames",
            "3",
        ],
    )

    out = capsys.readouterr().out
    assert "fast-skip: enabled" in out
    assert fake_cap.grab_calls == 1


def test_local_file_duplicate_completed_run_skips_startup(monkeypatch, tmp_path, capsys) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    _write_prior_local_run(
        tmp_path,
        run_uid="run_done",
        input_path=input_path,
        camera_id="cam_01",
        video_start="2026-03-11T10:00:00+07:00",
        completed=True,
    )

    def _open_impl(_source, **_kwargs):
        raise AssertionError("duplicate guard should skip before opening the video source")

    monkeypatch.setattr(main_module, "_open_source_with_first_frame", _open_impl)

    _run_main(
        monkeypatch,
        [
            "--backend",
            "motion",
            "--input",
            str(input_path),
            "--spool-dir",
            str(tmp_path),
            "--site-id",
            "site_a",
            "--camera-id",
            "cam_01",
            "--video-start",
            "2026-03-11T10:00:00+07:00",
            "--no-write",
            "--no-draw",
            "--no-progress",
            "--max-frames",
            "1",
        ],
    )

    out = capsys.readouterr().out
    assert "Duplicate local input already processed; skipping startup." in out
    assert "previous_run_uid=run_done" in out
    assert len(list(tmp_path.glob("*/*/run.json"))) == 1


def test_local_file_duplicate_can_be_overridden(monkeypatch, tmp_path) -> None:
    fake_cap = _FakeCap()
    frame0 = np.zeros((16, 16, 3), dtype=np.uint8)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    _write_prior_local_run(
        tmp_path,
        run_uid="run_done",
        input_path=input_path,
        camera_id="cam_01",
        video_start="2026-03-11T10:00:00+07:00",
        completed=True,
    )

    def _open_impl(source, **_kwargs):
        _ = source
        return fake_cap, frame0, 30.0, 16, 16, "FAKE"

    monkeypatch.setattr(main_module, "_open_source_with_first_frame", _open_impl)
    monkeypatch.setattr(main_module, "Detector", _FakeDetector)
    monkeypatch.setattr(main_module, "Tracker", _FakeTracker)
    monkeypatch.setattr(main_module, "LineCounter", _FakeLineCounter)

    _run_main(
        monkeypatch,
        [
            "--backend",
            "motion",
            "--input",
            str(input_path),
            "--spool-dir",
            str(tmp_path),
            "--site-id",
            "site_a",
            "--camera-id",
            "cam_01",
            "--video-start",
            "2026-03-11T10:00:00+07:00",
            "--allow-duplicate-local-input",
            "--no-write",
            "--no-draw",
            "--no-progress",
            "--max-frames",
            "1",
        ],
    )

    assert fake_cap.released is True
    assert len(list(tmp_path.glob("*/*/run.json"))) == 2


def test_local_file_duplicate_check_uses_video_start_identity(monkeypatch, tmp_path) -> None:
    fake_cap = _FakeCap()
    frame0 = np.zeros((16, 16, 3), dtype=np.uint8)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    _write_prior_local_run(
        tmp_path,
        run_uid="run_done",
        input_path=input_path,
        camera_id="cam_01",
        video_start="2026-03-11T10:00:00+07:00",
        completed=True,
    )

    def _open_impl(source, **_kwargs):
        _ = source
        return fake_cap, frame0, 30.0, 16, 16, "FAKE"

    monkeypatch.setattr(main_module, "_open_source_with_first_frame", _open_impl)
    monkeypatch.setattr(main_module, "Detector", _FakeDetector)
    monkeypatch.setattr(main_module, "Tracker", _FakeTracker)
    monkeypatch.setattr(main_module, "LineCounter", _FakeLineCounter)

    _run_main(
        monkeypatch,
        [
            "--backend",
            "motion",
            "--input",
            str(input_path),
            "--spool-dir",
            str(tmp_path),
            "--site-id",
            "site_a",
            "--camera-id",
            "cam_01",
            "--video-start",
            "2026-03-11T11:00:00+07:00",
            "--no-write",
            "--no-draw",
            "--no-progress",
            "--max-frames",
            "1",
        ],
    )

    assert fake_cap.released is True
    assert len(list(tmp_path.glob("*/*/run.json"))) == 2


def test_local_file_duplicate_check_ignores_incomplete_prior_runs(monkeypatch, tmp_path) -> None:
    fake_cap = _FakeCap()
    frame0 = np.zeros((16, 16, 3), dtype=np.uint8)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake")
    _write_prior_local_run(
        tmp_path,
        run_uid="run_incomplete",
        input_path=input_path,
        camera_id="cam_01",
        video_start="2026-03-11T10:00:00+07:00",
        completed=False,
    )

    def _open_impl(source, **_kwargs):
        _ = source
        return fake_cap, frame0, 30.0, 16, 16, "FAKE"

    monkeypatch.setattr(main_module, "_open_source_with_first_frame", _open_impl)
    monkeypatch.setattr(main_module, "Detector", _FakeDetector)
    monkeypatch.setattr(main_module, "Tracker", _FakeTracker)
    monkeypatch.setattr(main_module, "LineCounter", _FakeLineCounter)

    _run_main(
        monkeypatch,
        [
            "--backend",
            "motion",
            "--input",
            str(input_path),
            "--spool-dir",
            str(tmp_path),
            "--site-id",
            "site_a",
            "--camera-id",
            "cam_01",
            "--video-start",
            "2026-03-11T10:00:00+07:00",
            "--no-write",
            "--no-draw",
            "--no-progress",
            "--max-frames",
            "1",
        ],
    )

    assert fake_cap.released is True
    assert len(list(tmp_path.glob("*/*/run.json"))) == 2
