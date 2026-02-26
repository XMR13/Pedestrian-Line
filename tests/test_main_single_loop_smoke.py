import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

import pedestrian_line_counter.main as main_module
from pedestrian_line_counter.portal_uploader import SyncSummary
from pedestrian_line_counter.stream_reader import ReaderPoll


class _FakeCap:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, _prop: int) -> float:
        return 30.0

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
