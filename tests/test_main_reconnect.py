from __future__ import annotations

import sys
import time
from typing import Callable, List, Optional

import numpy as np

import pedestrian_line_counter.main as main_module
from pedestrian_line_counter.stream_reader import ReaderPoll


class _FakeCap:
    def __init__(self, name: str) -> None:
        self.name = name
        self.released = False

    def isOpened(self) -> bool:  # pragma: no cover - used by contract only
        return True

    def get(self, _prop: int) -> float:
        return 0.0

    def read(self):  # pragma: no cover - live path uses StreamReader
        return False, None

    def release(self) -> None:
        self.released = True

    def set(self, _prop: int, _value: float) -> bool:  # pragma: no cover - file mode only
        return True

    def grab(self) -> bool:  # pragma: no cover - file fast-skip only
        return False


def _install_live_fakes(
    monkeypatch,
    *,
    open_impl: Callable[..., tuple[_FakeCap, np.ndarray, float, int, int, Optional[str]]],
    reader_plans: List[List[ReaderPoll]],
):
    class _FakeDetector:
        def __init__(self, _cfg) -> None:
            pass

        def detect(self, _frame):
            return []

    class _FakeTracker:
        instances: List["_FakeTracker"] = []

        def __init__(self, _cfg) -> None:
            self.clear_calls = 0
            _FakeTracker.instances.append(self)

        def update(self, _detections, _frame_index):
            return []

        def clear_runtime_state(self) -> None:
            self.clear_calls += 1

    class _FakeLineCounter:
        instances: List["_FakeLineCounter"] = []

        def __init__(self, **_kwargs) -> None:
            self.count_a_to_b = 7
            self.count_b_to_a = 5
            self.count_by_class_dir = {"a_to_b": {1: 7}, "b_to_a": {2: 5}}
            self.clear_calls = 0
            _FakeLineCounter.instances.append(self)

        def update(self, _tracks, frame_index=0):
            _ = frame_index
            return []

        def clear_runtime_state(self) -> None:
            self.clear_calls += 1

    class _FakeStreamReader:
        instances: List["_FakeStreamReader"] = []

        def __init__(self, cap, *, queue_size: int = 3, overflow_policy: str = "drop_oldest") -> None:
            _ = queue_size
            _ = overflow_policy
            self._cap = cap
            self.stop_reasons: List[str] = []
            self.read_frames = 0
            idx = len(_FakeStreamReader.instances)
            self._plan = list(reader_plans[idx]) if idx < len(reader_plans) else []
            _FakeStreamReader.instances.append(self)

        def start(self) -> None:
            return None

        def get_with_status(self, *, timeout_s: float = 1.0) -> ReaderPoll:
            _ = timeout_s
            if self._plan:
                poll = self._plan.pop(0)
                if poll.status == "timeout":
                    # Simulate real poll blocking so stall-timeout logic can elapse.
                    time.sleep(max(float(timeout_s), 0.0))
                return poll
            return ReaderPoll(item=None, status="stopped", reason="fake_eos")

        def stop(self, *, join_timeout_s: float = 2.0, reason: str = "stopped_by_caller") -> None:
            _ = join_timeout_s
            self.stop_reasons.append(reason)
            self._cap.release()

    monkeypatch.setattr(main_module, "Detector", _FakeDetector)
    monkeypatch.setattr(main_module, "Tracker", _FakeTracker)
    monkeypatch.setattr(main_module, "LineCounter", _FakeLineCounter)
    monkeypatch.setattr(main_module, "StreamReader", _FakeStreamReader)
    monkeypatch.setattr(main_module, "_open_source_with_first_frame", open_impl)

    return _FakeTracker, _FakeLineCounter, _FakeStreamReader


def _run_live_main(monkeypatch, extra_args: List[str]) -> None:
    argv = [
        "prog",
        "--rtsp-url",
        "rtsp://unit-test",
        "--no-write",
        "--no-draw",
        "--no-progress",
        "--rtsp-reconnect-max-attempts",
        "2",
        "--rtsp-reconnect-initial-delay",
        "0.001",
        "--rtsp-reconnect-max-delay",
        "0.001",
        "--rtsp-reconnect-backoff",
        "1.0",
    ] + extra_args
    monkeypatch.setattr(sys, "argv", argv)
    main_module.main()


def test_live_timeout_reconnect_resets_transient_state_and_preserves_totals(monkeypatch, capsys) -> None:
    open_calls: List[str] = []
    created_caps: List[_FakeCap] = []

    def _open_impl(source, **_kwargs):
        open_calls.append(str(source))
        cap = _FakeCap(f"cap{len(open_calls)}")
        created_caps.append(cap)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        return cap, frame, 30.0, 8, 8, "FAKE"

    tracker_cls, line_counter_cls, reader_cls = _install_live_fakes(
        monkeypatch,
        open_impl=_open_impl,
        reader_plans=[[ReaderPoll(item=None, status="timeout")], []],
    )

    _run_live_main(
        monkeypatch,
        [
            "--max-frames",
            "2",
            "--rtsp-stall-timeout",
            "0.000001",
        ],
    )

    out = capsys.readouterr().out
    assert len(open_calls) == 2
    assert tracker_cls.instances[0].clear_calls == 1
    assert line_counter_cls.instances[0].clear_calls == 1
    assert line_counter_cls.instances[0].count_a_to_b == 7
    assert line_counter_cls.instances[0].count_b_to_a == 5
    assert "reconnect:stall_timeout" in reader_cls.instances[0].stop_reasons
    assert "Reconnect state reset: transient tracker/counter state cleared; totals preserved." in out
    assert "[main] Done. A->B: 7, B->A: 5" in out
    assert reader_cls.instances[1].stop_reasons
    assert created_caps[0].released is True
    assert created_caps[1].released is True


def test_live_reconnect_attempts_exhausted_stops(monkeypatch, capsys) -> None:
    open_calls: List[str] = []
    first_cap = _FakeCap("cap0")

    def _open_impl(source, **_kwargs):
        open_calls.append(str(source))
        if len(open_calls) == 1:
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            return first_cap, frame, 30.0, 8, 8, "FAKE"
        raise RuntimeError("forced reconnect failure")

    tracker_cls, line_counter_cls, reader_cls = _install_live_fakes(
        monkeypatch,
        open_impl=_open_impl,
        reader_plans=[[ReaderPoll(item=None, status="stopped", reason="read_failed")]],
    )

    _run_live_main(
        monkeypatch,
        [
            "--max-frames",
            "20",
            "--rtsp-stall-timeout",
            "0.000001",
        ],
    )

    out = capsys.readouterr().out
    assert len(open_calls) == 3
    assert "reconnect:read_failed" in reader_cls.instances[0].stop_reasons
    assert tracker_cls.instances[0].clear_calls == 0
    assert line_counter_cls.instances[0].clear_calls == 0
    assert "Reconnect cycle 1 exhausted attempts; stopping." in out
    assert first_cap.released is True
