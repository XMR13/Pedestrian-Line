from __future__ import annotations

from typing import List, Tuple

from pedestrian_line_counter import videoio_utils


def test_build_jetson_rtsp_gstreamer_pipeline_h264() -> None:
    pipeline = videoio_utils.build_jetson_rtsp_gstreamer_pipeline(
        "rtsp://user:pass@cam/stream",
        codec="h264",
        transport="tcp",
        latency_ms=250,
        appsink_drop=True,
        appsink_max_buffers=1,
    )
    assert "rtspsrc" in pipeline
    assert "protocols=tcp" in pipeline
    assert "latency=250" in pipeline
    assert "rtph264depay ! h264parse" in pipeline
    assert "nvv4l2decoder" in pipeline
    assert "appsink" in pipeline
    assert "drop=true" in pipeline


def test_build_jetson_rtsp_gstreamer_pipeline_h265() -> None:
    pipeline = videoio_utils.build_jetson_rtsp_gstreamer_pipeline(
        "rtsp://cam/stream",
        codec="h265",
        transport="udp",
        latency_ms=100,
        appsink_drop=False,
        appsink_max_buffers=8,
    )
    assert "protocols=udp" in pipeline
    assert "rtph265depay ! h265parse" in pipeline
    assert "drop=false" in pipeline
    assert "max-buffers=8" in pipeline


def test_open_video_capture_gstreamer_fallbacks_to_ffmpeg(monkeypatch) -> None:
    calls: List[Tuple[object, ...]] = []

    class _FakeCapture:
        def __init__(self, opened: bool) -> None:
            self._opened = opened
            self.released = False

        def isOpened(self) -> bool:
            return self._opened

        def release(self) -> None:
            self.released = True

        def set(self, _prop: int, _value: float) -> bool:
            return True

        def open(self, _source: str, _api: int) -> bool:
            self._opened = True
            return True

    def _factory(*args):
        calls.append(tuple(args))
        if len(args) >= 2 and args[1] == videoio_utils.cv2.CAP_GSTREAMER:
            return _FakeCapture(opened=False)
        if len(args) >= 2 and args[1] == videoio_utils.cv2.CAP_FFMPEG:
            return _FakeCapture(opened=True)
        return _FakeCapture(opened=False)

    monkeypatch.setattr(videoio_utils.cv2, "VideoCapture", _factory)

    cap = videoio_utils.open_video_capture(
        "rtsp://cam/stream",
        rtsp_capture_backend="gstreamer",
    )

    assert cap.isOpened()
    assert len(calls) >= 2
    assert calls[0][1] == videoio_utils.cv2.CAP_GSTREAMER
    assert isinstance(calls[0][0], str)
    assert "nvv4l2decoder" in str(calls[0][0])
    assert calls[1][1] == videoio_utils.cv2.CAP_FFMPEG


def test_open_video_capture_gstreamer_success_stops_fallback(monkeypatch) -> None:
    calls: List[Tuple[object, ...]] = []

    class _FakeCapture:
        def __init__(self, opened: bool) -> None:
            self._opened = opened

        def isOpened(self) -> bool:
            return self._opened

        def release(self) -> None:
            return None

        def set(self, _prop: int, _value: float) -> bool:
            return True

        def open(self, _source: str, _api: int) -> bool:
            return True

    def _factory(*args):
        calls.append(tuple(args))
        if len(args) >= 2 and args[1] == videoio_utils.cv2.CAP_GSTREAMER:
            return _FakeCapture(opened=True)
        return _FakeCapture(opened=False)

    monkeypatch.setattr(videoio_utils.cv2, "VideoCapture", _factory)

    cap = videoio_utils.open_video_capture(
        "rtsp://cam/stream",
        rtsp_capture_backend="gstreamer",
    )

    assert cap.isOpened()
    assert len(calls) == 1
    assert calls[0][1] == videoio_utils.cv2.CAP_GSTREAMER
