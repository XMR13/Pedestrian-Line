from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


def is_rtsp_source(source: Union[str, Path]) -> bool:
    src = str(source).strip().lower()
    return src.startswith("rtsp://") or src.startswith("rtsps://")


def build_jetson_rtsp_gstreamer_pipeline(
    rtsp_url: str,
    *,
    codec: str = "h264",
    transport: str = "tcp",
    latency_ms: int = 200,
    appsink_drop: bool = True,
    appsink_max_buffers: int = 1,
) -> str:
    """
    Build a Jetson-friendly RTSP GStreamer pipeline for OpenCV CAP_GSTREAMER.

    Uses NVIDIA decode (`nvv4l2decoder`) and converts frames to BGR for OpenCV.
    """

    codec_norm = str(codec).strip().lower()
    transport_norm = str(transport).strip().lower()
    if codec_norm not in {"h264", "h265"}:
        raise ValueError("codec must be 'h264' or 'h265'")
    if transport_norm not in {"tcp", "udp"}:
        raise ValueError("transport must be 'tcp' or 'udp'")

    depay = "rtph264depay ! h264parse" if codec_norm == "h264" else "rtph265depay ! h265parse"
    url_escaped = str(rtsp_url).replace('"', '\\"')
    sink_drop = "true" if appsink_drop else "false"
    max_buf = max(int(appsink_max_buffers), 1)
    latency = max(int(latency_ms), 0)

    return (
        f'rtspsrc location="{url_escaped}" protocols={transport_norm} latency={latency} ! '
        f"{depay} ! nvv4l2decoder ! nvvidconv ! "
        "video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
        f"appsink sync=false max-buffers={max_buf} drop={sink_drop}"
    )


def open_video_capture(
    source: Union[str, Path],
    *,
    open_timeout_ms: Optional[int] = None,
    read_timeout_ms: Optional[int] = None,
    rtsp_capture_backend: str = "opencv",
    rtsp_transport: str = "tcp",
    rtsp_latency_ms: int = 200,
    rtsp_codec: str = "h264",
    rtsp_gst_pipeline: Optional[str] = None,
    rtsp_appsink_drop: bool = True,
    rtsp_appsink_max_buffers: int = 1,
) -> cv2.VideoCapture:
    """
    Menjalankan video dengan openCV backend
    Open a video source robustly across OpenCV backends.

    Pada beberapa build/CAP_GSTREAMER bisa mengeluarkan error untuk beberapa file path 
    (terkhususnya dengan spaces) ketika diberikan suatu nama. Menggunakan ffmpeg jika tersedia, 
    apabila tidak, maka gunakan CAP_ANY

    Supports:
    - Local file paths
    - Stream URLs (e.g. RTSP)
    """

    source_str = str(source)
    backend_norm = str(rtsp_capture_backend).strip().lower()
    if backend_norm not in {"opencv", "gstreamer"}:
        raise ValueError("rtsp_capture_backend must be 'opencv' or 'gstreamer'")

    def _set_timeouts(cap: cv2.VideoCapture) -> None:
        if open_timeout_ms is not None and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(open_timeout_ms))
            except Exception:
                pass
        if read_timeout_ms is not None and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            try:
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(read_timeout_ms))
            except Exception:
                pass

    if is_rtsp_source(source_str) and backend_norm == "gstreamer":
        pipeline = str(rtsp_gst_pipeline).strip() if rtsp_gst_pipeline else ""
        if pipeline == "":
            pipeline = build_jetson_rtsp_gstreamer_pipeline(
                source_str,
                codec=rtsp_codec,
                transport=rtsp_transport,
                latency_ms=int(rtsp_latency_ms),
                appsink_drop=bool(rtsp_appsink_drop),
                appsink_max_buffers=int(rtsp_appsink_max_buffers),
            )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()

    # Prefer FFPMEG jika tersedia.
    if open_timeout_ms is not None or read_timeout_ms is not None:
        cap = cv2.VideoCapture()
        _set_timeouts(cap)
        try:
            cap.open(source_str, cv2.CAP_FFMPEG)
        except Exception:
            pass
        if cap.isOpened():
            return cap
        cap.release()
    else:
        cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        cap.release()

    if open_timeout_ms is not None or read_timeout_ms is not None:
        cap = cv2.VideoCapture()
        _set_timeouts(cap)
        try:
            cap.open(source_str, cv2.CAP_ANY)
        except Exception:
            pass
        return cap

    cap = cv2.VideoCapture(source_str, cv2.CAP_ANY)
    return cap


def read_first_frame(path: Path) -> Tuple[Optional[np.ndarray], Optional[cv2.VideoCapture]]:
    """
    Membaca frame pertama dari video, mengemnalikan frame dan membuka capture
    Caller memiliki capture tersebut dan harus mengembalikannya setelah selesai
    """

    cap = open_video_capture(path)
    if not cap.isOpened():
        return None, None
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None, None
    return frame, cap
