from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


def open_video_capture(
    source: Union[str, Path],
    *,
    open_timeout_ms: Optional[int] = None,
    read_timeout_ms: Optional[int] = None,
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
