from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def open_video_capture(path: Path) -> cv2.VideoCapture:
    """
    Open a video file robustly across OpenCV backends.

    On some builds/platforms, CAP_GSTREAMER can struggle with certain file paths
    (notably those with spaces) when given as a plain filename. Prefer FFmpeg
    when available, then fall back to CAP_ANY.
    """

    # Prefer FFmpeg when present.
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    cap.release()

    cap = cv2.VideoCapture(str(path), cv2.CAP_ANY)
    return cap


def read_first_frame(path: Path) -> Tuple[Optional[np.ndarray], Optional[cv2.VideoCapture]]:
    """
    Read the first frame of a video, returning both the frame and the opened capture.
    The caller owns the capture and should release it when done.
    """

    cap = open_video_capture(path)
    if not cap.isOpened():
        return None, None
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None, None
    return frame, cap

