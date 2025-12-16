from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class _CaptureLike(Protocol):
    def read(self) -> Tuple[bool, np.ndarray]: ...

    def isOpened(self) -> bool: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class FrameItem:
    frame: np.ndarray
    timestamp: float
    source_index: int


class StreamReader:
    """
    Frame reader untuk kamera live (e.g RTSP)

    Reader menjaga adanya queuq untuk frame terlama ketika penuh sehingga loop processing
    terus terjaga dan mendekati waktu secara real time.
    Background frame reader for live streams (e.g. RTSP).
    """

    def __init__(self, cap: _CaptureLike, *, queue_size: int = 3) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be > 0")
        self._cap = cap
        self._queue: queue.Queue[FrameItem] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.dropped: int = 0
        self.read_failures: int = 0
        self.read_frames: int = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="StreamReader", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
        try:
            self._cap.release()
        except Exception:
            pass

    def get(self, *, timeout_s: float = 1.0) -> Optional[FrameItem]:
        """
        Mengembalikan queue frame, atau tidak jika reader stop dan tidak ada frame
        yang pendiing.
        """

        while True:
            if self._stop.is_set() and self._queue.empty():
                return None
            try:
                return self._queue.get(timeout=timeout_s)
            except queue.Empty:
                if self._stop.is_set():
                    return None

    def _run(self) -> None:
        source_index = 0
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                self.read_failures += 1
                self._stop.set()
                break

            item = FrameItem(frame=frame, timestamp=time.time(), source_index=source_index)
            source_index += 1
            self.read_frames += 1

            if self._queue.full():
                try:
                    _ = self._queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self.dropped += 1
                continue

