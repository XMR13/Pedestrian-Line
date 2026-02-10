from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Literal, Optional, Protocol, Tuple, runtime_checkable

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


ReaderPollStatus = Literal["frame", "timeout", "stopped"]

# Terminal reasons for live reader shutdown.
STOP_REASON_STOPPED_BY_CALLER = "stopped_by_caller"
STOP_REASON_READ_FAILED = "read_failed"
STOP_REASON_READ_EXCEPTION = "read_exception"


@dataclass(frozen=True)
class ReaderPoll:
    item: Optional[FrameItem]
    status: ReaderPollStatus
    reason: Optional[str] = None


class StreamReader:
    """
    Background frame reader for live streams (e.g. RTSP).

    Uses a bounded queue and drops the oldest frame when full to keep the
    processing loop close to real-time.
    """

    def __init__(self, cap: _CaptureLike, *, queue_size: int = 3) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be > 0")
        self._cap = cap
        self._queue: queue.Queue[FrameItem] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._terminal_reason_lock = threading.Lock()
        self._terminal_reason: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self.dropped: int = 0
        self.read_failures: int = 0
        self.read_frames: int = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="StreamReader", daemon=True)
        self._thread.start()

    @property
    def terminal_reason(self) -> Optional[str]:
        with self._terminal_reason_lock:
            return self._terminal_reason

    def stop(
        self,
        *,
        join_timeout_s: float = 2.0,
        reason: str = STOP_REASON_STOPPED_BY_CALLER,
    ) -> None:
        self._request_stop(reason)
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
        try:
            self._cap.release()
        except Exception:
            pass

    def get(self, *, timeout_s: float = 1.0) -> Optional[FrameItem]:
        while True:
            poll = self.get_with_status(timeout_s=timeout_s)
            if poll.status == "frame":
                return poll.item
            if poll.status == "stopped":
                return None

    def get_with_status(self, *, timeout_s: float = 1.0) -> ReaderPoll:
        if self._stop.is_set() and self._queue.empty():
            return ReaderPoll(item=None, status="stopped", reason=self.terminal_reason)
        try:
            item = self._queue.get(timeout=timeout_s)
            return ReaderPoll(item=item, status="frame")
        except queue.Empty:
            if self._stop.is_set():
                return ReaderPoll(item=None, status="stopped", reason=self.terminal_reason)
            return ReaderPoll(item=None, status="timeout")

    def _request_stop(self, reason: str) -> None:
        with self._terminal_reason_lock:
            if self._terminal_reason is None:
                self._terminal_reason = reason
        self._stop.set()

    def _run(self) -> None:
        source_index = 0
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except Exception:
                ok, frame = False, None
                self.read_failures += 1
                self._request_stop(STOP_REASON_READ_EXCEPTION)
                break

            if not ok or frame is None:
                self.read_failures += 1
                self._request_stop(STOP_REASON_READ_FAILED)
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
