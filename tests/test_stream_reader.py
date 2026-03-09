from __future__ import annotations

import time

import numpy as np

from pedestrian_line_counter.stream_reader import StreamReader


class _FakeCapture:
    def __init__(self, *, frames: int, delay_s: float = 0.0) -> None:
        self._frames_left = int(frames)
        self._delay_s = float(delay_s)
        self._released = False
        self._index = 0

    def isOpened(self) -> bool:  # pragma: no cover - not used by StreamReader
        return not self._released

    def read(self):
        if self._released:
            return False, None
        if self._frames_left <= 0:
            return False, None
        if self._delay_s:
            time.sleep(self._delay_s)
        img = np.full((2, 2, 3), self._index % 255, dtype=np.uint8)
        self._index += 1
        self._frames_left -= 1
        return True, img

    def release(self) -> None:
        self._released = True


def test_stream_reader_drops_oldest_when_full() -> None:
    cap = _FakeCapture(frames=6, delay_s=0.001)
    reader = StreamReader(cap, queue_size=2)
    reader.start()

    # Wait for the producer to hit end-of-stream so the queue should hold only the last frames.
    deadline = time.time() + 1.0
    while reader.read_failures == 0 and time.time() < deadline:
        time.sleep(0.001)
    assert reader.read_frames == 6

    items = []
    while True:
        item = reader.get(timeout_s=0.1)
        if item is None:
            break
        items.append(item)

    reader.stop()

    assert reader.read_frames == 6
    assert len(items) == 2
    assert [it.source_index for it in items] == [4, 5]
    assert reader.dropped == 4


def test_stream_reader_block_policy_preserves_order_without_drops() -> None:
    cap = _FakeCapture(frames=5, delay_s=0.0005)
    reader = StreamReader(cap, queue_size=1, overflow_policy="block")
    reader.start()

    items = []
    deadline = time.time() + 2.0
    while time.time() < deadline:
        poll = reader.get_with_status(timeout_s=0.05)
        if poll.status == "frame" and poll.item is not None:
            items.append(poll.item)
            continue
        if poll.status == "stopped":
            break

    reader.stop()

    assert [it.source_index for it in items] == [0, 1, 2, 3, 4]
    assert reader.read_frames == 5
    assert reader.dropped == 0
