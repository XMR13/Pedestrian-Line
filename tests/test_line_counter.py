import pytest

from pedestrian_line_counter.line_counter import LineCounter
from pedestrian_line_counter.structures import Track


def _make_track(
    track_id: int,
    x: float,
    y: float,
    class_id: int,
    frame_index: int,
) -> Track:
    # Bottom-center at (x, y)
    return Track(
        track_id=track_id,
        x1=x - 5,
        y1=y - 10,
        x2=x + 5,
        y2=y,
        score=1.0,
        class_id=class_id,
        last_seen_frame=frame_index,
    )


def test_counts_a_to_b_once_with_class() -> None:
    # Vertical line at x=50 from top to bottom.
    lc = LineCounter(p1=(50, 0), p2=(50, 100))

    # Track starts on right side (x>50), crosses to left while moving down.
    positions = [
        (60, 10),
        (58, 15),
        (48, 20),  # sign flip here
        (46, 25),
        (44, 30),  # enough post frames to finalise
        (42, 35),
    ]

    events = []
    for i, (x, y) in enumerate(positions):
        # class_id here is an arbitrary "vehicle subclass" ID for testing.
        events.extend(lc.update([_make_track(1, x, y, class_id=2, frame_index=i)], frame_index=i))

    assert lc.count_a_to_b == 1
    assert lc.count_b_to_a == 0
    assert lc.count_by_class_dir["a_to_b"].get(2) == 1
    assert len(events) == 1
    assert events[0].direction == "A_TO_B"
    assert events[0].class_id == 2


def test_counts_b_to_a_once_with_class() -> None:
    lc = LineCounter(p1=(50, 0), p2=(50, 100))

    # Track starts on left side (x<50), crosses to right while moving up.
    positions = [
        (40, 80),
        (42, 75),
        (52, 70),  # sign flip here
        (55, 65),
        (58, 60),  # enough post frames to finalise
        (60, 55),
    ]

    events = []
    for i, (x, y) in enumerate(positions):
        # class_id here is an arbitrary "vehicle subclass" ID for testing.
        events.extend(lc.update([_make_track(2, x, y, class_id=7, frame_index=i)], frame_index=i))

    assert lc.count_a_to_b == 0
    assert lc.count_b_to_a == 1
    assert lc.count_by_class_dir["b_to_a"].get(7) == 1
    assert len(events) == 1
    assert events[0].direction == "B_TO_A"
    assert events[0].class_id == 7
