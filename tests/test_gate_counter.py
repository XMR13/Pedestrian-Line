from pedestrian_line_counter.line_counter import TwoLineGateCounter
from pedestrian_line_counter.structures import Track


def _make_track(track_id: int, x: float, y: float, class_id: int, frame_index: int) -> Track:
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


def test_gate_counts_a_to_b_when_crossing_line1_then_line2() -> None:
    # Two parallel vertical lines: x=40 (line1) and x=60 (line2).
    gate = TwoLineGateCounter(
        line1_p1=(40, 0),
        line1_p2=(40, 100),
        line2_p1=(60, 0),
        line2_p2=(60, 100),
        confirm_frames=2,
        max_gap_frames=30,
    )

    positions = [
        (30, 50),
        (38, 50),
        (42, 50),  # cross line1 candidate starts
        (45, 50),  # line1 crossing confirmed here
        (55, 50),
        (62, 50),  # cross line2 candidate starts
        (65, 50),  # line2 crossing confirmed here => A->B
        (70, 50),
    ]

    for i, (x, y) in enumerate(positions):
        # class_id here is an arbitrary "vehicle subclass" ID for testing.
        gate.update([_make_track(1, x, y, class_id=2, frame_index=i)], frame_index=i)

    assert gate.count_a_to_b == 1
    assert gate.count_b_to_a == 0
    assert gate.count_by_class_dir["a_to_b"].get(2) == 1


def test_gate_counts_b_to_a_when_crossing_line2_then_line1() -> None:
    gate = TwoLineGateCounter(
        line1_p1=(40, 0),
        line1_p2=(40, 100),
        line2_p1=(60, 0),
        line2_p2=(60, 100),
        confirm_frames=2,
        max_gap_frames=30,
    )

    positions = [
        (70, 50),
        (62, 50),
        (58, 50),  # cross line2 candidate starts
        (55, 50),  # line2 crossing confirmed here
        (45, 50),
        (38, 50),  # cross line1 candidate starts
        (35, 50),  # line1 crossing confirmed here => B->A
    ]

    for i, (x, y) in enumerate(positions):
        # class_id here is an arbitrary "vehicle subclass" ID for testing.
        gate.update([_make_track(2, x, y, class_id=7, frame_index=i)], frame_index=i)

    assert gate.count_a_to_b == 0
    assert gate.count_b_to_a == 1
    assert gate.count_by_class_dir["b_to_a"].get(7) == 1


def test_gate_does_not_count_if_only_one_line_crossed() -> None:
    gate = TwoLineGateCounter(
        line1_p1=(40, 0),
        line1_p2=(40, 100),
        line2_p1=(60, 0),
        line2_p2=(60, 100),
        confirm_frames=2,
        max_gap_frames=30,
    )

    positions = [
        (30, 50),
        (42, 50),  # cross line1 candidate starts
        (45, 50),  # line1 crossing confirmed
        (48, 50),
        (50, 50),
        (52, 50),
    ]

    for i, (x, y) in enumerate(positions):
        gate.update([_make_track(3, x, y, class_id=3, frame_index=i)], frame_index=i)

    assert gate.count_a_to_b == 0
    assert gate.count_b_to_a == 0
