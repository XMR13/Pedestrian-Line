from __future__ import annotations

from pedestrian_line_counter.config import TrackerConfig
from pedestrian_line_counter.structures import Detection
from pedestrian_line_counter.tracker import Tracker


def _det(x: float, class_id: int, score: float = 0.9) -> Detection:
    return Detection(
        x1=x,
        y1=10.0,
        x2=x + 20.0,
        y2=40.0,
        score=score,
        class_id=class_id,
    )


def test_weighted_majority_keeps_stable_class_when_last_frames_jitter() -> None:
    tracker = Tracker(
        TrackerConfig(
            class_vote_mode="weighted_majority",
            class_vote_min_frames=1,
            class_vote_ambiguity_ratio=1.05,
        )
    )

    track = None
    frame = 0
    for x in (100, 102, 104, 106, 108, 110):
        track = tracker.update([_det(float(x), class_id=2, score=0.90)], frame)[0]
        frame += 1
    for x in (112, 114):
        track = tracker.update([_det(float(x), class_id=5, score=0.40)], frame)[0]
        frame += 1

    assert track is not None
    assert track.class_id == 5
    assert track.stable_class_id == 2


def test_instant_mode_matches_latest_class() -> None:
    tracker = Tracker(
        TrackerConfig(
            class_vote_mode="instant",
            class_vote_min_frames=1,
            class_vote_ambiguity_ratio=1.1,
        )
    )

    t0 = tracker.update([_det(100.0, class_id=2, score=0.9)], 0)[0]
    stable0 = t0.stable_class_id
    t1 = tracker.update([_det(102.0, class_id=5, score=0.9)], 1)[0]

    assert stable0 == 2
    assert t1.class_id == 5
    assert t1.stable_class_id == 5


def test_ambiguity_ratio_prevents_unstable_class_flip() -> None:
    tracker = Tracker(
        TrackerConfig(
            class_vote_mode="majority",
            class_vote_min_frames=1,
            class_vote_ambiguity_ratio=1.6,
        )
    )

    sequence = [2, 2, 5, 5, 5]
    track = None
    for i, cid in enumerate(sequence):
        track = tracker.update([_det(100.0 + (i * 2.0), class_id=cid, score=0.9)], i)[0]

    assert track is not None
    assert track.class_id == 5
    assert track.stable_class_id == 2
