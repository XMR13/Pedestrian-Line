from __future__ import annotations

import numpy as np

from pedestrian_line_counter.draw_utils import draw_tracks
from pedestrian_line_counter.structures import Track


def test_draw_tracks_does_not_crash_when_class_id_is_none() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    tracks = [
        Track(
            track_id=1,
            x1=10,
            y1=10,
            x2=30,
            y2=30,
            score=0.9,
            class_id=None,
            last_seen_frame=0,
        )
    ]
    draw_tracks(frame, tracks, frame_index=0)

