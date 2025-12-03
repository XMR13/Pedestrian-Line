from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from line_counter import LineCounter
from structures import Track


Color = Tuple[int, int, int]


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    color: Color = (0, 255, 0),
) -> None:
    """
    Draw bounding boxes for tracks on the frame in-place.

    Track IDs are kept internally for counting but are not drawn, to keep
    the visualization focused on the objects themselves.
    """

    for track in tracks:
        x1, y1, x2, y2 = map(int, track.as_xyxy())
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_line_and_counts(
    frame: np.ndarray,
    line_counter: LineCounter,
    color: Color = (0, 255, 255),
) -> None:
    """
    Draw the virtual line and the current counts on the frame in-place.
    """

    p1 = tuple(map(int, line_counter.p1))
    p2 = tuple(map(int, line_counter.p2))

    cv2.line(frame, p1, p2, color, 2)

    text = f"A->B: {line_counter.count_a_to_b} | B->A: {line_counter.count_b_to_a}"

    # Put the text near the top-left corner of the frame
    x, y = 20, 30
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
    )
    cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
