from __future__ import annotations

from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from .line_counter import LineCounter
from .structures import Track


Color = Tuple[int, int, int]

# COCO class names for labeling boxes (0–79).
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    frame_index: int | None = None,
    color: Color = (0, 255, 0),
) -> None:
    """
    Draw bounding boxes for tracks on the frame in-place.

    Track IDs are kept internally for counting but are not drawn, to keep
    the visualization focused on the objects themselves.
    """

    for track in tracks:
        # Optionally draw only tracks that were updated on this frame.
        if frame_index is not None and track.last_seen_frame != frame_index:
            continue
        x1, y1, x2, y2 = map(int, track.as_xyxy())
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Optional label with class name/id and score
        label = None
        if track.class_id is not None and 0 <= track.class_id < len(COCO_NAMES):
            label = COCO_NAMES[track.class_id]
        elif track.class_id is not None:
            label = str(track.class_id)

        if label is not None:
            text = f"{label}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1
            )
            # Background box for readability
            cv2.rectangle(
                frame,
                (x1, y1 - th - 6),
                (x1 + tw + 4, y1),
                (0, 0, 0),
                thickness=-1,
            )
            # Text
            cv2.putText(
                frame,
                text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


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
    a_to_b_text = _format_class_counts_dir(
        line_counter.count_by_class_dir.get("a_to_b", {})
    )
    b_to_a_text = _format_class_counts_dir(
        line_counter.count_by_class_dir.get("b_to_a", {})
    )

    # Letakkan text di atas kiri frame tersebut
    x, y = 18, 28
    lines = [(text, (255, 255, 255))]
    if a_to_b_text:
        lines.append((f"A->B top: {a_to_b_text}", (180, 250, 180)))
    if b_to_a_text:
        lines.append((f"B->A top: {b_to_a_text}", (250, 220, 180)))

    for idx, (line, color_text) in enumerate(lines):
        y_line = y + idx * 22
        (tw, th), _ = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, thickness=2
        )
        cv2.rectangle(
            frame, (x - 6, y_line - th - 6), (x + tw + 6, y_line + 6), (0, 0, 0), -1
        )
        cv2.putText(
            frame,
            line,
            (x, y_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color_text,
            2,
            cv2.LINE_AA,
        )


def _format_class_counts_dir(counts: Dict[int, int]) -> str:
    if not counts:
        return ""

    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
    parts = []
    for cid, val in items:
        name = COCO_NAMES[cid] if 0 <= cid < len(COCO_NAMES) else str(cid)
        parts.append(f"{name} {val}")
    return " | ".join(parts)
