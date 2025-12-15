from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np

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
    stale_max_age: int = 2,
) -> None:
    """
    Menggambar bounding box di track on per frame jika objek tersebut terdeteksi

    Track ID digambar denagn label yang compact dan membantu mendiagnosis pergantian ID 
    dan under/over count, terutama pada saat kondisi ketika banyak kendaraan yang melewati garis
    """

    for track in tracks:
        is_updated = frame_index is None or track.last_seen_frame == frame_index
        if not is_updated and frame_index is not None:
            age = int(frame_index - track.last_seen_frame)
            if age > int(max(stale_max_age, 0)):
                continue

        x1, y1, x2, y2 = map(int, track.as_xyxy())
        if is_updated:
            box_color = color
            thickness = 2
            text_color = (255, 255, 255)
        else:
            box_color = (140, 140, 140)
            thickness = 1
            text_color = (220, 220, 220)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        #  Label yang tertate rapi : "<class>  #<track_id>" (atau hanya "#<track_id>")
        cid = track.class_id
        disp_ud = None
        if cid is not None:
            cls_key = int(cid)
            disp_id = track.display_ids_by_class.get(cls_key)
        if disp_id is None and track.display_id is not None and track.display_class_id == cid:
            disp_id = track.display_id
        if disp_id == None:
            disp_id = track.track_id

        if cid is not None and 0 <= cid < len(COCO_NAMES):
            cls = COCO_NAMES[cid]
            text = f"{cls}-{disp_id}"
        elif cid is not None:
            text = f"{cid}-{disp_id}"
        else:
            text = f"#{disp_id}"

        if text:
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
                text_color,
                1,
                cv2.LINE_AA,
            )


def draw_line_and_counts(
    frame: np.ndarray,
    line_counter: Any,
    color: Color = (0, 255, 255),
) -> None:
    """
    Menggambar garis vritual dan perhitungan sekarang dengan frame yang ditempat (in-place)
    """

    lines = None
    if hasattr(line_counter, "lines"):
        try:
            lines = list(line_counter.lines)
        except Exception:
            lines = None
    if not lines and hasattr(line_counter, "p1") and hasattr(line_counter, "p2"):
        lines = [(line_counter.p1, line_counter.p2)]

    if lines:
        line_colors = [color, (255, 255, 0)]
        for idx, (p1_raw, p2_raw) in enumerate(lines):
            p1 = tuple(map(int, p1_raw))
            p2 = tuple(map(int, p2_raw))
            c = line_colors[idx % len(line_colors)]
            cv2.line(frame, p1, p2, c, 2)

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
