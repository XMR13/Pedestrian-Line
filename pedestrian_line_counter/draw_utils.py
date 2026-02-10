from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .structures import Track


Color = Tuple[int, int, int]
PALETTE = Tuple[Color, ...]

# Default BGR palette used to color updated tracks.
DEFAULT_TRACK_PALETTE: PALETTE = (
    (46, 204, 113),   # green
    (52, 152, 219),   # blue
    (0, 159, 255),    # orange-ish
    (231, 76, 60),    # red
    (155, 89, 182),   # purple
    (241, 196, 15),   # yellow
    (26, 188, 156),   # turquoise
    (230, 126, 34),   # orange
)

def _to_color(value: Sequence[int] | Color) -> Color:
    if len(value) < 3:
        return (0, 255, 0)
    b = int(max(0, min(255, value[0])))
    g = int(max(0, min(255, value[1])))
    r = int(max(0, min(255, value[2])))
    return (b, g, r)

def _pick_box_color(
    track: Track,
    *,
    default_color: Color,
    class_colors: Optional[Mapping[int, Sequence[int] | Color]],
    palette: Optional[Sequence[Sequence[int] | Color]],
    color_by: str,
) -> Color:
    """
    Resolve a stable color for a track.

    Priority:
    1) Explicit `class_colors[class_id]` override.
    2) Palette color indexed by class_id or track_id.
    3) `default_color`.
    """

    cid = None if track.class_id is None else int(track.class_id)
    if cid is not None and class_colors:
        explicit = class_colors.get(cid)
        if explicit is not None:
            return _to_color(explicit)

    color_key = int(track.track_id)
    if color_by == "class" and cid is not None:
        color_key = cid

    palette_src = palette if palette else DEFAULT_TRACK_PALETTE
    if palette_src:
        return _to_color(palette_src[color_key % len(palette_src)])

    return _to_color(default_color)

def _class_name(class_id: int, class_names: Optional[Mapping[int, str]] = None) -> str:
    if class_names and int(class_id) in class_names:
        return str(class_names[int(class_id)])
    return str(int(class_id))


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    frame_index: int | None = None,
    color: Color = (46, 204, 113),
    stale_max_age: int = 2,
    class_names: Optional[Mapping[int, str]] = None,
    class_colors: Optional[Mapping[int, Sequence[int] | Color]] = None,
    palette: Optional[Sequence[Sequence[int] | Color]] = None,
    color_by: str = "class",
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
            box_color = _pick_box_color(
                track,
                default_color=color,
                class_colors=class_colors,
                palette=palette,
                color_by=color_by,
            )
            thickness = 3
            text_color = (255, 255, 255)
        else:
            box_color = (140, 140, 140)
            thickness = 1
            text_color = (220, 220, 220)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        #  Label yang tertate rapi : "<class>  #<track_id>" (atau hanya "#<track_id>")
        cid = track.class_id
        score = track.score
        disp_id = None
        if cid is not None:
            cls_key = int(cid)
            disp_id = track.display_ids_by_class.get(cls_key)
        if disp_id is None and track.display_id is not None and track.display_class_id == cid:
            disp_id = track.display_id
        if disp_id is None:
            disp_id = track.track_id

        if cid is not None:
            cls = _class_name(int(cid), class_names=class_names)
            text = f"{cls} | acc : {score:.2f}"
        else:
            text = f"#{disp_id}"

        if text:
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2
            )
            # Background box for readability
            cv2.rectangle(
                frame,
                (x1, y1 - th - 6),
                (x1 + tw + 10, y1 + 10),
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
                2,
                cv2.LINE_AA,
            )


def draw_line_and_counts(
    frame: np.ndarray,
    line_counter: Any,
    color: Color = (0, 255, 255),
    class_names: Optional[Mapping[int, str]] = None,
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
        line_counter.count_by_class_dir.get("a_to_b", {}), class_names=class_names
    )
    b_to_a_text = _format_class_counts_dir(
        line_counter.count_by_class_dir.get("b_to_a", {}), class_names=class_names
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


def _format_class_counts_dir(counts: Dict[int, int], class_names: Optional[Mapping[int, str]] = None) -> str:
    if not counts:
        return ""

    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
    parts = []
    for cid, val in items:
        name = _class_name(int(cid), class_names=class_names)
        parts.append(f"{name} {val}")
    return " | ".join(parts)
