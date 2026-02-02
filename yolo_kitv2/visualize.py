from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .types import Detection


def _color_for_class_id(class_id: Optional[int]) -> Tuple[int, int, int]:
    """
    Deterministic BGR color for a class id (OpenCV expects BGR).
    """

    if class_id is None:
        return (0, 255, 255)

    # Small deterministic palette, then fallback to a seeded RNG for larger IDs.
    palette = [
        (255, 56, 56),
        (255, 157, 151),
        (255, 112, 31),
        (255, 178, 29),
        (207, 210, 49),
        (72, 249, 10),
        (146, 204, 23),
        (61, 219, 134),
        (26, 147, 52),
        (0, 212, 187),
        (44, 153, 168),
        (0, 194, 255),
        (52, 69, 147),
        (100, 115, 255),
        (0, 24, 236),
        (132, 56, 255),
        (82, 0, 133),
        (203, 56, 255),
        (255, 149, 200),
        (255, 55, 199),
    ]
    if 0 <= class_id < len(palette):
        return palette[class_id]

    rng = np.random.default_rng(int(class_id))
    bgr = rng.integers(0, 256, size=3, dtype=np.uint8)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_detections(
    image_bgr: np.ndarray,
    detections: Iterable[Detection],
    *,
    class_names: Optional[Dict[int, str]] = None,
    show_score: bool = True,
    box_thickness: int = 1,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray:
    """
    Draw bounding boxes + labels on an OpenCV BGR image and return a copy.

    Args:
        image_bgr: input image in BGR (H, W, 3).
        detections: iterable of Detection with xyxy in original image coordinates.
        class_names: optional mapping {class_id: class_name}.
    """

    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("OpenCV is required for draw_detections(). Install with `pip install opencv-python`.") from e

    if image_bgr is None or not hasattr(image_bgr, "shape"):
        raise TypeError("image_bgr must be a NumPy array (BGR).")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {getattr(image_bgr, 'shape', None)}")

    out = image_bgr.copy()
    h, w = out.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det.as_xyxy()
        x1i = int(np.clip(round(x1), 0, w - 1))
        y1i = int(np.clip(round(y1), 0, h - 1))
        x2i = int(np.clip(round(x2), 0, w - 1))
        y2i = int(np.clip(round(y2), 0, h - 1))

        color = _color_for_class_id(det.class_id)
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, thickness=box_thickness)

        if det.class_id is None:
            label = "object"
        else:
            label = class_names.get(det.class_id, str(det.class_id)) if class_names else str(det.class_id)

        if show_score:
            label = f"{label} {det.score:.2f}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        # Place label above the box if possible, else inside.
        y_text_top = y1i - th - baseline
        if y_text_top < 0:
            y_text_top = y1i

        x_text_right = min(x1i + tw, w - 1)
        y_text_bottom = min(y_text_top + th + baseline, h - 1)

        cv2.rectangle(out, (x1i, y_text_top), (x_text_right, y_text_bottom), color, thickness=-1)
        cv2.putText(
            out,
            label,
            (x1i, min(y_text_top + th, h - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

    return out

