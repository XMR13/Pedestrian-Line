from typing import Tuple

import numpy as np


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
):
    """
    Resize and pad image to meet stride-multiple constraints, matching common YOLO exports.

    Returns:
        padded: resized + padded image
        ratio: (w_ratio, h_ratio)
        pad: (dw, dh) padding applied to width/height (left/top only; right/bottom equal)
    """
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("OpenCV is required for letterbox(). Install with `pip install opencv-python`.") from e

    shape = image.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h, w = shape
    new_w, new_h = new_shape

    # Scale ratio (new / old)
    r = min(new_w / w, new_h / h)
    if not scaleup:  # only scale down
        r = min(r, 1.0)

    # Compute padding
    ratio = (r, r)
    resized_w, resized_h = int(round(w * r)), int(round(h * r))
    dw, dh = new_w - resized_w, new_h - resized_h

    if auto:  # make sure padding is a multiple of stride
        dw %= stride
        dh %= stride
    elif scale_fill:  # stretch to fill
        resized_w, resized_h = new_w, new_h
        dw, dh = 0.0, 0.0
        ratio = (new_w / w, new_h / h)

    dw /= 2
    dh /= 2

    # Resize
    if (w, h) != (resized_w, resized_h):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded, ratio, (dw, dh)
