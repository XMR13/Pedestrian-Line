from dataclasses import dataclass
import numpy as np


@dataclass
class NMSConfig:
    iou_threshold: float = 0.45
    max_detections: int = 300

def nms(boxes: np.ndarray, scores: np.ndarray, cfg: NMSConfig) -> np.ndarray:
    """
    Simple NumPy NMS. Expects boxes shape (N,4) in xyxy and scores shape (N,).
    Returns indices of boxes to keep.
    """ 

    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0 and len(keep) < cfg.max_detections:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= cfg.iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)
