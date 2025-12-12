import numpy as np

from pedestrian_line_counter.config import ModelConfig
from pedestrian_line_counter.detector import Detector



def _make_detector() -> Detector:
    cfg = ModelConfig(backend="motion")
    cfg.confidence_threshold = 0.0
    cfg.nms_iou_threshold = 0.99
    return Detector(cfg)


def test_postprocess_end2end_decoded_layout() -> None:
    det = _make_detector()

    preds = np.array(
        [
            [
                [10, 10, 20, 20, 0.9, 2],
                [30, 30, 40, 40, 0.8, 3],
            ]
        ],
        dtype=np.float32,
    )

    boxes, scores, class_ids = det._postprocess_yolo_generic(
        preds, orig_w=100, orig_h=100, dw=0, dh=0, scale=1.0
    )

    assert boxes.shape == (2, 4)
    assert np.allclose(scores, [0.9, 0.8])
    assert class_ids.tolist() == [2, 3]


def test_postprocess_ultralytics_84xk_layout() -> None:
    det = _make_detector()

    k = 3
    preds = np.zeros((1, 84, k), dtype=np.float32)

    # Anchor 0: cx, cy, w, h
    preds[0, 0, 0] = 50
    preds[0, 1, 0] = 50
    preds[0, 2, 0] = 20
    preds[0, 3, 0] = 10
    preds[0, 4 + 2, 0] = 0.9  # class 2

    # Anchor 1
    preds[0, 0, 1] = 20
    preds[0, 1, 1] = 20
    preds[0, 2, 1] = 10
    preds[0, 3, 1] = 10
    preds[0, 4 + 7, 1] = 0.8  # class 7

    # Anchor 2
    preds[0, 0, 2] = 80
    preds[0, 1, 2] = 40
    preds[0, 2, 2] = 12
    preds[0, 3, 2] = 8
    preds[0, 4 + 5, 2] = 0.7  # class 5

    boxes, scores, class_ids = det._postprocess_yolo_generic(
        preds, orig_w=100, orig_h=100, dw=0, dh=0, scale=1.0
    )

    assert boxes.shape[1] == 4
    assert boxes.shape[0] == 3
    assert class_ids.tolist() == [2, 7, 5]
    assert np.allclose(scores, [0.9, 0.8, 0.7])


def test_postprocess_generic_5_plus_c_layout() -> None:
    det = _make_detector()

    # (1, N, 5 + C) with cx, cy, w, h, obj, class_scores...
    preds = np.array(
        [
            [
                [50, 50, 20, 10, 0.8, 0.1, 0.9, 0.0],
                [20, 20, 10, 10, 0.5, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )

    boxes, scores, class_ids = det._postprocess_yolo_generic(
        preds, orig_w=100, orig_h=100, dw=0, dh=0, scale=1.0
    )

    assert boxes.shape == (2, 4)
    assert class_ids.tolist() == [1, 2]
    assert np.allclose(scores, [0.8 * 0.9, 0.5 * 1.0])

