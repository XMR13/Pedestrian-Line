from __future__ import annotations

import numpy as np

from yolo_kitv2 import LetterboxConfig, YoloPipeline, YoloPostConfig, YoloPostprocessor
from yolo_kitv2.letterbox import letterbox


def _make_post() -> YoloPostprocessor:
    return YoloPostprocessor(
        YoloPostConfig(
            conf_threshold=0.0,
            iou_threshold=0.99,
            apply_nms=False,
            max_detections=1000,
        )
    )


def test_postprocess_end2end_decoded_layout() -> None:
    post = _make_post()

    preds = np.array(
        [
            [
                [10, 10, 20, 20, 0.9, 2],
                [30, 30, 40, 40, 0.8, 3],
            ]
        ],
        dtype=np.float32,
    )

    dets = post.process(preds, orig_size=(100, 100), pad=(0.0, 0.0), ratio=(1.0, 1.0))

    assert len(dets) == 2
    assert np.allclose([d.score for d in dets], [0.9, 0.8])
    assert [d.class_id for d in dets] == [2, 3]


def test_postprocess_ultralytics_84xk_layout() -> None:
    post = _make_post()

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

    dets = post.process(preds, orig_size=(100, 100), pad=(0.0, 0.0), ratio=(1.0, 1.0))

    assert len(dets) == 3
    assert [d.class_id for d in dets] == [2, 7, 5]
    assert np.allclose([d.score for d in dets], [0.9, 0.8, 0.7])


def test_postprocess_kx84_layout() -> None:
    post = _make_post()

    k = 3
    preds = np.zeros((1, k, 84), dtype=np.float32)

    # Anchor 0
    preds[0, 0, 0:4] = [50, 50, 20, 10]
    preds[0, 0, 4 + 2] = 0.9  # class 2

    # Anchor 1
    preds[0, 1, 0:4] = [20, 20, 10, 10]
    preds[0, 1, 4 + 7] = 0.8  # class 7

    # Anchor 2
    preds[0, 2, 0:4] = [80, 40, 12, 8]
    preds[0, 2, 4 + 5] = 0.7  # class 5

    dets = post.process(preds, orig_size=(100, 100), pad=(0.0, 0.0), ratio=(1.0, 1.0))

    assert len(dets) == 3
    assert [d.class_id for d in dets] == [2, 7, 5]
    assert np.allclose([d.score for d in dets], [0.9, 0.8, 0.7])


def test_postprocess_generic_5_plus_c_layout() -> None:
    post = _make_post()

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

    dets = post.process(preds, orig_size=(100, 100), pad=(0.0, 0.0), ratio=(1.0, 1.0))

    assert len(dets) == 2
    assert [d.class_id for d in dets] == [1, 2]
    assert np.allclose([d.score for d in dets], [0.8 * 0.9, 0.5 * 1.0])


def test_preprocess_blobfromimage_matches_previous_blob_layout() -> None:
    image = np.array(
        [
            [[0, 10, 20], [30, 40, 50], [60, 70, 80]],
            [[90, 100, 110], [120, 130, 140], [150, 160, 170]],
        ],
        dtype=np.uint8,
    )

    pipeline = YoloPipeline(
        lambda blob: blob,
        letterbox_cfg=LetterboxConfig(new_shape=(4, 4), auto=False, scale_fill=False, scaleup=True, stride=32),
    )

    prep = pipeline.preprocess(image)

    img, ratio, pad = letterbox(
        image,
        new_shape=(4, 4),
        color=(114, 114, 114),
        auto=False,
        scale_fill=False,
        scaleup=True,
        stride=32,
    )
    expected = img[:, :, ::-1].astype(np.float32) / 255.0
    expected = np.transpose(expected, (2, 0, 1))[None, ...]

    assert prep.blob.shape == expected.shape
    assert prep.blob.dtype == np.float32
    assert prep.orig_size == (3, 2)
    assert prep.ratio == ratio
    assert prep.pad == pad
    assert np.allclose(prep.blob, expected, atol=1e-7)
