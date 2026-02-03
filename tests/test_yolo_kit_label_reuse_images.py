from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from yolo_kitv2.datasets.label import label_images_to_coco
from yolo_kitv2.types import Detection


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_label_reuse_images_references_existing_paths(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    images_dir = dataset / "images"
    images_dir.mkdir(parents=True)
    img_path = images_dir / "foo.png"
    _write_image(img_path)

    def pipe(_frame: np.ndarray) -> list[object]:
        return [Detection(x1=1, y1=2, x2=10, y2=12, score=0.9, class_id=0)]

    before = sorted(p.name for p in images_dir.glob("*.png"))

    processed, included, images, annotations, seen, next_image_id, next_ann_id = label_images_to_coco(
        image_files=[img_path],
        output_dir=dataset,
        images_dir=images_dir,
        pipe=pipe,
        mode="coco",
        output_ext="png",
        every_n=1,
        max_images=None,
        resize_to=None,
        reuse_images=True,
        skip_empty=False,
        min_box_area_ratio=0.0,
        include_score=False,
        start_image_id=1,
        start_ann_id=1,
    )

    after = sorted(p.name for p in images_dir.glob("*.png"))
    assert before == after  # no new images written

    assert processed == 1
    assert included == 1
    assert len(images) == 1
    assert images[0]["file_name"] == "images/foo.png"
    assert len(annotations) == 1
    assert annotations[0]["image_id"] == 1
    assert seen == [0]
    assert next_image_id == 2
    assert next_ann_id == 2

