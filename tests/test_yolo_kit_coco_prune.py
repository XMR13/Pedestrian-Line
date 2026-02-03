from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from yolo_kitv2.datasets.coco_prune import prune_coco


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_coco_prune_keeps_images_with_images_prefix_and_writes_backup(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    images_dir = dataset / "images"
    images_dir.mkdir(parents=True)
    _write_image(images_dir / "foo.png")

    ann_path = dataset / "annotations.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "images\\foo.png", "width": 10, "height": 10},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0},
                ],
                "categories": [{"id": 0, "name": "cls0"}],
            }
        ),
        encoding="utf-8",
    )

    out_path, stats = prune_coco(
        ann_path=ann_path,
        images_dir=images_dir,
        in_place=True,
        output=None,
        dry_run=False,
        force=False,
        make_backup=True,
    )

    assert out_path == ann_path
    assert stats["images_missing"] == 0
    assert (dataset / "annotations.json.bak").exists()

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    assert len(data.get("images", [])) == 1
    assert len(data.get("annotations", [])) == 1


def test_coco_prune_refuses_to_wipe_everything_without_force(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    images_dir = dataset / "images"
    images_dir.mkdir(parents=True)

    ann_path = dataset / "annotations.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "images/missing.png", "width": 10, "height": 10}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2]}],
                "categories": [{"id": 0, "name": "cls0"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        prune_coco(
            ann_path=ann_path,
            images_dir=images_dir,
            in_place=True,
            output=None,
            dry_run=True,
            force=False,
            make_backup=False,
        )

    assert "Refusing to prune" in str(exc.value)

