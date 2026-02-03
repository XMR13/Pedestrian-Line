from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from yolo_kitv2.datasets.coco_merge import merge_coco


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def _write_coco_dataset(root: Path, *, tag: str, category_name: str = "cls0") -> None:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    _write_image(images_dir / "same.png")
    (root / "annotations.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "images/same.png", "width": 8, "height": 8}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0}],
                "categories": [{"id": 0, "name": category_name}],
                "info": {"tag": tag},
            }
        ),
        encoding="utf-8",
    )


def test_coco_merge_copies_and_renames_images(tmp_path: Path) -> None:
    ds_a = tmp_path / "ds_a"
    ds_b = tmp_path / "ds_b"
    _write_coco_dataset(ds_a, tag="a")
    _write_coco_dataset(ds_b, tag="b")

    out_dir = tmp_path / "merged"
    out_ann, stats = merge_coco(
        inputs=[str(ds_a), str(ds_b)],
        output_dir=out_dir,
        annotations_path=None,
    )

    assert out_ann.exists()
    assert stats["images"] == 2
    assert stats["annotations"] == 2

    merged = json.loads(out_ann.read_text(encoding="utf-8"))
    assert len(merged.get("images", [])) == 2
    assert len(merged.get("annotations", [])) == 2

    out_images = list((out_dir / "images").glob("*.png"))
    assert len(out_images) == 2


def test_coco_merge_category_conflict_fails(tmp_path: Path) -> None:
    ds_a = tmp_path / "ds_a"
    ds_b = tmp_path / "ds_b"
    _write_coco_dataset(ds_a, tag="a", category_name="cls0")
    _write_coco_dataset(ds_b, tag="b", category_name="different")

    out_dir = tmp_path / "merged"
    with pytest.raises(SystemExit) as exc:
        merge_coco(inputs=[str(ds_a), str(ds_b)], output_dir=out_dir, annotations_path=None)

    assert "Category conflict" in str(exc.value)

