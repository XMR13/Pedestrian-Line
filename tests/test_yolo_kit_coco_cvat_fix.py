from __future__ import annotations

import json
from pathlib import Path

from yolo_kitv2.datasets.coco_cvat_fix import cvat_fix_coco_ids


def test_cvat_fix_remaps_category_ids_from_zero_based() -> None:
    coco = {
        "images": [{"id": 1, "file_name": "images/a.png", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0}],
        "categories": [{"id": 0, "name": "cls0"}],
    }
    fixed, mapping = cvat_fix_coco_ids(coco=coco, start_category_id=1)
    assert mapping == {0: 1}
    assert fixed["categories"][0]["id"] == 1
    assert fixed["annotations"][0]["category_id"] == 1


def test_cvat_fix_can_basename_file_names() -> None:
    coco = {
        "images": [{"id": 1, "file_name": "images/a.png", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0}],
        "categories": [{"id": 0, "name": "cls0"}],
    }
    fixed, _ = cvat_fix_coco_ids(coco=coco, start_category_id=1, basename_file_names=True)
    assert fixed["images"][0]["file_name"] == "a.png"

def test_cvat_fix_basename_handles_windows_separators() -> None:
    coco = {
        "images": [{"id": 1, "file_name": "images\\a.png", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0}],
        "categories": [{"id": 0, "name": "cls0"}],
    }
    fixed, _ = cvat_fix_coco_ids(coco=coco, start_category_id=1, basename_file_names=True)
    assert fixed["images"][0]["file_name"] == "a.png"


def test_cvat_fix_noop_when_already_one_based() -> None:
    coco = {
        "images": [{"id": 1, "file_name": "images/a.png", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "cls0"}],
    }
    fixed, mapping = cvat_fix_coco_ids(coco=coco, start_category_id=1)
    assert mapping == {1: 1}
    assert fixed["categories"][0]["id"] == 1
    assert fixed["annotations"][0]["category_id"] == 1


def test_cvat_fix_basename_applies_even_if_ids_already_one_based() -> None:
    coco = {
        "images": [{"id": 1, "file_name": "images/a.png", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "cls0"}],
    }
    fixed, mapping = cvat_fix_coco_ids(coco=coco, start_category_id=1, basename_file_names=True)
    assert mapping == {1: 1}
    assert fixed["images"][0]["file_name"] == "a.png"
