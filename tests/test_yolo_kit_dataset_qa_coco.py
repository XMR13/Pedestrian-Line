from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from yolo_kitv2.datasets.qa import run


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((6, 6, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_dataset_qa_coco_reports_orphan_images(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    images_dir = dataset / "images"
    images_dir.mkdir(parents=True)
    _write_image(images_dir / "a.png")
    _write_image(images_dir / "b.png")

    (dataset / "annotations.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "images/a.png", "width": 6, "height": 6}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0}],
                "categories": [{"id": 0, "name": "cls0"}],
            }
        ),
        encoding="utf-8",
    )

    out_json = tmp_path / "report.json"
    code = run(
        [
            "--dataset-dir",
            str(dataset),
            "--format",
            "coco",
            "--output-json",
            str(out_json),
        ]
    )
    assert code == 0

    report = json.loads(out_json.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["images_total"] == 1
    assert summary["image_files_total"] == 2
    assert summary["orphan_images"] == 1

