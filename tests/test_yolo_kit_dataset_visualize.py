from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from yolo_kitv2.datasets.visualize import run


def _write_image(path: Path, shape: tuple[int, int, int] = (32, 48, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros(shape, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_dataset_viz_coco_generates_report_files(tmp_path: Path) -> None:
    dataset = tmp_path / "ds_coco"
    images_dir = dataset / "images"
    _write_image(images_dir / "a.png")
    _write_image(images_dir / "b.png")

    (dataset / "annotations.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "images/a.png", "width": 48, "height": 32},
                    {"id": 2, "file_name": "images/b.png", "width": 48, "height": 32},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 0, "bbox": [4, 4, 10, 8], "area": 80, "iscrowd": 0},
                ],
                "categories": [{"id": 0, "name": "truck"}],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "viz_coco"
    code = run(
        [
            "--dataset-dir",
            str(dataset),
            "--format",
            "coco",
            "--output-dir",
            str(output_dir),
            "--sample-count",
            "2",
        ]
    )
    assert code == 0

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "class_distribution.png").exists()
    assert (output_dir / "boxes_per_image_hist.png").exists()
    assert (output_dir / "box_area_ratio_hist.png").exists()
    assert (output_dir / "box_aspect_ratio_hist.png").exists()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["images_total"] == 2
    assert summary["summary"]["total_boxes"] == 1
    assert summary["class_counts"] == {"0": 1}
    assert len(list((output_dir / "samples").glob("*.jpg"))) == 1


def test_dataset_viz_yolo_counts_classes(tmp_path: Path) -> None:
    dataset = tmp_path / "ds_yolo"
    images_dir = dataset / "images"
    labels_dir = dataset / "labels"
    _write_image(images_dir / "a.png")
    _write_image(images_dir / "b.png")
    labels_dir.mkdir(parents=True, exist_ok=True)

    (labels_dir / "a.txt").write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.4 0.4",
                "1 0.3 0.4 0.2 0.3",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "viz_yolo"
    code = run(
        [
            "--dataset-dir",
            str(dataset),
            "--format",
            "yolo",
            "--output-dir",
            str(output_dir),
            "--sample-count",
            "2",
        ]
    )
    assert code == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["images_total"] == 2
    assert summary["summary"]["images_with_labels"] == 1
    assert summary["summary"]["total_boxes"] == 2
    assert summary["class_counts"] == {"0": 1, "1": 1}
    assert len(list((output_dir / "samples").glob("*.jpg"))) == 1


def test_dataset_viz_annotations_only_distribution(tmp_path: Path) -> None:
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "img_a.jpg", "width": 1920, "height": 1080},
                    {"id": 2, "file_name": "img_b.jpg", "width": 1920, "height": 1080},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 3, "bbox": [100, 200, 400, 250], "area": 100000, "iscrowd": 0},
                    {"id": 2, "image_id": 2, "category_id": 3, "bbox": [300, 220, 500, 260], "area": 130000, "iscrowd": 0},
                ],
                "categories": [{"id": 3, "name": "truck"}],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "viz_json_only"
    code = run(
        [
            "--annotations",
            str(ann_path),
            "--distribution-only",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert code == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["images_total"] == 2
    assert summary["summary"]["total_boxes"] == 2
    assert summary["class_counts"] == {"3": 2}
    assert (output_dir / "class_distribution.png").exists()
    assert (output_dir / "report.md").exists()
    assert not (output_dir / "boxes_per_image_hist.png").exists()
    assert not (output_dir / "samples").exists()
