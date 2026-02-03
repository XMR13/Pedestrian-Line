from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

from ..metadata import load_class_names


@dataclass
class LabelIssue:
    label_path: str
    line_number: int
    message: str


@dataclass
class ImageIssue:
    image_path: str
    message: str


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA tool for YOLO/COCO/VOC datasets.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset root directory (auto-detects images/labels).",
    )
    parser.add_argument("--images-dir", type=str, default=None, help="Root directory for images.")
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Root directory for label files (YOLO/VOC) or COCO JSON path/dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subdirectories under images/labels.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["auto", "yolo", "coco", "voc"],
        default="auto",
        help="Dataset label format (default: auto).",
    )
    parser.add_argument(
        "--image-exts",
        type=str,
        default="jpg,jpeg,png,bmp,webp",
        help="Comma-separated image extensions (default: jpg,jpeg,png,bmp,webp).",
    )
    parser.add_argument(
        "--label-ext",
        type=str,
        default=".txt",
        help="Label file extension (default: .txt).",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Optional class names mapping file (YAML/JSON) to validate class IDs.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write JSON report to this path.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Write per-image CSV summary to this path.",
    )
    parser.add_argument(
        "--max-issue-items",
        type=int,
        default=200,
        help="Max items to list per issue type in the report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any issues are found.",
    )
    return parser.parse_args(argv)


def _normalize_extensions(spec: str) -> Sequence[str]:
    extensions = []
    for item in (spec or "").split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        extensions.append(item)
    return extensions


def _iter_files(root: Path, recursive: bool, extensions: Sequence[str]) -> Iterable[Path]:
    if not root.exists():
        return []
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in sorted(root.glob(pattern))
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return files


def _find_class_names_file(images_dir: Path, labels_dir: Path, dataset_dir: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    folders = [labels_dir, images_dir, labels_dir.parent, images_dir.parent]
    if dataset_dir is not None:
        folders.extend([dataset_dir, dataset_dir / "data", dataset_dir / "config"])
    for folder in folders:
        if not folder.exists() or not folder.is_dir():
            continue
        candidates.extend(sorted(folder.glob("*.yml")))
        candidates.extend(sorted(folder.glob("*.yaml")))
        candidates.extend(sorted(folder.glob("*.json")))
    if not candidates:
        return None
    preferred = [
        path
        for path in candidates
        if path.name in {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml", "metadata.yaml", "metadata.json"}
    ]
    if len(preferred) == 1:
        return preferred[0]
    return candidates[0]


def _key_for_path(root: Path, file_path: Path) -> str:
    relative = file_path.relative_to(root)
    return str(relative.with_suffix("")).replace("\\", "/")


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _read_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return int(width), int(height)


def _parse_label_lines(
    label_path: Path,
    class_names_map: Optional[Dict[int, str]],
    max_issue_items: int,
) -> Tuple[int, Counter, List[LabelIssue], List[int]]:
    issues: List[LabelIssue] = []
    class_counts: Counter = Counter()
    unknown_classes: List[int] = []
    total_boxes = 0

    content = label_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    for line_number, line in enumerate(content, start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(label_path), line_number, "Too few fields (need 5)."))
            continue
        if len(parts) > 5 and len(issues) < max_issue_items:
            issues.append(LabelIssue(str(label_path), line_number, "Extra fields beyond 5."))
        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except Exception:
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(label_path), line_number, "Non-numeric fields."))
            continue

        numeric_values = [x_center, y_center, width, height]
        if any(not _is_finite(value) for value in numeric_values):
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(label_path), line_number, "Non-finite values."))
            continue

        if width <= 0 or height <= 0:
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(label_path), line_number, "Non-positive width/height."))
            continue

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 <= width <= 1.0
            and 0.0 <= height <= 1.0
        ):
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(label_path), line_number, "Values outside 0..1 range."))
            continue

        left = x_center - width / 2.0
        right = x_center + width / 2.0
        top = y_center - height / 2.0
        bottom = y_center + height / 2.0
        if left < 0.0 or top < 0.0 or right > 1.0 or bottom > 1.0:
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(label_path), line_number, "Box extends outside image."))
            continue

        if class_names_map is not None and class_id not in class_names_map:
            if len(unknown_classes) < max_issue_items:
                unknown_classes.append(class_id)

        class_counts[class_id] += 1
        total_boxes += 1

    return total_boxes, class_counts, issues, unknown_classes


def _load_coco_json(labels_dir: Path) -> Dict[str, object]:
    if labels_dir.is_file():
        return json.loads(labels_dir.read_text(encoding="utf-8"))
    json_files = sorted(labels_dir.glob("*.json"))
    if len(json_files) == 1:
        return json.loads(json_files[0].read_text(encoding="utf-8"))
    if not json_files:
        raise SystemExit(f"No COCO JSON found in: {labels_dir}")
    raise SystemExit(f"Multiple COCO JSON files found in: {labels_dir}")


def _resolve_coco_image_path(images_dir: Path, file_name: str) -> Path:
    raw = Path(str(file_name).replace("\\", "/").lstrip("./"))
    if raw.is_absolute():
        return raw

    candidate = images_dir / raw
    if candidate.exists():
        return candidate

    if images_dir.name.lower() == "images":
        candidate = images_dir.parent / raw
        if candidate.exists():
            return candidate

    return images_dir / raw.name


def _parse_coco(
    coco: Dict[str, object],
    images_dir: Path,
    class_names_map: Optional[Dict[int, str]],
    max_issue_items: int,
) -> Tuple[Dict[str, Dict[str, object]], Counter, List[LabelIssue], List[ImageIssue], List[int], List[str]]:
    images = {image["id"]: image for image in coco.get("images", []) if "id" in image}
    categories = {cat["id"]: cat for cat in coco.get("categories", []) if "id" in cat}
    annotations = coco.get("annotations", [])

    per_image: Dict[str, Dict[str, object]] = {}
    class_counts: Counter = Counter()
    label_issues: List[LabelIssue] = []
    image_issues: List[ImageIssue] = []
    unknown_class_ids: List[int] = []

    for image_id, image in images.items():
        file_name = image.get("file_name", "")
        key = Path(file_name).with_suffix("").as_posix()
        per_image[key] = {
            "image_id": image_id,
            "file_name": file_name,
            "width": image.get("width"),
            "height": image.get("height"),
            "boxes": 0,
        }
        if file_name:
            path = _resolve_coco_image_path(images_dir, str(file_name))
            if not path.exists() and len(image_issues) < max_issue_items:
                image_issues.append(ImageIssue(str(path), "Image file missing."))

    for idx, ann in enumerate(annotations, start=1):
        image_id = ann.get("image_id")
        if image_id not in images:
            if len(label_issues) < max_issue_items:
                label_issues.append(LabelIssue("coco", idx, "annotation references missing image_id."))
            continue
        image = images[image_id]
        file_name = image.get("file_name", "")
        key = Path(file_name).with_suffix("").as_posix() if file_name else f"image_{image_id}"
        entry = per_image.setdefault(
            key,
            {
                "image_id": image_id,
                "file_name": file_name,
                "width": image.get("width"),
                "height": image.get("height"),
                "boxes": 0,
            },
        )

        bbox = ann.get("bbox")
        category_id = ann.get("category_id")
        if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            if len(label_issues) < max_issue_items:
                label_issues.append(LabelIssue("coco", idx, "Invalid bbox format."))
            continue
        try:
            x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except Exception:
            if len(label_issues) < max_issue_items:
                label_issues.append(LabelIssue("coco", idx, "Non-numeric bbox."))
            continue
        if w <= 0 or h <= 0:
            if len(label_issues) < max_issue_items:
                label_issues.append(LabelIssue("coco", idx, "Non-positive bbox size."))
            continue

        width = image.get("width")
        height = image.get("height")
        if width and height:
            if x < 0 or y < 0 or (x + w) > width or (y + h) > height:
                if len(label_issues) < max_issue_items:
                    label_issues.append(LabelIssue("coco", idx, "BBox outside image bounds."))
                continue

        if class_names_map is not None and category_id not in class_names_map:
            if len(unknown_class_ids) < max_issue_items:
                unknown_class_ids.append(int(category_id))

        if category_id in categories:
            class_counts[int(category_id)] += 1
        else:
            if len(label_issues) < max_issue_items:
                label_issues.append(LabelIssue("coco", idx, "Unknown category_id."))
            continue

        entry["boxes"] += 1

    images_without_annotations = [key for key, entry in per_image.items() if entry.get("boxes", 0) == 0]
    return per_image, class_counts, label_issues, image_issues, unknown_class_ids, images_without_annotations


def _parse_voc_xml(
    xml_path: Path,
    class_name_to_id: Optional[Dict[str, int]],
    max_issue_items: int,
) -> Tuple[int, Counter, List[LabelIssue], Optional[Tuple[int, int]]]:
    issues: List[LabelIssue] = []
    class_counts: Counter = Counter()
    total_boxes = 0
    image_size = None

    try:
        tree = ET.parse(str(xml_path))
    except Exception:
        return 0, Counter(), [LabelIssue(str(xml_path), 0, "Failed to parse XML.")], None

    root = tree.getroot()
    size = root.find("size")
    if size is not None:
        try:
            width = int(float(size.findtext("width", default="0")))
            height = int(float(size.findtext("height", default="0")))
            if width > 0 and height > 0:
                image_size = (width, height)
        except Exception:
            image_size = None

    objects = root.findall("object")
    for idx, obj in enumerate(objects, start=1):
        name = obj.findtext("name", default="").strip()
        if not name:
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(xml_path), idx, "Missing class name."))
            continue
        class_counts[name] += 1

        bnd = obj.find("bndbox")
        if bnd is None:
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(xml_path), idx, "Missing bndbox."))
            continue
        try:
            xmin = float(bnd.findtext("xmin", default="nan"))
            ymin = float(bnd.findtext("ymin", default="nan"))
            xmax = float(bnd.findtext("xmax", default="nan"))
            ymax = float(bnd.findtext("ymax", default="nan"))
        except Exception:
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(xml_path), idx, "Non-numeric bbox."))
            continue

        if any(not _is_finite(value) for value in (xmin, ymin, xmax, ymax)):
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(xml_path), idx, "Non-finite bbox values."))
            continue
        if xmax <= xmin or ymax <= ymin:
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(xml_path), idx, "Non-positive bbox size."))
            continue
        if image_size:
            width, height = image_size
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                if len(issues) < max_issue_items:
                    issues.append(LabelIssue(str(xml_path), idx, "BBox outside image bounds."))
                continue

        if class_name_to_id is not None and name not in class_name_to_id:
            if len(issues) < max_issue_items:
                issues.append(LabelIssue(str(xml_path), idx, f"Unknown class name '{name}'."))

        total_boxes += 1

    return total_boxes, class_counts, issues, image_size


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            value_text = str(value).replace('"', '""')
            if "," in value_text or "\n" in value_text:
                value_text = f'"{value_text}"'
            values.append(value_text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines), encoding="utf-8")


def _dataset_root_from_images_dir(images_dir: Path) -> Path:
    return images_dir.parent if images_dir.name.lower() == "images" else images_dir


def _detect_format(labels_dir: Path, recursive: bool) -> str:
    json_files = list(_iter_files(labels_dir, recursive, [".json"]))
    if json_files:
        return "coco"
    xml_files = list(_iter_files(labels_dir, recursive, [".xml"]))
    txt_files = list(_iter_files(labels_dir, recursive, [".txt"]))
    if xml_files and not txt_files:
        return "voc"
    if txt_files and not xml_files:
        return "yolo"
    if txt_files and xml_files:
        raise SystemExit("Ambiguous label format: found both .txt and .xml. Use --format.")
    raise SystemExit("Could not detect label format. Use --format to specify.")


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    if dataset_dir is not None and not dataset_dir.exists():
        raise SystemExit(f"Dataset dir not found: {dataset_dir}")

    images_dir = Path(args.images_dir) if args.images_dir else None
    labels_dir = Path(args.labels_dir) if args.labels_dir else None

    if dataset_dir is not None:
        if images_dir is None:
            candidate = dataset_dir / "images"
            images_dir = candidate if candidate.exists() else dataset_dir
        if labels_dir is None:
            candidate = dataset_dir / "labels"
            labels_dir = candidate if candidate.exists() else dataset_dir

    if images_dir is None or labels_dir is None:
        raise SystemExit("Provide --dataset-dir or both --images-dir and --labels-dir.")

    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {labels_dir}")

    image_exts = _normalize_extensions(args.image_exts)
    label_format = args.format
    if label_format == "auto":
        label_format = _detect_format(labels_dir, args.recursive)

    class_names_map = None
    if args.class_names:
        class_names_map = load_class_names(args.class_names)

    label_issues: List[LabelIssue] = []
    image_issues: List[ImageIssue] = []
    unknown_class_ids: List[int] = []
    per_class_counts: Counter = Counter()
    per_image_rows: List[Dict[str, object]] = []
    empty_label_files: List[str] = []
    total_boxes = 0

    if label_format == "yolo":
        if class_names_map is None:
            auto_names = _find_class_names_file(images_dir, labels_dir, dataset_dir)
            if auto_names is not None:
                class_names_map = load_class_names(str(auto_names))
        label_ext = args.label_ext if args.label_ext.startswith(".") else f".{args.label_ext}"
        image_files = list(_iter_files(images_dir, args.recursive, image_exts))
        label_files = list(_iter_files(labels_dir, args.recursive, [label_ext]))

        image_keys = {_key_for_path(images_dir, path): path for path in image_files}
        label_keys = {_key_for_path(labels_dir, path): path for path in label_files}

        missing_labels = sorted(key for key in image_keys.keys() if key not in label_keys)
        orphan_labels = sorted(key for key in label_keys.keys() if key not in image_keys)

        for key, label_path in label_keys.items():
            total, class_counts, issues, unknown_classes = _parse_label_lines(
                label_path,
                class_names_map,
                args.max_issue_items,
            )
            total_boxes += total
            per_class_counts.update(class_counts)
            if total == 0:
                empty_label_files.append(str(label_path))
            label_issues.extend(issues)
            unknown_class_ids.extend(unknown_classes)

            image_path = image_keys.get(key)
            image_size = None
            if image_path is not None:
                image_size = _read_image_size(image_path)
                if image_size is None and len(image_issues) < args.max_issue_items:
                    image_issues.append(ImageIssue(str(image_path), "Failed to read image."))

            per_image_rows.append(
                {
                    "key": key,
                    "image_path": str(image_path) if image_path else "",
                    "label_path": str(label_path),
                    "boxes": total,
                    "image_width": image_size[0] if image_size else "",
                    "image_height": image_size[1] if image_size else "",
                }
            )

        summary = {
            "images_total": len(image_files),
            "labels_total": len(label_files),
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
            "empty_label_files": len(empty_label_files),
            "label_issues": len(label_issues),
            "image_issues": len(image_issues),
            "unknown_class_ids": len(unknown_class_ids),
            "total_boxes": total_boxes,
            "classes": dict(sorted(per_class_counts.items(), key=lambda item: item[0])),
        }

        report = {
            "summary": summary,
            "missing_labels": missing_labels[: args.max_issue_items],
            "orphan_labels": orphan_labels[: args.max_issue_items],
            "empty_label_files": empty_label_files[: args.max_issue_items],
            "label_issues": [issue.__dict__ for issue in label_issues[: args.max_issue_items]],
            "image_issues": [issue.__dict__ for issue in image_issues[: args.max_issue_items]],
            "unknown_class_ids": unknown_class_ids[: args.max_issue_items],
        }

    elif label_format == "coco":
        coco = _load_coco_json(labels_dir)
        per_image, class_counts, issues, img_issues, unknown_ids, images_without_annotations = _parse_coco(
            coco,
            images_dir,
            class_names_map,
            args.max_issue_items,
        )
        total_boxes = sum(int(entry.get("boxes", 0)) for entry in per_image.values())
        per_class_counts.update(class_counts)
        label_issues.extend(issues)
        image_issues.extend(img_issues)
        unknown_class_ids.extend(unknown_ids)

        dataset_root = _dataset_root_from_images_dir(images_dir)
        disk_images = list(_iter_files(images_dir, args.recursive, image_exts))
        disk_rel = {path.relative_to(dataset_root).as_posix() for path in disk_images}
        coco_rel = {
            str(entry.get("file_name", "")).replace("\\", "/")
            for entry in per_image.values()
            if entry.get("file_name")
        }
        orphan_images = sorted(rel for rel in disk_rel if rel not in coco_rel)

        for key, entry in per_image.items():
            per_image_rows.append(
                {
                    "key": key,
                    "image_path": str(images_dir / entry.get("file_name", "")) if entry.get("file_name") else "",
                    "label_path": str(labels_dir),
                    "boxes": entry.get("boxes", 0),
                    "image_width": entry.get("width", ""),
                    "image_height": entry.get("height", ""),
                }
            )

        summary = {
            "images_total": len(per_image),
            "image_files_total": len(disk_images),
            "orphan_images": len(orphan_images),
            "labels_total": 1,
            "missing_labels": 0,
            "orphan_labels": 0,
            "empty_label_files": 0,
            "images_without_annotations": len(images_without_annotations),
            "label_issues": len(label_issues),
            "image_issues": len(image_issues),
            "unknown_class_ids": len(unknown_class_ids),
            "total_boxes": total_boxes,
            "classes": dict(sorted(per_class_counts.items(), key=lambda item: item[0])),
        }

        report = {
            "summary": summary,
            "orphan_images": orphan_images[: args.max_issue_items],
            "images_without_annotations": images_without_annotations[: args.max_issue_items],
            "label_issues": [issue.__dict__ for issue in label_issues[: args.max_issue_items]],
            "image_issues": [issue.__dict__ for issue in image_issues[: args.max_issue_items]],
            "unknown_class_ids": unknown_class_ids[: args.max_issue_items],
        }

    else:  # voc
        label_ext_value = args.label_ext
        if args.label_ext == ".txt":
            label_ext_value = ".xml"
        label_ext = label_ext_value if label_ext_value.startswith(".") else f".{label_ext_value}"
        image_files = list(_iter_files(images_dir, args.recursive, image_exts))
        label_files = list(_iter_files(labels_dir, args.recursive, [label_ext]))

        image_keys = {_key_for_path(images_dir, path): path for path in image_files}
        label_keys = {_key_for_path(labels_dir, path): path for path in label_files}

        missing_labels = sorted(key for key in image_keys.keys() if key not in label_keys)
        orphan_labels = sorted(key for key in label_keys.keys() if key not in image_keys)

        class_name_to_id = None
        if class_names_map:
            class_name_to_id = {name: class_id for class_id, name in class_names_map.items()}

        for key, label_path in label_keys.items():
            total, class_counts, issues, image_size = _parse_voc_xml(
                label_path,
                class_name_to_id,
                args.max_issue_items,
            )
            total_boxes += total
            per_class_counts.update(class_counts)
            if total == 0:
                empty_label_files.append(str(label_path))
            label_issues.extend(issues)

            image_path = image_keys.get(key)
            if image_path is None and len(image_issues) < args.max_issue_items:
                image_issues.append(ImageIssue(str(label_path), "Missing image for label."))

            per_image_rows.append(
                {
                    "key": key,
                    "image_path": str(image_path) if image_path else "",
                    "label_path": str(label_path),
                    "boxes": total,
                    "image_width": image_size[0] if image_size else "",
                    "image_height": image_size[1] if image_size else "",
                }
            )

        summary = {
            "images_total": len(image_files),
            "labels_total": len(label_files),
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
            "empty_label_files": len(empty_label_files),
            "label_issues": len(label_issues),
            "image_issues": len(image_issues),
            "unknown_class_ids": len(unknown_class_ids),
            "total_boxes": total_boxes,
            "classes": dict(sorted(per_class_counts.items(), key=lambda item: str(item[0]))),
        }

        report = {
            "summary": summary,
            "missing_labels": missing_labels[: args.max_issue_items],
            "orphan_labels": orphan_labels[: args.max_issue_items],
            "empty_label_files": empty_label_files[: args.max_issue_items],
            "label_issues": [issue.__dict__ for issue in label_issues[: args.max_issue_items]],
            "image_issues": [issue.__dict__ for issue in image_issues[: args.max_issue_items]],
            "unknown_class_ids": unknown_class_ids[: args.max_issue_items],
        }

    print("[dataset_qa] summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[dataset_qa] wrote JSON report: {output_path}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(output_path, per_image_rows)
        print(f"[dataset_qa] wrote CSV report: {output_path}")

    has_issues = any(
        value > 0
        for value in (
            summary.get("missing_labels", 0),
            summary.get("orphan_labels", 0),
            summary.get("empty_label_files", 0),
            summary.get("label_issues", 0),
            summary.get("image_issues", 0),
            summary.get("unknown_class_ids", 0),
            summary.get("orphan_images", 0),
        )
    )
    if args.strict and has_issues:
        raise SystemExit(1)

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)
