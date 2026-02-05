from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..metadata import load_class_names
from ..types import Detection
from ..visualize import draw_detections


@dataclass
class ImageSample:
    key: str
    image_path: Optional[Path]
    width: int
    height: int
    detections: List[Detection] = field(default_factory=list)


@dataclass
class VizDataset:
    label_format: str
    class_names: Dict[int, str]
    images: Dict[str, ImageSample]
    class_counts: Counter
    area_ratios: List[float]
    aspect_ratios: List[float]
    invalid_boxes: int
    missing_images: int
    parse_errors: int
    label_files_total: int


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dataset labels (class distribution, box stats, samples).")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset root directory.")
    parser.add_argument("--images-dir", type=str, default=None, help="Images root directory.")
    parser.add_argument("--labels-dir", type=str, default=None, help="Labels root directory or COCO JSON path/dir.")
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="COCO annotations.json path (shortcut; useful for JSON-only distribution reports).",
    )
    parser.add_argument("--format", type=str, choices=["auto", "yolo", "coco"], default="auto")
    parser.add_argument("--recursive", action="store_true", help="Scan files recursively.")
    parser.add_argument(
        "--image-exts",
        type=str,
        default="jpg,jpeg,png,bmp,webp",
        help="Comma-separated image extensions.",
    )
    parser.add_argument("--label-ext", type=str, default=".txt", help="YOLO label extension.")
    parser.add_argument("--class-names", type=str, default=None, help="Optional class names YAML/JSON file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Report output directory (default: <dataset-dir>/visualizations or ./visualizations).",
    )
    parser.add_argument("--sample-count", type=int, default=24, help="How many sample overlays to generate.")
    parser.add_argument("--sample-max-side", type=int, default=1280, help="Max side for saved sample overlays.")
    parser.add_argument("--max-class-bars", type=int, default=30, help="Top N classes in class bar chart.")
    parser.add_argument("--hist-bins", type=int, default=20, help="Histogram bins for box metrics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample tie-breaking.")
    parser.add_argument(
        "--distribution-only",
        action="store_true",
        help="Only generate class distribution outputs (summary + class chart + report), skip extra charts and samples.",
    )
    return parser.parse_args(argv)


def _normalize_extensions(spec: str) -> List[str]:
    out: List[str] = []
    for part in (spec or "").split(","):
        item = part.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        out.append(item)
    return out


def _iter_files(root: Path, recursive: bool, extensions: Sequence[str]) -> List[Path]:
    if not root.exists():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in extensions)


def _read_image_size(path: Path) -> Optional[Tuple[int, int]]:
    image = cv2.imread(str(path))
    if image is None:
        return None
    h, w = image.shape[:2]
    return int(w), int(h)


def _resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Optional[Path], Path]:
    if args.annotations:
        ann_path = Path(args.annotations)
        if not ann_path.exists():
            raise SystemExit(f"Annotations not found: {ann_path}")
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else ann_path.parent
        images_dir = Path(args.images_dir) if args.images_dir else None
        if images_dir is not None and not images_dir.exists():
            raise SystemExit(f"Images dir not found: {images_dir}")
        return dataset_dir, images_dir, ann_path

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
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

    if dataset_dir is None:
        dataset_dir = images_dir.parent if images_dir.name.lower() == "images" else images_dir
    return dataset_dir, images_dir, labels_dir


def _detect_format(labels_dir: Path, recursive: bool) -> str:
    json_files = _iter_files(labels_dir, recursive, [".json"]) if labels_dir.is_dir() else [labels_dir]
    if any(path.suffix.lower() == ".json" for path in json_files):
        return "coco"
    txt_files = _iter_files(labels_dir, recursive, [".txt"])
    if txt_files:
        return "yolo"
    raise SystemExit("Could not detect dataset label format. Use --format.")


def _key_for_path(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    return str(rel.with_suffix("")).replace("\\", "/")


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


def _safe_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except Exception:
        return None


def _parse_yolo(
    *,
    images_dir: Path,
    labels_dir: Path,
    recursive: bool,
    image_exts: Sequence[str],
    label_ext: str,
    class_names: Dict[int, str],
) -> VizDataset:
    label_ext = label_ext if label_ext.startswith(".") else f".{label_ext}"
    image_files = _iter_files(images_dir, recursive, image_exts)
    label_files = _iter_files(labels_dir, recursive, [label_ext])

    image_by_key = {_key_for_path(images_dir, path): path for path in image_files}
    label_by_key = {_key_for_path(labels_dir, path): path for path in label_files}

    images: Dict[str, ImageSample] = {}
    for key, image_path in image_by_key.items():
        size = _read_image_size(image_path)
        if size is None:
            continue
        images[key] = ImageSample(key=key, image_path=image_path, width=size[0], height=size[1], detections=[])

    class_counts: Counter = Counter()
    area_ratios: List[float] = []
    aspect_ratios: List[float] = []
    invalid_boxes = 0
    missing_images = 0
    parse_errors = 0

    for key, label_path in label_by_key.items():
        sample = images.get(key)
        if sample is None:
            missing_images += 1
            continue

        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) < 5:
                parse_errors += 1
                continue
            class_id = _safe_float(parts[0])
            xc = _safe_float(parts[1])
            yc = _safe_float(parts[2])
            bw = _safe_float(parts[3])
            bh = _safe_float(parts[4])
            if class_id is None or xc is None or yc is None or bw is None or bh is None:
                parse_errors += 1
                continue
            if bw <= 0.0 or bh <= 0.0:
                invalid_boxes += 1
                continue
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                invalid_boxes += 1
                continue

            width, height = sample.width, sample.height
            x1 = (xc - bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            x2 = (xc + bw / 2.0) * width
            y2 = (yc + bh / 2.0) * height
            x1 = max(0.0, min(float(width - 1), x1))
            y1 = max(0.0, min(float(height - 1), y1))
            x2 = max(0.0, min(float(width - 1), x2))
            y2 = max(0.0, min(float(height - 1), y2))
            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
                continue

            class_idx = int(class_id)
            sample.detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=1.0, class_id=class_idx))
            class_counts[class_idx] += 1
            area_ratios.append(((x2 - x1) * (y2 - y1)) / float(width * height))
            aspect_ratios.append((x2 - x1) / (y2 - y1))

    inferred_class_names = dict(class_names)
    for class_id in class_counts.keys():
        inferred_class_names.setdefault(int(class_id), f"class_{int(class_id)}")

    return VizDataset(
        label_format="yolo",
        class_names=inferred_class_names,
        images=images,
        class_counts=class_counts,
        area_ratios=area_ratios,
        aspect_ratios=aspect_ratios,
        invalid_boxes=invalid_boxes,
        missing_images=missing_images,
        parse_errors=parse_errors,
        label_files_total=len(label_files),
    )


def _parse_coco(
    *,
    images_dir: Optional[Path],
    labels_dir: Path,
    class_names: Dict[int, str],
) -> VizDataset:
    coco = _load_coco_json(labels_dir)
    categories = {int(cat["id"]): str(cat.get("name", f"class_{int(cat['id'])}")) for cat in coco.get("categories", []) if "id" in cat}
    images_raw = {int(img["id"]): img for img in coco.get("images", []) if "id" in img}

    images: Dict[str, ImageSample] = {}
    by_image_id: Dict[int, str] = {}
    for image_id, img in images_raw.items():
        file_name = str(img.get("file_name", ""))
        key = Path(file_name).with_suffix("").as_posix() if file_name else f"image_{image_id}"
        image_path = _resolve_coco_image_path(images_dir, file_name) if images_dir is not None and file_name else None

        width = int(img.get("width") or 0)
        height = int(img.get("height") or 0)
        if (width <= 0 or height <= 0) and image_path is not None and image_path.exists():
            size = _read_image_size(image_path)
            if size is not None:
                width, height = size

        images[key] = ImageSample(
            key=key,
            image_path=image_path if image_path is not None and image_path.exists() else None,
            width=max(0, width),
            height=max(0, height),
            detections=[],
        )
        by_image_id[image_id] = key

    class_counts: Counter = Counter()
    area_ratios: List[float] = []
    aspect_ratios: List[float] = []
    invalid_boxes = 0
    missing_images = 0
    parse_errors = 0

    for sample in images.values():
        if sample.image_path is None:
            missing_images += 1

    annotations = coco.get("annotations", [])
    for ann in annotations:
        if not isinstance(ann, dict):
            parse_errors += 1
            continue
        image_id_raw = ann.get("image_id")
        category_id_raw = ann.get("category_id")
        bbox = ann.get("bbox")
        if image_id_raw is None or category_id_raw is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            parse_errors += 1
            continue
        try:
            image_id = int(image_id_raw)
            class_id = int(category_id_raw)
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        except Exception:
            parse_errors += 1
            continue

        key = by_image_id.get(image_id)
        if key is None:
            parse_errors += 1
            continue
        sample = images[key]

        if w <= 0.0 or h <= 0.0:
            invalid_boxes += 1
            continue

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        if sample.width > 0 and sample.height > 0:
            x1 = max(0.0, min(float(sample.width - 1), x1))
            y1 = max(0.0, min(float(sample.height - 1), y1))
            x2 = max(0.0, min(float(sample.width - 1), x2))
            y2 = max(0.0, min(float(sample.height - 1), y2))
            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
                continue
            area_ratios.append(((x2 - x1) * (y2 - y1)) / float(sample.width * sample.height))
        else:
            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
                continue

        sample.detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=1.0, class_id=class_id))
        class_counts[class_id] += 1
        aspect_ratios.append((x2 - x1) / (y2 - y1))

    inferred_class_names = dict(class_names)
    for class_id, cat_name in categories.items():
        inferred_class_names.setdefault(class_id, cat_name)
    for class_id in class_counts.keys():
        inferred_class_names.setdefault(int(class_id), f"class_{int(class_id)}")

    return VizDataset(
        label_format="coco",
        class_names=inferred_class_names,
        images=images,
        class_counts=class_counts,
        area_ratios=area_ratios,
        aspect_ratios=aspect_ratios,
        invalid_boxes=invalid_boxes,
        missing_images=missing_images,
        parse_errors=parse_errors,
        label_files_total=1,
    )


def _make_canvas(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _draw_text(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.6, color: Tuple[int, int, int] = (20, 20, 20)) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _write_class_distribution_png(
    *,
    class_counts: Counter,
    class_names: Dict[int, str],
    output_path: Path,
    max_bars: int,
) -> None:
    items = sorted(((int(k), int(v)) for k, v in class_counts.items()), key=lambda x: x[1], reverse=True)[:max(1, max_bars)]
    if not items:
        canvas = _make_canvas(1200, 400)
        _draw_text(canvas, "Class Distribution", 40, 50, 1.0)
        _draw_text(canvas, "No labels found.", 40, 120, 0.8)
        cv2.imwrite(str(output_path), canvas)
        return

    row_h = 30
    height = max(360, 120 + len(items) * row_h)
    canvas = _make_canvas(1400, height)
    _draw_text(canvas, "Class Distribution (Top classes)", 30, 50, 1.0)

    left = 340
    right = canvas.shape[1] - 80
    top = 90
    max_count = max(count for _, count in items)
    bar_w = max(1, right - left)

    for idx, (class_id, count) in enumerate(items):
        y1 = top + idx * row_h
        y2 = y1 + 18
        label = class_names.get(class_id, f"class_{class_id}")
        _draw_text(canvas, f"{label} ({class_id})", 30, y2, 0.55)
        width = int(round((count / max_count) * bar_w)) if max_count > 0 else 0
        cv2.rectangle(canvas, (left, y1), (left + width, y2), (76, 144, 245), thickness=-1)
        _draw_text(canvas, str(count), left + width + 8, y2, 0.55)

    cv2.imwrite(str(output_path), canvas)


def _write_histogram_png(
    *,
    values: List[float],
    title: str,
    x_label: str,
    output_path: Path,
    bins: int,
    value_range: Optional[Tuple[float, float]] = None,
) -> None:
    canvas = _make_canvas(1200, 650)
    _draw_text(canvas, title, 40, 50, 1.0)
    if not values:
        _draw_text(canvas, "No data.", 40, 120, 0.8)
        cv2.imwrite(str(output_path), canvas)
        return

    data = np.asarray(values, dtype=np.float64)
    hist_bins = max(5, int(bins))
    counts, edges = np.histogram(data, bins=hist_bins, range=value_range)
    max_count = int(np.max(counts)) if len(counts) else 0

    left = 80
    right = canvas.shape[1] - 40
    top = 80
    bottom = canvas.shape[0] - 130
    chart_w = max(1, right - left)
    chart_h = max(1, bottom - top)

    cv2.rectangle(canvas, (left, top), (right, bottom), (220, 220, 220), thickness=1)
    if max_count <= 0:
        _draw_text(canvas, "No data after histogram binning.", 40, 120, 0.8)
        cv2.imwrite(str(output_path), canvas)
        return

    bar_gap = 2
    slot_w = chart_w / float(len(counts))
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        x1 = int(left + idx * slot_w + bar_gap)
        x2 = int(left + (idx + 1) * slot_w - bar_gap)
        y2 = bottom
        y1 = int(bottom - (count / max_count) * chart_h)
        cv2.rectangle(canvas, (x1, y1), (max(x1 + 1, x2), y2), (49, 177, 105), thickness=-1)

    _draw_text(canvas, x_label, left, canvas.shape[0] - 35, 0.65)
    _draw_text(canvas, f"count max={max_count}", right - 170, top - 15, 0.5)
    _draw_text(canvas, f"{edges[0]:.4f}", left, bottom + 30, 0.5)
    _draw_text(canvas, f"{edges[-1]:.4f}", right - 80, bottom + 30, 0.5)

    cv2.imwrite(str(output_path), canvas)


def _resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image
    scale = max_side / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _safe_stem(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    collapsed = "".join(out).strip("_")
    return collapsed or "image"


def _write_sample_overlays(
    *,
    dataset: VizDataset,
    output_dir: Path,
    sample_count: int,
    sample_max_side: int,
    seed: int,
) -> List[str]:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    candidates = [sample for sample in dataset.images.values() if sample.image_path is not None and sample.detections]
    if not candidates or sample_count <= 0:
        return []

    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    candidates.sort(key=lambda s: len(s.detections), reverse=True)
    selected = candidates[:sample_count]

    written: List[str] = []
    used_names: Counter = Counter()
    for idx, sample in enumerate(selected, start=1):
        if sample.image_path is None:
            continue
        image = cv2.imread(str(sample.image_path))
        if image is None:
            continue

        image = _resize_max_side(image, sample_max_side)
        sx = image.shape[1] / float(max(1, sample.width))
        sy = image.shape[0] / float(max(1, sample.height))

        scaled: List[Detection] = []
        for det in sample.detections:
            scaled.append(
                Detection(
                    x1=det.x1 * sx,
                    y1=det.y1 * sy,
                    x2=det.x2 * sx,
                    y2=det.y2 * sy,
                    score=1.0,
                    class_id=det.class_id,
                )
            )

        rendered = draw_detections(
            image,
            scaled,
            class_names=dataset.class_names,
            show_score=False,
            box_thickness=2,
            font_scale=0.5,
            font_thickness=1,
        )
        stem = _safe_stem(sample.key.replace("/", "_"))
        used_names[stem] += 1
        suffix = f"_{used_names[stem]}" if used_names[stem] > 1 else ""
        name = f"{idx:03d}_{stem}{suffix}.jpg"
        out_path = sample_dir / name
        cv2.imwrite(str(out_path), rendered)
        written.append(str(Path("samples") / name))

    return written


def _build_summary(dataset: VizDataset) -> Dict[str, object]:
    boxes_per_image = [len(sample.detections) for sample in dataset.images.values()]
    total_boxes = int(sum(boxes_per_image))
    images_with_labels = int(sum(1 for count in boxes_per_image if count > 0))
    images_total = int(len(boxes_per_image))
    empty_images = images_total - images_with_labels

    summary = {
        "format": dataset.label_format,
        "images_total": images_total,
        "images_with_labels": images_with_labels,
        "images_without_labels": empty_images,
        "label_files_total": int(dataset.label_files_total),
        "total_boxes": total_boxes,
        "classes_total": int(len(dataset.class_counts)),
        "missing_images": int(dataset.missing_images),
        "invalid_boxes": int(dataset.invalid_boxes),
        "parse_errors": int(dataset.parse_errors),
        "boxes_per_image_mean": float(np.mean(boxes_per_image)) if boxes_per_image else 0.0,
        "boxes_per_image_p95": float(np.percentile(boxes_per_image, 95)) if boxes_per_image else 0.0,
        "box_area_ratio_mean": float(np.mean(dataset.area_ratios)) if dataset.area_ratios else 0.0,
        "box_area_ratio_p95": float(np.percentile(dataset.area_ratios, 95)) if dataset.area_ratios else 0.0,
        "box_aspect_ratio_mean": float(np.mean(dataset.aspect_ratios)) if dataset.aspect_ratios else 0.0,
        "box_aspect_ratio_p95": float(np.percentile(dataset.aspect_ratios, 95)) if dataset.aspect_ratios else 0.0,
    }
    return summary


def _write_report_markdown(
    *,
    output_dir: Path,
    summary: Dict[str, object],
    dataset: VizDataset,
    sample_images: List[str],
    include_extra_charts: bool,
) -> None:
    class_rows = sorted(((int(k), int(v)) for k, v in dataset.class_counts.items()), key=lambda x: x[1], reverse=True)
    lines: List[str] = []
    lines.append("# Dataset Label Visualization Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Format: {summary.get('format')}")
    lines.append(f"- Images total: {summary.get('images_total')}")
    lines.append(f"- Images with labels: {summary.get('images_with_labels')}")
    lines.append(f"- Total boxes: {summary.get('total_boxes')}")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    lines.append("### Class Distribution")
    lines.append("![Class distribution](class_distribution.png)")
    if include_extra_charts:
        lines.append("")
        lines.append("### Boxes per Image")
        lines.append("![Boxes per image](boxes_per_image_hist.png)")
        lines.append("")
        lines.append("### Box Area Ratio")
        lines.append("![Box area ratio](box_area_ratio_hist.png)")
        lines.append("")
        lines.append("### Box Aspect Ratio")
        lines.append("![Box aspect ratio](box_aspect_ratio_hist.png)")
    lines.append("")
    lines.append("## Top Classes")
    lines.append("")
    lines.append("| class_id | class_name | boxes |")
    lines.append("|---|---|---:|")
    for class_id, count in class_rows[:50]:
        class_name = dataset.class_names.get(class_id, f"class_{class_id}")
        lines.append(f"| {class_id} | {class_name} | {count} |")

    if sample_images:
        lines.append("")
        lines.append("## Sample Overlays")
        lines.append("")
        for rel in sample_images[:24]:
            lines.append(f"![{rel}]({rel})")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    dataset_dir, images_dir, labels_dir = _resolve_inputs(args)

    label_format = args.format if args.format != "auto" else _detect_format(labels_dir, args.recursive)
    if label_format not in {"yolo", "coco"}:
        raise SystemExit(f"Unsupported format for visualization: {label_format}")

    class_names: Dict[int, str] = {}
    if args.class_names:
        class_names = load_class_names(args.class_names)

    if label_format == "yolo":
        if images_dir is None:
            raise SystemExit("YOLO visualization requires images. Provide --dataset-dir or --images-dir.")
        dataset = _parse_yolo(
            images_dir=images_dir,
            labels_dir=labels_dir,
            recursive=bool(args.recursive),
            image_exts=_normalize_extensions(args.image_exts),
            label_ext=args.label_ext,
            class_names=class_names,
        )
    else:
        dataset = _parse_coco(
            images_dir=images_dir,
            labels_dir=labels_dir,
            class_names=class_names,
        )

    output_dir = Path(args.output_dir) if args.output_dir else (dataset_dir / "visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_summary(dataset)
    class_counts_json = {str(k): int(v) for k, v in sorted(dataset.class_counts.items(), key=lambda item: int(item[0]))}
    summary_payload = {
        "summary": summary,
        "class_counts": class_counts_json,
        "class_names": {str(k): v for k, v in sorted(dataset.class_names.items(), key=lambda item: int(item[0]))},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    _write_class_distribution_png(
        class_counts=dataset.class_counts,
        class_names=dataset.class_names,
        output_path=output_dir / "class_distribution.png",
        max_bars=int(args.max_class_bars),
    )

    include_extra_charts = not bool(args.distribution_only)
    if include_extra_charts:
        per_image_counts = [len(sample.detections) for sample in dataset.images.values()]
        _write_histogram_png(
            values=[float(value) for value in per_image_counts],
            title="Boxes per Image Distribution",
            x_label="boxes per image",
            output_path=output_dir / "boxes_per_image_hist.png",
            bins=int(args.hist_bins),
            value_range=None,
        )
        _write_histogram_png(
            values=[float(value) for value in dataset.area_ratios],
            title="Bounding Box Area Ratio Distribution",
            x_label="box area / image area",
            output_path=output_dir / "box_area_ratio_hist.png",
            bins=int(args.hist_bins),
            value_range=(0.0, 1.0),
        )
        _write_histogram_png(
            values=[float(value) for value in dataset.aspect_ratios],
            title="Bounding Box Aspect Ratio Distribution",
            x_label="box width / box height",
            output_path=output_dir / "box_aspect_ratio_hist.png",
            bins=int(args.hist_bins),
            value_range=None,
        )

    sample_images: List[str] = []
    if not bool(args.distribution_only) and images_dir is not None:
        sample_images = _write_sample_overlays(
            dataset=dataset,
            output_dir=output_dir,
            sample_count=int(args.sample_count),
            sample_max_side=int(args.sample_max_side),
            seed=int(args.seed),
        )
    _write_report_markdown(
        output_dir=output_dir,
        summary=summary,
        dataset=dataset,
        sample_images=sample_images,
        include_extra_charts=include_extra_charts,
    )

    print(f"[dataset_viz] wrote report -> {output_dir}")
    print(
        "[dataset_viz] summary: "
        f"images={summary['images_total']} labeled_images={summary['images_with_labels']} "
        f"boxes={summary['total_boxes']} classes={summary['classes_total']}"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)
