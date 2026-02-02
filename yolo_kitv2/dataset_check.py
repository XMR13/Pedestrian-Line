from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .runtime import resolve_path


PathLike = Union[str, Path]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SplitReport:
    name: str
    images: int = 0
    labels_found: int = 0
    labels_missing: int = 0
    labels_empty: int = 0
    labels_orphan: int = 0
    boxes_total: int = 0
    boxes_invalid: int = 0
    boxes_out_of_range: int = 0
    invalid_lines: int = 0
    class_counts: Dict[int, int] = field(default_factory=dict)
    image_paths: List[Path] = field(default_factory=list)
    image_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    box_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    image_size_samples: int = 0
    image_size_missing: int = 0


@dataclass
class DatasetReport:
    format: str
    source: Path
    root: Optional[Path]
    splits: Dict[str, SplitReport] = field(default_factory=dict)
    classes: Dict[int, str] = field(default_factory=dict)
    extra: Dict[str, int] = field(default_factory=dict)
    quality: str = "standard"


def _read_text_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen: Set[Path] = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def _list_images_in_dir(path: Path) -> List[Path]:
    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return files


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        median = sorted_vals[mid]
    else:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return {
        "count": float(n),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "mean": float(total / n),
        "median": float(median),
    }


def _augment_with_duplicates(report: DatasetReport) -> None:
    image_paths: List[Path] = []
    for split in report.splits.values():
        if split.image_paths:
            image_paths.extend(split.image_paths)
    if not image_paths:
        return
    hashes: Dict[str, List[Path]] = {}
    failures = 0
    for p in image_paths:
        h = _hash_file(p)
        if h is None:
            failures += 1
            continue
        hashes.setdefault(h, []).append(p)
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    if duplicate_groups:
        report.extra["duplicate_images"] = sum(len(g) for g in duplicate_groups)
        report.extra["duplicate_groups"] = len(duplicate_groups)
    if failures:
        report.extra["hash_failures"] = failures


def _get_image_size_reader() -> Optional[Callable[[Path], Optional[Tuple[int, int]]]]:
    try:
        from PIL import Image  # type: ignore

        def _pil_reader(p: Path) -> Optional[Tuple[int, int]]:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                return int(w), int(h)
            except Exception:
                return None

        return _pil_reader
    except Exception:
        pass
    try:
        import cv2  # type: ignore

        def _cv_reader(p: Path) -> Optional[Tuple[int, int]]:
            try:
                img = cv2.imread(str(p))
            except Exception:
                return None
            if img is None or not hasattr(img, "shape"):
                return None
            h, w = img.shape[:2]
            return int(w), int(h)

        return _cv_reader
    except Exception:
        return None


def _resolve_image_spec(spec: Union[str, Path], base: Path) -> Tuple[List[Path], List[Path]]:
    spec_path = Path(spec)
    if not spec_path.is_absolute():
        spec_path = (base / spec_path).resolve()

    image_dirs: List[Path] = []
    images: List[Path] = []

    if spec_path.exists() and spec_path.is_dir():
        image_dirs.append(spec_path)
        images.extend(_list_images_in_dir(spec_path))
        return _dedupe_paths(images), image_dirs

    if spec_path.exists() and spec_path.is_file() and spec_path.suffix.lower() == ".txt":
        for line in _read_text_lines(spec_path):
            p = Path(line)
            if not p.is_absolute():
                p = (spec_path.parent / p).resolve()
            if p.exists() and p.is_file():
                images.append(p)
        return _dedupe_paths(images), image_dirs

    if any(ch in str(spec) for ch in ["*", "?", "["]):
        pattern = str(spec_path) if spec_path.is_absolute() else str(base / spec)
        for p in glob.glob(pattern, recursive=True):
            pp = Path(p)
            if pp.is_file() and pp.suffix.lower() in IMAGE_EXTS:
                images.append(pp.resolve())
        return _dedupe_paths(images), image_dirs

    return _dedupe_paths(images), image_dirs


def _infer_labels_dir(images_dir: Path) -> Optional[Path]:
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        candidate = Path(*parts[:idx], "labels", *parts[idx + 1 :])
        if candidate.exists():
            return candidate
    candidate = images_dir.parent / "labels" / images_dir.name
    if candidate.exists():
        return candidate
    candidate = images_dir.parent / "labels"
    if candidate.exists():
        return candidate
    return None


def _label_path_for_image(image_path: Path, labels_dir: Optional[Path]) -> Path:
    if labels_dir is not None:
        return (labels_dir / image_path.stem).with_suffix(".txt")
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _parse_class_id(token: str) -> Optional[int]:
    try:
        v = float(token)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    if abs(v - round(v)) > 1e-3:
        return None
    return int(round(v))


def _parse_float(token: str) -> Optional[float]:
    try:
        v = float(token)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _load_yaml_minimal(path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    lines = _read_text_lines(path)
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" not in line:
            i += 1
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if key == "names" and rest == "":
            names_map: Dict[int, str] = {}
            names_list: List[str] = []
            j = i + 1
            while j < len(lines):
                nline = lines[j]
                if nline.startswith("-"):
                    item = nline[1:].strip().strip("'").strip('"')
                    names_list.append(item)
                    j += 1
                    continue
                if ":" in nline:
                    left, right = nline.split(":", 1)
                    left = left.strip()
                    right = right.strip().strip("'").strip('"')
                    if left.isdigit():
                        names_map[int(left)] = right
                        j += 1
                        continue
                break
            if names_list:
                data["names"] = names_list
            elif names_map:
                data["names"] = names_map
            i = j
            continue

        if rest.startswith("[") or rest.startswith("{"):
            try:
                import ast

                data[key] = ast.literal_eval(rest)
            except Exception:
                data[key] = rest
        elif rest == "":
            data[key] = ""
        else:
            data[key] = rest.strip("'").strip('"')
        i += 1
    return data


def _load_yaml(path: Path) -> Dict[str, object]:
    try:
        import yaml  # type: ignore
    except Exception:
        return _load_yaml_minimal(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping at top level.")
    return data


def _load_config(path: Path) -> Dict[str, object]:
    if path.suffix.lower() in {".yml", ".yaml"}:
        return _load_yaml(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported config file: {path}")


def _detect_format(cfg: Optional[Dict[str, object]], data_path: Path, user_format: str) -> str:
    if user_format != "auto":
        return user_format
    if cfg and isinstance(cfg.get("format"), str):
        fmt = str(cfg["format"]).lower()
        if fmt in {"yolo", "yolo-folders", "coco"}:
            return fmt
    if data_path.is_dir():
        return "yolo-folders"
    if data_path.suffix.lower() == ".json" and cfg:
        if "images" in cfg and "annotations" in cfg:
            return "coco"
    if data_path.suffix.lower() in {".yml", ".yaml"}:
        return "yolo"
    raise ValueError("Could not infer dataset format; pass --format explicitly.")


def _as_class_names(names_obj: object) -> Dict[int, str]:
    if isinstance(names_obj, dict):
        out: Dict[int, str] = {}
        for k, v in names_obj.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    if isinstance(names_obj, list):
        return {i: str(v) for i, v in enumerate(names_obj)}
    return {}


def _resolve_split_paths(
    cfg: Dict[str, object],
    *,
    data_path: Path,
    root: Optional[Path],
    split_key: str,
    override: Optional[Union[str, Sequence[str]]],
) -> Tuple[List[Path], List[Path], Optional[Path]]:
    base = root or data_path.parent
    if cfg.get("path"):
        base = resolve_path(str(cfg["path"]), root=base)
    images_dirs: List[Path] = []
    images: List[Path] = []

    spec = override if override is not None else cfg.get(split_key)
    if spec is None:
        return [], [], None
    if isinstance(spec, (list, tuple)):
        for s in spec:
            imgs, dirs = _resolve_image_spec(s, base)
            images.extend(imgs)
            images_dirs.extend(dirs)
    else:
        imgs, dirs = _resolve_image_spec(spec, base)
        images.extend(imgs)
        images_dirs.extend(dirs)
    labels_dir = None
    if images_dirs:
        labels_dir = _infer_labels_dir(images_dirs[0])
    return images, images_dirs, labels_dir


def _check_yolo_split(
    name: str,
    images: List[Path],
    labels_dir: Optional[Path],
    *,
    explicit_labels_dir: Optional[Path] = None,
    quality: str = "standard",
    size_reader: Optional[Callable[[Path], Optional[Tuple[int, int]]]] = None,
) -> SplitReport:
    report = SplitReport(name=name)
    report.images = len(images)
    if explicit_labels_dir is not None:
        labels_dir = explicit_labels_dir
    if quality == "extended":
        report.image_paths = [p.resolve() for p in images]

    image_sizes: Dict[Path, Tuple[int, int]] = {}
    if quality == "extended" and size_reader is not None:
        for img in images:
            size = size_reader(img)
            if size is None:
                report.image_size_missing += 1
                continue
            image_sizes[img.resolve()] = size
            report.image_size_samples += 1

    expected_labels: Dict[Path, Path] = {}
    for img in images:
        label_path = _label_path_for_image(img, labels_dir)
        expected_labels[img.resolve()] = label_path.resolve()

    widths: List[float] = []
    heights: List[float] = []
    box_w_norm: List[float] = []
    box_h_norm: List[float] = []
    box_area_norm: List[float] = []
    box_aspect_norm: List[float] = []
    box_w_px: List[float] = []
    box_h_px: List[float] = []
    box_area_px: List[float] = []

    for img, label_path in expected_labels.items():
        if not label_path.exists():
            report.labels_missing += 1
            continue
        report.labels_found += 1
        valid_boxes = 0
        try:
            lines = _read_text_lines(label_path)
        except Exception:
            report.invalid_lines += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                report.invalid_lines += 1
                continue
            class_id = _parse_class_id(parts[0])
            coords = [_parse_float(p) for p in parts[1:5]]
            if class_id is None or any(v is None for v in coords):
                report.invalid_lines += 1
                continue
            x, y, w, h = coords
            if w <= 0 or h <= 0:
                report.boxes_invalid += 1
            in_range = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0
            if not in_range:
                report.boxes_out_of_range += 1
            report.boxes_total += 1
            if quality in {"standard", "extended"}:
                report.class_counts[class_id] = report.class_counts.get(class_id, 0) + 1
            valid_boxes += 1

            if quality == "extended" and in_range and w > 0 and h > 0:
                box_w_norm.append(float(w))
                box_h_norm.append(float(h))
                box_area_norm.append(float(w * h))
                box_aspect_norm.append(float(w / h))
                if img in image_sizes:
                    iw, ih = image_sizes[img]
                    box_w_px.append(float(w * iw))
                    box_h_px.append(float(h * ih))
                    box_area_px.append(float(w * h * iw * ih))
        if valid_boxes == 0:
            report.labels_empty += 1

    if labels_dir is not None and labels_dir.exists():
        label_files = {p.resolve() for p in labels_dir.rglob("*.txt") if p.is_file()}
        expected = set(expected_labels.values())
        report.labels_orphan = len(label_files - expected)

    if quality == "extended":
        if image_sizes:
            widths = [float(sz[0]) for sz in image_sizes.values()]
            heights = [float(sz[1]) for sz in image_sizes.values()]
            report.image_stats["width"] = _stats(widths)
            report.image_stats["height"] = _stats(heights)
        if box_w_norm:
            report.box_stats["w_norm"] = _stats(box_w_norm)
            report.box_stats["h_norm"] = _stats(box_h_norm)
            report.box_stats["area_norm"] = _stats(box_area_norm)
            report.box_stats["aspect"] = _stats(box_aspect_norm)
        if box_w_px:
            report.box_stats["w_px"] = _stats(box_w_px)
            report.box_stats["h_px"] = _stats(box_h_px)
            report.box_stats["area_px"] = _stats(box_area_px)

    return report


def _check_yolo(
    cfg: Dict[str, object],
    *,
    data_path: Path,
    root: Optional[Path],
    override_train: Optional[Union[str, Sequence[str]]],
    override_val: Optional[Union[str, Sequence[str]]],
    override_test: Optional[Union[str, Sequence[str]]],
    labels_dir: Optional[Path],
    quality: str,
    size_reader: Optional[Callable[[Path], Optional[Tuple[int, int]]]],
) -> DatasetReport:
    report = DatasetReport(format="yolo", source=data_path, root=root, quality=quality)
    report.classes = _as_class_names(cfg.get("names", {}))

    for split_name, override in [
        ("train", override_train),
        ("val", override_val),
        ("test", override_test),
    ]:
        cfg_key = split_name
        if cfg_key not in cfg and override is None:
            continue
        images, _, inferred_labels = _resolve_split_paths(
            cfg,
            data_path=data_path,
            root=root,
            split_key=cfg_key,
            override=override,
        )
        if not images:
            continue
        split_report = _check_yolo_split(
            split_name,
            images,
            inferred_labels,
            explicit_labels_dir=labels_dir,
            quality=quality,
            size_reader=size_reader,
        )
        report.splits[split_name] = split_report

    return report


def _check_yolo_folders(
    cfg: Optional[Dict[str, object]],
    *,
    data_path: Path,
    root: Optional[Path],
    train: Optional[str],
    val: Optional[str],
    test: Optional[str],
    labels_dir: Optional[Path],
    quality: str,
    size_reader: Optional[Callable[[Path], Optional[Tuple[int, int]]]],
) -> DatasetReport:
    report = DatasetReport(format="yolo-folders", source=data_path, root=root, quality=quality)

    base = root or (data_path.parent if data_path.is_file() else data_path)
    if cfg and cfg.get("path"):
        base = resolve_path(str(cfg["path"]), root=base)

    splits = {
        "train": train or (str(cfg.get("train")) if cfg and cfg.get("train") else None),
        "val": val or (str(cfg.get("val")) if cfg and cfg.get("val") else None),
        "test": test or (str(cfg.get("test")) if cfg and cfg.get("test") else None),
    }
    if all(v is None for v in splits.values()):
        for split in ["train", "val", "test"]:
            images_dir = base / "images" / split
            if images_dir.exists():
                splits[split] = str(images_dir)

    for split_name, spec in splits.items():
        if spec is None:
            continue
        images, image_dirs = _resolve_image_spec(spec, base)
        if not images:
            continue
        inferred = _infer_labels_dir(image_dirs[0]) if image_dirs else None
        split_report = _check_yolo_split(
            split_name,
            images,
            inferred,
            explicit_labels_dir=labels_dir,
            quality=quality,
            size_reader=size_reader,
        )
        report.splits[split_name] = split_report
    return report


def _check_coco(
    cfg: Dict[str, object],
    *,
    data_path: Path,
    root: Optional[Path],
    image_root: Optional[Path],
    quality: str,
) -> DatasetReport:
    report = DatasetReport(format="coco", source=data_path, root=root, quality=quality)
    images = cfg.get("images") or []
    annotations = cfg.get("annotations") or []
    categories = cfg.get("categories") or []

    if isinstance(categories, list):
        for c in categories:
            if isinstance(c, dict) and "id" in c and "name" in c:
                report.classes[int(c["id"])] = str(c["name"])

    image_index: Dict[int, Dict[str, object]] = {}
    for img in images:
        if not isinstance(img, dict):
            continue
        if "id" not in img:
            continue
        image_index[int(img["id"])] = img

    image_count = len(image_index)
    ann_count = 0
    invalid_boxes = 0
    out_of_range = 0
    orphan_annotations = 0
    class_counts: Dict[int, int] = {}
    widths: List[float] = []
    heights: List[float] = []
    box_w: List[float] = []
    box_h: List[float] = []
    box_area: List[float] = []
    box_aspect: List[float] = []

    if quality == "extended":
        for img in image_index.values():
            iw = img.get("width")
            ih = img.get("height")
            if isinstance(iw, (int, float)) and isinstance(ih, (int, float)):
                widths.append(float(iw))
                heights.append(float(ih))
    images_with_ann = set()

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        ann_count += 1
        image_id = ann.get("image_id")
        if image_id is None or int(image_id) not in image_index:
            orphan_annotations += 1
            continue
        images_with_ann.add(int(image_id))
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            invalid_boxes += 1
        else:
            x, y, w, h = bbox[:4]
            if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in [x, y, w, h]):
                invalid_boxes += 1
            elif w <= 0 or h <= 0:
                invalid_boxes += 1
            else:
                img = image_index[int(image_id)]
                iw = img.get("width")
                ih = img.get("height")
                if isinstance(iw, (int, float)) and isinstance(ih, (int, float)):
                    if x < 0 or y < 0 or x + w > iw or y + h > ih:
                        out_of_range += 1
        cid = ann.get("category_id")
        if cid is not None:
            try:
                cid_i = int(cid)
                class_counts[cid_i] = class_counts.get(cid_i, 0) + 1
            except Exception:
                pass
        if quality == "extended":
            bbox = ann.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                if all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in [x, y, w, h]):
                    if w > 0 and h > 0:
                        box_w.append(float(w))
                        box_h.append(float(h))
                        box_area.append(float(w * h))
                        box_aspect.append(float(w / h))

    missing_images = 0
    image_paths: List[Path] = []
    if image_root is not None:
        for img in image_index.values():
            fname = img.get("file_name")
            if not isinstance(fname, str):
                continue
            p = Path(fname)
            if not p.is_absolute():
                p = (image_root / p).resolve()
            if not p.exists():
                missing_images += 1
            else:
                image_paths.append(p)
    elif quality == "extended":
        for img in image_index.values():
            fname = img.get("file_name")
            if not isinstance(fname, str):
                continue
            p = Path(fname)
            if p.is_absolute() and p.exists():
                image_paths.append(p)

    split_report = SplitReport(name="all")
    split_report.images = image_count
    split_report.labels_found = ann_count
    split_report.labels_missing = image_count - len(images_with_ann)
    split_report.labels_orphan = orphan_annotations
    split_report.boxes_total = ann_count
    split_report.boxes_invalid = invalid_boxes
    split_report.boxes_out_of_range = out_of_range
    split_report.class_counts = class_counts
    if quality == "extended":
        if widths:
            split_report.image_stats["width"] = _stats(widths)
            split_report.image_stats["height"] = _stats(heights)
        if box_w:
            split_report.box_stats["w_px"] = _stats(box_w)
            split_report.box_stats["h_px"] = _stats(box_h)
            split_report.box_stats["area_px"] = _stats(box_area)
            split_report.box_stats["aspect"] = _stats(box_aspect)
    if quality == "extended" and image_paths:
        split_report.image_paths = [p.resolve() for p in image_paths]
    report.splits["all"] = split_report
    if missing_images:
        report.extra["missing_images"] = missing_images
    return report


def _format_class_counts(class_counts: Dict[int, int], classes: Dict[int, str]) -> str:
    if not class_counts:
        return "-"
    items = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)
    parts = []
    for cid, count in items:
        name = classes.get(cid, str(cid))
        parts.append(f"{name}:{count}")
    return ", ".join(parts)


def _format_stats_block(stats: Dict[str, float]) -> str:
    if not stats:
        return "-"
    count = int(stats.get("count", 0))
    return (
        f"n={count} min={stats['min']:.2f} mean={stats['mean']:.2f} "
        f"median={stats['median']:.2f} max={stats['max']:.2f}"
    )


def _print_yolo_report(report: DatasetReport) -> None:
    print(f"Dataset check (format: {report.format})")
    print(f"Source: {report.source}")
    if report.root:
        print(f"Root: {report.root}")
    print("")
    total_images = 0
    total_labels_found = 0
    total_labels_missing = 0
    total_labels_empty = 0
    total_labels_orphan = 0
    total_boxes = 0
    total_invalid = 0
    total_out = 0
    total_invalid_lines = 0
    total_class_counts: Dict[int, int] = {}

    for split in report.splits.values():
        total_images += split.images
        total_labels_found += split.labels_found
        total_labels_missing += split.labels_missing
        total_labels_empty += split.labels_empty
        total_labels_orphan += split.labels_orphan
        total_boxes += split.boxes_total
        total_invalid += split.boxes_invalid
        total_out += split.boxes_out_of_range
        total_invalid_lines += split.invalid_lines
        for cid, cnt in split.class_counts.items():
            total_class_counts[cid] = total_class_counts.get(cid, 0) + cnt

    print("Summary")
    print(f"  Images: {total_images}")
    print(f"  Labels: found {total_labels_found} | missing {total_labels_missing} | empty {total_labels_empty} | orphan {total_labels_orphan}")
    print(f"  Boxes: total {total_boxes} | invalid {total_invalid} | out_of_range {total_out}")
    if total_invalid_lines:
        print(f"  Invalid lines: {total_invalid_lines}")
    print(f"  Classes: {_format_class_counts(total_class_counts, report.classes)}")
    if report.quality == "extended":
        if report.extra.get("duplicate_images"):
            print(
                f"  Duplicates: {report.extra.get('duplicate_images', 0)} images "
                f"in {report.extra.get('duplicate_groups', 0)} groups"
            )
        if report.extra.get("hash_failures"):
            print(f"  Hash failures: {report.extra.get('hash_failures', 0)}")
        size_stats_available = any(s.image_stats for s in report.splits.values())
        if not size_stats_available and total_images > 0:
            print("  Image size stats: unavailable (install pillow or opencv-python)")
    print("")

    for split in report.splits.values():
        print(f"{split.name}")
        print(f"  Images: {split.images}")
        print(f"  Labels: found {split.labels_found} | missing {split.labels_missing} | empty {split.labels_empty} | orphan {split.labels_orphan}")
        print(f"  Boxes: total {split.boxes_total} | invalid {split.boxes_invalid} | out_of_range {split.boxes_out_of_range}")
        if split.invalid_lines:
            print(f"  Invalid lines: {split.invalid_lines}")
        print(f"  Classes: {_format_class_counts(split.class_counts, report.classes)}")
        if report.quality == "extended":
            if split.image_stats:
                print(f"  Image width: {_format_stats_block(split.image_stats.get('width', {}))}")
                print(f"  Image height: {_format_stats_block(split.image_stats.get('height', {}))}")
            if split.box_stats:
                if "w_norm" in split.box_stats:
                    print(f"  Box w (norm): {_format_stats_block(split.box_stats.get('w_norm', {}))}")
                    print(f"  Box h (norm): {_format_stats_block(split.box_stats.get('h_norm', {}))}")
                    print(f"  Box area (norm): {_format_stats_block(split.box_stats.get('area_norm', {}))}")
                if "w_px" in split.box_stats:
                    print(f"  Box w (px): {_format_stats_block(split.box_stats.get('w_px', {}))}")
                    print(f"  Box h (px): {_format_stats_block(split.box_stats.get('h_px', {}))}")
                    print(f"  Box area (px): {_format_stats_block(split.box_stats.get('area_px', {}))}")
                if "aspect" in split.box_stats:
                    print(f"  Box aspect: {_format_stats_block(split.box_stats.get('aspect', {}))}")
        print("")


def _print_coco_report(report: DatasetReport) -> None:
    split = report.splits.get("all")
    print("Dataset check (format: coco)")
    print(f"Source: {report.source}")
    if report.root:
        print(f"Root: {report.root}")
    print("")
    if split is None:
        print("No data found.")
        return
    print("Summary")
    print(f"  Images: {split.images}")
    print(f"  Annotations: {split.labels_found}")
    print(f"  Images without annotations: {split.labels_missing}")
    print(f"  Orphan annotations: {split.labels_orphan}")
    print(f"  Boxes: total {split.boxes_total} | invalid {split.boxes_invalid} | out_of_range {split.boxes_out_of_range}")
    if report.extra.get("missing_images"):
        print(f"  Missing image files: {report.extra['missing_images']}")
    print(f"  Classes: {_format_class_counts(split.class_counts, report.classes)}")
    if report.quality == "extended":
        if report.extra.get("duplicate_images"):
            print(
                f"  Duplicates: {report.extra.get('duplicate_images', 0)} images "
                f"in {report.extra.get('duplicate_groups', 0)} groups"
            )
        if report.extra.get("hash_failures"):
            print(f"  Hash failures: {report.extra.get('hash_failures', 0)}")
        if split.image_stats:
            print(f"  Image width: {_format_stats_block(split.image_stats.get('width', {}))}")
            print(f"  Image height: {_format_stats_block(split.image_stats.get('height', {}))}")
        if split.box_stats:
            if "w_px" in split.box_stats:
                print(f"  Box w (px): {_format_stats_block(split.box_stats.get('w_px', {}))}")
                print(f"  Box h (px): {_format_stats_block(split.box_stats.get('h_px', {}))}")
                print(f"  Box area (px): {_format_stats_block(split.box_stats.get('area_px', {}))}")
            if "aspect" in split.box_stats:
                print(f"  Box aspect: {_format_stats_block(split.box_stats.get('aspect', {}))}")
    print("")


def run_dataset_check(
    *,
    data: PathLike,
    fmt: str = "auto",
    root: Optional[PathLike] = None,
    quality: str = "extended",
    train: Optional[str] = None,
    val: Optional[str] = None,
    test: Optional[str] = None,
    labels: Optional[PathLike] = None,
    image_root: Optional[PathLike] = None,
) -> DatasetReport:
    data_path = resolve_path(data, root=root)
    root_path = resolve_path(root, root="auto") if root is not None else None
    cfg = _load_config(data_path) if data_path.is_file() else None
    fmt = _detect_format(cfg, data_path, fmt)

    labels_dir = resolve_path(labels, root=root_path) if labels is not None else None
    image_root_path = resolve_path(image_root, root=root_path) if image_root is not None else None
    size_reader = _get_image_size_reader() if quality == "extended" else None

    if fmt == "yolo":
        if cfg is None:
            raise ValueError("YOLO format requires a YAML config file.")
        report = _check_yolo(
            cfg,
            data_path=data_path,
            root=root_path,
            override_train=train,
            override_val=val,
            override_test=test,
            labels_dir=labels_dir,
            quality=quality,
            size_reader=size_reader,
        )
        if quality == "extended":
            _augment_with_duplicates(report)
        return report
    if fmt == "yolo-folders":
        report = _check_yolo_folders(
            cfg,
            data_path=data_path,
            root=root_path,
            train=train,
            val=val,
            test=test,
            labels_dir=labels_dir,
            quality=quality,
            size_reader=size_reader,
        )
        if quality == "extended":
            _augment_with_duplicates(report)
        return report
    if fmt == "coco":
        if cfg is None:
            raise ValueError("COCO format requires a JSON annotation file.")
        report = _check_coco(cfg, data_path=data_path, root=root_path, image_root=image_root_path, quality=quality)
        if quality == "extended":
            _augment_with_duplicates(report)
        return report
    raise ValueError(f"Unsupported format: {fmt}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="YOLO Kit dataset checker")
    parser.add_argument("--data", required=True, help="Dataset config (data.yaml, coco.json) or dataset root dir")
    parser.add_argument("--format", default="auto", choices=["auto", "yolo", "yolo-folders", "coco"])
    parser.add_argument("--quality", default="extended", choices=["minimal", "standard", "extended"])
    parser.add_argument("--root", help="Base path to resolve relative dataset paths")
    parser.add_argument("--train", help="Override train path (dir or txt list)")
    parser.add_argument("--val", help="Override val path (dir or txt list)")
    parser.add_argument("--test", help="Override test path (dir or txt list)")
    parser.add_argument("--labels", help="Explicit labels directory (for YOLO)")
    parser.add_argument("--image-root", help="Root folder for COCO image files")

    args = parser.parse_args(argv)

    try:
        report = run_dataset_check(
            data=args.data,
            fmt=args.format,
            quality=args.quality,
            root=args.root,
            train=args.train,
            val=args.val,
            test=args.test,
            labels=args.labels,
            image_root=args.image_root,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    if report.format in {"yolo", "yolo-folders"}:
        _print_yolo_report(report)
    else:
        _print_coco_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
