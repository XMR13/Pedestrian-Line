from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from pathlib import PurePosixPath
from typing import Dict, List, Tuple


def _backup_path_for(path: Path) -> Path:
    base = path.with_suffix(path.suffix + ".bak")
    if not base.exists():
        return base
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return path.with_suffix(path.suffix + f".bak_{stamp}")


def _resolve_paths(dataset_dir: Path | None, annotations: Path | None) -> Tuple[Path, Path | None]:
    if dataset_dir is not None:
        ann_path = annotations if annotations is not None else (dataset_dir / "annotations.json")
        return ann_path, dataset_dir
    if annotations is None:
        raise SystemExit("Provide --dataset-dir or --annotations.")
    return annotations, None


def cvat_fix_coco_ids(
    *,
    coco: Dict[str, object],
    start_category_id: int = 1,
    basename_file_names: bool = False,
) -> Tuple[Dict[str, object], Dict[int, int]]:
    """
    Make COCO categories compatible with CVAT imports by ensuring category IDs are >= 1
    and remapping annotation `category_id` accordingly.

    Strategy:
    - Build a deterministic mapping old_id -> new_id by sorting categories by old id.
    - Re-write categories and annotations using the new ids.
    - Add `supercategory` if missing (CVAT doesn't require it consistently, but it's harmless).
    """
    categories = coco.get("categories", [])
    if not isinstance(categories, list):
        raise SystemExit("Invalid COCO: `categories` must be a list.")
    if not categories:
        raise SystemExit("Invalid COCO: missing `categories` (CVAT needs labels).")

    # Validate + build mapping.
    cat_ids: List[int] = []
    for c in categories:
        if not isinstance(c, dict) or "id" not in c or "name" not in c:
            raise SystemExit("Invalid COCO: each category must contain `id` and `name`.")
        try:
            cat_ids.append(int(c["id"]))
        except Exception:
            raise SystemExit("Invalid COCO: category `id` must be int-like.")

    old_ids_sorted = sorted(set(cat_ids))
    need_remap = bool(old_ids_sorted) and min(old_ids_sorted) < int(start_category_id)

    id_map: Dict[int, int] = {}
    if need_remap:
        next_id = int(start_category_id)
        for old in old_ids_sorted:
            id_map[int(old)] = next_id
            next_id += 1
    else:
        id_map = {int(i): int(i) for i in old_ids_sorted}

    new_categories: List[Dict[str, object]] = []
    for c in categories:
        old = int(c["id"])
        new = id_map[old]
        cc = dict(c)
        cc["id"] = int(new)
        cc.setdefault("supercategory", "")
        new_categories.append(cc)

    annotations = coco.get("annotations", [])
    if not isinstance(annotations, list):
        raise SystemExit("Invalid COCO: `annotations` must be a list.")
    new_annotations: List[Dict[str, object]] = []
    for a in annotations:
        if not isinstance(a, dict):
            continue
        if "category_id" not in a:
            raise SystemExit("Invalid COCO: annotation missing `category_id`.")
        old = int(a["category_id"])
        if old not in id_map:
            raise SystemExit(f"Invalid COCO: annotation category_id={old} not found in categories.")
        aa = dict(a)
        aa["category_id"] = int(id_map[old])
        new_annotations.append(aa)

    out = dict(coco)
    out["categories"] = new_categories
    out["annotations"] = new_annotations

    if basename_file_names:
        images = out.get("images", [])
        if not isinstance(images, list):
            raise SystemExit("Invalid COCO: `images` must be a list.")
        for img in images:
            if not isinstance(img, dict):
                continue
            file_name = img.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                continue
            normalized = file_name.replace("\\", "/").lstrip("./")
            img["file_name"] = PurePosixPath(normalized).name

    info = out.get("info")
    if not isinstance(info, dict):
        info = {}
    info = dict(info)
    info.setdefault("yolo_kitv2", {})
    if isinstance(info["yolo_kitv2"], dict):
        info["yolo_kitv2"].setdefault("category_id_map", {str(k): v for k, v in id_map.items()})
        if basename_file_names:
            info["yolo_kitv2"].setdefault("file_name_basename", True)
    out["info"] = info

    return out, id_map


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Make COCO JSON more compatible with CVAT imports.")
    p.add_argument("--dataset-dir", type=str, default=None, help="Dataset dir containing annotations.json.")
    p.add_argument("--annotations", type=str, default=None, help="Path to COCO annotations JSON.")
    p.add_argument("--in-place", action="store_true", help="Overwrite the annotations file.")
    p.add_argument("--no-backup", action="store_true", help="Disable automatic .bak backup when using --in-place.")
    p.add_argument("--output", type=str, default=None, help="Output JSON path when not using --in-place.")
    p.add_argument(
        "--start-category-id",
        type=int,
        default=1,
        help="First category id to use when remapping (default 1).",
    )
    p.add_argument(
        "--basename-file-names",
        action="store_true",
        help="Rewrite images[].file_name to basename only (strip folders like 'images/').",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    annotations = Path(args.annotations) if args.annotations else None

    ann_path, _ = _resolve_paths(dataset_dir, annotations)
    if not ann_path.exists():
        raise SystemExit(f"Annotations not found: {ann_path}")

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    fixed, id_map = cvat_fix_coco_ids(
        coco=coco,
        start_category_id=int(args.start_category_id),
        basename_file_names=bool(args.basename_file_names),
    )

    if args.in_place:
        if not args.no_backup:
            backup = _backup_path_for(ann_path)
            shutil.copy2(ann_path, backup)
            print(f"[coco_cvat_fix] wrote backup: {backup}")
        out_path = ann_path
    else:
        out_path = Path(args.output) if args.output else ann_path.with_name(ann_path.stem + "_cvat.json")

    out_path.write_text(json.dumps(fixed, indent=2), encoding="utf-8")
    print(f"[coco_cvat_fix] remapped {len(id_map)} categories -> {out_path}")
    return 0
