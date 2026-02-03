from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _resolve_paths(
    dataset_dir: Path | None,
    annotations: Path | None,
    images_dir: Path | None,
) -> Tuple[Path, Path]:
    ann_path: Path
    img_dir: Path

    if dataset_dir is not None:
        ann_path = annotations if annotations is not None else (dataset_dir / "annotations.json")
        img_dir = images_dir if images_dir is not None else (dataset_dir / "images")
    else:
        if annotations is None:
            raise SystemExit("Provide --dataset-dir or --annotations.")
        ann_path = annotations
        img_dir = images_dir if images_dir is not None else (ann_path.parent / "images")

    if not ann_path.exists():
        raise SystemExit(f"Annotations not found: {ann_path}")
    if not img_dir.exists():
        raise SystemExit(f"Images dir not found: {img_dir}")

    return ann_path, img_dir


def _backup_path_for(ann_path: Path) -> Path:
    base = ann_path.with_suffix(ann_path.suffix + ".bak")
    if not base.exists():
        return base
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return ann_path.with_suffix(ann_path.suffix + f".bak_{stamp}")


def _image_path(images_dir: Path, file_name: str) -> Path:
    """
    Resolve COCO `file_name` to an on-disk path.

    Common layouts:
    - dataset root has `images/` and COCO file_name is `images/foo.png`
    - dataset root has `images/` and COCO file_name is `foo.png`
    """
    normalized = str(file_name).replace("\\", "/").lstrip("./")
    raw = Path(normalized)
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


def prune_coco(
    *,
    ann_path: Path,
    images_dir: Path,
    in_place: bool,
    output: Path | None,
    dry_run: bool,
    force: bool,
    make_backup: bool,
) -> Tuple[Path | None, Dict[str, int]]:
    coco: Dict[str, object] = json.loads(ann_path.read_text(encoding="utf-8"))
    images: List[Dict[str, object]] = list(coco.get("images", []))
    annotations_list: List[Dict[str, object]] = list(coco.get("annotations", []))

    kept_images: List[Dict[str, object]] = []
    kept_image_ids: Set[int] = set()
    missing_images = 0

    for image in images:
        file_name = str(image.get("file_name", ""))
        path = _image_path(images_dir, file_name)
        if path.exists():
            kept_images.append(image)
            kept_image_ids.add(int(image["id"]))
        else:
            missing_images += 1

    kept_annotations: List[Dict[str, object]] = [
        ann for ann in annotations_list if int(ann.get("image_id", -1)) in kept_image_ids
    ]

    # Safety: a path mismatch frequently looks like "everything missing".
    if not force:
        if images and not kept_images:
            raise SystemExit(
                "[coco_prune] Refusing to prune: would remove ALL images from COCO JSON. "
                "This is usually a path mismatch (e.g. file_name includes 'images/...' while images_dir is also '.../images'). "
                "Use --dry-run to inspect, fix paths, or pass --force if intended."
            )
        if annotations_list and not kept_annotations:
            raise SystemExit(
                "[coco_prune] Refusing to prune: would remove ALL annotations from COCO JSON. "
                "Use --force if intended."
            )

    pruned = dict(coco)
    pruned["images"] = kept_images
    pruned["annotations"] = kept_annotations

    if in_place:
        output_path = ann_path
    elif output is not None:
        output_path = output
    else:
        output_path = ann_path.with_name(ann_path.stem + "_pruned.json")

    if dry_run:
        return None, {
            "images_total": len(images),
            "images_missing": missing_images,
            "images_kept": len(kept_images),
            "annotations_total": len(annotations_list),
            "annotations_kept": len(kept_annotations),
        }

    if in_place and make_backup:
        backup_path = _backup_path_for(ann_path)
        shutil.copy2(ann_path, backup_path)
        print(f"[coco_prune] wrote backup: {backup_path}")

    output_path.write_text(json.dumps(pruned, indent=2), encoding="utf-8")
    return output_path, {
        "images_total": len(images),
        "images_missing": missing_images,
        "images_kept": len(kept_images),
        "annotations_total": len(annotations_list),
        "annotations_kept": len(kept_annotations),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prune COCO annotations for missing images.")
    p.add_argument("--dataset-dir", type=str, default=None, help="Dataset directory (expects images + annotations).")
    p.add_argument("--annotations", type=str, default=None, help="COCO annotations JSON path.")
    p.add_argument("--images-dir", type=str, default=None, help="Override images directory.")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <annotations>_pruned.json).",
    )
    p.add_argument("--in-place", action="store_true", help="Overwrite the input annotations file.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow pruning that would remove all images/annotations (dangerous).",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic .bak backup when using --in-place.",
    )
    p.add_argument("--dry-run", action="store_true", help="Report only, do not write output.")
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    annotations = Path(args.annotations) if args.annotations else None
    images_dir = Path(args.images_dir) if args.images_dir else None

    ann_path, img_dir = _resolve_paths(dataset_dir, annotations, images_dir)

    output_path = Path(args.output) if args.output else None

    out_path, stats = prune_coco(
        ann_path=ann_path,
        images_dir=img_dir,
        in_place=bool(args.in_place),
        output=output_path,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        make_backup=bool(args.in_place) and (not bool(args.no_backup)),
    )

    suffix = "(dry-run)" if args.dry_run else str(out_path)
    print(
        "[coco_prune] images_total={} images_missing={} images_kept={} "
        "annotations_total={} annotations_kept={} -> {}".format(
            stats["images_total"],
            stats["images_missing"],
            stats["images_kept"],
            stats["annotations_total"],
            stats["annotations_kept"],
            suffix,
        )
    )
    return 0

