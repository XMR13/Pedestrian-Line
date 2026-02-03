from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _resolve_paths(dataset_dir: Path | None, annotations: Path | None, images_dir: Path | None) -> Tuple[Path, Path]:
    ann_path: Path
    img_dir: Path

    if dataset_dir is not None:
        ann_path = annotations if annotations is not None else (dataset_dir / "annotations.json")
        img_dir = images_dir if images_dir is not None else (dataset_dir / "images")
    else:
        if annotations is None:
            raise SystemExit("Provide --dataset-dir or --annotations.")
        ann_path = annotations
        if images_dir is not None:
            img_dir = images_dir
        else:
            img_dir = ann_path.parent / "images"

    if not ann_path.exists():
        raise SystemExit(f"Annotations not found: {ann_path}")
    if not img_dir.exists():
        raise SystemExit(f"Images dir not found: {img_dir}")

    return ann_path, img_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune COCO annotations for missing images.")
    p.add_argument("--dataset-dir", type=str, default=None, help="Dataset yang tedapat images")
    p.add_argument("--annotations", type=str, default=None, help="Path COCO annotations")
    p.add_argument("--images-dir", type=str, default=None, help="Image directory ovverridee")
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
        help="Allow pruning that would remove all images/annotations (dangerous; keep backups!).",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic .bak backup when using --in-place.",
    )
    p.add_argument("--dry-run", action="store_true", help="Report only, do not write output.")
    return p.parse_args()

def _image_path(images_dir: Path, file_name:str) -> Path:
    """
    Resolve COCO `file_name` to an on-disk path.

    In this repo we commonly write COCO `file_name` as a path relative to the
    dataset root, e.g. `images/foo.png`. When users pass `--dataset-dir`, this
    script defaults `images_dir` to `<dataset>/images`, so a naive join would
    incorrectly look under `<dataset>/images/images/foo.png`.
    """
    # COCO exports sometimes contain backslashes (Windows) even though COCO paths are typically POSIX-like.
    normalized = str(file_name).replace("\\", "/").lstrip("./")
    raw = Path(normalized)
    if raw.is_absolute():
        return raw

    # Most common: `images_dir` points to `<dataset>/images`.
    candidate = images_dir / raw
    if candidate.exists():
        return candidate

    # If `file_name` already includes `images/...`, try resolving from the dataset root.
    if images_dir.name.lower() == "images":
        candidate = images_dir.parent / raw
        if candidate.exists():
            return candidate

    # Fallback: treat file_name as a bare basename.
    candidate = images_dir / raw.name
    return candidate


def _backup_path_for(ann_path: Path) -> Path:
    # Default backup next to the file (easy to find). Avoid overwriting an existing backup.
    base = ann_path.with_suffix(ann_path.suffix + ".bak")
    if not base.exists():
        return base
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return ann_path.with_suffix(ann_path.suffix + f".bak_{stamp}")

def main() -> None:
    args = _parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    annotations = Path(args.annotations) if args.annotations else None
    images_dir = Path(args.images_dir) if args.images_dir else None

    ann_path, img_dir = _resolve_paths(dataset_dir, annotations, images_dir)

    coco: Dict[str, object] = json.loads(ann_path.read_text(encoding="utf-8"))
    images: List[Dict[str, object]] = list(coco.get("images", []))
    annotations_list: List[Dict[str, object]] = list(coco.get("annotations", []))

    kept_images: List[Dict[str, object]] = []
    kept_image_ids: Set[int] = set()
    missing_images = 0

    for image in images:
        file_name = str(image.get("file_name", ""))
        path = _image_path(img_dir, file_name)
        if path.exists():
            kept_images.append(image)
            kept_image_ids.add(int(image["id"]))
        else:
            missing_images += 1

    kept_annotations: List[Dict[str, object]] = [
        ann for ann in annotations_list if int(ann.get("image_id", -1)) in kept_image_ids
    ]

    # Safety: if we are about to wipe everything, it's usually a path-resolution/config mistake.
    if not args.force:
        if images and not kept_images:
            raise SystemExit(
                "[prune_coco] Refusing to prune: would remove ALL images from COCO JSON. "
                "This is usually a path mismatch (e.g. file_name includes 'images/...' while you also set images_dir). "
                "Re-run with --dry-run, or pass --images-dir correctly, or use --force if you really intend this."
            )
        if annotations_list and not kept_annotations:
            raise SystemExit(
                "[prune_coco] Refusing to prune: would remove ALL annotations from COCO JSON. "
                "This is usually a path mismatch or bad image_id references. Use --force if intended."
            )

    pruned = dict(coco)
    pruned["images"] = kept_images
    pruned["annotations"] = kept_annotations

    output_path = ann_path if args.in_place else Path(args.output) if args.output else ann_path.with_name(
        ann_path.stem + "_pruned.json"
    )

    if not args.dry_run:
        if args.in_place and not args.no_backup:
            backup_path = _backup_path_for(ann_path)
            shutil.copy2(ann_path, backup_path)
            print(f"[prune_coco] wrote backup: {backup_path}")
        output_path.write_text(json.dumps(pruned, indent=2), encoding="utf-8")

    print(
        "[prune_coco] images_total={} images_missing={} images_kept={} "
        "annotations_total={} annotations_kept={} -> {}".format(
            len(images),
            missing_images,
            len(kept_images),
            len(annotations_list),
            len(kept_annotations),
            "(dry-run)" if args.dry_run else output_path,
        )
    )


if __name__ == "__main__":
    main()
