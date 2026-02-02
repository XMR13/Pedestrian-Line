from __future__ import annotations

import argparse
import json
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
    p.add_argument("--dry-run", action="store_true", help="Report only, do not write output.")
    return p.parse_args()

def _image_path(images_dir: Path, file_name:str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    return images_dir/path

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

    pruned = dict(coco)
    pruned["images"] = kept_images
    pruned["annotations"] = kept_annotations

    output_path = ann_path if args.in_place else Path(args.output) if args.output else ann_path.with_name(
        ann_path.stem + "_pruned.json"
    )

    if not args.dry_run:
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
