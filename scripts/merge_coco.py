from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def _safe_tag(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
    return "".join(cleaned) or "dataset"


def _unique_name(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        used.add(candidate)
        return candidate
    stem = Path(candidate).stem
    suffix = Path(candidate).suffix
    idx = 1
    while True:
        name = f"{stem}__{idx}{suffix}"
        if name not in used:
            used.add(name)
            return name
        idx += 1


def _resolve_image_path(root: Path, file_name: str) -> Path:
    raw = Path(file_name)
    if raw.is_absolute() and raw.exists():
        return raw
    candidate = root / file_name
    if candidate.exists():
        return candidate
    alt = root / "images" / raw.name
    if alt.exists():
        return alt
    raise SystemExit(f"Image file not found: {file_name} (root={root})")


def _load_coco(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge multiple COCO datasets into one directory.")
    p.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Input dataset directories (containing annotations.json) or COCO JSON files.",
    )
    p.add_argument("--output-dir", type=str, required=True, help="Output dataset directory.")
    p.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Output COCO JSON path (default: <output_dir>/annotations.json).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    out_ann_path = Path(args.annotations) if args.annotations else (output_dir / "annotations.json")

    categories: Dict[int, str] = {}
    images_out: List[Dict[str, object]] = []
    annotations_out: List[Dict[str, object]] = []
    used_names: set[str] = set()
    next_image_id = 1
    next_ann_id = 1
    sources: List[str] = []

    for idx, item in enumerate(args.inputs, start=1):
        input_path = Path(item)
        if input_path.is_dir():
            ann_path = input_path / "annotations.json"
            dataset_root = input_path
            tag = _safe_tag(input_path.name)
        else:
            ann_path = input_path
            dataset_root = input_path.parent
            tag = _safe_tag(input_path.stem)

        if not ann_path.exists():
            raise SystemExit(f"Missing COCO JSON: {ann_path}")

        if tag in {s.split(":")[0] for s in sources}:
            tag = f"{tag}_{idx}"

        coco = _load_coco(ann_path)
        sources.append(f"{tag}:{ann_path}")

        for cat in coco.get("categories", []):
            cid = int(cat["id"])
            name = str(cat.get("name", cid))
            if cid in categories and categories[cid] != name:
                raise SystemExit(f"Category conflict id={cid}: '{categories[cid]}' vs '{name}' in {ann_path}")
            categories[cid] = name

        id_map: Dict[int, int] = {}
        for image in coco.get("images", []):
            old_id = int(image["id"])
            file_name = str(image["file_name"])
            src_path = _resolve_image_path(dataset_root, file_name)
            new_name = _unique_name(f"{tag}__{Path(file_name).name}", used_names)
            dst_rel = Path("images") / new_name
            dst_path = output_dir / dst_rel
            shutil.copy2(src_path, dst_path)

            new_image = dict(image)
            new_image["id"] = next_image_id
            new_image["file_name"] = str(dst_rel).replace("\\", "/")
            images_out.append(new_image)
            id_map[old_id] = next_image_id
            next_image_id += 1

        for ann in coco.get("annotations", []):
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = id_map[int(ann["image_id"])]
            annotations_out.append(new_ann)
            next_ann_id += 1

    categories_out = [{"id": cid, "name": categories[cid]} for cid in sorted(categories.keys())]
    merged = {
        "info": {
            "description": "Merged COCO dataset",
            "sources": sources,
        },
        "licenses": [],
        "images": images_out,
        "annotations": annotations_out,
        "categories": categories_out,
    }
    out_ann_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(
        f"[merge_coco] done inputs={len(args.inputs)} images={len(images_out)} "
        f"annotations={len(annotations_out)} -> {out_ann_path}"
    )


if __name__ == "__main__":
    main()
