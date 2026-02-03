from __future__ import annotations

import sys
from typing import Callable, Dict, List, Sequence, Tuple


def _help() -> None:
    print(
        "\n".join(
            [
                "yolo_kitv2 CLI (vendored).",
                "",
                "Usage:",
                "  python -m yolo_kitv2 <group> <command> [args...]",
                "",
                "Groups/commands:",
                "  dataset qa            Validate datasets (COCO/YOLO/VOC).",
                "  coco prune            Remove COCO entries for missing images (safe by default).",
                "  coco merge            Merge multiple COCO datasets into one output directory.",
                "  label run             Auto-label video/images into COCO, or extract frames/candidates.",
                "",
                "Examples:",
                "  python -m yolo_kitv2 dataset qa --dataset-dir data/ds --format coco --strict",
                "  python -m yolo_kitv2 coco prune --dataset-dir data/ds --dry-run",
                "  python -m yolo_kitv2 coco merge --inputs ds_a ds_b --output-dir merged",
                "  python -m yolo_kitv2 label run --mode coco --input images/ --output-dir ds --reuse-images --model model.onnx",
                "",
                "Tip: append -h after any command to see full help.",
            ]
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = list(argv)
    if not args or args[0] in {"-h", "--help"}:
        _help()
        return 0

    # Dispatch by prefix tokens, and let each tool handle its own argparse and `-h`.
    dispatch: Dict[Tuple[str, ...], Callable[[List[str]], int]] = {
        ("dataset", "qa"): _dataset_qa,
        ("coco", "prune"): _coco_prune,
        ("coco", "merge"): _coco_merge,
        ("label", "run"): _label_run,
    }

    for key, fn in dispatch.items():
        if tuple(args[: len(key)]) == key:
            rest = args[len(key) :]
            return fn(rest)

    print("Unknown command. Run: python -m yolo_kitv2 --help", file=sys.stderr)
    return 2


def _dataset_qa(argv: List[str]) -> int:
    from .datasets.qa import main as tool_main

    return tool_main(argv)


def _coco_prune(argv: List[str]) -> int:
    from .datasets.coco_prune import main as tool_main

    return tool_main(argv)


def _coco_merge(argv: List[str]) -> int:
    from .datasets.coco_merge import main as tool_main

    return tool_main(argv)


def _label_run(argv: List[str]) -> int:
    from .datasets.label import main as tool_main

    return tool_main(argv)

