from __future__ import annotations

import sys


def main() -> int:
    print(
        "[deprecated] scripts/merge_coco.py -> use: python -m yolo_kitv2 coco merge ...",
        file=sys.stderr,
    )
    from yolo_kitv2.datasets.coco_merge import main as tool_main

    return tool_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

