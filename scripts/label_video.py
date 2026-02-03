from __future__ import annotations

import sys


def main() -> int:
    print(
        "[deprecated] scripts/label_video.py -> use: python -m yolo_kitv2 label run ...",
        file=sys.stderr,
    )
    from yolo_kitv2.datasets.label import main as tool_main

    return tool_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

