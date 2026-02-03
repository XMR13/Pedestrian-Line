from __future__ import annotations

import sys


def main() -> int:
    print(
        "[deprecated] scripts/dataset_qa.py -> use: python -m yolo_kitv2 dataset qa ...",
        file=sys.stderr,
    )
    from yolo_kitv2.datasets.qa import main as tool_main

    return tool_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

