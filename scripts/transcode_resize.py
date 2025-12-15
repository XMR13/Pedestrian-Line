"""
Mengubah video menjadi target size yang digunakan untuk openCV VideoCapture/VideoWriter

Hal ini berguna ketika input beresolusi tinggi lebih lambat ketika di devode, meskipun
pipelinneya melakukan reize lagi

Usage:
  python3 scripts/transcode_resize.py --input media/video5.mp4 --output media/video5_848x478.mp4 --size 848x478
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pedestrian_line_counter.videoio_utils import open_video_capture  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--size", required=True, type=str, help="Target size like 848x478")
    return p.parse_args()


def parse_size(spec: str) -> tuple[int, int]:
    spec = (spec or "").strip().lower()
    if "x" not in spec:
        raise ValueError(f"Invalid size '{spec}'. Expected format like 848x478.")
    w_str, h_str = spec.split("x", 1)
    w = int(w_str)
    h = int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid size '{spec}': width/height must be > 0.")
    return w, h


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    w, h = parse_size(args.size)

    cap = open_video_capture(inp)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open: {inp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Failed to open writer: {out}")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        n += 1
        if n % 500 == 0:
            print(f"[transcode] frames={n}")

    cap.release()
    writer.release()
    print(f"[transcode] done frames={n} -> {out}")


if __name__ == "__main__":
    main()

