"""
Mengecek properti video dengan mneggunakan openCV.

Usage:
  python3 scripts/inspect_video.py --input media/video.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from pedestrian_line_counter.videoio_utils import open_video_capture


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--samples", type=int, default=12, help="How many frame indices to sample.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    cap = open_video_capture(path)
    print("path:", path)
    print("opened:", cap.isOpened())
    if hasattr(cap, "getBackendName"):
        try:
            print("backend:", cap.getBackendName())
        except Exception:
            pass

    fps = cap.get(cv2.CAP_PROP_FPS)
    rep_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    rep_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT``) or 0)
    rep_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print("reported:", f"{rep_w}x{rep_h}", "fps=", fps, "frames=", rep_n)

    ok, fr0 = cap.read()
    if not ok or fr0 is None:
        print("first frame: FAIL")
        cap.release()
        return
    h0, w0 = fr0.shape[:2]
    print("decoded first frame:", f"{w0}x{h0}")

    # Sample frames for resolution changes / decode failures
    sample_idxs = [0, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1000, 2000][: max(args.samples, 1)]
    sizes = {(w0, h0)}
    fails = 0
    for idx in sample_idxs[1:]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok2, fr = cap.read()
        if not ok2 or fr is None:
            fails += 1
            continue
        sizes.add((fr.shape[1], fr.shape[0]))
    print("sample_sizes:", sorted(sizes))
    print("sample_failures:", fails)
    cap.release()


if __name__ == "__main__":
    main()

