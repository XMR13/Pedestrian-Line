from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2


def _parse_size(spec: str) -> Tuple[int, int]:
    spec = (spec or "").strip().lower()
    if "x" not in spec:
        raise ValueError(f"Invalid size '{spec}'. Expected format like 1280x720.")
    w_str, h_str = spec.split("x", 1)
    w = int(w_str)
    h = int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid size '{spec}': width/height must be > 0.")
    return w, h


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frames from a video for annotation.")
    p.add_argument("--input", type=str, required=True, help="Input video path.")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to write extracted frames.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--fps", type=float, help="Sample at this FPS (time-based, best-effort).")
    group.add_argument("--every-n", type=int, help="Sample every N frames.")
    p.add_argument("--start-seconds", type=float, default=0.0, help="Skip the first N seconds.")
    p.add_argument("--max-seconds", type=float, default=None, help="Stop after N seconds (after start).")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after writing N frames.")
    p.add_argument("--resize-to", type=str, default=None, help="Optional resize, e.g. '1280x720'.")
    p.add_argument("--ext", type=str, default="jpg", help="Output image extension (jpg/png).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open: {input_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0.0:
        src_fps = 30.0

    start_frames = int(max(args.start_seconds, 0.0) * src_fps)

    resize_to: Optional[Tuple[int, int]] = None
    if args.resize_to:
        try:
            resize_to = _parse_size(args.resize_to)
        except ValueError as exc:
            raise SystemExit(str(exc))

    if args.fps is not None:
        if args.fps <= 0:
            raise SystemExit("--fps must be > 0")
        every_n = max(int(round(src_fps / float(args.fps))), 1)
    else:
        if args.every_n is None or args.every_n <= 0:
            raise SystemExit("--every-n must be > 0")
        every_n = int(args.every_n)

    max_end_frame: Optional[int] = None
    if args.max_seconds is not None:
        max_end_frame = start_frames + int(max(float(args.max_seconds), 0.0) * src_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frames)
    frame_index = start_frames
    written = 0

    try:
        while True:
            if max_end_frame is not None and frame_index >= max_end_frame:
                break
            if args.max_frames is not None and written >= int(args.max_frames):
                break

            ret, frame = cap.read()
            if not ret:
                break

            if (frame_index - start_frames) % every_n == 0:
                if resize_to is not None:
                    frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
                out_path = out_dir / f"frame_{frame_index:06d}.{args.ext.lstrip('.')}"
                ok = cv2.imwrite(str(out_path), frame)
                if not ok:
                    raise SystemExit(f"Failed to write: {out_path}")
                written += 1

            frame_index += 1
    finally:
        cap.release()

    print(f"[extract_frames] done input={input_path.name} written={written} every_n={every_n} src_fps={src_fps:.2f}")


if __name__ == "__main__":
    main()

