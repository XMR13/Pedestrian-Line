"""
Interactive line picker for counting.

Usage (Windows/GUI environment):

    uv run -p 3.8 python line_picker.py --input media/WhatsApp\ Video\ 2025-12-03\ at\ 11.23.31_60de7c28.mp4 --lines 1 --save lines.json

Controls:
    - Left click: add a point.
    - R or C: reset all points.
    - Enter/Space: finish when required points are placed.
    - Esc/Q: cancel.

Notes:
    - One oriented line is enough to count both directions (A->B and B->A).
    - Two lines are optional if you later want separate zones; this picker supports 1 or 2.
    - Requires a GUI. In WSL without an X server, OpenCV windows will crash; run on Windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2


Point = Tuple[int, int]
NormPoint = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively pick 1–2 lines on a frame.")
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to a video file or an image.",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=1,
        choices=[1, 2],
        help="How many lines to pick (1 or 2). Default: 1.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save picked lines as JSON.",
    )
    return parser.parse_args()


def load_first_frame(path: Path):
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(str(path))
        return img

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def pick_lines(frame, n_lines: int) -> List[Tuple[Point, Point]]:
    window = "Pick lines (L-click pts, Enter=ok, R/C=reset, Esc/Q=cancel)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    points: List[Point] = []

    def on_mouse(event, x, y, _flags, _param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 2 * n_lines:
                points = []
            points.append((x, y))

    cv2.setMouseCallback(window, on_mouse)

    while True:
        display = frame.copy()
        # Draw points
        for pt in points:
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
        # Draw lines for each completed pair
        for i in range(0, len(points), 2):
            if i + 1 < len(points):
                cv2.line(display, points[i], points[i + 1], (0, 255, 255), 2)

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # Enter or Space
            if len(points) == 2 * n_lines:
                break
        elif key in (27, ord("q")):  # Esc or q
            points = []
            break
        elif key in (ord("r"), ord("c")):
            points = []

    cv2.destroyWindow(window)

    if len(points) != 2 * n_lines:
        return []

    lines = []
    for i in range(0, len(points), 2):
        lines.append((points[i], points[i + 1]))
    return lines


def normalize(point: Point, width: int, height: int) -> NormPoint:
    x, y = point
    return (x / float(width), y / float(height))


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    frame = load_first_frame(path)
    if frame is None:
        raise SystemExit(f"Could not read image/video: {path}")

    h, w = frame.shape[:2]
    lines = pick_lines(frame, args.lines)
    if not lines:
        print("No lines selected (cancelled).")
        return

    result = []
    for idx, (p1, p2) in enumerate(lines, start=1):
        n1 = normalize(p1, w, h)
        n2 = normalize(p2, w, h)
        result.append(
            {
                "line": idx,
                "pixels": {"start": p1, "end": p2},
                "normalized": {"start": n1, "end": n2},
            }
        )
        print(f"Line {idx} pixels: {p1} -> {p2}")
        print(f"Line {idx} normalized: start={n1}, end={n2}")

    if args.save:
        save_path = Path(args.save)
        save_path.write_text(json.dumps(result, indent=2))
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
