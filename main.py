from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import json

from config import AppConfig, get_default_config
from detector import Detector
from draw_utils import draw_line_and_counts, draw_tracks
from line_counter import LineCounter
from tracker import Tracker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count people/vehicles crossing a virtual line in a video."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input video file (defaults to config's IO settings).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output video file (defaults to config's IO settings).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["motion", "onnx"],
        help="Detector backend: 'motion' (no model) or 'onnx'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to ONNX model (when backend is 'onnx'). Overrides config.",
    )
    parser.add_argument(
        "--line-json",
        type=str,
        help="Path to line JSON from line_picker.py (uses first line's normalized coords).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display a window while processing (press 'q' to quit).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most this many frames (for quick tests).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Process at most this many seconds of video (for quick tests).",
    )
    parser.add_argument(
        "--select-line",
        action="store_true",
        help="Interactively pick the counting line on the first frame.",
    )
    return parser.parse_args()


def _build_line_points(cfg: AppConfig, width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    sx, sy = cfg.line.start_norm
    ex, ey = cfg.line.end_norm
    p1 = (int(sx * width), int(sy * height))
    p2 = (int(ex * width), int(ey * height))
    return p1, p2


def _select_line_interactively(
    video_path: Path,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[float, float], Tuple[float, float]]]:
    """
    Let the user click two points on the first frame to define the line.
    Returns (p1, p2, start_norm, end_norm) or None if cancelled.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[main] Failed to open video for line selection: {video_path}")
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[main] Could not read frame for line selection.")
        return None

    height, width = frame.shape[:2]
    window_name = "Select Line (L-click 2 points, Enter=OK, R/C=reset, Esc/Q=cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    points: list[Tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # If already had 2 points, start over
            if len(points) >= 2:
                points.clear()
            points.append((x, y))

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = frame.copy()
        # Draw clicked points
        for pt in points:
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
        # Draw line if two points selected
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (0, 255, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # Enter or Space
            if len(points) == 2:
                break
        elif key in (27, ord("q")):  # Esc or q
            points.clear()
            break
        elif key in (ord("r"), ord("c")):
            points.clear()

    cv2.destroyWindow(window_name)

    if len(points) != 2:
        return None

    (x1, y1), (x2, y2) = points
    start_norm = (x1 / float(width), y1 / float(height))
    end_norm = (x2 / float(width), y2 / float(height))
    return (points[0], points[1], start_norm, end_norm)


def main() -> None:
    args = _parse_args()
    cfg = get_default_config()

    # Allow CLI overrides
    if args.input:
        cfg.io.input_path = Path(args.input)
    if args.output:
        cfg.io.output_path = Path(args.output)
    if args.backend:
        cfg.model.backend = args.backend
    if args.model:
        cfg.model.model_path = Path(args.model)
    if args.line_json:
        line_path = Path(args.line_json)
        if not line_path.exists():
            raise SystemExit(f"Line JSON not found: {line_path}")
        data = json.loads(line_path.read_text())
        if not data:
            raise SystemExit(f"Line JSON is empty: {line_path}")
        first = data[0]
        start_norm = tuple(first["normalized"]["start"])
        end_norm = tuple(first["normalized"]["end"])
        cfg.line.start_norm = (float(start_norm[0]), float(start_norm[1]))
        cfg.line.end_norm = (float(end_norm[0]), float(end_norm[1]))

    input_path = cfg.io.input_path
    output_path = cfg.io.output_path

    if not input_path.exists():
        raise SystemExit(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Always infer the actual frame size from the first frame so that
    # normalized coordinates from line_picker.py map back exactly, even if
    # the container metadata reports a slightly different height/width.
    ret, frame0 = cap.read()
    if not ret:
        raise SystemExit("Could not read any frame from input video.")
    height, width = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Derive a frame limit if the user provided one in seconds
    max_frames: Optional[int] = args.max_frames
    if args.max_seconds is not None:
        seconds_limit = max(args.max_seconds, 0)
        frames_from_seconds = int(math.ceil(seconds_limit * fps))
        max_frames = (
            frames_from_seconds
            if max_frames is None
            else min(max_frames, frames_from_seconds)
        )

    # Optionally let the user pick the line interactively
    if args.select_line:
        selection = _select_line_interactively(input_path)
        if selection is None:
            print("[main] Line selection cancelled; exiting.")
            cap.release()
            return
        p1, p2, start_norm, end_norm = selection
        cfg.line.start_norm = start_norm
        cfg.line.end_norm = end_norm
        print(f"[main] Selected line pixels: {p1} -> {p2}")
        print(
            f"[main] Selected normalized line: "
            f"start_norm={start_norm}, end_norm={end_norm}"
        )
        # Restart video from the beginning for processing
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        p1, p2 = _build_line_points(cfg, width, height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    detector = Detector(cfg.model)
    tracker = Tracker(cfg.tracker)

    line_counter = LineCounter(p1=p1, p2=p2)

    print(f"[main] Processing {input_path} -> {output_path}")
    print(f"[main] Resolution: {width}x{height} @ {fps:.2f} FPS")
    print(f"[main] Detector backend: {cfg.model.backend}")
    print(f"[main] Line: {p1} -> {p2}")
    if max_frames is not None:
        print(f"[main] Frame limit: {max_frames} (via --max-frames/--max-seconds)")

    frame_index = 0

    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_index)
        line_counter.update(tracks)

        # Draw only tracks that were actually updated in this frame to
        # avoid "stuck" boxes when a track is kept alive across occlusions.
        draw_tracks(frame, tracks, frame_index=frame_index)
        draw_line_and_counts(frame, line_counter)

        writer.write(frame)

        if args.show:
            cv2.imshow("Pedestrian/Vehicle Line Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(
        f"[main] Done. A->B: {line_counter.count_a_to_b}, "
        f"B->A: {line_counter.count_b_to_a}"
    )


if __name__ == "__main__":
    main()
