from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2

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
        "--show",
        action="store_true",
        help="Display a window while processing (press 'q' to quit).",
    )
    return parser.parse_args()


def _build_line_points(cfg: AppConfig, width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    sx, sy = cfg.line.start_norm
    ex, ey = cfg.line.end_norm
    p1 = (int(sx * width), int(sy * height))
    p2 = (int(ex * width), int(ey * height))
    return p1, p2


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

    input_path = cfg.io.input_path
    output_path = cfg.io.output_path

    if not input_path.exists():
        raise SystemExit(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        # Fallback: read one frame to infer size
        ret, frame = cap.read()
        if not ret:
            raise SystemExit("Could not read any frame from input video.")
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    detector = Detector(cfg.model)
    tracker = Tracker(cfg.tracker)

    p1, p2 = _build_line_points(cfg, width, height)
    line_counter = LineCounter(p1=p1, p2=p2)

    print(f"[main] Processing {input_path} -> {output_path}")
    print(f"[main] Resolution: {width}x{height} @ {fps:.2f} FPS")
    print(f"[main] Detector backend: {cfg.model.backend}")
    print(f"[main] Line: {p1} -> {p2}")

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_index)
        line_counter.update(tracks)

        draw_tracks(frame, tracks)
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
