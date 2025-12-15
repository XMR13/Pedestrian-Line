from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import json
import sys

from .config import AppConfig, ROOT_DIR, get_default_config
from .detector import Detector
from .draw_utils import draw_line_and_counts, draw_tracks
from .line_counter import LineCounter, TwoLineGateCounter
from .tracker import Tracker

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover - safe fallback
    tqdm = None

# mendefinisikan command terlebih dahulu (later integration
# bisa langsung dijadikan argumen dari command
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Menghtitung orang dan kendaraan ."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path ke input yang diigninkan",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path ke output yang ingin disimpan (jika kosong default ke I/O setting).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["motion", "onnx", "torch"],
        help="Detector backend: 'motion' (no model), 'onnx', atau 'torch'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Tambahkan model yang diinginkan (bisa berupa .pt atau .onnx). Overrides config.",
    )
    parser.add_argument(
        "--line-json",
        type=str,
        help=(
            "Path to line JSON from line_picker.py. "
            "For --line-mode single: uses the first line. "
            "For --line-mode gate: uses the first two lines."
        ),
    )
    parser.add_argument(
        "--camera",
        type=str,
        help=(
            "Nama camera atau path per-camera line untuk JSON"
            "Jika diberikan sebuah nama, melihat ke config/camera/<name>.json atau config/<name>.json"
            "Camera name or path to a per-camera line JSON. "
            "If a name is given, looks for config/cameras/<name>.json then config/<name>.json."
        ),
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
        help="Tentukan berapa banyak frame yang ingin digunakan sebagai contoh dari input (prototype).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Menentukan berapa detik dari input yang diinginkan untuk diproses ",
    )
    parser.add_argument(
        "--select-line",
        action="store_true",
        help="Interactively pick the counting line on the first frame.",
    )
    parser.add_argument(
        "--line-mode",
        type=str,
        default="single",
        choices=["single", "gate"],
        help=(
            "Menentukan mode garis yang ingin digunakan"
            "'single' menghitung hanya dengan menggunakan 1 garis (default behaviour)"
            "'gate' menggunakan 2 buah garis dan meentukan arahnya berdasarkan urutan"
            "(line1->line2 = A->B, line2->line1 = B->A)."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (useful if tqdm is not installed or when logging).",
    )
    return parser.parse_args()


def _build_line_points(cfg: AppConfig, width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    #adalah fungsi yang mengembalikan build line points
    sx, sy = cfg.line.start_norm
    ex, ey = cfg.line.end_norm
    p1 = (int(sx * width), int(sy * height))
    p2 = (int(ex * width), int(ey * height))
    return p1, p2


def _build_line_points_from_norm(
    start_norm: Tuple[float, float],
    end_norm: Tuple[float, float],
    width: int,
    height: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    sx, sy = start_norm
    ex, ey = end_norm
    return (int(sx * width), int(sy * height)), (int(ex * width), int(ey * height))


def _load_lines_from_json(path: Path) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    data = json.loads(path.read_text())
    if not data:
        return []

    # Current format: list[{ "normalized": {"start":[x,y],"end":[x,y]}, ... }]
    if isinstance(data, list):
        out: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            norm = item.get("normalized")
            if not isinstance(norm, dict):
                continue
            start = norm.get("start")
            end = norm.get("end")
            if start is None or end is None:
                continue
            out.append(((float(start[0]), float(start[1])), (float(end[0]), float(end[1]))))
        return out

    # Legacy-ish fallback: dict with normalized keys.
    if isinstance(data, dict) and "normalized" in data:
        norm = data["normalized"]
        start = norm.get("start")
        end = norm.get("end")
        if start is None or end is None:
            return []
        return [((float(start[0]), float(start[1])), (float(end[0]), float(end[1])))]

    return []


def _select_line_interactively(
    video_path: Path,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[float, float], Tuple[float, float]]]:
    """
    Memungkinkan user untuk memilih garis nya
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

    line_path: Optional[Path] = None
    if args.line_json:
        line_path = Path(args.line_json)
    elif args.camera:
        cam_arg = Path(args.camera)
        if cam_arg.exists():
            line_path = cam_arg
        else:
            candidate1 = ROOT_DIR / "config" / "cameras" / f"{args.camera}.json"
            candidate2 = ROOT_DIR / "config" / f"{args.camera}.json"
            if candidate1.exists():
                line_path = candidate1
            elif candidate2.exists():
                line_path = candidate2
            else:
                raise SystemExit(
                    f"Camera line JSON not found for '{args.camera}'. "
                    f"Tried: {candidate1} and {candidate2}"
                )

    lines_norm: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if line_path is not None:
        if not line_path.exists():
            raise SystemExit(f"Line JSON not found: {line_path}")
        lines_norm = _load_lines_from_json(line_path)
        if not lines_norm:
            raise SystemExit(f"Line JSON is empty or invalid: {line_path}")
        # Default: populate cfg with the first line (single-line mode).
        (start_norm, end_norm) = lines_norm[0]
        cfg.line.start_norm = start_norm
        cfg.line.end_norm = end_norm

    input_path = cfg.io.input_path
    output_path = cfg.io.output_path

    if not input_path.exists():
        raise SystemExit(f"Input video not found: {input_path}")

    if args.line_mode == "gate" and args.select_line:
        raise SystemExit("--select-line is only supported for --line-mode single. Use line_picker.py to save 2 lines.")

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

    # Menentukan maksimla frame dari waktu seconds
    max_frames: Optional[int] = args.max_frames
    if args.max_seconds is not None:
        seconds_limit = max(args.max_seconds, 0)
        frames_from_seconds = int(math.ceil(seconds_limit * fps))
        max_frames = (
            frames_from_seconds
            if max_frames is None
            else min(max_frames, frames_from_seconds)
        )

    
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

    if args.line_mode == "gate":
        if len(lines_norm) < 2:
            raise SystemExit(
                "--line-mode gate requires 2 lines in --line-json/--camera JSON (pick with line_picker.py --lines 2)."
            )
        (l1_start, l1_end) = lines_norm[0]
        (l2_start, l2_end) = lines_norm[1]
        l1_p1, l1_p2 = _build_line_points_from_norm(l1_start, l1_end, width, height)
        l2_p1, l2_p2 = _build_line_points_from_norm(l2_start, l2_end, width, height)
        line_counter = TwoLineGateCounter(
            line1_p1=l1_p1,
            line1_p2=l1_p2,
            line2_p1=l2_p1,
            line2_p2=l2_p2,
        )
    else:
        line_counter = LineCounter(p1=p1, p2=p2)

    print(f"[main] Processing {input_path} -> {output_path}")
    print(f"[main] Resolution: {width}x{height} @ {fps:.2f} FPS")
    print(f"[main] Detector backend: {cfg.model.backend}")
    if args.line_mode == "gate":
        (a1, a2), (b1, b2) = line_counter.lines
        print(f"[main] Line mode: gate")
        print(f"[main] Line 1: {a1} -> {a2}")
        print(f"[main] Line 2: {b1} -> {b2}")
    else:
        print(f"[main] Line mode: single")
        print(f"[main] Line: {p1} -> {p2}")
    if max_frames is not None:
        print(f"[main] Frame limit: {max_frames} (via --max-frames/--max-seconds)")

    # Prepare progress bar
    progress_total = None
    # If user set a frame cap, use that; else if video reports length, use it.
    reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None:
        progress_total = max_frames
    elif reported_total > 0:
        progress_total = reported_total

    progress_bar = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(
            total=progress_total,
            unit="frame",
            desc="Processing",
            file=sys.stdout,
            leave=False,
        )
    elif not args.no_progress and tqdm is None:
        print("[main] tqdm is not installed; run `uv run pip install tqdm` to enable a progress bar.")

    frame_index = 0

    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_index)
        if args.line_mode == "gate":
            tracks_updated = [t for t in tracks if t.last_seen_frame == frame_index]
            line_counter.update(tracks_updated, frame_index)
        else:
            line_counter.update(tracks)

        # Gambar setiap gambar yang telah diupdate 
        # avoid "stuck" boxes when a track is kept alive across occlusions.
        draw_tracks(frame, tracks, frame_index=frame_index)
        draw_line_and_counts(frame, line_counter)

        writer.write(frame)

        if args.show:
            cv2.imshow("Pedestrian/Vehicle Line Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1
        if progress_bar is not None:
            progress_bar.update(1)

    cap.release()
    writer.release()
    if progress_bar is not None:
        progress_bar.close()
    if args.show:
        cv2.destroyAllWindows()

    print(
        f"[main] Done. A->B: {line_counter.count_a_to_b}, "
        f"B->A: {line_counter.count_b_to_a}"
    )


if __name__ == "__main__":
    main()
