from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2

from pedestrian_line_counter.config import ModelConfig, TrackerConfig, load_class_names
from pedestrian_line_counter.detector import Detector
from pedestrian_line_counter.tracker import Tracker
from pedestrian_line_counter.videoio_utils import open_video_capture

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover - safe fallback
    tqdm = None

def _parse_class_ids(value: str) -> list[int]:
    ids: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ids.append(int(item))
    return ids


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


def _safe_name(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
    return "".join(cleaned) or "class"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract candidate frames for annotation using detector + tracker (no line counting)."
    )
    p.add_argument("--input", type=str, required=True, help="Input video path or directory.")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to write extracted frames.")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="If --input is a directory, scan recursively for video files.",
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=["onnx", "tensorrt", "torch", "motion"],
        default="onnx",
        help="Detector backend to use for candidate extraction.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path override (.onnx/.engine/.pt).",
    )
    p.add_argument(
        "--class-ids",
        type=str,
        default=None,
        help="Comma-separated target class IDs (required for model backends). Example: '0,1,2'.",
    )
    p.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Optional class names mapping file (YAML/JSON) for filenames.",
    )
    p.add_argument(
        "--allow-all-classes",
        action="store_true",
        help="Debug only: allow running model backends without --class-ids.",
    )
    p.add_argument(
        "--max-per-track",
        type=int,
        default=3,
        help="Max frames to save per track ID (default: 3).",
    )
    p.add_argument(
        "--min-seconds-between",
        type=float,
        default=1.0,
        help="Minimum seconds between saves for the same track (default: 1.0).",
    )
    p.add_argument(
        "--min-frames-between",
        type=int,
        default=None,
        help="Minimum frames between saves for the same track (overrides --min-seconds-between).",
    )
    p.add_argument(
        "--baseline-fps",
        type=float,
        default=0.0,
        help="Optional baseline sampling FPS (0 disables).",
    )
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Wait N frames after a track first appears before saving (helps capture full vehicle).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing N frames.",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Stop after N seconds of video.",
    )
    p.add_argument(
        "--resize-to",
        type=str,
        default=None,
        help="Optional resize for saved frames, e.g. '1280x720'.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar.",
    )
    return p.parse_args()


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}

def _iter_input_files(input_path: Path, recursive: bool) -> Iterable[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        return []
    pattern = "**/*" if recursive else "*"

    #do the iteration and for checking
    files = [
        path 
        for path in sorted(input_path.glob(pattern))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return files

def _process_clip(
    input_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    cfg: ModelConfig,
    class_names_map: Optional[Dict[int, str]],
    resize_to: Optional[Tuple[int, int]],
) -> Tuple[int, int]:
    cap = open_video_capture(input_path)
    if not cap.isOpened():
        print(f"[extract_candidates] failed to open: {input_path}")
        return 0, 0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = 30.0

    max_frames = args.max_frames
    if args.max_seconds is not None:
        seconds_limit = max(float(args.max_seconds), 0.0)
        frames_from_seconds = int(round(seconds_limit * fps))
        max_frames = frames_from_seconds if max_frames is None else min(max_frames, frames_from_seconds)

    if args.min_frames_between is not None:
        min_frames_between = max(int(args.min_frames_between), 0)
    else:
        min_frames_between = max(int(round(float(args.min_seconds_between) * fps)), 0)

    baseline_every_n = 0
    if args.baseline_fps and args.baseline_fps > 0:
        baseline_every_n = max(int(round(fps / float(args.baseline_fps))), 1)

    detector = Detector(cfg)
    tracker = Tracker(TrackerConfig())

    clip_name = _safe_name(input_path.stem)

    per_track_state: Dict[int, Dict[str, int]] = {}
    saved_total = 0
    frame_index = 0
    progress_bar = None
    progress_total = None
    reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None:
        progress_total = max_frames
    elif reported_total > 0:
        progress_total = reported_total

    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(
            total=progress_total,
            unit="frame",
            desc=f"Extracting {input_path.name}",
            file=sys.stdout,
            leave=False,
        )
    elif not args.no_progress and tqdm is None:
        print("[extract_candidates] tqdm is not installed; run `uv run pip install tqdm` to enable a progress bar.")

    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame_index)
            updated_tracks = [t for t in tracks if t.last_seen_frame == frame_index]

            # Optional baseline sampling regardless of detection
            if baseline_every_n and frame_index % baseline_every_n == 0:
                out_path = out_dir / f"{clip_name}_f{frame_index:06d}_baseline.jpg"
                img = frame if resize_to is None else cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_path), img)
                saved_total += 1

            for track in updated_tracks:
                if track.class_id is None:
                    continue
                if cfg.track_class_ids and int(track.class_id) not in cfg.track_class_ids and not cfg.allow_all_classes:
                    continue

                state = per_track_state.setdefault(
                    track.track_id,
                    {"count": 0, "last_frame": -10**9, "first_frame": frame_index},
                )
                warmup_frames = max(int(args.warmup_frames), 0)
                if (frame_index - state["first_frame"]) < warmup_frames:
                    continue
                if state["count"] >= int(args.max_per_track):
                    continue
                if (frame_index - state["last_frame"]) < min_frames_between:
                    continue

                class_id = int(track.class_id)
                class_label = str(class_id)
                if class_names_map and class_id in class_names_map:
                    class_label = _safe_name(class_names_map[class_id])

                filename = (
                    f"{clip_name}_f{frame_index:06d}_tid{track.track_id:04d}"
                    f"_cid{class_id:03d}_{class_label}.jpg"
                )
                out_path = out_dir / filename

                img = frame if resize_to is None else cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_path), img)
                saved_total += 1

                state["count"] += 1
                state["last_frame"] = frame_index

            frame_index += 1
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        cap.release()
        if progress_bar is not None:
            progress_bar.close()

    print(
        f"[extract_candidates] done clip={input_path.name} frames={frame_index} "
        f"saved={saved_total} min_gap_frames={min_frames_between}"
    )
    return frame_index, saved_total


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input)
    input_files = _iter_input_files(input_path, args.recursive)
    if not input_files:
        raise SystemExit(f"No input files found under: {input_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(backend=args.backend)
    if args.model:
        cfg.model_path = Path(args.model)
    if args.allow_all_classes:
        cfg.allow_all_classes = True
    if args.class_ids:
        try:
            cfg.track_class_ids = _parse_class_ids(args.class_ids)
        except Exception:
            raise SystemExit(f"Invalid --class-ids value: '{args.class_ids}'")

    if cfg.backend in {"onnx", "tensorrt", "torch"} and not cfg.track_class_ids and not cfg.allow_all_classes:
        raise SystemExit(
            "Model-based backend requires explicit target classes. "
            "Pass --class-ids or use --allow-all-classes for debugging."
        )

    class_names_map = None
    if args.class_names:
        class_names_map = load_class_names(Path(args.class_names))

    resize_to: Optional[Tuple[int, int]] = None
    if args.resize_to:
        try:
            resize_to = _parse_size(args.resize_to)
        except ValueError as exc:
            raise SystemExit(str(exc))

    total_frames = 0
    total_saved = 0
    for clip in input_files:
        frames, saved = _process_clip(clip, out_dir, args, cfg, class_names_map, resize_to)
        total_frames += frames
        total_saved += saved

    if len(input_files) > 1:
        print(
            f"[extract_candidates] all done clips={len(input_files)} "
            f"frames={total_frames} saved={total_saved}"
        )


if __name__ == "__main__":
    main()
