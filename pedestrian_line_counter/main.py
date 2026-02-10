from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional, Tuple
import time

import cv2
import json
import sys

from .config import AppConfig, ROOT_DIR, get_default_config, infer_track_class_ids_from_class_names, load_class_names
from .config_io import apply_config_overrides, load_config_overrides, split_overrides, write_app_config_json
from .detector import Detector
from .draw_utils import draw_line_and_counts, draw_tracks
from .line_counter import LineCounter, TwoLineGateCounter
from .stream_reader import StreamReader
from .traffic_spool import TrafficSpoolConfig, TrafficSpoolWriter
from .tracker import Tracker
from .videoio_utils import open_video_capture

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover - safe fallback
    tqdm = None

# mendefinisikan command terlebih dahulu (later integration
# bisa langsung dijadikan argumen dari command
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Menghitung kendaraan target (vehicle subclasses) yang melintasi garis virtual."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path ke input yang diigninkan",
    )
    parser.add_argument(
        "--rtsp-url",
        type=str,
        default=None,
        help="RTSP URL for live mode. If set, --input is not used.",
    )
    parser.add_argument(
        "--rtsp-url-env",
        type=str,
        default=None,
        help=(
            "Environment variable name that stores RTSP URL (e.g. PLC_RTSP_URL). "
            "Used when --rtsp-url is not set."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path ke output yang ingin disimpan (jika kosong default ke I/O setting).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a JSON config override file (for local experimentation). "
            "Overrides are applied on top of defaults; CLI flags still take precedence."
        ),
    )
    parser.add_argument(
        "--dump-config",
        type=str,
        default=None,
        help="Write the resolved config (after defaults/--config/CLI overrides) to this JSON path and exit.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["motion", "onnx", "tensorrt", "torch"],
        help="Detector backend: 'motion' (no model), 'onnx', 'tensorrt' (.engine), atau 'torch'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model path override (.onnx for onnx, .engine for tensorrt, .pt/.pth for torch).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override detector confidence threshold (e.g. 0.5).",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=None,
        help="Override NMS IoU threshold (e.g. 0.45).",
    )
    parser.add_argument(
        "--pre-nms-topk",
        type=int,
        default=None,
        help="Cap candidates before NMS (performance). Typical values: 300..2000. 0 disables.",
    )
    parser.add_argument(
        "--class-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated integer class IDs to track/count (required for model backends). "
            "Example: '0,1,2'."
        ),
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help=(
            "Optional class names mapping file (YAML/JSON) for overlays/reports. "
            "Supports YOLO-style `data.yaml` with `names:`."
        ),
    )
    parser.add_argument(
        "--allow-all-classes",
        action="store_true",
        help="Debug only: allow running model backends without --class-ids (counts all detected classes).",
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
        "--frame-stride",
        type=int,
        default=1,
        help=(
            "Process only every Nth frame for detect/track/count (1 = every frame). "
            "Higher values speed up inference but may reduce counting accuracy."
        ),
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
    parser.add_argument(
        "--stale-frames",
        type=int,
        default=2,
        help=(
            "Draw tracks that were not updated for up to N frames (helps debug "
            "short detection dropouts). Use 0 to draw only tracks updated on the current frame."
        ),
    )
    parser.add_argument(
        "--resize-to",
        type=str,
        default=None,
        help=(
            "Optional processing/output size, e.g. '848x478'. "
            "If set, frames are resized to this size to match performance/visuals across videos."
        ),
    )
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="Disable drawing overlays (useful for benchmarking).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Disable writing output video (useful for benchmarking).",
    )
    parser.add_argument(
        "--write-processed-only",
        action="store_true",
        help="When using --frame-stride > 1, write only processed frames to output (reduces encoder cost).",
    )
    parser.add_argument(
        "--fast-skip",
        action="store_true",
        help=(
            "File mode only: when combined with --frame-stride > 1 and --write-processed-only, "
            "skip decoding intermediate frames using VideoCapture.grab() for higher throughput."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print simple per-stage timing stats while processing.",
    )
    parser.add_argument(
        "--benchmark-every",
        type=int,
        default=250,
        help="How often (in frames) to print benchmark stats when --benchmark is set.",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Draw runtime FPS counters on output frames (loop FPS, processed FPS, and live source FPS).",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=3,
        help="Live mode only: bounded reader queue size (drops oldest when full).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Live mode only: process at most this FPS by dropping frames.",
    )
    parser.add_argument(
        "--log-every-seconds",
        type=float,
        default=None,
        help="Print current totals every N seconds (useful in headless live mode).",
    )
    parser.add_argument(
        "--open-timeout-ms",
        type=int,
        default=None,
        help="VideoCapture open timeout in ms (best-effort; mainly for RTSP).",
    )
    parser.add_argument(
        "--read-timeout-ms",
        type=int,
        default=None,
        help="VideoCapture read timeout in ms (best-effort; mainly for RTSP).",
    )
    parser.add_argument(
        "--rtsp-reconnect",
        dest="rtsp_reconnect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable RTSP reconnect policy in live mode.",
    )
    parser.add_argument(
        "--rtsp-reconnect-max-attempts",
        type=int,
        default=None,
        help="Max RTSP reconnect attempts (0 means unlimited).",
    )
    parser.add_argument(
        "--rtsp-reconnect-initial-delay",
        type=float,
        default=None,
        help="Initial RTSP reconnect delay in seconds.",
    )
    parser.add_argument(
        "--rtsp-reconnect-max-delay",
        type=float,
        default=None,
        help="Maximum RTSP reconnect delay in seconds.",
    )
    parser.add_argument(
        "--rtsp-reconnect-backoff",
        type=float,
        default=None,
        help="RTSP reconnect exponential backoff factor (>= 1.0).",
    )
    parser.add_argument(
        "--rtsp-stall-timeout",
        type=float,
        default=None,
        help="Live read stall timeout in seconds before considering stream unhealthy.",
    )

    # Filesystem-first traffic spool (for later portal upload).
    parser.add_argument(
        "--spool-dir",
        type=str,
        default=None,
        help="Write per-crossing traffic events (events.jsonl) + thumbnails into this directory.",
    )
    parser.add_argument("--site-id", type=str, default=None, help="Site ID for traffic spool metadata.")
    parser.add_argument(
        "--camera-id",
        type=str,
        default=None,
        help="Camera ID for traffic spool metadata (defaults to --camera if set).",
    )
    parser.add_argument(
        "--spool-thumbnails",
        dest="spool_thumbnails",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable writing thumbnails for crossing events (default: enabled).",
    )
    parser.add_argument("--spool-thumb-pad", type=int, default=None, help="Padding (pixels) around bbox when saving thumbnails.")
    parser.add_argument("--spool-thumb-max-side", type=int, default=None, help="Max side (pixels) for saved thumbnails.")
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


def _parse_size(spec: str) -> Tuple[int, int]:
    spec = (spec or "").strip().lower()
    if "x" not in spec:
        raise ValueError(f"Invalid size '{spec}'. Expected format like 848x478.")
    w_str, h_str = spec.split("x", 1)
    w = int(w_str)
    h = int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid size '{spec}': width/height must be > 0.")
    return w, h

def _draw_fps_overlay(
    frame,
    *,
    fps_loop: float,
    fps_processed: float,
    fps_source: float = 0.0,
    is_live: bool = False,
) -> None:
    text = f"FPS loop:{fps_loop:.1f} proc:{fps_processed:.1f}"
    if is_live and fps_source > 0.0:
        text += f" src:{fps_source:.1f}"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.62, thickness=2)
    fw = int(frame.shape[1])
    x = max(fw - tw - 14, 12)
    y = max(th + 12, 28)
    cv2.rectangle(frame, (x - 6, y - th - 8), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


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


def _resolve_camera_line_path(camera_value: str) -> Path:
    cam_arg = Path(camera_value)
    if cam_arg.exists():
        return cam_arg

    candidate1 = ROOT_DIR / "config" / "cameras" / f"{camera_value}.json"
    candidate2 = ROOT_DIR / "config" / f"{camera_value}.json"
    if candidate1.exists():
        return candidate1
    if candidate2.exists():
        return candidate2
    raise SystemExit(
        f"Camera line JSON not found for '{camera_value}'. "
        f"Tried: {candidate1} and {candidate2}"
    )


def _select_line_interactively(
    video_path: Path,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[float, float], Tuple[float, float]]]:
    """
    Memungkinkan user untuk memilih garis nya
    """

    cap = open_video_capture(video_path)
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

    for flag, value in (
        ("--queue-size", args.queue_size),
        ("--frame-stride", args.frame_stride),
    ):
        if value <= 0:
            raise SystemExit(f"{flag} must be > 0")
    for flag, value in (
        ("--target-fps", args.target_fps),
        ("--log-every-seconds", args.log_every_seconds),
        ("--rtsp-reconnect-initial-delay", args.rtsp_reconnect_initial_delay),
        ("--rtsp-reconnect-max-delay", args.rtsp_reconnect_max_delay),
        ("--rtsp-stall-timeout", args.rtsp_stall_timeout),
    ):
        if value is not None and value <= 0:
            raise SystemExit(f"{flag} must be > 0")
    for flag, value in (
        ("--rtsp-reconnect-max-attempts", args.rtsp_reconnect_max_attempts),
    ):
        if value is not None and value < 0:
            raise SystemExit(f"{flag} must be >= 0")
    for flag, value in (
        ("--rtsp-reconnect-backoff", args.rtsp_reconnect_backoff),
    ):
        if value is not None and value < 1.0:
            raise SystemExit(f"{flag} must be >= 1.0")

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = ROOT_DIR / cfg_path
        if not cfg_path.exists():
            raise SystemExit(f"Config override JSON not found: {cfg_path}")
        overrides_all = load_config_overrides(cfg_path)
        overrides_app, _overrides_extra = split_overrides(overrides_all)
        # Resolve relative paths in overrides against repo root by default.
        apply_config_overrides(cfg, overrides_app, path_base_dir=ROOT_DIR)

    rtsp_url: Optional[str] = None
    if args.rtsp_url is not None:
        rtsp_url = args.rtsp_url
    elif args.rtsp_url_env is not None:
        env_name = str(args.rtsp_url_env).strip()
        if env_name == "":
            raise SystemExit("--rtsp-url-env must be a non-empty environment variable name.")
        env_val = os.environ.get(env_name)
        if env_val is None or str(env_val).strip() == "":
            raise SystemExit(f"Environment variable '{env_name}' is not set or empty.")
        rtsp_url = str(env_val).strip()
    else:
        rtsp_url = cfg.io.rtsp_url
    is_live = rtsp_url is not None
    if is_live and args.input:
        raise SystemExit("Use only one source: either --input (file) or --rtsp-url (live).")

    # Allow CLI overrides
    if args.input:
        cfg.io.input_path = Path(args.input)
    if args.output:
        cfg.io.output_path = Path(args.output)
    if args.backend:
        cfg.model.backend = args.backend
    if args.model:
        cfg.model.model_path = Path(args.model)
    if args.confidence is not None:
        cfg.model.confidence_threshold = float(args.confidence)
    if args.nms_iou is not None:
        cfg.model.nms_iou_threshold = float(args.nms_iou)
    if args.pre_nms_topk is not None:
        cfg.model.pre_nms_topk = None if int(args.pre_nms_topk) <= 0 else int(args.pre_nms_topk)
    if args.allow_all_classes:
        cfg.model.allow_all_classes = True
    if args.class_ids:
        try:
            cfg.model.track_class_ids = [int(x.strip()) for x in args.class_ids.split(",") if x.strip() != ""]
        except Exception:
            raise SystemExit(f"Invalid --class-ids value: '{args.class_ids}'. Expected comma-separated integers.")
    if args.class_names:
        cfg.model.class_names_path = Path(args.class_names)
    if args.rtsp_reconnect is not None:
        cfg.io.rtsp_reconnect_enabled = bool(args.rtsp_reconnect)
    if args.rtsp_reconnect_max_attempts is not None:
        cfg.io.rtsp_reconnect_max_attempts = int(args.rtsp_reconnect_max_attempts)
    if args.rtsp_reconnect_initial_delay is not None:
        cfg.io.rtsp_reconnect_initial_delay_s = float(args.rtsp_reconnect_initial_delay)
    if args.rtsp_reconnect_max_delay is not None:
        cfg.io.rtsp_reconnect_max_delay_s = float(args.rtsp_reconnect_max_delay)
    if args.rtsp_reconnect_backoff is not None:
        cfg.io.rtsp_reconnect_backoff_factor = float(args.rtsp_reconnect_backoff)
    if args.rtsp_stall_timeout is not None:
        cfg.io.rtsp_stall_timeout_s = float(args.rtsp_stall_timeout)

    if cfg.io.rtsp_reconnect_max_attempts < 0:
        raise SystemExit("config io.rtsp_reconnect_max_attempts must be >= 0")
    for key, value in (
        ("io.rtsp_reconnect_initial_delay_s", cfg.io.rtsp_reconnect_initial_delay_s),
        ("io.rtsp_reconnect_max_delay_s", cfg.io.rtsp_reconnect_max_delay_s),
        ("io.rtsp_stall_timeout_s", cfg.io.rtsp_stall_timeout_s),
    ):
        if value <= 0:
            raise SystemExit(f"config {key} must be > 0")
    if cfg.io.rtsp_reconnect_max_delay_s < cfg.io.rtsp_reconnect_initial_delay_s:
        raise SystemExit("config io.rtsp_reconnect_max_delay_s must be >= io.rtsp_reconnect_initial_delay_s")
    if cfg.io.rtsp_reconnect_backoff_factor < 1.0:
        raise SystemExit("config io.rtsp_reconnect_backoff_factor must be >= 1.0")

    line_path: Optional[Path] = None
    if args.line_json:
        line_path = Path(args.line_json)
    elif args.camera:
        line_path = _resolve_camera_line_path(args.camera)
    elif cfg.line.line_json_path is not None:
        line_path = Path(cfg.line.line_json_path)
    elif cfg.line.camera_name:
        line_path = _resolve_camera_line_path(str(cfg.line.camera_name))

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

    if args.dump_config:
        dump_path = Path(args.dump_config)
        if not dump_path.is_absolute():
            dump_path = ROOT_DIR / dump_path
        write_app_config_json(dump_path, cfg)
        print(f"[main] Wrote resolved config JSON: {dump_path}")
        return

    if not is_live and not input_path.exists():
        raise SystemExit(f"Input video not found: {input_path}")

    if args.line_mode == "gate" and args.select_line:
        raise SystemExit("--select-line is only supported for --line-mode single. Use line_picker.py to save 2 lines.")
    if is_live and args.select_line:
        raise SystemExit("--select-line is not supported in --rtsp-url mode yet. Use --line-json/--camera instead.")
    if args.resize_to and args.select_line:
        raise SystemExit("--select-line is not supported together with --resize-to. Use line_picker.py to save the line JSON.")

    source_label = rtsp_url if is_live else str(input_path)
    cap = open_video_capture(
        rtsp_url if is_live else input_path,
        open_timeout_ms=args.open_timeout_ms,
        read_timeout_ms=args.read_timeout_ms,
    )
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video source: {source_label}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = float(args.target_fps) if (is_live and args.target_fps) else 30.0
    writer_fps = fps
    if is_live and args.target_fps:
        # In live mode we may drop frames (via StreamReader's bounded queue) to
        # honor --target-fps. If we write the output with the source-reported FPS,
        # playback becomes sped up. Clamp to the effective processing rate.
        writer_fps = min(float(args.target_fps), fps) if fps > 0.0 else float(args.target_fps)
    if args.write_processed_only and args.frame_stride > 1:
        # Keep playback speed sensible when writing only a subset of frames.
        writer_fps = max(float(writer_fps) / float(args.frame_stride), 1e-3)
    reported_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    reported_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    backend_name = None
    if hasattr(cap, "getBackendName"):
        try:
            backend_name = cap.getBackendName()
        except Exception:
            backend_name = None

    # Always infer the actual frame size from the first frame so that
    # normalized coordinates from line_picker.py map back exactly, even if
    # the container metadata reports a slightly different height/width.
    ret, frame0 = cap.read()
    if not ret:
        raise SystemExit("Could not read any frame from input video.")
    input_height, input_width = frame0.shape[:2]
    height, width = input_height, input_width
    if args.resize_to:
        try:
            width, height = _parse_size(args.resize_to)
        except ValueError as exc:
            raise SystemExit(str(exc))
    if not is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Menentukan maksimla frame dari waktu seconds
    max_frames: Optional[int] = args.max_frames
    max_end_time_s: Optional[float] = None
    if args.max_seconds is not None:
        seconds_limit = max(args.max_seconds, 0)
        if is_live:
            max_end_time_s = time.time() + float(seconds_limit)
        else:
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
    writer = None
    if is_live and args.output is None and not args.no_write and max_end_time_s is None and max_frames is None:
        print(
            "[main] Live mode: default output is disabled to avoid unbounded file growth. "
            "Use --output or --max-seconds (or --max-frames), or pass --no-write explicitly."
        )
        args.no_write = True
    if not args.no_write:
        writer = cv2.VideoWriter(str(output_path), fourcc, writer_fps, (width, height))

    class_names_map = None
    if cfg.model.class_names_path is not None:
        if not cfg.model.class_names_path.exists():
            raise SystemExit(f"Class names file not found: {cfg.model.class_names_path}")
        class_names_map = load_class_names(cfg.model.class_names_path)
        if (
            cfg.model.backend.lower() in {"onnx", "tensorrt", "torch"}
            and not cfg.model.track_class_ids
            and not cfg.model.allow_all_classes
        ):
            inferred = infer_track_class_ids_from_class_names(class_names_map)
            if inferred:
                cfg.model.track_class_ids = inferred
                print(f"[main] Inferred target class IDs from --class-names: {cfg.model.track_class_ids}")

    if cfg.model.backend.lower() in {"onnx", "tensorrt", "torch"} and not cfg.model.track_class_ids and not cfg.model.allow_all_classes:
        raise SystemExit(
            "Model-based backend selected but no target classes configured. "
            "Pass --class-ids (comma-separated) to specify which vehicle subclasses to count, "
            "or provide --class-names so class IDs can be inferred, "
            "or use --allow-all-classes for debugging."
        )

    track_color_by = str(cfg.visual.track_color_by).strip().lower()
    if track_color_by not in {"class", "track"}:
        print(
            f"[main] Warning: invalid app.visual.track_color_by='{cfg.visual.track_color_by}'. "
            "Using 'class'."
        )
        track_color_by = "class"
    track_class_colors = cfg.visual.class_colors if cfg.visual.class_colors else None
    track_palette = cfg.visual.track_palette if cfg.visual.track_palette else None
    track_default_color = tuple(cfg.visual.track_default_color)

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

    spool: Optional[TrafficSpoolWriter] = None
    spool_dir = Path(args.spool_dir) if args.spool_dir else cfg.spool.root_dir
    if spool_dir is not None:
        cam_id = args.camera_id or cfg.spool.camera_id or (args.camera if args.camera else "camera")
        site_id = str(args.site_id) if args.site_id is not None else str(cfg.spool.site_id)
        write_thumbs = cfg.spool.write_thumbnails if args.spool_thumbnails is None else bool(args.spool_thumbnails)
        thumb_pad = int(cfg.spool.thumb_pad) if args.spool_thumb_pad is None else int(args.spool_thumb_pad)
        thumb_max_side = int(cfg.spool.thumb_max_side) if args.spool_thumb_max_side is None else int(args.spool_thumb_max_side)

        model_version = (
            "motion"
            if cfg.model.backend.lower() == "motion"
            else str(Path(cfg.model.model_path).name)
        )
        cfg_version = str(args.camera) if args.camera else "default"
        source = {"type": "rtsp", "value": str(rtsp_url)} if is_live else {"type": "video", "value": str(input_path)}
        spool_cfg = TrafficSpoolConfig(
            root_dir=Path(spool_dir),
            site_id=str(site_id),
            camera_id=str(cam_id),
            write_thumbnails=bool(write_thumbs),
            thumb_pad=int(thumb_pad),
            thumb_max_side=int(thumb_max_side),
        )
        spool = TrafficSpoolWriter(
            spool_cfg,
            source=source,
            model_version=model_version,
            cfg_version=cfg_version,
            line_mode=("gate" if args.line_mode == "gate" else "line"),
            line_id=("gate_1" if args.line_mode == "gate" else "line_1"),
            fps=float(fps),
            frame_size=(int(width), int(height)),
            class_names=class_names_map,
        )

    print(
        f"[main] Processing {source_label} -> "
        f"{output_path if not args.no_write else '(no-write)'}"
    )
    if (input_width, input_height) != (width, height):
        print(f"[main] Input decoded size: {input_width}x{input_height} @ {fps:.2f} FPS")
        print(f"[main] Processing size: {width}x{height}")
    else:
        print(f"[main] Resolution: {width}x{height} @ {fps:.2f} FPS")
    if backend_name:
        print(f"[main] Video backend: {backend_name}")
    if reported_w and reported_h and (reported_w != input_width or reported_h != input_height):
        print(
            f"[main] Reported size differs: {reported_w}x{reported_h} "
            f"(using decoded {input_width}x{input_height})"
        )
    print(f"[main] Detector backend: {cfg.model.backend}")
    if cfg.model.track_class_ids:
        print(f"[main] Target class IDs: {sorted(set(cfg.model.track_class_ids))}")
    if class_names_map:
        print(f"[main] Class names loaded: {len(class_names_map)} classes")
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
    if args.frame_stride > 1:
        print(
            f"[main] Frame stride: {args.frame_stride} "
            f"(running detect/track/count on every {args.frame_stride}th frame)"
        )
    if args.write_processed_only:
        print(f"[main] Output write mode: processed-only (fps={writer_fps:.3f})")
    fast_skip_enabled = (
        (not is_live)
        and bool(args.fast_skip)
        and int(args.frame_stride) > 1
        and bool(args.write_processed_only)
        and (not args.show)
    )
    if args.fast_skip and not fast_skip_enabled:
        print(
            "[main] fast-skip requested but not enabled. "
            "Requirements: file input (no --rtsp-url), --frame-stride > 1, --write-processed-only, and no --show."
        )
    if fast_skip_enabled:
        print("[main] fast-skip: enabled (using VideoCapture.grab() for skipped frames)")
    if cfg.model.backend.lower() in {"onnx", "tensorrt", "torch"}:
        print(f"[main] Thresholds: conf={cfg.model.confidence_threshold} nms_iou={cfg.model.nms_iou_threshold} pre_nms_topk={cfg.model.pre_nms_topk}")
    if args.show_fps:
        print("[main] FPS overlay: enabled")

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
    process_index = 0
    warned_size_mismatch = False
    t_read = 0.0
    t_detect = 0.0
    t_track = 0.0
    t_count = 0.0
    t_draw = 0.0
    t_write = 0.0
    t_total = 0.0
    t_frames = 0
    t_processed = 0
    run_started_perf_s = time.perf_counter()
    fps_loop_rt = 0.0
    fps_processed_rt = 0.0
    fps_source_rt = 0.0
    fps_prev_time_s = run_started_perf_s
    fps_prev_frames = 0
    fps_prev_processed = 0
    fps_prev_source = 0
    fps_update_interval_s = 0.5
    last_log_time_s = time.time()
    next_due_time_s: Optional[float] = None

    pending_first_frame = frame0 if is_live else None
    latest_tracks = []
    reader: Optional[StreamReader] = None
    if is_live:
        reader = StreamReader(cap, queue_size=args.queue_size)
        reader.start()

    try:
        def refresh_realtime_fps(
            *,
            frame_total: int,
            processed_total: int,
            force: bool = False,
        ) -> None:
            nonlocal fps_loop_rt
            nonlocal fps_processed_rt
            nonlocal fps_source_rt
            nonlocal fps_prev_time_s
            nonlocal fps_prev_frames
            nonlocal fps_prev_processed
            nonlocal fps_prev_source

            now_perf = time.perf_counter()
            dt = now_perf - fps_prev_time_s
            if dt <= 0:
                return
            if (not force) and dt < fps_update_interval_s:
                return

            loop_delta = max(frame_total - fps_prev_frames, 0)
            processed_delta = max(processed_total - fps_prev_processed, 0)
            fps_loop_rt = float(loop_delta) / dt
            fps_processed_rt = float(processed_delta) / dt

            if is_live and reader is not None:
                source_total = int(reader.read_frames)
                source_delta = max(source_total - fps_prev_source, 0)
                fps_source_rt = float(source_delta) / dt
                fps_prev_source = source_total
            else:
                fps_source_rt = 0.0

            fps_prev_time_s = now_perf
            fps_prev_frames = frame_total
            fps_prev_processed = processed_total

        while True:
            t0_total = time.perf_counter()
            if max_frames is not None and frame_index >= max_frames:
                break
            if max_end_time_s is not None and time.time() >= max_end_time_s:
                break

            if is_live and args.target_fps:
                now = time.time()
                if next_due_time_s is None:
                    next_due_time_s = now
                if now < next_due_time_s:
                    time.sleep(max(next_due_time_s - now, 0.0))

            t0 = time.perf_counter()
            if pending_first_frame is not None:
                frame = pending_first_frame
                pending_first_frame = None
                frame_ts = time.time()
            elif reader is not None:
                item = reader.get(timeout_s=1.0)
                if item is None:
                    break
                frame = item.frame
                frame_ts = float(item.timestamp)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_ts = time.time()

            # Normalize frame size for the pipeline/output.
            if frame.shape[0] != height or frame.shape[1] != width:
                if not warned_size_mismatch:
                    print(
                        f"[main] Warning: decoded frame size {frame.shape[1]}x{frame.shape[0]} "
                        f"does not match processing size {width}x{height}; resizing."
                    )
                    warned_size_mismatch = True
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            t_read += time.perf_counter() - t0

            process_this_frame = (frame_index % args.frame_stride) == 0
            if process_this_frame:
                t0 = time.perf_counter()
                detections = detector.detect(frame)
                t_detect += time.perf_counter() - t0

                t0 = time.perf_counter()
                tracks = tracker.update(detections, process_index)
                t_track += time.perf_counter() - t0
                latest_tracks = tracks

                if args.line_mode == "gate":
                    tracks_updated = [t for t in tracks if t.last_seen_frame == process_index]
                    t0 = time.perf_counter()
                    events = line_counter.update(tracks_updated, process_index)
                    t_count += time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    events = line_counter.update(tracks, frame_index=process_index)
                    t_count += time.perf_counter() - t0

                if spool is not None and events:
                    spool.record_events(events, frame_bgr=frame, frame_ts=frame_ts)

                # Gambar setiap gambar yang telah diupdate
                # avoid "stuck" boxes when a track is kept alive across occlusions.
                if not args.no_draw:
                    t0 = time.perf_counter()
                    draw_tracks(
                        frame,
                        tracks,
                        frame_index=process_index,
                        stale_max_age=args.stale_frames,
                        class_names=class_names_map,
                        color=track_default_color,
                        class_colors=track_class_colors,
                        palette=track_palette,
                        color_by=track_color_by,
                    )
                    draw_line_and_counts(frame, line_counter, class_names=class_names_map)
                    t_draw += time.perf_counter() - t0
                process_index += 1
                t_processed += 1
            elif not args.no_draw:
                # For skipped frames, keep drawing latest tracks so overlays stay stable.
                t0 = time.perf_counter()
                if latest_tracks:
                    draw_tracks(
                        frame,
                        latest_tracks,
                        frame_index=max(process_index - 1, 0),
                        stale_max_age=args.stale_frames,
                        class_names=class_names_map,
                        color=track_default_color,
                        class_colors=track_class_colors,
                        palette=track_palette,
                        color_by=track_color_by,
                    )
                draw_line_and_counts(frame, line_counter, class_names=class_names_map)
                t_draw += time.perf_counter() - t0

            if args.show_fps:
                refresh_realtime_fps(
                    frame_total=(t_frames + 1),
                    processed_total=t_processed,
                )
                _draw_fps_overlay(
                    frame,
                    fps_loop=fps_loop_rt,
                    fps_processed=fps_processed_rt,
                    fps_source=fps_source_rt,
                    is_live=is_live,
                )

            if writer is not None and (not args.write_processed_only or process_this_frame):
                t0 = time.perf_counter()
                writer.write(frame)
                t_write += time.perf_counter() - t0

            if args.show:
                cv2.imshow("Vehicle Subclass Line Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if fast_skip_enabled:
                # We already processed the current frame. Now skip decoding the next (stride-1) frames.
                stride = int(args.frame_stride)
                to_skip = stride - 1
                if max_frames is not None:
                    remaining = int(max_frames - (frame_index + 1))
                    to_skip = max(min(to_skip, max(remaining, 0)), 0)

                t0_skip = time.perf_counter()
                skipped = 0
                for _ in range(to_skip):
                    ok = cap.grab()
                    if not ok:
                        break
                    skipped += 1
                t_read += time.perf_counter() - t0_skip

                frame_index += 1 + skipped
                if progress_bar is not None:
                    progress_bar.update(1 + skipped)
                t_total += time.perf_counter() - t0_total
                t_frames += 1 + skipped
            else:
                frame_index += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                t_total += time.perf_counter() - t0_total
                t_frames += 1

            if progress_bar is not None:
                # updated above
                pass
            if args.target_fps:
                next_due_time_s = time.time() + (1.0 / max(float(args.target_fps), 1e-9))

            if args.log_every_seconds:
                now = time.time()
                if (now - last_log_time_s) >= float(args.log_every_seconds):
                    refresh_realtime_fps(
                        frame_total=t_frames,
                        processed_total=t_processed,
                        force=True,
                    )
                    print(
                        f"[main] frames={t_frames} processed={t_processed} "
                        f"fps_loop_rt={fps_loop_rt:.2f} fps_proc_rt={fps_processed_rt:.2f} "
                        f"{f'fps_src_rt={fps_source_rt:.2f} ' if is_live and fps_source_rt > 0.0 else ''}"
                        f"A->B={line_counter.count_a_to_b} B->A={line_counter.count_b_to_a}"
                    )
                    last_log_time_s = now

            if args.benchmark and t_frames > 0 and (t_frames % max(args.benchmark_every, 1) == 0):
                fps_eff = float(t_frames) / max(t_total, 1e-9)
                det_avg = (1000 * t_detect / t_processed) if t_processed > 0 else 0.0
                track_avg = (1000 * t_track / t_processed) if t_processed > 0 else 0.0
                count_avg = (1000 * t_count / t_processed) if t_processed > 0 else 0.0
                print(
                    f"[bench] frames={t_frames} processed={t_processed} fps={fps_eff:.2f} "
                    f"read={1000*t_read/t_frames:.1f}ms "
                    f"det={det_avg:.1f}ms "
                    f"track={track_avg:.1f}ms "
                    f"count={count_avg:.1f}ms "
                    f"draw={1000*t_draw/t_frames:.1f}ms "
                    f"write={1000*t_write/t_frames:.1f}ms"
                )
    except KeyboardInterrupt:
        print("[main] Interrupted (Ctrl-C). Shutting down...")

    if reader is not None:
        reader.stop()
    else:
        cap.release()
    if writer is not None:
        writer.release()
    if spool is not None:
        spool.close()
    if progress_bar is not None:
        progress_bar.close()
    if args.show:
        cv2.destroyAllWindows()

    if args.benchmark and t_frames > 0:
        fps_eff = float(t_frames) / max(t_total, 1e-9)
        det_avg = (1000 * t_detect / t_processed) if t_processed > 0 else 0.0
        track_avg = (1000 * t_track / t_processed) if t_processed > 0 else 0.0
        count_avg = (1000 * t_count / t_processed) if t_processed > 0 else 0.0
        print(
            f"[bench] done frames={t_frames} processed={t_processed} fps={fps_eff:.2f} "
            f"read={1000*t_read/t_frames:.1f}ms "
            f"det={det_avg:.1f}ms "
            f"track={track_avg:.1f}ms "
            f"count={count_avg:.1f}ms "
            f"draw={1000*t_draw/t_frames:.1f}ms "
            f"write={1000*t_write/t_frames:.1f}ms"
        )

    print(
        f"[main] Done. A->B: {line_counter.count_a_to_b}, "
        f"B->A: {line_counter.count_b_to_a}"
    )


if __name__ == "__main__":
    main()
