from __future__ import annotations

import argparse
import copy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone, tzinfo
import difflib
import math
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple
import threading
import time

import cv2
import json
import sys
import numpy as np

from .config import AppConfig, ROOT_DIR, get_default_config, infer_track_class_ids_from_class_names, load_class_names
from .config_io import apply_config_overrides, load_config_overrides, split_overrides, write_app_config_json
from .detector import Detector
from .draw_utils import draw_line_and_counts, draw_tracks
from .line_counter import LineCounter, TwoLineGateCounter
from .portal_uploader import RetryConfig, UploaderConfig, process_pending_runs, resolve_portal_api_key
from .report_writer import ReportWriter, ReportWriterConfig
from .stream_reader import StreamReader
from .traffic_spool import TrafficSpoolConfig, TrafficSpoolWriter
from .tracker import Tracker
from .videoio_utils import open_video_capture

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover - safe fallback
    tqdm = None


def _add_bool_arg(
    parser: argparse.ArgumentParser,
    option: str,
    *,
    dest: str,
    default: Optional[bool],
    help: Optional[str] = None,
) -> None:
    """Add a --foo/--no-foo flag pair with a Python 3.8-compatible fallback."""

    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            option,
            dest=dest,
            action=argparse.BooleanOptionalAction,
            default=default,
            help=help,
        )
        return

    if not option.startswith("--"):
        raise ValueError(f"Expected long option starting with '--', got: {option}")

    parser.set_defaults(**{dest: default})
    parser.add_argument(option, dest=dest, action="store_true", help=help)
    parser.add_argument(
        f"--no-{option[2:]}",
        dest=dest,
        action="store_false",
        help=argparse.SUPPRESS,
    )

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
        "--class-vote-mode",
        type=str,
        choices=["instant", "majority", "weighted_majority"],
        default=None,
        help=(
            "Class decision strategy for counting/reporting per track: "
            "'instant' | 'majority' | 'weighted_majority'."
        ),
    )
    parser.add_argument(
        "--class-vote-min-frames",
        type=int,
        default=None,
        help="Minimum classified frames before majority-based class vote is used.",
    )
    parser.add_argument(
        "--class-vote-ambiguity-ratio",
        type=float,
        default=None,
        help=(
            "Ambiguity guard for class vote. If top/second vote is below this ratio, "
            "keep previous stable class. Use <= 1.0 to disable."
        ),
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
        help="Live mode only: bounded reader queue size.",
    )
    parser.add_argument(
        "--queue-policy",
        type=str,
        choices=["drop_oldest", "block"],
        default=None,
        help=(
            "Live mode queue overflow policy: "
            "'drop_oldest' keeps latency lower, 'block' favors frame completeness."
        ),
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
        "--headless-status-every-seconds",
        type=float,
        default=10.0,
        help=(
            "Write periodic runtime health snapshots for headless monitoring "
            "(run.json health_summary + status.json when spool is enabled)."
        ),
    )
    parser.add_argument(
        "--headless-status-json",
        type=str,
        default=None,
        help=(
            "Optional explicit path for standalone headless status JSON output. "
            "If omitted and spool is enabled, status is written to <run_dir>/status.json."
        ),
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
        "--rtsp-capture-backend",
        type=str,
        choices=["opencv", "gstreamer"],
        default=None,
        help="Live mode only: capture backend for RTSP open/reconnect.",
    )
    parser.add_argument(
        "--rtsp-transport",
        type=str,
        choices=["tcp", "udp"],
        default=None,
        help="Live mode only: RTSP transport preference for GStreamer backend.",
    )
    parser.add_argument(
        "--rtsp-latency-ms",
        type=int,
        default=None,
        help="Live mode only: RTSP latency (ms) for GStreamer backend.",
    )
    parser.add_argument(
        "--rtsp-codec",
        type=str,
        choices=["h264", "h265"],
        default=None,
        help="Live mode only: codec hint for GStreamer depay/parser selection.",
    )
    parser.add_argument(
        "--rtsp-gst-pipeline",
        type=str,
        default=None,
        help="Live mode only: full custom GStreamer pipeline string (advanced override).",
    )
    _add_bool_arg(
        parser,
        "--rtsp-reconnect",
        dest="rtsp_reconnect",
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
    _add_bool_arg(
        parser,
        "--spool-thumbnails",
        dest="spool_thumbnails",
        default=None,
        help="Enable/disable writing thumbnails for crossing events (default: enabled).",
    )
    parser.add_argument("--spool-thumb-pad", type=int, default=None, help="Padding (pixels) around bbox when saving thumbnails.")
    parser.add_argument("--spool-thumb-max-side", type=int, default=None, help="Max side (pixels) for saved thumbnails.")

    _add_bool_arg(
        parser,
        "--spool-scene-thumbnails",
        dest="spool_scene_thumbnails",
        default=None,
        help="Enable/disable writing scene-context thumbnails (default: enabled).",
    )
    parser.add_argument("--spool-scene-thumb-max-side", type=int, default=None, help="Max side (pixels) for scene thumbnails.")
    parser.add_argument("--spool-scene-thumb-quality", type=int, default=None, help="JPEG quality (10..100) for scene thumbnails.")
    
    _add_bool_arg(
        parser,
        "--report-csv",
        dest="report_csv",
        default=None,
        help="Enable/disable per-run report csv output (default: enabled when spool is enabled).",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="Report CSV Filename inside each run directory (default: report.csv)"
    )
    _add_bool_arg(
        parser,
        "--report-include-extra-cols",
        dest="report_include_extra_cols",
        default=None,
        help="Include technical columns (track_id), frame_index, class_id, confidence, thumb path).",
    )
    _add_bool_arg(
        parser,
        "--portal-upload",
        dest="portal_upload",
        default=False,
        help=(
            "Enable integrated portal uploader in this same process. "
            "Use this to run detect+spool+upload with a single command."
        ),
    )
    parser.add_argument(
        "--portal-api-base-url",
        type=str,
        default=None,
        help="Portal API base URL for integrated upload (e.g. http://localhost:5000).",
    )
    parser.add_argument(
        "--portal-api-key",
        type=str,
        default=None,
        help="Portal API key for integrated upload. If omitted, --portal-api-key-env is used.",
    )
    parser.add_argument(
        "--portal-api-key-env",
        type=str,
        default="PORTAL_API_KEY",
        help="Environment variable containing portal API key for integrated upload.",
    )
    parser.add_argument(
        "--portal-api-key-json-path",
        type=str,
        default=None,
        help=(
            "Optional path to appsettings.Local.json containing Portal.ApiKey. "
            "If omitted, integrated uploader tries ./portal/appsettings.Local.json."
        ),
    )
    parser.add_argument(
        "--portal-upload-interval-s",
        type=float,
        default=10.0,
        help="Live mode only: how often integrated uploader syncs spool runs.",
    )
    parser.add_argument(
        "--portal-upload-max-runs-per-pass",
        type=int,
        default=None,
        help="Integrated uploader: limit runs processed per sync pass.",
    )
    parser.add_argument(
        "--portal-upload-timeout-s",
        type=float,
        default=20.0,
        help="Integrated uploader HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--portal-upload-events-batch-size",
        type=int,
        default=200,
        help="Integrated uploader batch size for /api/events/upsert.",
    )
    _add_bool_arg(
        parser,
        "--portal-upload-thumbnails",
        dest="portal_upload_thumbnails",
        default=True,
        help="Integrated uploader: enable/disable event thumbnail upload.",
    )
    _add_bool_arg(
        parser,
        "--portal-upload-scene-thumbnails",
        dest="portal_upload_scene_thumbnails",
        default=False,
        help="Integrated uploader: enable/disable scene thumbnail upload.",
    )
    parser.add_argument(
        "--portal-upload-retry-max-attempts",
        type=int,
        default=8,
        help="Integrated uploader retry attempts per API request (0=unlimited).",
    )
    parser.add_argument(
        "--portal-upload-retry-initial-delay-s",
        type=float,
        default=1.0,
        help="Integrated uploader initial retry delay in seconds.",
    )
    parser.add_argument(
        "--portal-upload-retry-max-delay-s",
        type=float,
        default=30.0,
        help="Integrated uploader max retry delay in seconds.",
    )
    parser.add_argument(
        "--portal-upload-retry-backoff-factor",
        type=float,
        default=2.0,
        help="Integrated uploader retry backoff factor (>=1.0).",
    )

    parser.add_argument(
        "--video-start",
        type=str,
        default=None,
        help=(
            "Offline video only: RFC3339 timestamp with timezone for the start of the video. "
            "If set (and spool enabled), occurred_at_utc for events is derived from video timeline "
            "(start + frame_index/fps). Example: 2026-02-18T08:00:00+07:00"
        ),
    )
    parser.add_argument(
        "--source-fps-override",
        type=float,
        default=None,
        help="Override decoded source FPS when metadata is missing/wrong (used for timestamps and writer FPS).",
    )
    parser.add_argument(
        "--prompt-video-start",
        action="store_true",
        help=(
            "Offline video only convenience: if spool is enabled and --video-start is missing, "
            "prompt interactively for date/time/timezone to build a valid RFC3339 timestamp."
        ),
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


def _parse_rfc3339_to_epoch_utc(value: str) -> float:
    s = (value or "").strip()
    if not s:
        raise ValueError("empty timestamp")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError("timestamp must include timezone offset (e.g. +07:00 or Z)")
    return float(dt.astimezone(timezone.utc).timestamp())


def _prompt_video_start_rfc3339(*, attempts: int = 3) -> str:
    """
    Interactive prompt helper. Kept simple and CLI-only (no UI deps).
    """

    def ask(prompt: str) -> str:
        return str(input(prompt)).strip()

    for _ in range(max(int(attempts), 1)):
        print("[prompt] Enter the video start time (used for occurred_at_utc).")
        date = ask("[prompt] Date (YYYY-MM-DD): ")
        time_s = ask("[prompt] Time (HH:MM[:SS], default 00:00:00): ") or "00:00:00"
        if len(time_s.split(":")) == 2:
            time_s = f"{time_s}:00"
        print("[prompt] Timezone options:")
        print("[prompt]   1) WIB  (+07:00, Asia/Jakarta)")
        print("[prompt]   2) WITA (+08:00, Asia/Makassar)")
        print("[prompt]   3) WIT  (+09:00, Asia/Jayapura)")
        print("[prompt]   4) Custom offset (e.g. +07:00 or Z)")
        tz_sel = ask("[prompt] Choose (1-4, default 1): ") or "1"
        tz_map = {"1": "WIB", "2": "WITA", "3": "WIT", "4": "CUSTOM"}
        tz_sel_norm = tz_map.get(tz_sel.strip(), "WIB")
        if tz_sel_norm == "CUSTOM":
            tz = ask("[prompt] Offset (+HH:MM or Z): ")
        else:
            tz = tz_sel_norm

        tz_upper = (tz or "").strip().upper()
        if tz_upper in {"WIB", "JAKARTA", "ASIA/JAKARTA"}:
            tz = "+07:00"
        elif tz_upper in {"WITA", "MAKASSAR", "ASIA/MAKASSAR"}:
            tz = "+08:00"
        elif tz_upper in {"WIT", "JAYAPURA", "ASIA/JAYAPURA"}:
            tz = "+09:00"
        elif tz_upper == "Z":
            tz = "Z"
        candidate = f"{date}T{time_s}{tz}"
        try:
            _ = _parse_rfc3339_to_epoch_utc(candidate)
            return candidate
        except ValueError as exc:
            print(f"[prompt] Invalid value: {exc}")
            print("[prompt] Example: 2026-02-18T08:00:00+07:00")

    raise SystemExit("Failed to collect a valid --video-start via prompt.")


def _missing_path_hint(path: Path) -> str:
    """
    Give a helpful hint for common path mistakes (typos, Windows slashes in bash/WSL).
    """

    raw = str(path)
    normalized = raw.replace("\\", "/")
    if normalized != raw and Path(normalized).exists():
        return f"Resolved by replacing backslashes: {normalized}"

    parent = path.parent
    grand = parent.parent
    if not parent.exists() and grand.exists():
        try:
            siblings = [p.name for p in grand.iterdir()]
        except OSError:
            siblings = []
        if siblings:
            matches = difflib.get_close_matches(parent.name, siblings, n=1, cutoff=0.6)
            if matches:
                candidate_parent = grand / matches[0]
                candidate = candidate_parent / path.name
                if candidate.exists():
                    return f"Did you mean: {candidate} ?"
                return f"Did you mean folder: {candidate_parent} ?"

    return ""


def _open_source_with_first_frame(
    source: str | Path,
    *,
    open_timeout_ms: Optional[int] = None,
    read_timeout_ms: Optional[int] = None,
    source_label: Optional[str] = None,
    is_live: bool = False,
    rtsp_capture_backend: str = "opencv",
    rtsp_transport: str = "tcp",
    rtsp_latency_ms: int = 200,
    rtsp_codec: str = "h264",
    rtsp_gst_pipeline: Optional[str] = None,
    queue_policy: str = "drop_oldest",
) -> Tuple[cv2.VideoCapture, np.ndarray, float, int, int, Optional[str]]:
    appsink_drop = str(queue_policy).strip().lower() == "drop_oldest"
    appsink_max_buffers = 1 if appsink_drop else 8
    cap = open_video_capture(
        source,
        open_timeout_ms=open_timeout_ms,
        read_timeout_ms=read_timeout_ms,
        rtsp_capture_backend=(rtsp_capture_backend if is_live else "opencv"),
        rtsp_transport=rtsp_transport,
        rtsp_latency_ms=int(rtsp_latency_ms),
        rtsp_codec=rtsp_codec,
        rtsp_gst_pipeline=rtsp_gst_pipeline,
        rtsp_appsink_drop=appsink_drop,
        rtsp_appsink_max_buffers=appsink_max_buffers,
    )
    label = source_label if source_label is not None else str(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {label}")

    # get the parameters of the rtsp stream
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    reported_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    backend_name = None
    if hasattr(cap, "getBackendName"):
        try:
            backend_name = cap.getBackendName()
        except Exception:
            backend_name = None

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError(f"Could not read any frame from source: {label}")

    return cap, frame0, fps, reported_w, reported_h, backend_name


def _reconnect_delay_s(
    attempt_no: int,
    *,
    initial_delay_s: float,
    max_delay_s: float,
    backoff_factor: float,
) -> float:
    if attempt_no <= 1:
        return min(initial_delay_s, max_delay_s)
    delay = initial_delay_s * (backoff_factor ** float(attempt_no - 1))
    return min(delay, max_delay_s)


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

    def _normalize_cli_path(value: str) -> Path:
        """
        Normalize common Windows path forms when running under bash/WSL:
        - .\\foo\\bar -> ./foo/bar
        - ..\\foo\\bar -> ../foo/bar
        - C:\\Users\\x -> /mnt/c/Users/x  (best-effort for WSL)
        """

        raw = str(value or "").strip().strip('"').strip("'")
        if raw == "":
            return Path(raw)

        s = raw.replace("\\", "/")
        if s.startswith("./") or s.startswith("../") or s.startswith("/"):
            return Path(s)

        # Handle ".\x" or "..\x" that becomes "./x" or "../x" after replace.
        if s.startswith(".//"):
            s = "./" + s[3:]
        if s.startswith("..//"):
            s = "../" + s[4:]

        # Drive-letter best-effort: C:/... -> /mnt/c/...
        if len(s) >= 3 and s[1] == ":" and s[2] == "/":
            drive = s[0].lower()
            rest = s[3:].lstrip("/")
            mnt = Path(f"/mnt/{drive}")
            if mnt.exists():
                return mnt / rest
            # If not WSL, keep as-is.
            return Path(s)

        return Path(s)

    def _dataclass_diff(a: object, b: object, *, prefix: str = "") -> list[str]:
        if type(a) is not type(b):
            return [prefix.rstrip(".") or "<root>"]
        if is_dataclass(a) and is_dataclass(b):
            out: list[str] = []
            for f in fields(a):
                out.extend(
                    _dataclass_diff(
                        getattr(a, f.name),
                        getattr(b, f.name),
                        prefix=f"{prefix}{f.name}.",
                    )
                )
            return out
        if isinstance(a, Path) or isinstance(b, Path):
            return [] if str(a) == str(b) else [prefix.rstrip(".") or "<root>"]
        return [] if a == b else [prefix.rstrip(".") or "<root>"]

    for flag, value in (
        ("--queue-size", args.queue_size),
        ("--frame-stride", args.frame_stride),
    ):
        if value <= 0:
            raise SystemExit(f"{flag} must be > 0")
    for flag, value in (
        ("--class-vote-min-frames", args.class_vote_min_frames),
    ):
        if value is not None and value <= 0:
            raise SystemExit(f"{flag} must be > 0")
    for flag, value in (
        ("--target-fps", args.target_fps),
        ("--log-every-seconds", args.log_every_seconds),
        ("--rtsp-reconnect-initial-delay", args.rtsp_reconnect_initial_delay),
        ("--rtsp-reconnect-max-delay", args.rtsp_reconnect_max_delay),
        ("--rtsp-stall-timeout", args.rtsp_stall_timeout),
        ("--class-vote-ambiguity-ratio", args.class_vote_ambiguity_ratio),
    ):
        if value is not None and value <= 0:
            raise SystemExit(f"{flag} must be > 0")
    if args.headless_status_every_seconds is not None and float(args.headless_status_every_seconds) < 0:
        raise SystemExit("--headless-status-every-seconds must be >= 0")
    if args.rtsp_latency_ms is not None and args.rtsp_latency_ms < 0:
        raise SystemExit("--rtsp-latency-ms must be >= 0")
    for flag, value in (
        ("--rtsp-reconnect-max-attempts", args.rtsp_reconnect_max_attempts),
        ("--portal-upload-max-runs-per-pass", args.portal_upload_max_runs_per_pass),
        ("--portal-upload-retry-max-attempts", args.portal_upload_retry_max_attempts),
    ):
        if value is not None and value < 0:
            raise SystemExit(f"{flag} must be >= 0")
    for flag, value in (
        ("--rtsp-reconnect-backoff", args.rtsp_reconnect_backoff),
        ("--portal-upload-retry-backoff-factor", args.portal_upload_retry_backoff_factor),
    ):
        if value is not None and value < 1.0:
            raise SystemExit(f"{flag} must be >= 1.0")
    for flag, value in (
        ("--portal-upload-interval-s", args.portal_upload_interval_s),
        ("--portal-upload-timeout-s", args.portal_upload_timeout_s),
        ("--portal-upload-retry-initial-delay-s", args.portal_upload_retry_initial_delay_s),
        ("--portal-upload-retry-max-delay-s", args.portal_upload_retry_max_delay_s),
    ):
        if value is not None and value <= 0:
            raise SystemExit(f"{flag} must be > 0")
    if (
        args.portal_upload_retry_max_delay_s is not None
        and args.portal_upload_retry_initial_delay_s is not None
        and float(args.portal_upload_retry_max_delay_s) < float(args.portal_upload_retry_initial_delay_s)
    ):
        raise SystemExit("--portal-upload-retry-max-delay-s must be >= --portal-upload-retry-initial-delay-s")
    if args.portal_upload_events_batch_size is not None and int(args.portal_upload_events_batch_size) <= 0:
        raise SystemExit("--portal-upload-events-batch-size must be > 0")

    if args.config:
        cfg_before = copy.deepcopy(cfg)
        cfg_path = _normalize_cli_path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = ROOT_DIR / cfg_path
        if not cfg_path.exists():
            raise SystemExit(
                "Config override JSON not found.\n"
                f"- raw: {args.config}\n"
                f"- normalized: {cfg_path}\n"
                "Tip: in bash/WSL, prefer './config/local/exp.json' not '.\\\\config\\\\local\\\\exp.json'."
            )
        overrides_all = load_config_overrides(cfg_path)
        overrides_app, _overrides_extra = split_overrides(overrides_all)
        # Resolve relative paths in overrides against repo root by default.
        apply_config_overrides(cfg, overrides_app, path_base_dir=ROOT_DIR)
        changed = _dataclass_diff(cfg_before, cfg)
        if changed:
            preview = ", ".join(changed[:12])
            more = "" if len(changed) <= 12 else f" (+{len(changed) - 12} more)"
            print(f"[main] Loaded config overrides: {cfg_path} (changed {len(changed)} keys: {preview}{more})")
        else:
            print(f"[main] Loaded config overrides: {cfg_path} (no effective changes)")

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
        cfg.io.input_path = _normalize_cli_path(args.input)
    if args.output:
        cfg.io.output_path = _normalize_cli_path(args.output)
    if args.backend:
        cfg.model.backend = args.backend
    if args.model:
        cfg.model.model_path = _normalize_cli_path(args.model)
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
        cfg.model.class_names_path = _normalize_cli_path(args.class_names)
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
    if args.rtsp_capture_backend is not None:
        cfg.io.rtsp_capture_backend = str(args.rtsp_capture_backend)
    if args.rtsp_transport is not None:
        cfg.io.rtsp_transport = str(args.rtsp_transport)
    if args.rtsp_latency_ms is not None:
        cfg.io.rtsp_latency_ms = int(args.rtsp_latency_ms)
    if args.rtsp_codec is not None:
        cfg.io.rtsp_codec = str(args.rtsp_codec)
    if args.rtsp_gst_pipeline is not None:
        cfg.io.rtsp_gst_pipeline = str(args.rtsp_gst_pipeline)
    if args.queue_policy is not None:
        cfg.io.live_queue_policy = str(args.queue_policy)
    if args.video_start is not None:
        cfg.io.video_start = str(args.video_start)
    if args.source_fps_override is not None:
        cfg.io.source_fps_override = float(args.source_fps_override)
    if args.class_vote_mode is not None:
        cfg.tracker.class_vote_mode = str(args.class_vote_mode)
    if args.class_vote_min_frames is not None:
        cfg.tracker.class_vote_min_frames = int(args.class_vote_min_frames)
    if args.class_vote_ambiguity_ratio is not None:
        cfg.tracker.class_vote_ambiguity_ratio = float(args.class_vote_ambiguity_ratio)

    class_vote_mode = str(cfg.tracker.class_vote_mode).strip().lower()
    if class_vote_mode not in {"instant", "majority", "weighted_majority"}:
        raise SystemExit(
            "Invalid class vote mode. Use one of: instant, majority, weighted_majority."
        )
    if int(cfg.tracker.class_vote_min_frames) <= 0:
        raise SystemExit("tracker.class_vote_min_frames must be > 0.")
    if float(cfg.tracker.class_vote_ambiguity_ratio) <= 0:
        raise SystemExit("tracker.class_vote_ambiguity_ratio must be > 0.")

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
    if int(cfg.io.rtsp_latency_ms) < 0:
        raise SystemExit("config io.rtsp_latency_ms must be >= 0")

    rtsp_capture_backend = str(cfg.io.rtsp_capture_backend).strip().lower()
    if rtsp_capture_backend not in {"opencv", "gstreamer"}:
        raise SystemExit("config io.rtsp_capture_backend must be 'opencv' or 'gstreamer'")
    rtsp_transport = str(cfg.io.rtsp_transport).strip().lower()
    if rtsp_transport not in {"tcp", "udp"}:
        raise SystemExit("config io.rtsp_transport must be 'tcp' or 'udp'")
    rtsp_codec = str(cfg.io.rtsp_codec).strip().lower()
    if rtsp_codec not in {"h264", "h265"}:
        raise SystemExit("config io.rtsp_codec must be 'h264' or 'h265'")
    queue_policy = str(cfg.io.live_queue_policy).strip().lower()
    if queue_policy not in {"drop_oldest", "block"}:
        raise SystemExit("config io.live_queue_policy must be 'drop_oldest' or 'block'")
    rtsp_gst_pipeline = None
    if cfg.io.rtsp_gst_pipeline is not None:
        rtsp_gst_pipeline = str(cfg.io.rtsp_gst_pipeline).strip() or None

    line_path: Optional[Path] = None
    if args.line_json:
        line_path = _normalize_cli_path(args.line_json)
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
        dump_path = _normalize_cli_path(args.dump_config)
        if not dump_path.is_absolute():
            dump_path = ROOT_DIR / dump_path
        write_app_config_json(dump_path, cfg)
        print(f"[main] Wrote resolved config JSON: {dump_path}")
        return

    standalone_headless_status_path: Optional[Path] = None
    if args.headless_status_json is not None:
        standalone_headless_status_path = _normalize_cli_path(args.headless_status_json)
        if not standalone_headless_status_path.is_absolute():
            standalone_headless_status_path = ROOT_DIR / standalone_headless_status_path

    if not is_live and not input_path.exists():
        hint = _missing_path_hint(input_path)
        msg = f"Input video not found: {input_path}"
        if hint:
            msg += f"\n{hint}"
        raise SystemExit(msg)

    if args.line_mode == "gate" and args.select_line:
        raise SystemExit("--select-line is only supported for --line-mode single. Use line_picker.py to save 2 lines.")
    if is_live and args.select_line:
        raise SystemExit("--select-line is not supported in --rtsp-url mode yet. Use --line-json/--camera instead.")
    if args.resize_to and args.select_line:
        raise SystemExit("--select-line is not supported together with --resize-to. Use line_picker.py to save the line JSON.")

    source_label = rtsp_url if is_live else str(input_path)
    try:
        cap, frame0, fps, reported_w, reported_h, backend_name = _open_source_with_first_frame(
            rtsp_url if is_live else input_path,
            open_timeout_ms=args.open_timeout_ms,
            read_timeout_ms=args.read_timeout_ms,
            source_label=source_label,
            is_live=is_live,
            rtsp_capture_backend=rtsp_capture_backend,
            rtsp_transport=rtsp_transport,
            rtsp_latency_ms=int(cfg.io.rtsp_latency_ms),
            rtsp_codec=rtsp_codec,
            rtsp_gst_pipeline=rtsp_gst_pipeline,
            queue_policy=queue_policy,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    if fps <= 0.0:
        fps = float(args.target_fps) if (is_live and args.target_fps) else 30.0
    if cfg.io.source_fps_override is not None:
        override = float(cfg.io.source_fps_override)
        if override <= 0.0:
            raise SystemExit("io.source_fps_override must be > 0.")
        fps = override
    writer_fps = fps
    if is_live and args.target_fps:
        # In live mode we may drop frames (via StreamReader's bounded queue) to
        # honor --target-fps. If we write the output with the source-reported FPS,
        # playback becomes sped up. Clamp to the effective processing rate.
        writer_fps = min(float(args.target_fps), fps) if fps > 0.0 else float(args.target_fps)
    if args.write_processed_only and args.frame_stride > 1:
        # Keep playback speed sensible when writing only a subset of frames.
        writer_fps = max(float(writer_fps) / float(args.frame_stride), 1e-3)
    # Always infer the actual frame size from the first frame so that
    # normalized coordinates from line_picker.py map back exactly, even if
    # the container metadata reports a slightly different height/width.
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
    report_writer: Optional[ReportWriter] = None
    portal_uploader_cfg: Optional[UploaderConfig] = None
    portal_uploader_interval_s: float = float(args.portal_upload_interval_s)
    portal_uploader_max_runs: Optional[int] = args.portal_upload_max_runs_per_pass
    portal_uploader_stop_event: Optional[threading.Event] = None
    portal_uploader_thread: Optional[threading.Thread] = None
    portal_upload_lock = threading.Lock()
    portal_upload_runtime: Dict[str, object] = {
        "watch_passes_total": 0,
        "discovered_runs_total": 0,
        "completed_runs_total": 0,
        "skipped_runs_total": 0,
        "failed_runs_total": 0,
        "last_pass_reason": None,
        "last_pass_at_utc": None,
        "last_success_at_utc": None,
        "last_error": None,
        "last_error_at_utc": None,
    }
    headless_status_every_s = float(args.headless_status_every_seconds or 0.0)
    next_headless_status_s = time.time() + max(headless_status_every_s, 0.0)
    standalone_status_target: Optional[Path] = standalone_headless_status_path
    spool_events_written_total = 0
    spool_dir = Path(args.spool_dir) if args.spool_dir else cfg.spool.root_dir
    if spool_dir is not None:
        cam_id = args.camera_id or cfg.spool.camera_id or (args.camera if args.camera else "camera")
        site_id = str(args.site_id) if args.site_id is not None else str(cfg.spool.site_id)
        write_thumbs = cfg.spool.write_thumbnails if args.spool_thumbnails is None else bool(args.spool_thumbnails)
        write_scene_thumbs = (
            cfg.spool.write_scene_thumbnails
            if args.spool_scene_thumbnails is None
            else bool(args.spool_scene_thumbnails)
        )
        thumb_pad = int(cfg.spool.thumb_pad) if args.spool_thumb_pad is None else int(args.spool_thumb_pad)
        thumb_max_side = int(cfg.spool.thumb_max_side) if args.spool_thumb_max_side is None else int(args.spool_thumb_max_side)
        scene_max_side = int(cfg.spool.scene_thumb_max_side) if args.spool_scene_thumb_max_side is None else int(args.spool_scene_thumb_max_side)
        scene_quality = int(cfg.spool.scene_thumb_quality) if args.spool_scene_thumb_quality is None else int(args.spool_scene_thumb_quality)

        model_version = (
            "motion"
            if cfg.model.backend.lower() == "motion"
            else str(Path(cfg.model.model_path).name)
        )
        cfg_version = str(args.camera) if args.camera else "default"
        source = {"type": "rtsp", "value": str(rtsp_url)} if is_live else {"type": "video", "value": str(input_path)}
        spool_lines = None
        if hasattr(line_counter, "lines"):
            try:
                spool_lines = [
                    ((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
                    for (a, b) in list(line_counter.lines)
                ]
            except Exception:
                spool_lines = None
        if (not spool_lines) and hasattr(line_counter, "p1") and hasattr(line_counter, "p2"):
            try:
                spool_lines = [
                    (
                        (int(line_counter.p1[0]), int(line_counter.p1[1])),
                        (int(line_counter.p2[0]), int(line_counter.p2[1])),
                    )
                ]
            except Exception:
                spool_lines = None

        spool_cfg = TrafficSpoolConfig(
            root_dir=Path(spool_dir),
            site_id=str(site_id),
            camera_id=str(cam_id),
            write_thumbnails=bool(write_thumbs),
            write_scene_thumbnails=bool(write_scene_thumbs),
            thumb_pad=int(thumb_pad),
            thumb_max_side=int(thumb_max_side),
            scene_thumb_max_side=int(scene_max_side),
            scene_thumb_quality=int(scene_quality),
        )
        spool = TrafficSpoolWriter(
            spool_cfg,
            source=source,
            model_version=model_version,
            cfg_version=cfg_version,
            line_mode=("gate" if args.line_mode == "gate" else "line"),
            line_id=("gate_1" if args.line_mode == "gate" else "line_1"),
            lines=spool_lines,
            fps=float(fps),
            frame_size=(int(width), int(height)),
            class_names=class_names_map,
        )
        if standalone_status_target is None:
            standalone_status_target = spool.run_dir / "status.json"
    if args.portal_upload:
        if spool is None or spool_dir is None:
            raise SystemExit(
                "--portal-upload requires spool output. Set --spool-dir (or app.spool.root_dir) first."
            )
        if not args.portal_api_base_url:
            raise SystemExit("--portal-upload requires --portal-api-base-url.")
        portal_api_key = resolve_portal_api_key(
            args.portal_api_key,
            api_key_env=str(args.portal_api_key_env),
            appsettings_local_path=args.portal_api_key_json_path,
        )
        if not portal_api_key:
            raise SystemExit(
                "Missing portal API key for integrated upload. "
                "Pass --portal-api-key, set --portal-api-key-env, "
                "or set Portal.ApiKey in portal/appsettings.Local.json."
            )
        portal_uploader_cfg = UploaderConfig(
            spool_dir=Path(spool_dir),
            api_base_url=str(args.portal_api_base_url),
            api_key=str(portal_api_key),
            timeout_s=float(args.portal_upload_timeout_s),
            events_batch_size=int(args.portal_upload_events_batch_size),
            upload_thumbnails=bool(args.portal_upload_thumbnails),
            upload_scene_thumbnails=bool(args.portal_upload_scene_thumbnails),
            retry=RetryConfig(
                max_attempts=int(args.portal_upload_retry_max_attempts),
                initial_delay_s=float(args.portal_upload_retry_initial_delay_s),
                max_delay_s=float(args.portal_upload_retry_max_delay_s),
                backoff_factor=float(args.portal_upload_retry_backoff_factor),
            ),
        )

    report_enabled = (
        (cfg.report.enabled and spool is not None)
        if args.report_csv is None
        else bool(args.report_csv)
    )
    report_name = cfg.report.filename if args.report_name is None else str(args.report_name)
    report_include_extra_cols = (
        cfg.report.include_extra_cols
        if args.report_include_extra_cols is None
        else bool(args.report_include_extra_cols)
    )
    report_name = str(report_name or "").strip()
    if report_enabled:
        if spool is None:
            raise SystemExit(
                "Report CSV is enabled but spool output is disabled. "
                "Set --spool-dir (or app.spool.root_dir) to enable run report output."
            )
        if report_name == "":
            raise SystemExit("Report CSV is enabled but --report-name is empty.")
        report_path = spool.run_dir / report_name
        report_writer = ReportWriter(
            ReportWriterConfig(
                path=report_path,
                fps=float(fps),
                include_extra_cols=bool(report_include_extra_cols),
                notes_default="",
            ),
            class_names=class_names_map,
        )
        spool.update_run_metadata({"report_csv_relpath": report_name})

    video_start_epoch_utc: Optional[float] = None
    video_start_local_tz: Optional[tzinfo] = None
    if (not is_live) and spool is not None:
        video_start_val = cfg.io.video_start
        if not video_start_val:
            if args.prompt_video_start and sys.stdin.isatty():
                cfg.io.video_start = _prompt_video_start_rfc3339()
                video_start_val = cfg.io.video_start
            else:
                raise SystemExit(
                    "Offline video + spool enabled requires --video-start (RFC3339 with timezone). "
                    "Example: --video-start 2026-02-18T08:00:00+07:00 "
                    "(or pass --prompt-video-start for interactive entry)."
                )
        try:
            video_start_epoch_utc = _parse_rfc3339_to_epoch_utc(str(video_start_val))
            video_start_raw = str(video_start_val).strip()
            if video_start_raw.endswith("Z"):
                video_start_raw = video_start_raw[:-1] + "+00:00"
            video_start_dt = datetime.fromisoformat(video_start_raw)
            video_start_local_tz = video_start_dt.tzinfo
            # Persist the parsed value so run.json explains where occurred_at_utc came from.
            spool.update_run_metadata(
                {
                    "video_start": str(video_start_val),
                    "video_start_epoch_utc": float(video_start_epoch_utc),
                }
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid --video-start: {exc}")

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
    print(
        "[main] Class vote: "
        f"mode={class_vote_mode} "
        f"min_frames={int(cfg.tracker.class_vote_min_frames)} "
        f"ambiguity_ratio={float(cfg.tracker.class_vote_ambiguity_ratio):.2f}"
    )
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
    if report_writer is not None:
        print(f"[main] Report CSV: {report_writer.cfg.path}")
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
    if is_live:
        attempts_txt = (
            "unlimited"
            if int(cfg.io.rtsp_reconnect_max_attempts) == 0
            else str(int(cfg.io.rtsp_reconnect_max_attempts))
        )
        print(
            f"[main] RTSP capture: backend={rtsp_capture_backend} transport={rtsp_transport} "
            f"codec={rtsp_codec} latency_ms={int(cfg.io.rtsp_latency_ms)} queue_policy={queue_policy}"
        )
        if rtsp_capture_backend == "gstreamer":
            backend_upper = str(backend_name).upper() if backend_name else ""
            if "GSTREAMER" not in backend_upper:
                print(
                    "[main] Warning: requested GStreamer RTSP capture, "
                    "but OpenCV did not open with the GSTREAMER backend. Using fallback backend."
                )
        print(
            f"[main] RTSP reconnect: {'enabled' if cfg.io.rtsp_reconnect_enabled else 'disabled'} "
            f"(max_attempts={attempts_txt}, initial_delay={cfg.io.rtsp_reconnect_initial_delay_s:.2f}s, "
            f"max_delay={cfg.io.rtsp_reconnect_max_delay_s:.2f}s, backoff={cfg.io.rtsp_reconnect_backoff_factor:.2f}, "
            f"stall_timeout={cfg.io.rtsp_stall_timeout_s:.2f}s)"
        )
    if spool is not None or standalone_status_target is not None:
        if headless_status_every_s > 0:
            print(
                f"[main] Headless status: enabled every {headless_status_every_s:.1f}s "
                f"(target={standalone_status_target if standalone_status_target is not None else 'spool-only'})"
            )
        else:
            print("[main] Headless status: snapshots on startup/shutdown only.")

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
    reconnect_enabled = bool(is_live and cfg.io.rtsp_reconnect_enabled)
    live_poll_timeout_s = max(min(float(cfg.io.rtsp_stall_timeout_s), 1.0), 0.1)
    last_live_frame_time_s = time.time()
    reconnect_cycle_count = 0
    reconnect_attempt_total = 0
    reconnect_reason_counts: Dict[str, int] = {}
    stall_timeout_count = 0
    live_reader_read_frames_total = 0
    live_reader_dropped_total = 0
    live_reader_read_failures_total = 0
    if is_live:
        reader = StreamReader(
            cap,
            queue_size=args.queue_size,
            overflow_policy=queue_policy,
        )
        reader.start()

    def _utcnow_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _portal_upload_pass(reason: str) -> None:
        if portal_uploader_cfg is None:
            return
        now_iso = _utcnow_iso()
        try:
            summary = process_pending_runs(
                portal_uploader_cfg,
                force=False,
                dry_run=False,
                max_runs=portal_uploader_max_runs,
            )
            with portal_upload_lock:
                portal_upload_runtime["watch_passes_total"] = int(portal_upload_runtime["watch_passes_total"]) + 1
                portal_upload_runtime["discovered_runs_total"] = int(portal_upload_runtime["discovered_runs_total"]) + int(summary.discovered_runs)
                portal_upload_runtime["completed_runs_total"] = int(portal_upload_runtime["completed_runs_total"]) + int(summary.completed_runs)
                portal_upload_runtime["skipped_runs_total"] = int(portal_upload_runtime["skipped_runs_total"]) + int(summary.skipped_runs)
                portal_upload_runtime["failed_runs_total"] = int(portal_upload_runtime["failed_runs_total"]) + int(summary.failed_runs)
                portal_upload_runtime["last_pass_reason"] = str(reason)
                portal_upload_runtime["last_pass_at_utc"] = now_iso
                if int(summary.failed_runs) == 0:
                    portal_upload_runtime["last_success_at_utc"] = now_iso
                    portal_upload_runtime["last_error"] = None
                    portal_upload_runtime["last_error_at_utc"] = None
            if summary.completed_runs > 0 or summary.failed_runs > 0:
                print(
                    f"[main][portal] {reason}: discovered={summary.discovered_runs} "
                    f"completed={summary.completed_runs} skipped={summary.skipped_runs} failed={summary.failed_runs}"
                )
        except Exception as exc:
            with portal_upload_lock:
                portal_upload_runtime["watch_passes_total"] = int(portal_upload_runtime["watch_passes_total"]) + 1
                portal_upload_runtime["last_pass_reason"] = str(reason)
                portal_upload_runtime["last_pass_at_utc"] = now_iso
                portal_upload_runtime["last_error"] = str(exc)
                portal_upload_runtime["last_error_at_utc"] = now_iso
            print(f"[main][portal] {reason} upload pass failed: {exc}")

    def _build_health_summary(
        *,
        lifecycle_status: str,
        ended_at_utc: Optional[str] = None,
    ) -> Dict[str, object]:
        elapsed_s = max(time.perf_counter() - run_started_perf_s, 1e-9)
        current_reader_frames = int(getattr(reader, "read_frames", 0)) if reader is not None else 0
        current_reader_dropped = int(getattr(reader, "dropped", 0)) if reader is not None else 0
        current_reader_failures = int(getattr(reader, "read_failures", 0)) if reader is not None else 0
        with portal_upload_lock:
            portal_runtime_snapshot = dict(portal_upload_runtime)

        summary: Dict[str, object] = {
            "lifecycle_status": str(lifecycle_status),
            "status_updated_at_utc": _utcnow_iso(),
            "started_at_utc": spool.started_at_utc if spool is not None else None,
            "ended_at_utc": str(ended_at_utc) if ended_at_utc else None,
            "frames_total": int(t_frames),
            "frames_processed": int(t_processed),
            "events_emitted_total": int(spool_events_written_total),
            "count_a_to_b": int(line_counter.count_a_to_b),
            "count_b_to_a": int(line_counter.count_b_to_a),
            "effective_fps": float(t_frames) / elapsed_s if t_frames > 0 else 0.0,
            "processed_fps": float(t_processed) / elapsed_s if t_processed > 0 else 0.0,
            "reconnect_cycles": int(reconnect_cycle_count),
            "reconnect_attempts_total": int(reconnect_attempt_total),
            "reconnect_reason_counts": dict(reconnect_reason_counts),
            "stall_timeout_count": int(stall_timeout_count),
            "reader_read_frames": int(live_reader_read_frames_total) + current_reader_frames,
            "reader_dropped_frames": int(live_reader_dropped_total) + current_reader_dropped,
            "reader_read_failures": int(live_reader_read_failures_total) + current_reader_failures,
            "queue_policy": queue_policy,
            "queue_size": int(args.queue_size),
            "target_fps": float(args.target_fps) if args.target_fps else None,
            "rtsp_capture_backend": rtsp_capture_backend if is_live else None,
            "rtsp_transport": rtsp_transport if is_live else None,
            "rtsp_codec": rtsp_codec if is_live else None,
            "portal_upload_runtime": portal_runtime_snapshot if portal_uploader_cfg is not None else None,
        }
        return summary

    def _write_standalone_status(status_payload: Mapping[str, object]) -> None:
        if standalone_status_target is None:
            return
        target = Path(standalone_status_target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_suffix(target.suffix + ".tmp")
        temp_path.write_text(json.dumps(dict(status_payload), indent=2, ensure_ascii=True), encoding="utf-8")
        temp_path.replace(target)

    def _publish_headless_status(*, lifecycle_status: str, ended_at_utc: Optional[str] = None) -> None:
        health_summary = _build_health_summary(
            lifecycle_status=str(lifecycle_status),
            ended_at_utc=ended_at_utc,
        )
        run_status = {
            "lifecycle_status": str(lifecycle_status),
            "ended_at_utc": ended_at_utc,
            "health_summary": health_summary,
        }
        if spool is not None:
            spool.update_run_metadata(run_status)
            spool.write_headless_status(run_status)
            if standalone_status_target is not None and Path(standalone_status_target) == (spool.run_dir / "status.json"):
                return
        standalone_payload: Dict[str, object] = dict(run_status)
        if spool is not None:
            standalone_payload.update(
                {
                    "run_uid": spool.run_uid,
                    "site_id": spool.cfg.site_id,
                    "camera_id": spool.cfg.camera_id,
                    "started_at_utc": spool.started_at_utc,
                }
            )
        _write_standalone_status(standalone_payload)

    if portal_uploader_cfg is not None:
        if is_live:
            portal_uploader_stop_event = threading.Event()

            def _portal_upload_loop() -> None:
                assert portal_uploader_stop_event is not None
                while not portal_uploader_stop_event.is_set():
                    _portal_upload_pass("watch-pass")
                    portal_uploader_stop_event.wait(max(float(portal_uploader_interval_s), 0.5))

            portal_uploader_thread = threading.Thread(
                target=_portal_upload_loop,
                name="portal-uploader",
                daemon=True,
            )
            portal_uploader_thread.start()
            print(
                "[main][portal] integrated uploader enabled "
                f"(interval={float(portal_uploader_interval_s):.1f}s, max_runs_per_pass={portal_uploader_max_runs})"
            )
        else:
            print("[main][portal] integrated uploader enabled (single final upload pass at end).")

    if spool is not None or standalone_status_target is not None:
        _publish_headless_status(lifecycle_status="running")

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

        def _sleep_interruptible(total_s: float) -> bool:
            end_s = time.time() + max(float(total_s), 0.0)
            while True:
                now_s = time.time()
                if max_end_time_s is not None and now_s >= max_end_time_s:
                    return False
                remaining = end_s - now_s
                if remaining <= 0:
                    return True
                time.sleep(min(remaining, 0.25))

        def _reset_transient_runtime_state() -> None:
            """
            Reset only reconnect-sensitive runtime state.

            Keeps cumulative line totals in `line_counter` unchanged.
            """
            nonlocal latest_tracks
            tracker.clear_runtime_state()
            line_counter.clear_runtime_state()
            latest_tracks = []

        def _accumulate_reader_stats(reader_obj: Optional[StreamReader]) -> None:
            nonlocal live_reader_read_frames_total
            nonlocal live_reader_dropped_total
            nonlocal live_reader_read_failures_total
            if reader_obj is None:
                return
            live_reader_read_frames_total += int(getattr(reader_obj, "read_frames", 0))
            live_reader_dropped_total += int(getattr(reader_obj, "dropped", 0))
            live_reader_read_failures_total += int(getattr(reader_obj, "read_failures", 0))

        def _reconnect_live(trigger_reason: str) -> bool:
            nonlocal reader
            nonlocal cap
            nonlocal pending_first_frame
            nonlocal warned_size_mismatch
            nonlocal last_live_frame_time_s
            nonlocal next_due_time_s
            nonlocal fps_prev_source
            nonlocal reconnect_cycle_count
            nonlocal reconnect_attempt_total
            nonlocal stall_timeout_count

            if (not is_live) or (not reconnect_enabled) or (rtsp_url is None):
                return False
            reconnect_reason_counts[trigger_reason] = reconnect_reason_counts.get(trigger_reason, 0) + 1
            if trigger_reason == "stall_timeout":
                stall_timeout_count += 1

            reconnect_cycle_count += 1
            cycle_no = reconnect_cycle_count
            max_attempts = int(cfg.io.rtsp_reconnect_max_attempts)
            max_attempts_txt = "unlimited" if max_attempts == 0 else str(max_attempts)
            print(
                f"[main] Live stream interrupted (reason={trigger_reason}). "
                f"Starting reconnect cycle #{cycle_no} (max_attempts={max_attempts_txt})."
            )

            if reader is not None:
                _accumulate_reader_stats(reader)
                reader.stop(reason=f"reconnect:{trigger_reason}")
                reader = None

            attempt_no = 0
            while True:
                if max_end_time_s is not None and time.time() >= max_end_time_s:
                    print("[main] Reconnect aborted: --max-seconds limit reached.")
                    return False

                if max_attempts > 0 and attempt_no >= max_attempts:
                    print(
                        f"[main] Reconnect cycle #{cycle_no} failed after {attempt_no} attempts."
                    )
                    return False
                
                attempt_no += 1
                reconnect_attempt_total += 1
                delay_s = _reconnect_delay_s(
                    attempt_no,
                    initial_delay_s=float(cfg.io.rtsp_reconnect_initial_delay_s),
                    max_delay_s=float(cfg.io.rtsp_reconnect_max_delay_s),
                    backoff_factor=float(cfg.io.rtsp_reconnect_backoff_factor),
                )

                total_txt = max_attempts_txt if max_attempts > 0 else "inf"
                print(
                    f"[main] Reconnect attempt {attempt_no}/{total_txt}"
                    f"(cycle #{cycle_no}, global_attempt={reconnect_attempt_total})."
                )

                try:
                    new_cap, new_frame0, _new_fps, _new_w, _new_h, _new_backend = _open_source_with_first_frame(
                        rtsp_url,
                        open_timeout_ms=args.open_timeout_ms,
                        read_timeout_ms=args.read_timeout_ms,
                        source_label=source_label,
                        is_live=True,
                        rtsp_capture_backend=rtsp_capture_backend,
                        rtsp_transport=rtsp_transport,
                        rtsp_latency_ms=int(cfg.io.rtsp_latency_ms),
                        rtsp_codec=rtsp_codec,
                        rtsp_gst_pipeline=rtsp_gst_pipeline,
                        queue_policy=queue_policy,
                    )
                except RuntimeError as exc:
                    print(f"[main] Reconnect attempt {attempt_no} failed: {exc}")
                    if max_attempts > 0 and attempt_no >= max_attempts:
                        print(
                            f"[main] Reconnect cycle {cycle_no} exhausted attempts; stopping."
                        )
                        return False
                    print(f"[main] Backoff sleep {delay_s:.2f}s before next reconnect attempt:")
                    if not _sleep_interruptible(delay_s):
                        print(f"[main] Reconnect aborted while waiting for next attempt.")
                        return False
                    continue

                new_reader = StreamReader(
                    new_cap,
                    queue_size=args.queue_size,
                    overflow_policy=queue_policy,
                )
                new_reader.start()

                cap = new_cap
                reader = new_reader
                pending_first_frame = new_frame0
                warned_size_mismatch = False
                last_live_frame_time_s = time.time()
                next_due_time_s = None
                fps_prev_source = 0
                _reset_transient_runtime_state()
                print("[main] Reconnect state reset: transient tracker/counter state cleared; totals preserved.")

                print(
                    f"[main] Reconnected successfully on attempt {attempt_no}"
                    f"(cycle {cycle_no})"
                )
                return True

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
                last_live_frame_time_s = frame_ts
            elif reader is not None:
                poll = reader.get_with_status(timeout_s=live_poll_timeout_s)
                if poll.status == "frame" and poll.item is not None:
                    frame = poll.item.frame
                    frame_ts = float(poll.item.timestamp)
                    last_live_frame_time_s = time.time()
                elif poll.status == "timeout":
                    if (time.time() - last_live_frame_time_s) < float(cfg.io.rtsp_stall_timeout_s):
                        continue
                    if not _reconnect_live("stall_timeout"):
                        break
                    continue
                else:
                    stop_reason = poll.reason if poll.reason else "reader_stopped"
                    if not _reconnect_live(stop_reason):
                        break
                    continue
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
                tracks = tracker.update(detections, frame_index)
                t_track += time.perf_counter() - t0
                latest_tracks = tracks

                if args.line_mode == "gate":
                    tracks_updated = [t for t in tracks if t.last_seen_frame == frame_index]
                    t0 = time.perf_counter()
                    events = line_counter.update(tracks_updated, frame_index)
                    t_count += time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    events = line_counter.update(tracks, frame_index=frame_index)
                    t_count += time.perf_counter() - t0

                if events:
                    occurred_at_ts = frame_ts
                    occurred_at_source = "wallclock"
                    if (not is_live) and video_start_epoch_utc is not None:
                        occurred_at_ts = float(video_start_epoch_utc) + (float(frame_index) / float(fps))
                        occurred_at_source = "video_start"
                    event_records = [] if (spool is not None and report_writer is not None) else None
                    if spool is not None:
                        written_events = spool.record_events(
                            events,
                            frame_bgr=frame,
                            occurred_at_ts=float(occurred_at_ts),
                            occurred_at_utc_source=str(occurred_at_source),
                            occurred_at_local_tz=video_start_local_tz,
                            capture_records=event_records,
                        )
                        spool_events_written_total += int(written_events)
                    if report_writer is not None:
                        report_writer.record_events(events, event_records=event_records)

                # Gambar setiap gambar yang telah diupdate
                # avoid "stuck" boxes when a track is kept alive across occlusions.
                if not args.no_draw:
                    t0 = time.perf_counter()
                    draw_tracks(
                        frame,
                        tracks,
                        frame_index=frame_index,
                        stale_max_age=args.stale_frames,
                        class_names=class_names_map,
                        color=track_default_color,
                        class_colors=track_class_colors,
                        palette=track_palette,
                        color_by=track_color_by,
                    )
                    draw_line_and_counts(frame, line_counter, class_names=class_names_map)
                    t_draw += time.perf_counter() - t0
                t_processed += 1
            elif not args.no_draw:
                # For skipped frames, keep drawing latest tracks so overlays stay stable.
                t0 = time.perf_counter()
                if latest_tracks:
                    draw_tracks(
                        frame,
                        latest_tracks,
                        frame_index=frame_index,
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

            if (spool is not None or standalone_status_target is not None) and headless_status_every_s > 0:
                now_status = time.time()
                if now_status >= next_headless_status_s:
                    _publish_headless_status(lifecycle_status="running")
                    next_headless_status_s = now_status + headless_status_every_s

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

    if portal_uploader_stop_event is not None:
        portal_uploader_stop_event.set()
        if portal_uploader_thread is not None:
            portal_uploader_thread.join(timeout=max(float(portal_uploader_interval_s) + 1.0, 2.0))

    if reader is not None:
        live_reader_read_frames_total += int(getattr(reader, "read_frames", 0))
        live_reader_dropped_total += int(getattr(reader, "dropped", 0))
        live_reader_read_failures_total += int(getattr(reader, "read_failures", 0))
        reader.stop()
    else:
        cap.release()
    if writer is not None:
        writer.release()
    if spool is not None or standalone_status_target is not None:
        _publish_headless_status(lifecycle_status="stopped", ended_at_utc=_utcnow_iso())
    if spool is not None:
        spool.close()
    if report_writer is not None:
        report_writer.close()
    if portal_uploader_cfg is not None:
        _portal_upload_pass("final-pass")
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
