from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

from pedestrian_line_counter.videoio_utils import open_video_capture
from yolo_kitv2 import LetterboxConfig, YoloPostConfig, draw_detections, load_pipeline
from yolo_kitv2.metadata import load_class_names

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover - safe fallback
    tqdm = None


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

ALLOWED_IMAGE_EXTS = {"png", "bmp", "tif", "tiff"}


def _normalize_ext(value: str) -> str:
    ext = (value or "").strip().lower().lstrip(".")
    if not ext:
        raise ValueError("Output image extension cannot be empty.")
    if ext in {"jpg", "jpeg"}:
        raise ValueError("JPEG output is disabled. Use PNG (default) or another lossless format.")
    if ext not in ALLOWED_IMAGE_EXTS:
        raise ValueError(f"Unsupported image extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTS))}.")
    return ext


def _parse_size(spec: str) -> Tuple[int, int]:
    spec = (spec or "").strip().lower()
    if "x" not in spec:
        raise ValueError(f"Invalid size '{spec}'. Expected format like 640x640.")
    w_str, h_str = spec.split("x", 1)
    w = int(w_str)
    h = int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid size '{spec}': width/height must be > 0.")
    return w, h


def _parse_class_ids(text: Optional[str]) -> List[int]:
    if not text:
        return []
    items: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _safe_name(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
    return "".join(cleaned) or "clip"


def _normalize_backend(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip().lower()
    if v == "onnx":
        return "onnxruntime"
    if v == "torch":
        return "torchscript"
    return v


def _pick_onnx_providers() -> List[str]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return ["CPUExecutionProvider"]

    candidate_providers = [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]

    if hasattr(ort, "get_available_providers"):
        try:
            available = set(ort.get_available_providers())
            providers = [p for p in candidate_providers if p in available]
            return providers or ["CPUExecutionProvider"]
        except Exception:
            return candidate_providers

    return candidate_providers


def _iter_input_files(input_path: Path, recursive: bool) -> Iterable[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in sorted(input_path.glob(pattern))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return files


def _filter_by_min_area(
    dets: List[object],
    *,
    frame_w: int,
    frame_h: int,
    min_ratio: float,
) -> List[object]:
    if min_ratio <= 0:
        return dets
    min_area = float(frame_w * frame_h) * float(min_ratio)
    kept: List[object] = []
    for d in dets:
        x1, y1, x2, y2 = d.as_xyxy()
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        kept.append(d)
    return kept


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview YOLO detections with optional annotated outputs.")
    p.add_argument("--input", type=str, required=True, help="Input video path or directory.")
    p.add_argument("--recursive", action="store_true", help="If input is a directory, scan recursively.")
    p.add_argument("--model", type=str, required=True, help="Model path (.onnx/.engine/.pt).")
    p.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["onnx", "onnxruntime", "tensorrt", "torch", "torchscript"],
        help="Override backend; default infers from model extension.",
    )
    p.add_argument("--class-ids", type=str, default=None, help="Comma-separated class IDs to keep.")
    p.add_argument("--class-names", type=str, default=None, help="Class names YAML/JSON for overlays.")
    p.add_argument(
        "--class-names-id-offset",
        type=int,
        default=0,
        help=(
            "Offset applied to IDs loaded from --class-names before lookup. "
            "Example: if your file is 1-based (CVAT-style) but your model outputs 0-based IDs, use -1."
        ),
    )
    p.add_argument(
        "--allow-all-classes",
        action="store_true",
        help="Do not restrict classes even if --class-ids is empty.",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    p.add_argument("--max-detections", type=int, default=50, help="Max detections per frame.")
    p.add_argument("--input-size", type=str, default="640x640", help="Model input size, e.g. 640x640.")
    p.add_argument(
        "--min-box-area-ratio",
        type=float,
        default=0.0002,
        help="Drop boxes smaller than this fraction of frame area (set 0 to disable).",
    )

    group = p.add_mutually_exclusive_group()
    group.add_argument("--fps", type=float, default=None, help="Sample at this FPS (best-effort).")
    group.add_argument("--every-n", type=int, default=1, help="Sample every N frames (default 1).")

    p.add_argument("--start-seconds", type=float, default=0.0, help="Skip the first N seconds.")
    p.add_argument("--max-seconds", type=float, default=None, help="Stop after N seconds (after start).")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after processing N frames.")
    p.add_argument("--resize-to", type=str, default=None, help="Optional resize, e.g. 1280x720.")
    p.add_argument("--ext", type=str, default="png", help="Preview frame extension (lossless). Default: png.")

    p.add_argument("--show", action="store_true", help="Show preview window.")
    p.add_argument("--save-frames", action="store_true", help="Save annotated frames.")
    p.add_argument("--frames-dir", type=str, default=None, help="Directory to store preview frames.")
    p.add_argument("--save-video", action="store_true", help="Save annotated preview video.")

    video_group = p.add_mutually_exclusive_group()
    video_group.add_argument("--video-out", type=str, default=None, help="Preview video output file (single clip).")
    video_group.add_argument("--video-dir", type=str, default=None, help="Preview video output directory.")
    p.add_argument("--video-fps", type=float, default=None, help="Preview video FPS (default: source FPS).")
    p.add_argument("--output-dir", type=str, default="preview_outputs", help="Base output directory.")

    p.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input)
    input_files = list(_iter_input_files(input_path, args.recursive))
    if not input_files:
        raise SystemExit(f"No input files found under: {input_path}")

    if not (args.show or args.save_frames or args.save_video):
        raise SystemExit("Nothing to do: use --show, --save-frames, or --save-video.")

    try:
        output_ext = _normalize_ext(args.ext)
    except ValueError as exc:
        raise SystemExit(str(exc))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir: Optional[Path] = None
    if args.save_frames:
        frames_dir = Path(args.frames_dir) if args.frames_dir else (output_dir / "preview_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)

    if args.save_video:
        if args.video_out and len(input_files) > 1:
            raise SystemExit("--video-out can only be used with a single input clip.")
        if not args.video_out and not args.video_dir:
            args.video_dir = str(output_dir / "preview_videos")
        if args.video_dir:
            Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    class_ids = _parse_class_ids(args.class_ids)
    class_names: Optional[Dict[int, str]] = None
    if args.class_names:
        class_names = load_class_names(args.class_names)
        offset = int(args.class_names_id_offset)
        if offset:
            class_names = {int(k) + offset: v for k, v in class_names.items()}
        if not class_names:
            print(
                "[preview_video] warning: class names loaded as empty. "
                "Check file format (YAML/JSON) or install PyYAML if using YAML."
            )
            class_names = None

    try:
        input_size = _parse_size(args.input_size)
    except ValueError as exc:
        raise SystemExit(str(exc))

    resize_to: Optional[Tuple[int, int]] = None
    if args.resize_to:
        try:
            resize_to = _parse_size(args.resize_to)
        except ValueError as exc:
            raise SystemExit(str(exc))

    post_cfg = YoloPostConfig(
        conf_threshold=float(args.conf),
        iou_threshold=float(args.iou),
        max_detections=int(args.max_detections),
        class_ids=class_ids if (class_ids and not args.allow_all_classes) else None,
    )

    providers = _pick_onnx_providers()
    pipe = load_pipeline(
        Path(args.model),
        backend=_normalize_backend(args.backend),
        letterbox_cfg=LetterboxConfig(new_shape=input_size),
        post_cfg=post_cfg,
        onnx_providers=providers,
    )

    total_frames = 0
    for clip in input_files:
        cap = open_video_capture(clip)
        if not cap.isOpened():
            print(f"[preview_video] failed to open: {clip}")
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 30.0

        if args.fps is not None:
            if args.fps <= 0:
                raise SystemExit("--fps must be > 0")
            every_n = max(int(round(fps / float(args.fps))), 1)
        else:
            if args.every_n <= 0:
                raise SystemExit("--every-n must be > 0")
            every_n = int(args.every_n)

        start_frames = int(max(args.start_seconds, 0.0) * fps)
        max_frames = args.max_frames
        if args.max_seconds is not None:
            seconds_limit = max(float(args.max_seconds), 0.0)
            frames_from_seconds = int(round(seconds_limit * fps))
            max_frames = frames_from_seconds if max_frames is None else min(max_frames, frames_from_seconds)

        clip_name = _safe_name(clip.stem)
        frame_index = start_frames

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
                desc=f"Preview {clip.name}",
                file=sys.stdout,
                leave=False,
            )
        elif not args.no_progress and tqdm is None:
            print("[preview_video] tqdm is not installed; run `uv run pip install tqdm` to enable a progress bar.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frames)

        video_path: Optional[Path] = None
        if args.save_video:
            if args.video_out:
                video_path = Path(args.video_out)
                video_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                video_dir = Path(args.video_dir) if args.video_dir else (output_dir / "preview_videos")
                video_path = video_dir / f"{clip_name}_preview.mp4"

        writer: Optional[cv2.VideoWriter] = None

        try:
            while True:
                if max_frames is not None and (frame_index - start_frames) >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if resize_to is not None:
                    frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)

                if (frame_index - start_frames) % every_n != 0:
                    frame_index += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
                    continue

                h, w = frame.shape[:2]
                dets = pipe(frame)
                dets = _filter_by_min_area(dets, frame_w=w, frame_h=h, min_ratio=args.min_box_area_ratio)

                vis = draw_detections(frame, dets, class_names=class_names)

                if frames_dir is not None:
                    out_name = f"{clip_name}_f{frame_index:06d}_preview.{output_ext}"
                    cv2.imwrite(str(frames_dir / out_name), vis)

                if video_path is not None:
                    if writer is None:
                        out_fps = float(args.video_fps or fps)
                        if out_fps <= 0:
                            out_fps = fps
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(video_path), fourcc, out_fps, (w, h))
                        if not writer.isOpened():
                            raise SystemExit(f"Failed to open preview video: {video_path}")
                    writer.write(vis)

                if args.show:
                    cv2.imshow("preview_video", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        raise KeyboardInterrupt

                frame_index += 1
                if progress_bar is not None:
                    progress_bar.update(1)
        except KeyboardInterrupt:
            print("[preview_video] interrupted by user.")
            break
        finally:
            cap.release()
            if progress_bar is not None:
                progress_bar.close()
            if args.show:
                cv2.destroyAllWindows()
            if writer is not None:
                writer.release()

        total_frames += (frame_index - start_frames)
        print(f"[preview_video] done clip={clip.name} frames={frame_index - start_frames}")

    if len(input_files) > 1:
        print(f"[preview_video] all done clips={len(input_files)} frames={total_frames}")


if __name__ == "__main__":
    main()
