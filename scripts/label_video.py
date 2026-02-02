from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from pedestrian_line_counter.config import TrackerConfig, load_class_names
from pedestrian_line_counter.structures import Detection as TrackDetection
from pedestrian_line_counter.tracker import Tracker
from pedestrian_line_counter.videoio_utils import open_video_capture
from yolo_kitv2 import LetterboxConfig, YoloPostConfig, load_pipeline

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
    return "".join(cleaned) or "class"


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


def _build_categories(
    class_names: Optional[Dict[int, str]],
    class_ids: List[int],
    seen_ids: Iterable[int],
) -> List[Dict[str, object]]:
    if class_ids:
        ids = sorted(set(class_ids))
    elif class_names:
        ids = sorted(class_names.keys())
    else:
        ids = sorted(set(seen_ids))

    categories: List[Dict[str, object]] = []
    for cid in ids:
        name = str(cid)
        if class_names and cid in class_names:
            name = class_names[cid]
        categories.append({"id": int(cid), "name": name})
    return categories


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run YOLO on a video for candidate extraction or COCO auto-label export."
    )
    p.add_argument("--input", type=str, required=True, help="Input video path or directory.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    p.add_argument(
        "--mode",
        type=str,
        default="coco",
        choices=["coco", "candidates", "frames"],
        help="Output mode: coco (labels), candidates (track-based), frames (extract only).",
    )
    p.add_argument("--recursive", action="store_true", help="If input is a directory, scan recursively.")

    p.add_argument("--model", type=str, required=False, help="Model path (.onnx/.engine/.pt).")
    p.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["onnx", "onnxruntime", "tensorrt", "torch", "torchscript"],
        help="Override backend; default infers from model extension.",
    )
    p.add_argument("--class-ids", type=str, default=None, help="Comma-separated class IDs to keep.")
    p.add_argument("--class-names", type=str, default=None, help="Class names YAML/JSON for COCO categories.")
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
    p.add_argument("--resize-to", type=str, default=None, help="Optional resize for saved images, e.g. 1280x720.")
    p.add_argument("--ext", type=str, default="png", help="Output image extension (lossless). Default: png.")
    p.add_argument("--skip-empty", action="store_true", help="Skip frames with zero detections (coco mode).")
    p.add_argument("--images-dir", type=str, default=None, help="Optional images directory override.")
    p.add_argument("--annotations", type=str, default=None, help="COCO JSON path (coco mode).")
    p.add_argument("--include-score", action="store_true", help="Include `score` in COCO annotations.")

    # Candidate extraction options
    p.add_argument("--max-per-track", type=int, default=3, help="Max frames to save per track (candidates).")
    p.add_argument(
        "--min-seconds-between",
        type=float,
        default=1.0,
        help="Minimum seconds between saves for the same track (candidates).",
    )
    p.add_argument(
        "--min-frames-between",
        type=int,
        default=None,
        help="Minimum frames between saves for the same track (overrides --min-seconds-between).",
    )
    p.add_argument("--baseline-fps", type=float, default=0.0, help="Baseline sampling FPS (candidates).")
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Wait N frames after a track appears before saving (candidates).",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar.")

    return p.parse_args()


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


def _to_tracker_dets(dets: List[object]) -> List[TrackDetection]:
    out: List[TrackDetection] = []
    for d in dets:
        class_id = None if d.class_id is None else int(d.class_id)
        out.append(
            TrackDetection(
                x1=float(d.x1),
                y1=float(d.y1),
                x2=float(d.x2),
                y2=float(d.y2),
                score=float(d.score),
                class_id=class_id,
            )
        )
    return out


def _candidate_should_save(
    track: object,
    *,
    class_ids: List[int],
    allow_all_classes: bool,
    per_track_state: Dict[int, Dict[str, int]],
    frame_index: int,
    min_frames_between: int,
    max_per_track: int,
    warmup_frames: int,
) -> bool:
    if track.class_id is None:
        return False
    class_id = int(track.class_id)
    if class_ids and class_id not in class_ids and not allow_all_classes:
        return False

    state = per_track_state.setdefault(
        track.track_id,
        {"count": 0, "last_frame": -10**9, "first_frame": frame_index},
    )
    if (frame_index - state["first_frame"]) < warmup_frames:
        return False
    if state["count"] >= max_per_track:
        return False
    if (frame_index - state["last_frame"]) < min_frames_between:
        return False

    state["count"] += 1
    state["last_frame"] = frame_index
    return True


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input)
    input_files = _iter_input_files(input_path, args.recursive)
    if not input_files:
        raise SystemExit(f"No input files found under: {input_path}")

    try:
        output_ext = _normalize_ext(args.ext)
    except ValueError as exc:
        raise SystemExit(str(exc))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir) if args.images_dir else (output_dir / "images")
    images_dir.mkdir(parents=True, exist_ok=True)

    ann_path = Path(args.annotations) if args.annotations else (output_dir / "annotations.json")

    model_path: Optional[Path] = None
    if args.mode != "frames":
        if not args.model:
            raise SystemExit("--model is required for coco/candidates modes.")
        model_path = Path(args.model)
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path}")

    class_ids = _parse_class_ids(args.class_ids)
    class_names: Optional[Dict[int, str]] = None
    if args.class_names:
        class_names = load_class_names(Path(args.class_names))

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

    pipe = None
    if args.mode != "frames":
        post_cfg = YoloPostConfig(
            conf_threshold=float(args.conf),
            iou_threshold=float(args.iou),
            max_detections=int(args.max_detections),
            class_ids=class_ids if (class_ids and not args.allow_all_classes) else None,
        )

        providers = _pick_onnx_providers()
        pipe = load_pipeline(
            model_path,
            backend=_normalize_backend(args.backend),
            letterbox_cfg=LetterboxConfig(new_shape=input_size),
            post_cfg=post_cfg,
            onnx_providers=providers,
        )

    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []
    seen_class_ids: List[int] = []
    image_id = 1
    ann_id = 1

    total_frames = 0
    total_saved = 0

    for clip in input_files:
        cap = open_video_capture(clip)
        if not cap.isOpened():
            print(f"[label_video] failed to open: {clip}")
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

        if args.min_frames_between is not None:
            min_frames_between = max(int(args.min_frames_between), 0)
        else:
            min_frames_between = max(int(round(float(args.min_seconds_between) * fps)), 0)

        baseline_every_n = 0
        if args.baseline_fps and args.baseline_fps > 0:
            baseline_every_n = max(int(round(fps / float(args.baseline_fps))), 1)

        tracker = Tracker(TrackerConfig())
        per_track_state: Dict[int, Dict[str, int]] = {}

        clip_name = _safe_name(clip.stem)
        frame_index = start_frames
        saved_total = 0
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
                desc=f"Processing {clip.name}",
                file=sys.stdout,
                leave=False,
            )
        elif not args.no_progress and tqdm is None:
            print("[label_video] tqdm is not installed; run `uv run pip install tqdm` to enable a progress bar.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frames)

        try:
            while True:
                if max_frames is not None and (frame_index - start_frames) >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if resize_to is not None:
                    frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)

                h, w = frame.shape[:2]
                dets: List[object] = []

                if args.mode == "candidates":
                    if pipe is None:
                        raise SystemExit("Pipeline not initialized for candidates mode.")
                    dets = pipe(frame)
                    dets = _filter_by_min_area(dets, frame_w=w, frame_h=h, min_ratio=args.min_box_area_ratio)

                    tracks = tracker.update(_to_tracker_dets(dets), frame_index)
                    updated_tracks = [t for t in tracks if t.last_seen_frame == frame_index]

                    if baseline_every_n and frame_index % baseline_every_n == 0:
                        out_path = images_dir / f"{clip_name}_f{frame_index:06d}_baseline.{output_ext}"
                        cv2.imwrite(str(out_path), frame)
                        saved_total += 1

                    for track in updated_tracks:
                        if track.class_id is None:
                            continue
                        if class_ids and int(track.class_id) not in class_ids and not args.allow_all_classes:
                            continue

                        warmup_frames = max(int(args.warmup_frames), 0)
                        max_per_track = max(int(args.max_per_track), 0)
                        if not _candidate_should_save(
                            track,
                            class_ids=class_ids,
                            allow_all_classes=args.allow_all_classes,
                            per_track_state=per_track_state,
                            frame_index=frame_index,
                            min_frames_between=min_frames_between,
                            max_per_track=max_per_track,
                            warmup_frames=warmup_frames,
                        ):
                            continue

                        class_id = int(track.class_id)
                        class_label = str(class_id)
                        if class_names and class_id in class_names:
                            class_label = _safe_name(class_names[class_id])

                        filename = (
                            f"{clip_name}_f{frame_index:06d}_tid{track.track_id:04d}"
                            f"_cid{class_id:03d}_{class_label}.{output_ext}"
                        )
                        out_path = images_dir / filename
                        cv2.imwrite(str(out_path), frame)
                        saved_total += 1

                elif args.mode == "coco":
                    if pipe is None:
                        raise SystemExit("Pipeline not initialized for coco mode.")
                    dets = pipe(frame)
                    dets = _filter_by_min_area(dets, frame_w=w, frame_h=h, min_ratio=args.min_box_area_ratio)

                    tracks = tracker.update(_to_tracker_dets(dets), frame_index)
                    updated_tracks = [t for t in tracks if t.last_seen_frame == frame_index]
                    warmup_frames = max(int(args.warmup_frames), 0)
                    max_per_track = max(int(args.max_per_track), 0)
                    save_frame = False
                    for track in updated_tracks:
                        if _candidate_should_save(
                            track,
                            class_ids=class_ids,
                            allow_all_classes=args.allow_all_classes,
                            per_track_state=per_track_state,
                            frame_index=frame_index,
                            min_frames_between=min_frames_between,
                            max_per_track=max_per_track,
                            warmup_frames=warmup_frames,
                        ):
                            save_frame = True
                    if baseline_every_n and frame_index % baseline_every_n == 0:
                        save_frame = True
                    if not save_frame:
                        frame_index += 1
                        if progress_bar is not None:
                            progress_bar.update(1)
                        continue

                    if args.skip_empty and not dets:
                        frame_index += 1
                        if progress_bar is not None:
                            progress_bar.update(1)
                        continue

                    filename = f"{clip_name}_f{frame_index:06d}.{output_ext}"
                    image_path = images_dir / filename
                    if not cv2.imwrite(str(image_path), frame):
                        raise SystemExit(f"Failed to write: {image_path}")

                    try:
                        rel_path = str(image_path.relative_to(output_dir))
                    except Exception:
                        rel_path = str(image_path)

                    images.append(
                        {
                            "id": image_id,
                            "file_name": rel_path.replace("\\", "/"),
                            "width": int(w),
                            "height": int(h),
                            "frame_index": int(frame_index),
                        }
                    )

                    for d in dets:
                        if d.class_id is None:
                            continue
                        x1, y1, x2, y2 = d.as_xyxy()
                        x1 = float(np.clip(x1, 0, w - 1))
                        y1 = float(np.clip(y1, 0, h - 1))
                        x2 = float(np.clip(x2, 0, w - 1))
                        y2 = float(np.clip(y2, 0, h - 1))
                        bw = max(0.0, x2 - x1)
                        bh = max(0.0, y2 - y1)
                        if bw <= 0.0 or bh <= 0.0:
                            continue

                        ann = {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": int(d.class_id),
                            "bbox": [x1, y1, bw, bh],
                            "area": float(bw * bh),
                            "iscrowd": 0,
                        }
                        if args.include_score:
                            ann["score"] = float(d.score)

                        annotations.append(ann)
                        ann_id += 1
                        seen_class_ids.append(int(d.class_id))

                    image_id += 1
                    saved_total += 1

                else:  # frames
                    if (frame_index - start_frames) % every_n != 0:
                        frame_index += 1
                        if progress_bar is not None:
                            progress_bar.update(1)
                        continue

                    filename = f"{clip_name}_f{frame_index:06d}.{output_ext}"
                    image_path = images_dir / filename
                    if not cv2.imwrite(str(image_path), frame):
                        raise SystemExit(f"Failed to write: {image_path}")
                    saved_total += 1

                frame_index += 1
                if progress_bar is not None:
                    progress_bar.update(1)
        except KeyboardInterrupt:
            print("[label_video] interrupted by user.")
            break
        finally:
            cap.release()
            if progress_bar is not None:
                progress_bar.close()

        total_frames += (frame_index - start_frames)
        total_saved += saved_total

        print(
            f"[label_video] done clip={clip.name} frames={frame_index - start_frames} "
            f"saved={saved_total} mode={args.mode}"
        )

    if args.mode == "coco":
        categories = _build_categories(class_names, class_ids, seen_class_ids)
        coco = {
            "info": {
                "description": "Auto-labeled COCO dataset (video -> frames) via yolo_kitv2",
                "video": str(input_path),
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        ann_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        print(
            f"[label_video] coco saved images={len(images)} annotations={len(annotations)} -> {ann_path}"
        )

    if len(input_files) > 1:
        print(
            f"[label_video] all done clips={len(input_files)} "
            f"frames={total_frames} saved={total_saved} mode={args.mode}"
        )


if __name__ == "__main__":
    main()
