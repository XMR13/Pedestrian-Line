"""
Small helper script to inspect the raw ONNX output from the YOLOv9 model
and see how many boxes survive YOLO post-processing.

Usage (from project root):

    uv run python scripts/debug_onnx_inspect.py \
        --input media/video3.mp4 \
        --model Models/yolov9-s.onnx

This DOES NOT change any project code. It only prints shapes/stats so we can
adapt the detector if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from pedestrian_line_counter.config import load_class_names
from yolo_kitv2 import LetterboxConfig, YoloPostConfig, draw_detections, load_pipeline


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inspect YOLOv9 ONNX outputs and post-processing.")
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default="media/video3.mp4",
        help="Path to a test video (default: media/video3.mp4).",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        default=None,
        help="Optional image path (if set, --input is ignored)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the input frame and detections in a window.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="Models/yolov9-s.onnx",
        help="Path to YOLOv9 ONNX model (default: Models/yolov9-s.onnx).",
    )
    parser.add_argument(
        "--class-ids",
        type=str,
        default=None,
        help="Comma-separated class IDs to keep (optional).",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Optional class names mapping file (YAML/JSON) for labels.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Which frame index to inspect (default: 0).",
    )
    return parser.parse_args()


def load_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(index, 0))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit(f"Could not read frame {index} from {path}")
    return frame

def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Tidak bisa meload image melalui {path}")
    return img

def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    image_path = Path(args.image) if args.image else None

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if image_path is not None:
        if not image_path.exists():
            raise SystemExit(f"input image not found in: {image_path}")
        frame = load_image(image_path)
    #use video or anything that's not resembling an image
    else:
        if not input_path.exists():
            raise SystemExit(f"input video not found: {input_path}")
        frame = load_frame(input_path, args.frame_index)
    h, w = frame.shape[:2]
    print(f"[debug] Frame size: {w}x{h}")

    # Keep config explicit here so you can compare between models/export settings.
    input_size = (640, 640)
    class_ids = _parse_class_ids(args.class_ids)
    post_cfg = YoloPostConfig(
        conf_threshold=0.35,
        iou_threshold=0.45,
        class_ids=class_ids if class_ids else None,
    )

    class_names: Optional[Dict[int, str]] = None
    if args.class_names:
        class_names = load_class_names(Path(args.class_names))

    pipe = load_pipeline(
        model_path,
        backend="onnxruntime",
        letterbox_cfg=LetterboxConfig(new_shape=input_size),
        post_cfg=post_cfg,
    )

    if getattr(pipe, "backend", None) is not None and hasattr(pipe.backend, "providers_in_use"):
        try:
            print(f"[debug] providers_in_use: {list(pipe.backend.providers_in_use)}")
        except Exception:
            pass

    prep = pipe.preprocess(frame)
    print(f"[debug] Input tensor shape: {prep.blob.shape}")

    # Raw model output stats (primary output).
    raw = pipe.backend.infer(prep.blob) if getattr(pipe, "backend", None) is not None else None
    if raw is not None:
        arr = np.asarray(raw)
        try:
            mn = float(arr.min())
            mx = float(arr.max())
            mean = float(arr.mean())
        except Exception:
            mn = mx = mean = float("nan")
        print(f"[debug] raw shape={arr.shape} dtype={arr.dtype} min={mn:.4f} max={mx:.4f} mean={mean:.4f}")

    dets = pipe(frame)
    print(f"[debug] postprocess -> detections: {len(dets)}")
    for i, d in enumerate(dets[:10]):
        print(
            f"  det[{i}] xyxy=({d.x1:.1f},{d.y1:.1f},{d.x2:.1f},{d.y2:.1f}) "
            f"score={d.score:.3f} class_id={d.class_id}"
        )

    if args.show:
        vis = draw_detections(frame, dets, class_names=class_names)
        cv2.imshow("debug_input", frame)
        cv2.imshow("debug_detections", vis)
        print("[debug] Press any key to close the preview window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
