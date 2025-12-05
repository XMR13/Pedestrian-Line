"""
Small helper script to inspect the raw ONNX output from the YOLOv9 model
and see how many boxes survive our post-processing.

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

import cv2
import numpy as np
import onnxruntime as ort

from pedestrian_line_counter.config import ModelConfig
from pedestrian_line_counter.detector import Detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inspect YOLOv9 ONNX outputs and post-processing.")
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default="media/video3.mp4",
        help="Path to a test video frame (default: media/video3.mp4).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="Models/yolov9-s.onnx",
        help="Path to YOLOv9 ONNX model (default: Models/yolov9-s.onnx).",
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


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not input_path.exists():
        raise SystemExit(f"Input video not found: {input_path}")

    frame = load_frame(input_path, args.frame_index)
    h, w = frame.shape[:2]

    cfg = ModelConfig(
        backend="motion",  # avoid initializing ONNX inside Detector
        model_path=model_path,
    )
    detector = Detector(cfg)

    input_width, input_height = cfg.input_size

    # Letterbox resize – same logic as _detect_onnx
    scale = min(input_width / w, input_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    dw, dh = (input_width - new_w) // 2, (input_height - new_h) // 2
    canvas[dh : dh + new_h, dw : dw + new_w] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
    blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
    blob = np.expand_dims(blob, 0)

    print(f"[debug] Frame size: {w}x{h}")
    print(f"[debug] Input tensor shape: {blob.shape}")

    providers = ort.get_available_providers()
    print(f"[debug] onnxruntime providers: {providers}")

    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: blob})
    print(f"[debug] num outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        arr = np.asarray(out)
        print(
            f"[debug] out[{i}] shape={arr.shape} "
            f"dtype={arr.dtype} min={arr.min():.4f} "
            f"max={arr.max():.4f} mean={arr.mean():.4f}"
        )

    # Try to run the same post-processing as the main detector
    pred0 = outputs[0]
    try:
        boxes_xyxy, scores, class_ids = detector._postprocess_yolo_generic(
            np.asarray(pred0), w, h, dw, dh, scale
        )
    except Exception as exc:  # pragma: no cover - debug helper
        print(f"[debug] _postprocess_yolo_generic raised: {exc}")
        return

    print(f"[debug] postprocess -> boxes: {len(boxes_xyxy)}")
    for i in range(min(10, len(boxes_xyxy))):
        x1, y1, x2, y2 = boxes_xyxy[i]
        print(
            f"  box[{i}] xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
            f"score={scores[i]:.3f} class_id={int(class_ids[i])}"
        )


if __name__ == "__main__":
    main()
