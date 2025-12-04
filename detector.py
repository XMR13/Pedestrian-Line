from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from config import ModelConfig
from structures import Detection


@dataclass
class Detector:
    """
    High-level detector wrapper that can use different backends.

    Supported backends:
    - \"motion\": simple background-subtraction detector (no classes).
    - \"onnx\": ONNXRuntime-based detector (YOLO-style), if onnxruntime
      and a suitable model are available.
    """

    config: ModelConfig
    _backend: str = None
    _motion_subtractor: Optional[cv2.BackgroundSubtractor] = None
    _onnx_session: Optional[object] = None

    def __post_init__(self) -> None:
        backend = (self.config.backend or "motion").lower()

        if backend == "onnx":
            try:
                import onnxruntime as ort  # type: ignore

                available = ort.get_available_providers()
                providers = []
                # Prefer GPU if available, but always keep CPU as fallback
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                if "DmlExecutionProvider" in available:
                    providers.append("DmlExecutionProvider")
                providers.append("CPUExecutionProvider")

                self._onnx_session = ort.InferenceSession(
                    str(self.config.model_path),
                    providers=providers,
                )
                self._backend = "onnx"
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[detector] Failed to initialize ONNX backend: {exc}")
                print("[detector] Falling back to motion-based detector.")
                backend = "motion"

        if backend == "motion":
            self._motion_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True
            )
            self._backend = "motion"

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame and return a list of Detection.
        """

        if self._backend == "onnx" and self._onnx_session is not None:
            return self._detect_onnx(frame_bgr)

        return self._detect_motion(frame_bgr)

    # --------------------------------------------------------------------- #
    # Motion-based detector
    # --------------------------------------------------------------------- #
    def _detect_motion(self, frame_bgr: np.ndarray) -> List[Detection]:
        fgmask = self._motion_subtractor.apply(frame_bgr)

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []
        h, w = fgmask.shape[:2]
        min_area = (w * h) * 0.0005  # heuristic

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            detections.append(
                Detection(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + bw),
                    y2=float(y + bh),
                    score=1.0,
                    class_id=None,
                )
            )
        return detections

    # --------------------------------------------------------------------- #
    # ONNX (YOLO-style) detector
    # --------------------------------------------------------------------- #
    def _detect_onnx(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        A generic YOLO-style ONNX inference path.

        This implementation assumes a single output tensor shaped like
        (batch, num_detections, 5 + num_classes):
        [cx, cy, w, h, objectness, class_scores...].

        You may need to adapt this function if your model uses a different
        layout. The rest of the pipeline (tracker, line counter, drawing)
        will stay the same.
        """

        assert self._onnx_session is not None
        input_width, input_height = self.config.input_size

        img = frame_bgr
        h, w = img.shape[:2]
        frame_area = float(w * h)

        # Letterbox resize
        scale = min(input_width / w, input_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        dw, dh = (input_width - new_w) // 2, (input_height - new_h) // 2
        canvas[dh : dh + new_h, dw : dw + new_w] = resized

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(blob, 0)

        input_name = self._onnx_session.get_inputs()[0].name
        outputs = self._onnx_session.run(None, {input_name: blob})

        # Some YOLO exports (e.g., Ultralytics YOLOv8/9) return shape (1, 84, 8400)
        # and may include duplicate outputs. Handle this explicitly.
        pred0 = outputs[0]

        boxes_xyxy, scores, class_ids = self._postprocess_yolo_generic(
            pred0, w, h, dw, dh, scale
        )

        detections: List[Detection] = []
        min_area = self.config.min_box_area_ratio * frame_area
        for (x1, y1, x2, y2), score, class_id in zip(boxes_xyxy, scores, class_ids):
            if score < self.config.confidence_threshold:
                continue
            # Filter out very small boxes that are likely noise
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < min_area:
                continue
            # Filter out detections whose center falls inside any ignore region
            if self._is_in_ignore_region((x1 + x2) * 0.5, (y1 + y2) * 0.5, w, h):
                continue
            if (
                self.config.track_class_ids
                and class_id not in self.config.track_class_ids
            ):
                continue
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(score),
                    class_id=int(class_id),
                )
            )
        return detections

    def _postprocess_yolo_generic(
        self,
        preds: np.ndarray,
        orig_w: int,
        orig_h: int,
        dw: int,
        dh: int,
        scale: float,
    ):
        """
        Convert YOLO-style predictions into (xyxy, score, class_id) arrays.

        Supports two common layouts:
        - (1, N, 5 + C): [cx, cy, w, h, obj, class_scores...]
        - (1, 84, 8400): Ultralytics YOLOv8/9 export (no explicit obj column,
          just class scores after internal objectness); we take max class score.
        """

        # Case 0: Some YOLO exports (end2end) return already-decoded boxes:
        # (1, N, 6) -> [x1, y1, x2, y2, score, class_id]
        if (
            preds.ndim == 3
            and preds.shape[0] == 1
            and preds.shape[2] == 6
        ):
            preds = np.squeeze(preds, axis=0)
            boxes_xyxy = preds[:, 0:4]
            scores = preds[:, 4]
            class_ids = preds[:, 5].astype(int)
            return boxes_xyxy, scores, class_ids

        # Important: handle the YOLOv8/YOLOv9 (1, 84, 8400) layout
        # *before* the generic (1, N, 5 + C) case, otherwise it will be
        # misinterpreted and produce nonsense box coordinates and class IDs.
        if preds.ndim == 3 and preds.shape[0] == 1 and preds.shape[1] == 84:
            # Match the working oi.py script:
            #   - output[0] has shape (84, 8400)
            #   - transpose to (8400, 84)
            #   - first 4 values are [cx, cy, w, h] in the letterboxed space
            #   - the remaining 80 values are per-class scores (logits or probs)
            preds = np.squeeze(preds, axis=0)  # (84, 8400)
            preds = preds.T  # (8400, 84)
            boxes = preds[:, :4]
            class_scores = preds[:, 4:]

            # Use the same semantics as oi.py:
            # pick the best class per anchor and treat that value as the score.
            class_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

        # Case A: generic YOLO layout (1, N, 5 + C)
        elif preds.ndim == 3 and preds.shape[0] == 1 and preds.shape[2] >= 5:
            preds = np.squeeze(preds, axis=0)
            boxes = preds[:, 0:4]
            objectness = preds[:, 4:5]
            class_scores = preds[:, 5:]

            class_ids = np.argmax(class_scores, axis=1)
            class_conf = class_scores[np.arange(class_scores.shape[0]), class_ids]
            scores = objectness.squeeze(-1) * class_conf

        else:
            raise ValueError(f"Unsupported YOLO output shape: {preds.shape}")

        # Filter by confidence
        mask = scores >= self.config.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if boxes.size == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        # cx, cy, w, h -> x1, y1, x2, y2 (in letterboxed space)
        cx, cy, w_box, h_box = boxes.T
        x1 = cx - w_box / 2
        y1 = cy - h_box / 2
        x2 = cx + w_box / 2
        y2 = cy + h_box / 2

        # Undo letterbox to map back to original image coordinates
        x1 = (x1 - dw) / scale
        x2 = (x2 - dw) / scale
        y1 = (y1 - dh) / scale
        y2 = (y2 - dh) / scale

        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Non-maximum suppression
        indices = self._nms(boxes_xyxy, scores, self.config.nms_iou_threshold)

        return boxes_xyxy[indices], scores[indices], class_ids[indices]

    def _is_in_ignore_region(self, cx: float, cy: float, width: int, height: int) -> bool:
        """
        Return True if the point (cx, cy) lies inside any normalized ignore region.
        """

        if not self.config.ignore_regions:
            return False

        nx = cx / float(width)
        ny = cy / float(height)
        for x1, y1, x2, y2 in self.config.ignore_regions:
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                return True
        return False

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """
        Simple NMS implementation.
        """

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Standard area computation; no +1 so IoU matches common YOLO post-processing
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / np.maximum(union, 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int32)
