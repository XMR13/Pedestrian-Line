from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from .config import ModelConfig, ROOT_DIR
from .structures import Detection


@dataclass
class Detector:
    """
    High level detector wrapper yang bisa menggunakan beberapa macam backend.   

    Supported backends:
    - \"motion\": dengan melakukan perhitungan melalui background yang bergerak (no classes).
    - \"onnx\": Berbasis ONNXruntime dengan style (model YOLO) (.onnx)
    - \"tensorrt\": TensorRT engine runtime (.engine) for Jetson / NVIDIA GPUs.
    - \"torch\" : Menggunakan backend model pytoch (.pt) 
    """

    config: ModelConfig
    _backend: str = None #deafult value aadalah None, dan akan dilakukan pengecekan terhadap jenis backend yang tersedia
    _motion_subtractor: Optional[cv2.BackgroundSubtractor] = None
    _trt_runner: Optional[object] = None
    _torch_detector: Optional[object] = None
    _yolo_pipe: Optional[object] = None

    def __post_init__(self) -> None:
        backend = (self.config.backend or "motion").lower()

        if backend in {"onnx", "tensorrt", "torch"} and not self.config.track_class_ids and not getattr(
            self.config, "allow_all_classes", False
        ):
            raise ValueError(
                "Model-based backends require an explicit target class filter. "
                "Set `ModelConfig.track_class_ids` (or pass `--class-ids`) to the vehicle subclass IDs "
                "you want to count, or set `ModelConfig.allow_all_classes=True` for debugging."
            )

        if backend == "onnx":
            try:
                from yolo_kitv2 import LetterboxConfig, YoloPostConfig, load_pipeline  # type: ignore

                providers = self._pick_onnx_providers()
                self._yolo_pipe = load_pipeline(
                    self.config.model_path,
                    backend="onnxruntime",
                    letterbox_cfg=LetterboxConfig(new_shape=self.config.input_size),
                    post_cfg=self._build_yolo_post_cfg(),
                    onnx_providers=providers,
                )
                # Print providers actually in use so it's obvious whether CUDA is active.
                try:
                    ort_backend = getattr(self._yolo_pipe, "backend", None)
                    providers_in_use = list(getattr(ort_backend, "providers_in_use", []) or [])
                    if providers_in_use:
                        using_cuda = "CUDAExecutionProvider" in providers_in_use
                        print(
                            "[detector] ONNXRuntime providers in use: "
                            f"{providers_in_use} (cuda={'yes' if using_cuda else 'no'})"
                        )
                    else:
                        # Fallback: still show what we requested.
                        print(f"[detector] ONNXRuntime providers requested: {list(providers)}")
                except Exception:
                    print(f"[detector] ONNXRuntime providers requested: {list(providers)}")
                self._backend = "onnx"
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[detector] Failed to initialize ONNX backend: {exc}")
                print("[detector] Falling back to motion-based detector.")
                backend = "motion"

        if backend == "tensorrt":
            if str(self.config.model_path).lower().endswith(".onnx"):
                raise RuntimeError(
                    "TensorRT backend expects a TensorRT engine file (.engine). "
                    "Build an engine on the target device and pass it via --model."
                )
            from .tensorrt_runner import TensorRTRunner

            trt_model_path = Path(self.config.model_path)
            if not trt_model_path.is_absolute() and not trt_model_path.exists():
                trt_model_path = ROOT_DIR / trt_model_path

            self._trt_runner = TensorRTRunner(
                trt_model_path,
                input_size=self.config.input_size,
            )

            from yolo_kitv2 import LetterboxConfig, YoloPipeline  # type: ignore

            infer_fn = self._build_trt_infer_fn(self._trt_runner)
            self._yolo_pipe = YoloPipeline(
                infer_fn,
                backend=self._trt_runner,
                backend_name="tensorrt",
                letterbox_cfg=LetterboxConfig(new_shape=self._trt_runner.input_hw),
                post_cfg=self._build_yolo_post_cfg(),
            )
            self._backend = "tensorrt"

        if backend == "torch":
            # Lazy import untuk mereka yang memungkinkan hal ini
            # Jika menggunakan torch
            from .torch_detector import TorchDetector

            self._torch_detector = TorchDetector(self.config)
            self._backend = "torch"

        if backend == "motion":
            self._motion_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True
            )
            self._backend = "motion"

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Lakukan deteksi pada satu frame dan kembalian list deteksi yagn diperlukan.
        """

        if self._backend in {"onnx", "tensorrt"} and self._yolo_pipe is not None:
            return self._detect_yolo(frame_bgr)
        if self._backend == "torch" and self._torch_detector is not None:
            return self._torch_detector.detect(frame_bgr)

        # Default / fallback
        return self._detect_motion(frame_bgr)

    # --------------------------------------------------------------------- #
    # Motion-based detector
    # --------------------------------------------------------------------- #
    def _detect_motion(self, frame_bgr: np.ndarray) -> List[Detection]:
        fgmask = self._motion_subtractor.apply(frame_bgr)

        # Hilangi noise dengan beberap operation yang digunakan pada open cv
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

    def _detect_yolo(self, frame_bgr: np.ndarray) -> List[Detection]:
        assert self._yolo_pipe is not None

        img_h, img_w = frame_bgr.shape[:2]
        frame_area = float(img_w * img_h)
        min_area = self.config.min_box_area_ratio * frame_area

        raw_dets = self._yolo_pipe(frame_bgr)
        detections: List[Detection] = []

        for d in raw_dets:
            x1, y1, x2, y2 = float(d.x1), float(d.y1), float(d.x2), float(d.y2)
            score = float(d.score)
            class_id = None if getattr(d, "class_id", None) is None else int(d.class_id)

            if score < self.config.confidence_threshold:
                continue
            if class_id is not None and self.config.track_class_ids and class_id not in self.config.track_class_ids:
                continue

            box_area = (x2 - x1) * (y2 - y1)
            if box_area < min_area:
                continue

            if self._is_in_ignore_region((x1 + x2) * 0.5, (y1 + y2) * 0.5, img_w, img_h):
                continue

            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    class_id=class_id,
                )
            )

        return detections

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

    def _build_yolo_post_cfg(self) -> "object":
        from yolo_kitv2 import YoloPostConfig  # type: ignore

        class_ids = None
        if self.config.track_class_ids and not getattr(self.config, "allow_all_classes", False):
            class_ids = list(self.config.track_class_ids)

        return YoloPostConfig(
            conf_threshold=float(self.config.confidence_threshold),
            iou_threshold=float(self.config.nms_iou_threshold),
            pre_nms_topk=self.config.pre_nms_topk,
            class_ids=class_ids,
        )

    @staticmethod
    def _pick_onnx_providers() -> List[str]:
        """
        Best-effort provider selection for ONNX Runtime.

        Uses CUDA if available, then DirectML (Windows), then CPU.
        """

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

    @staticmethod
    def _build_trt_infer_fn(trt_runner: object) -> Callable[[np.ndarray], np.ndarray]:
        def infer(blob: np.ndarray) -> np.ndarray:
            outputs = trt_runner.infer(blob)
            if not outputs:
                raise RuntimeError("TensorRT backend returned no outputs.")

            yolo_like = [
                out
                for out in outputs
                if getattr(out, "ndim", 0) == 3 and len(getattr(out, "shape", ())) >= 1 and out.shape[0] == 1
            ]
            if len(yolo_like) == 1:
                return np.asarray(yolo_like[0])
            if len(outputs) == 1:
                return np.asarray(outputs[0])

            shapes = [tuple(int(dim) for dim in getattr(out, "shape", ())) for out in outputs]
            raise RuntimeError(
                "TensorRT backend could not identify a unique YOLO-style output tensor. "
                f"Engine outputs: {shapes}. Use a single-output raw YOLO engine for this backend."
            )

        return infer
