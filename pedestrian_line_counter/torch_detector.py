from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import ModelConfig
from .structures import Detection


@dataclass
class TorchDetector:
    """
    Lightweight Torch-based detector backend.
    Backend ringaan berbasis dibasis oleh pytorch

    Backend ini bersifat optional
    It is intentionally simple and makes a few assumptions to stay generic:

    - Torch model ini diload dengan menggunakan ModelConfig.model_path dan program ini menghaurskan 
      untuk menerima tensor dedngan bentuk `(1, 3, H, W)` dengan nilai berada `[0, 1]` <-- normalisasi

    - Model ini harus mengoutptukan deteksi dengan format yang valid, antarnya adalah
        * `(N, 6)` or `(1, N, 6)` with `[x1, y1, x2, y2, score, class_id]`
          in **resized-image pixel coordinates**.

      Colum lain akan diabaikan.
    - Kelas ini akan melakuakn skala ulang ke bentuk originalnya
      Panjang/lebar ratiio da kemudian menerapkan
        * confidence threshold (yang berada di config)
        * minimum ration
        * ignore region
        *id track

    If your model uses a different output format, you can adapt this module or
    wrap your model so that it produces the expected `(N, 6)` layout.
    """

    config: ModelConfig
    _model: Optional[object] = None
    _device: Optional[object] = None
    _torch: Optional[object] = None

    def __post_init__(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Torch backend selected but PyTorch is not installed. "
                "Install it first (for example: `pip install torch` or via `uv`), "
                "or switch ModelConfig.backend back to 'onnx' or 'motion'."
            ) from exc

        self._torch = torch

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Torch model file not found at {model_path}. "
                "Set `ModelConfig.model_path` to a valid `.pt` / `.pth` file."
            )

        self._device = self._select_device()

        # Mencoba torchscript, jika tidak terinstal kembali ke load generic torch.load.
        try:
            model = torch.jit.load(str(model_path), map_location=self._device)
        except Exception:
            try:
                model = torch.load(str(model_path), map_location=self._device)
            except Exception as exc:  # pragma: no cover - model specific
                raise RuntimeError(
                    f"Tidak bisa mendapatkan model torch {model_path}. "
                    "Harus berupa objek torchscript atau objek torch native"
                ) from exc

        if hasattr(model, "eval"):
            model.eval()

        self._model = model

    # ------------------------------------------------------------------ #
    # APU
    # ------------------------------------------------------------------ #
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run inference pada satu frame dan mengembalikan objek deteksi.
        """

        assert self._model is not None, "TorchDetector model is not initialised"
        torch = self._torch
        assert torch is not None

        h, w = frame_bgr.shape[:2]
        input_width, input_height = self.config.input_size

        # Resize sederhana
        resized = frame_bgr
        if (w, h) != (input_width, input_height):
            resized = cv2.resize(frame_bgr, (input_width, input_height), interpolation=cv2.INTER_LINEAR)  # type: ignore[name-defined]

        # BGR -> RGB, HWC -> CHW, normalise to [0, 1]
        img_rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        tensor = torch.from_numpy(img_chw).unsqueeze(0)  # (1, 3, H, W)

        tensor = tensor.to(self._device)
        if self.config.torch_use_half and self._device.type == "cuda":
            tensor = tensor.half()
        else:
            tensor = tensor.float()

        with torch.no_grad():
            outputs = self._model(tensor)

        preds = self._to_numpy(outputs)
        if preds is None or preds.size == 0:
            return []

        # Expecting (N, 6) with [x1, y1, x2, y2, score, class_id] in resized coords
        boxes_resized = preds[:, :4]
        scores = preds[:, 4]
        class_ids = preds[:, 5].astype(int)

        # Map box untuk menyesuaikan dengan koordinat awal
        scale_x = w / float(input_width)
        scale_y = h / float(input_height)

        x1 = boxes_resized[:, 0] * scale_x
        y1 = boxes_resized[:, 1] * scale_y
        x2 = boxes_resized[:, 2] * scale_x
        y2 = boxes_resized[:, 3] * scale_y

        # Clip to frame bounds
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Terapkan confidence threshold
        mask = scores >= self.config.confidence_threshold
        boxes_xyxy = boxes_xyxy[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if boxes_xyxy.size == 0:
            return []

        # Optional NMS if there are many boxes (assumes the model did not already apply NMS)
        keep = self._nms(
            boxes_xyxy,
            scores,
            iou_threshold=self.config.nms_iou_threshold,
        )
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Filter area, ignore region, dan kelas yang diperlukan
        frame_area = float(w * h)
        min_area = self.config.min_box_area_ratio * frame_area

        detections: List[Detection] = []
        for (bx1, by1, bx2, by2), score, class_id in zip(
            boxes_xyxy, scores, class_ids
        ):
            box_area = (bx2 - bx1) * (by2 - by1)
            if box_area < min_area:
                continue
            if self._is_in_ignore_region((bx1 + bx2) * 0.5, (by1 + by2) * 0.5, w, h):
                continue
            if (
                self.config.track_class_ids
                and class_id not in self.config.track_class_ids
            ):
                continue

            detections.append(
                Detection(
                    x1=float(bx1),
                    y1=float(by1),
                    x2=float(bx2),
                    y2=float(by2),
                    score=float(score),
                    class_id=int(class_id),
                )
            )

        return detections

    # ------------------------------------------------------------------ #
    # helper
    # ------------------------------------------------------------------ #
    def _select_device(self):
        torch = self._torch
        assert torch is not None

        cfg_device = (self.config.torch_device or "auto").lower()
        if cfg_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        # ("cpu", "cuda:0", etc.)
        return torch.device(cfg_device)

    @staticmethod
    def _to_numpy(outputs) -> Optional[np.ndarray]:
        """
        Normalisas itorch model output ke nummpy array (N,6)
        Ekpektasi layoyt
        - Tensor dengan shape (N,6)
        - Tensor array dengan shape (1, N, 6) dengan catatan dimensi didepan bisa dibuang
        - Elemen single yang memiliki salah satu dari bentuk objek diatas
        """

        import torch  # local import to avoid hard dependency at module import time

        if isinstance(outputs, (list, tuple)) and outputs:
            outputs = outputs[0]

        if isinstance(outputs, torch.Tensor):
            arr = outputs.detach().cpu().numpy()
        else:
            arr = np.asarray(outputs)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)

        if arr.ndim != 2 or arr.shape[1] < 6:
            return None

        return arr[:, :6]

    def _is_in_ignore_region(
        self, cx: float, cy: float, width: int, height: int
    ) -> bool:
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
        Implementasi NMS sederhana pada kotak (x1, y1, x2, y2)
        """

        if boxes.size == 0:
            return np.empty((0,), dtype=np.int64)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / np.maximum(union, 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int64)

