from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .letterbox import letterbox
from .postprocess import YoloPostConfig, YoloPostprocessor
from .types import Detection


PathLike = Union[str, Path]


def find_project_root(
    start: Optional[PathLike] = None,
    markers: Sequence[str] = ("pyproject.toml", "setup.py", ".git", "requirements.txt"),
) -> Path:
    """
    Best-effort project root discovery.

    Useful when `yolo_kit` is vendored as `A/yolo_kit` and models live in `A/models`.
    """

    p = Path(start) if start is not None else Path.cwd()
    p = p.resolve()

    # If a file is provided, start from its directory.
    if p.is_file():
        p = p.parent

    for parent in (p, *p.parents):
        for m in markers:
            if (parent / m).exists():
                return parent
    return p


def resolve_path(path: PathLike, root: Optional[PathLike] = "auto") -> Path:
    """
    Resolve `path` to an absolute Path.

    - Absolute paths are returned as-is.
    - Relative paths are resolved against:
      - `root` if provided
      - project root (auto) otherwise
    """

    p = Path(path)
    if p.is_absolute():
        return p

    if root == "auto" or root is None:
        base = find_project_root()
    else:
        base = Path(root).resolve()

    return (base / p).resolve()


@dataclass(frozen=True)
class LetterboxConfig:
    new_shape: Tuple[int, int] = (640, 640)
    color: Tuple[int, int, int] = (114, 114, 114)
    auto: bool = False
    scale_fill: bool = False
    scaleup: bool = True
    stride: int = 32


@dataclass(frozen=True)
class PreprocessResult:
    blob: np.ndarray
    orig_size: Tuple[int, int]
    ratio: Tuple[float, float]
    pad: Tuple[float, float]


class YoloPipeline:
    """
    Plug-and-play pipeline: preprocess (letterbox) -> inference -> postprocess.

    The pipeline expects BGR images (OpenCV-style) as `np.ndarray` and returns
    a list of `Detection` in original image coordinates.
    """

    def __init__(
        self,
        infer_fn: Callable[[np.ndarray], np.ndarray],
        *,
        backend: Optional[object] = None,
        backend_name: Optional[str] = None,
        letterbox_cfg: LetterboxConfig = LetterboxConfig(),
        post_cfg: YoloPostConfig = YoloPostConfig(),
    ):
        self._infer_fn = infer_fn
        self.backend = backend
        self.backend_name = backend_name
        self.letterbox_cfg = letterbox_cfg
        self.post = YoloPostprocessor(post_cfg)

    def preprocess(self, image_bgr: np.ndarray) -> PreprocessResult:
        if image_bgr is None or not hasattr(image_bgr, "shape"):
            raise TypeError("image_bgr must be a NumPy array (BGR).")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {getattr(image_bgr, 'shape', None)}")

        orig_h, orig_w = image_bgr.shape[:2]
        img, ratio, pad = letterbox(
            image_bgr,
            new_shape=self.letterbox_cfg.new_shape,
            color=self.letterbox_cfg.color,
            auto=self.letterbox_cfg.auto,
            scale_fill=self.letterbox_cfg.scale_fill,
            scaleup=self.letterbox_cfg.scaleup,
            stride=self.letterbox_cfg.stride,
        )

        # BGR -> RGB, normalize, HWC -> CHW, add batch
        blob = img[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        return PreprocessResult(blob=blob, orig_size=(orig_w, orig_h), ratio=ratio, pad=pad)

    def __call__(self, image_bgr: np.ndarray) -> List[Detection]:
        prep = self.preprocess(image_bgr)
        preds = self._infer_fn(prep.blob)
        return self.post.process(preds, orig_size=prep.orig_size, pad=prep.pad, ratio=prep.ratio)


def load_pipeline(
    model_path: PathLike,
    *,
    backend: Optional[str] = None,
    root: Optional[PathLike] = "auto",
    letterbox_cfg: LetterboxConfig = LetterboxConfig(),
    post_cfg: YoloPostConfig = YoloPostConfig(),
    onnx_providers: Optional[Sequence[str]] = None,
    onnx_input_name: Optional[str] = None,
    onnx_output_name: Optional[str] = None,
    torch_device: str = "cpu",
    torch_half: bool = False,
    torch_output_index: int = 0,
    trt_device: str = "cuda",
    trt_input_name: Optional[str] = None,
    trt_output_name: Optional[str] = None,
    trt_output_index: int = 0,
) -> YoloPipeline:
    """
    Create a plug-and-play pipeline for a model on disk.

    Typical usage when `yolo_kit` is vendored into another repo:
        pipe = load_pipeline("models/yolov8n.onnx")  # resolves from project root by default

    Args:
        model_path: path to weights/model file; relative paths resolve against project root by default
        backend: "onnxruntime" (default for .onnx) or None to infer from extension
        root: base directory for resolving relative model paths ("auto" uses best-effort project root)
    """

    resolved = resolve_path(model_path, root=root)
    chosen = backend
    if chosen is None:
        suffix = resolved.suffix.lower()
        if suffix == ".onnx":
            chosen = "onnxruntime"
        elif suffix in {".engine", ".plan"}:
            chosen = "tensorrt"
        elif suffix in {".torchscript", ".ts", ".pt"}:
            chosen = "torchscript"
        else:
            raise ValueError(
                f"Could not infer backend from extension '{suffix}'. Pass backend=... explicitly."
            )

    chosen = chosen.lower()
    if chosen == "onnxruntime":
        from .backends.onnxruntime_backend import OnnxRuntimeBackend, OnnxRuntimeBackendConfig

        ort_backend = OnnxRuntimeBackend(
            resolved,
            OnnxRuntimeBackendConfig(
                providers=onnx_providers,
                input_name=onnx_input_name,
                output_name=onnx_output_name,
            ),
        )
        return YoloPipeline(
            ort_backend.infer,
            backend=ort_backend,
            backend_name="onnxruntime",
            letterbox_cfg=letterbox_cfg,
            post_cfg=post_cfg,
        )

    if chosen == "torchscript":
        from .backends.torchscript_backend import TorchScriptBackend, TorchScriptBackendConfig

        ts_backend = TorchScriptBackend(
            resolved,
            TorchScriptBackendConfig(device=torch_device, half=torch_half, output_index=torch_output_index),
        )
        return YoloPipeline(
            ts_backend.infer,
            backend=ts_backend,
            backend_name="torchscript",
            letterbox_cfg=letterbox_cfg,
            post_cfg=post_cfg,
        )

    if chosen == "tensorrt":
        from .backends.tensorrt_backend import TensorRTBackend, TensorRTBackendConfig

        trt_backend = TensorRTBackend(
            resolved,
            TensorRTBackendConfig(
                device=trt_device,
                input_name=trt_input_name,
                output_name=trt_output_name,
                output_index=trt_output_index,
            ),
        )
        return YoloPipeline(
            trt_backend.infer,
            backend=trt_backend,
            backend_name="tensorrt",
            letterbox_cfg=letterbox_cfg,
            post_cfg=post_cfg,
        )

    raise ValueError(f"Unsupported backend: {backend!r}")
