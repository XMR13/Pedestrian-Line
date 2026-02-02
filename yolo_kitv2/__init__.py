"""
Lightweight, reusable YOLO post-processing helpers.

Designed to be framework-agnostic: works with NumPy arrays emitted by ONNX
Runtime or PyTorch tensors converted to NumPy. No external dependencies beyond
NumPy and OpenCV for letterboxing.
"""

from .types import Detection
from .letterbox import letterbox
from .nms import nms
from .postprocess import YoloPostprocessor, YoloPostConfig
from .runtime import YoloPipeline, load_pipeline, find_project_root, resolve_path, LetterboxConfig
from .metadata import load_class_names
from .visualize import draw_detections
from .dataset_check import run_dataset_check

__all__ = [
    "Detection",
    "letterbox",
    "nms",
    "YoloPostprocessor",
    "YoloPostConfig",
    "YoloPipeline",
    "load_pipeline",
    "find_project_root",
    "resolve_path",
    "LetterboxConfig",
    "load_class_names",
    "draw_detections",
    "run_dataset_check",
]
