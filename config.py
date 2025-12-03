from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


ROOT_DIR = Path(__file__).resolve().parent


@dataclass
class ModelConfig:
    """
    Configuration for the object detector.

    By default we *design for* an ONNX detector, but also allow a lightweight
    motion-based fallback that does not require any ML model file.
    """

    # "onnx" | "motion"
    backend: str = "motion"

    # Path to an ONNX model (used when backend == "onnx")
    model_path: Path = ROOT_DIR / "models" / "yolov9-c.onnx"

    # Expected input size for the ONNX model (width, height).
    # Common YOLO-style models use 640x640 or 416x416.
    input_size: Tuple[int, int] = (640, 640)

    # Confidence threshold for detections
    confidence_threshold: float = 0.2

    # NMS IoU threshold (only used by ONNX backend)
    nms_iou_threshold: float = 0.2

    # Class IDs to track (COCO-style IDs by default)
    track_class_ids: List[int] = None

    def __post_init__(self) -> None:
        # Default to common vehicle classes in COCO if not specified:
        # 2=car, 3=motorcycle, 5=bus, 7=truck
        if self.track_class_ids is None:
            self.track_class_ids = [2, 3, 5, 7]


@dataclass
class TrackerConfig:
    """
    Simple multi-object tracker configuration.

    The tracker is a lightweight, greedy, distance-based tracker inspired by SORT
    (without Kalman filtering).
    """

    max_distance: float = 60.0  # pixels, center-to-center distance
    max_lost: int = 30  # max frames to keep a lost track


@dataclass
class LineConfig:
    """
    Virtual counting line configuration.

    The line is defined in *normalized* coordinates (0-1) relative to the
    video frame size. This makes it easier to reuse the same config on
    different resolutions.
    """

    # (x, y) in [0, 1] x [0, 1]
    start_norm: Tuple[float, float] = (0.35, 0.45)
    end_norm: Tuple[float, float] = (0.85, 0.75)


@dataclass
class IOConfig:
    """
    Input/output configuration for processing a single video file.
    """

    # Default to one of the sample videos if present; otherwise the user should
    # override this via CLI.
    input_path: Path = ROOT_DIR / "media" / "WhatsApp Video 2025-12-03 at 11.23.31_60de7c28.mp4"
    output_path: Path = ROOT_DIR / "output.mp4"


@dataclass
class AppConfig:
    """
    Aggregated configuration for the whole application.
    """

    model: ModelConfig = ModelConfig()
    tracker: TrackerConfig = TrackerConfig()
    line: LineConfig = LineConfig()
    io: IOConfig = IOConfig()


def get_default_config() -> AppConfig:
    """
    Return a fresh AppConfig instance.
    """

    return AppConfig()
