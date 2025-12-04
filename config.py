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
    model_path: Path = ROOT_DIR / "models" / "yolov9-s_v2.onnx"

    # Expected input size for the ONNX model (width, height).
    # Common YOLO-style models use 640x640 or 416x416.
    input_size: Tuple[int, int] = (640, 640)

    # Confidence threshold for detections.
    # Slightly lower so distant / mildly blurred vehicles are kept.
    confidence_threshold: float = 0.35

    # NMS IoU threshold (only used by ONNX backend)
    # Lowered slightly to suppress near-duplicate boxes produced by adjacent anchors.
    nms_iou_threshold: float = 0.45

    # Minimum box area as a fraction of the full frame area.
    # Lowered so distant vehicles are not filtered out too aggressively.
    min_box_area_ratio: float = 0.0002

    # Class IDs to track (COCO-style IDs by default)
    track_class_ids: List[int] = None

    # Normalized ignore regions: list of (x1, y1, x2, y2) in [0,1] coords.
    # Any detection whose box center lies inside one of these regions is dropped.
    # Defaults aim to ignore the left banana leaves and the top-right corner leaf.
    ignore_regions: List[Tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        # Default to common road-user classes in COCO if not specified:
        # 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
        if self.track_class_ids is None:
            self.track_class_ids = [1, 2, 3, 5, 7]
        if self.ignore_regions is None:
            self.ignore_regions = [
                # Left foliage / pole area
                (0.0, 0.0, 0.45, 1.0),
                # Top-right leaf occlusion
                (0.82, 0.0, 1.0, 0.25),
            ]


@dataclass
class TrackerConfig:
    """
    Simple multi-object tracker configuration.

    The tracker is a lightweight, greedy, distance-based tracker inspired by SORT
    (without Kalman filtering).
    """

    # How far a detection is allowed to move (in pixels, center-to-center)
    # between frames while still being matched to the same track. Increased
    # a bit so fast-moving vehicles remain linked.
    max_distance: float = 80.0

    # How many frames we keep a track alive without a detection before
    # considering it lost. Slightly higher to tolerate brief occlusions
    # or blurred frames for fast vehicles.
    max_lost: int = 45


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
