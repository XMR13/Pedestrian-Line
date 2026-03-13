from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Project root (one level above the package directory)
ROOT_DIR = Path(__file__).resolve().parent.parent

"""
There's 5 Config (represented by a dataclass)
1. Model Config for all things related to the model (class, changes, iou, confidence score, path, etc)
2. Track Config : Config for tracking behaviour 
3. IO config : COnfig for Input/Output 
4. Line Configuration : Configuration for line separated by the 
5. Visual config : for changing visual configuration (colour box, font colors, etc)
"""


@dataclass
class ModelConfig:
    """
    
    Konfirugasi yang bisa di set untuk detektor, pada config ini juga terdapat
    metode alternatif untuk menjalankan program, seperti model torch dan kemudian non AI model.
    """
    # "onnx" | "tensorrt" | "motion"
    backend: str = "motion"

    # Path to a model file.
    # - When backend == "onnx": expected to be an ONNX file.
    # - When backend == "tensorrt": expected to be a TensorRT `.engine` file (built on target device).
    # - When backend == "torch": expected to be a Torch `.pt` / `.pth` file.
    # Defaults to a typical YOLOv9-s export under the local Models/ folder.
    model_path: Path = ROOT_DIR / "Models" / "vehicle_subclasses.onnx"

    # Common YOLO-style models use 640x640 or 416x416.
    input_size: Tuple[int, int] = (640, 640)

    # Confidence threshold sebelum model bisa menentukan bahwa objek tersebut patut diteteksi.
    confidence_threshold: float = 0.5

    # NMS IoU Threshold (only used by ONNX backend)
    nms_iou_threshold: float = 0.45

    # Performance knob: cap candidates before NMS (YOLO anchors can be large).
    # Typical values: 300..2000. None disables.
    pre_nms_topk: Optional[int] = 1000

    # Minimum box area dibandingkan dengan bagian area yang lain.
    # Lowered so distant vehicles are not filtered out too aggressively.
    min_box_area_ratio: float = 0.0002

    # ID kelas yang akan termasuk bagian dari tracking + counting.
    #
    # Penting: untuk backend berbasis model ("onnx"/"tensorrt"/"torch"), project ini
    # didesain untuk menghitung *hanya* kelas kendaraan target. Jadi biasanya kamu
    # akan mengisi list ini dengan class_id kendaraan yang relevan untuk use-case ini.
    #
    # Jika list ini kosong, model-based backend dapat menolak untuk jalan (lihat `allow_all_classes`).
    track_class_ids: List[int] = field(default_factory=list)

    # Optional mapping `class_id -> name` untuk overlay dan laporan.
    # Format yang didukung:
    # - YAML: `{names: [..]}` atau `{names: {0: "truck", 1: "pickup"}}`
    # - JSON: list atau dict dengan key int/str-int.
    class_names_path: Optional[Path] = None

    # Normalisasi area area ini menjadi dengan rentang: list of (x1, y1, x2, y2) in [0,1] coords.
    # Any detection whose box center lies inside one of these regions is dropped..
    ignore_regions: List[Tuple[float, float, float, float]] = field(default_factory=list)

    # Advanced escape hatch: allow running model-based backends without class filtering.
    # Default False to avoid accidentally counting irrelevant classes.
    allow_all_classes: bool = False

    # Opsi yang spesifik dengan torch (used only when backend == "torch")
    # device:
    #   - "auto": Menggunakan CUDA jika tersedia, jika belum terrsedia, gunakan CPU.
    #   - "cpu", "cuda", or any valid torch device string.
    torch_device: str = "auto"
    # Jalnkakn model ini dengan half precision or not?.
    torch_use_half: bool = False

def load_class_names(path: Path) -> Dict[int, str]:
    """
    Load a class-name map from YAML or JSON.

    Supported shapes:
    - YAML dict with `names: [...]` or `names: {id: name, ...}`
    - JSON list: ["truck", "pickup", ...]  (index is class_id)
    - JSON dict: {"0": "truck", "1": "pickup", ...}

    Returns a dict mapping integer class IDs to display names.
    """

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text())
    elif suffix == ".json":
        import json

        data = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported class names file type: {path}")

    if isinstance(data, dict) and "names" in data:
        data = data["names"]

    if isinstance(data, list):
        return {int(i): str(name) for i, name in enumerate(data)}

    if isinstance(data, dict):
        out: Dict[int, str] = {}
        for k, v in data.items():
            try:
                cid = int(k)
            except Exception:
                continue
            out[cid] = str(v)
        return out

    raise ValueError(f"Invalid class names file contents: {path}")


def infer_track_class_ids_from_class_names(class_names: Dict[int, str]) -> List[int]:
    """
    Infer a default class-id allowlist from a class-name mapping.

    This enables a simple workflow where users provide only --class-names, and the
    program counts all classes defined in that mapping.
    """

    ids: List[int] = []
    for k in class_names.keys():
        try:
            ids.append(int(k))
        except Exception:
            continue
    return sorted(set(ids))


@dataclass
class TrackerConfig:
    """
    Multi object tracker yang sederhana
    
    The tracker ini ringan, dan diimplementasikan dengan menggunakan SORT.

    """

    # Seberapa jauh deteksi diperbolehkan untuk bergreak pada pixel (in pixels, center-to-center)
    # Nilai agak besar karena untuk kendaraan yang melaju dengan kecepatan di atas rata-rata.
    max_distance: float = 80.0
    max_lost: int = 45
    # Track re-association tweaks:
    # - When a track is not updated for a few frames (detector dropout / occlusion),
    #   allow a larger matching radius to prevent track fragmentation.
    max_distance_scale_cap: int = 5
    # Class decision used for counting/reporting:
    # - "instant": use current frame class directly (legacy behavior).
    # - "majority": use most frequent class across track lifetime.
    # - "weighted_majority": use highest confidence-weighted class across lifetime.
    class_vote_mode: str = "weighted_majority"
    # Minimum classified frames before majority/weighted vote is trusted.
    class_vote_min_frames: int = 3
    # Ambiguity guard:
    # if top_vote / second_vote is below this ratio, keep previous stable class.
    # Use <= 1.0 to disable.
    class_vote_ambiguity_ratio: float = 1.15


@dataclass
class LineConfig:
    """
    Konfigurasi perhitungan virtual.

    Garis didefinisikan dengan koordinat hasil normalisasi relatif terhadap
    vidoe framenya
.
    """

    # Optional line source:
    # - line_json_path: path to line JSON produced by line_picker.py
    # - camera_name: resolves to config/cameras/<name>.json then config/<name>.json
    # If both are None, fallback to start_norm/end_norm below.
    line_json_path: Optional[Path] = None
    camera_name: Optional[str] = None

    # (x, y) in [0, 1] x [0, 1]
    start_norm: Tuple[float, float] = (0.35, 0.45)
    end_norm: Tuple[float, float] = (0.85, 0.75)


@dataclass
class IOConfig:
    """
    Konfigurasi untuk input dan output dari video atau media yang ingin digunakan
    """

    # Optional RTSP URL for live mode (alternative to input_path).
    # Keep credentials in a local-only config (e.g. config/local/*.local.json).
    rtsp_url: Optional[str] = None

    # RTSP reconnect policy (used in live mode).
    rtsp_reconnect_enabled: bool = True
    rtsp_reconnect_max_attempts: int = 0  # 0 = unlimited
    rtsp_reconnect_initial_delay_s: float = 1.0
    rtsp_reconnect_max_delay_s: float = 30.0
    rtsp_reconnect_backoff_factor: float = 2.0
    rtsp_stall_timeout_s: float = 5.0

    # RTSP capture backend configuration.
    # - "opencv": default OpenCV/FFmpeg path (portable).
    # - "gstreamer": OpenCV CAP_GSTREAMER pipeline (Jetson-friendly, NVDEC).
    rtsp_capture_backend: str = "opencv"
    rtsp_transport: str = "tcp"  # "tcp" | "udp"
    rtsp_latency_ms: int = 200
    rtsp_codec: str = "h264"  # "h264" | "h265"
    rtsp_gst_pipeline: Optional[str] = None

    # Live reader queue behaviour.
    # - "drop_oldest": keep latency bounded by dropping stale frames.
    # - "block": keep frames when possible (may increase end-to-end delay).
    live_queue_policy: str = "drop_oldest"

    # Dapat di ovveride dengan menggunakan Command Line Interfae.
    input_path: Path = ROOT_DIR / "media" / "input.mp4"
    # Offline file capture backend configuration.
    # - "opencv": default OpenCV/FFmpeg path (portable).
    # - "gstreamer": OpenCV CAP_GSTREAMER pipeline for local file input.
    input_capture_backend: str = "opencv"
    input_gst_pipeline: Optional[str] = None
    output_path: Path = ROOT_DIR / "output.mp4"

    # Offline video timing (optional).
    # If set (RFC3339 with timezone), event timestamps will be derived from the video
    # timeline instead of wall-clock time: occurred_at_utc = video_start + frame_index/fps.
    video_start: Optional[str] = None
    # Override decoded source FPS when CAP_PROP_FPS is missing/wrong.
    source_fps_override: Optional[float] = None

@dataclass
class SpoolRetentionConfig:
    """
    Konfigurasi retensi untuk folder spool lokal.

    V1 sengaja conservative:
    - default 90 hari,
    - hanya hapus whole run directory,
    - run tanpa state upload yang valid dianggap protected.
    """

    enabled: bool = False
    max_age_days: int = 90
    protect_incomplete_runs: bool = True
    state_filename: str = ".portal_upload_state.json"


@dataclass
class SpoolConfig:
    """
    Konfigurasi untuk filesystem-first traffic spool (events.jsonl + thumbnails).

    Spool is disabled when `root_dir` is None.
    """

    root_dir: Optional[Path] = None
    site_id: str = "default"
    camera_id: Optional[str] = None
    write_thumbnails: bool = True
    write_scene_thumbnails: bool = True
    thumb_pad: int = 20
    thumb_max_side: int = 320
    scene_thumb_max_side: int = 640
    scene_thumb_quality: int = 85
    retention: SpoolRetentionConfig = field(default_factory=SpoolRetentionConfig)


@dataclass
class ReportConfig:
    """
    Konfigurasi report CSV per-run (human-readable) dari crossing events.

    Report berada di dalam folder run spool agar 1:1 dengan events.jsonl.
    """

    enabled: bool = True
    filename: str = "report.csv"
    include_extra_cols: bool = True


@dataclass
class VisualConfig:
    """
    Konfigurasi visual untuk overlay box/label.
    Semua warna menggunakan format BGR (OpenCV).
    """

    # "class" -> warna stabil per class_id
    # "track" -> warna stabil per track_id
    track_color_by: str = "class"

    # Fallback color jika palette kosong.
    track_default_color: Tuple[int, int, int] = (46, 204, 113)

    # Palette default untuk box 
    # use opencv BGR default order (NOT RGB)
    track_palette: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (46, 204, 113),
            (52, 152, 219),
            (0, 159, 255),
            (231, 76, 60),
            (155, 89, 182),
            (241, 196, 15),
            (26, 188, 156),
            (230, 126, 34),
        ]
    )

    # Override warna per class_id, contoh:
    # {"0": [0, 255, 0], "1": [0, 0, 255]}
    # Key string dipakai agar mudah ditulis di JSON.
    class_colors: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class AppConfig:
    """
    Menjadikan konfigurasi ini dalam 1 kelas agar mudah digunakan dan diimplementasikan ke bagian project yang lain
    .
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    line: LineConfig = field(default_factory=LineConfig)
    io: IOConfig = field(default_factory=IOConfig)
    spool: SpoolConfig = field(default_factory=SpoolConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)


def get_default_config() -> AppConfig:
    """
    Mengembalikan config app.
    """

    # melakukan return config untuk semua yang default
    return AppConfig()
