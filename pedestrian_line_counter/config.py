from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Project root (one level above the package directory)
ROOT_DIR = Path(__file__).resolve().parent.parent


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
    confidence_threshold: float = 0.45

    # NMS IoU Threshold (only used by ONNX backend)
    nms_iou_threshold: float = 0.45

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


@dataclass
class LineConfig:
    """
    Konfigurasi perhitungan virtual.

    Garis didefinisikan dengan koordinat hasil normalisasi relatif terhadap
    vidoe framenya
.
    """

    # (x, y) in [0, 1] x [0, 1]
    start_norm: Tuple[float, float] = (0.35, 0.45)
    end_norm: Tuple[float, float] = (0.85, 0.75)


@dataclass
class IOConfig:
    """
    Konfigurasi untuk input dan output dari video atau media yang ingin digunakan
    """

    # Dapat di ovveride dengan menggunakan Command Line Interfae.
    input_path: Path = ROOT_DIR / "media" / "input.mp4"
    output_path: Path = ROOT_DIR / "output.mp4"


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


def get_default_config() -> AppConfig:
    """
    Mengembalikan config app.
    """

    # melakukan return config untuk semua yang default
    return AppConfig()
