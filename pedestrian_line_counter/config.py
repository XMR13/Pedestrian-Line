from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# Project root (one level above the package directory)
ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    """
    
    Konfirugasi yang bisa di set untuk detektor, pada config ini juga terdapat
    metode alternatif untuk menjalankan program, seperti model torch dan kemudian non AI model.
    """

    # "onnx" | "motion"
    backend: str = "motion"

    # Path to a model file.
    # - When backend == "onnx": expected to be an ONNX file.
    # - When backend == "torch": expected to be a Torch `.pt` / `.pth` file.
    # Defaults to a typical YOLOv9-s export under the local Models/ folder.
    model_path: Path = ROOT_DIR / "Models" / "yolov9-s_v2.onnx"

    # Common YOLO-style models use 640x640 or 416x416.
    input_size: Tuple[int, int] = (640, 640)

    # Confidence threshold sebelum model bisa menentukan bahwa objek tersebut patut diteteksi.
    confidence_threshold: float = 0.5

    # NMS IoU Threshold (only used by ONNX backend)
    nms_iou_threshold: float = 0.45

    # Minimum box area dibandingkan dengan bagian area yang lain.
    # Lowered so distant vehicles are not filtered out too aggressively.
    min_box_area_ratio: float = 0.0002

    # ID kelas yang akan termasuk bagian dari tracking
    track_class_ids: List[int] = None

    # Normalisasi area area ini menjadi dengan rentang: list of (x1, y1, x2, y2) in [0,1] coords.
    # Any detection whose box center lies inside one of these regions is dropped..
    ignore_regions: List[Tuple[float, float, float, float]] = None

    # Opsi yang spesifik dengan torch (used only when backend == "torch")
    # device:
    #   - "auto": Menggunakan CUDA jika tersedia, jika belum terrsedia, gunakan CPU.
    #   - "cpu", "cuda", or any valid torch device string.
    torch_device: str = "auto"
    # Jalnkakn model ini dengan half precision or not?.
    torch_use_half: bool = False

    def __post_init__(self) -> None:
        # Penggunaan kelas koko pada projek ini, untuk data lengkapnya bisa dilihat di Models/metadata.yaml:
        # 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
        # hasil dari ini bis dapat diatur dengan menggunakan config file
        if self.track_class_ids is None:
            self.track_class_ids =  [1, 2, 3, 5, 7]
        if self.ignore_regions is None:
            # Default: do not suppress any region. Configure per-camera if needed.
            self.ignore_regions = []


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
    input_path: Path = ROOT_DIR / "media" / "WhatsApp Video 2025-12-03 at 11.23.31_60de7c28.mp4"
    output_path: Path = ROOT_DIR / "output.mp4"


@dataclass
class AppConfig:
    """
    Menjadikan konfigurasi ini dalam 1 kelas agar mudah digunakan dan diimplementasikan ke bagian project yang lain
    .
    """

    model: ModelConfig = ModelConfig()
    tracker: TrackerConfig = TrackerConfig()
    line: LineConfig = LineConfig()
    io: IOConfig = IOConfig()


def get_default_config() -> AppConfig:
    """
    Mengembalikan config app.
    """

    # melakukan return config untuk semua yang default
    return AppConfig()
