from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal


@dataclass
class Detection:
    """
    Hasil deteksi objek.
    """

    #koordinat
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: Optional[int] = None

    #return the coordinate 
    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def bottom_center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, self.y2


@dataclass
class Track:
    """
    Objek yang di track dengan stable ID.
    """

    #koordinat track dan ditambah dengan variabel data last_seen_frame
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: Optional[int]
    last_seen_frame: int
    # Display identifier (untuk per track event)
    display_class_id: Optional[int] = None
    display_id: Optional[int] = None
    display_ids_by_class: Dict[int, int] = field(default_factory=dict)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def bottom_center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, self.y2


Direction = Literal["A_TO_B", "B_TO_A"]
LineMode = Literal["line", "gate"]


@dataclass(frozen=True)
class CrossingEvent:
    """
    A single line-crossing event produced by the counter.

    This is intentionally small and edge-friendly; the portal uploader can enrich
    it (site/camera/run IDs, timestamps, etc.) before ingestion.
    """

    track_id: int
    direction: Direction
    frame_index: int
    class_id: Optional[int] = None
    confidence: Optional[float] = None
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    line_mode: LineMode = "line"
