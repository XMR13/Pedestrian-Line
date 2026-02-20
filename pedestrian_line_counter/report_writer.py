from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

from .structures import CrossingEvent


@dataclass
class ReportWriterConfig:
    #The configuration of the report writer
    path: Path
    fps: float
    include_extra_cols: bool = True
    notes_default: str = ""


class ReportWriter:
    """
    Append-only CSV writer for per-crossing human-readable reports.
    """

    #the columns that will created with the report
    BASE_COLUMNS = (
        "event_no",
        "timestamp_s",
        "vehicle_type",
        "direction",
        "notes",
    )
    EXTRA_COLUMNS = (
        "track_id",
        "frame_index",
        "class_id",
        "confidence",
        "waktu lewat (WIB)",
        "thumb_relpath",
        "scene_relpath",
        "line_mode",
    )

    def __init__(
        self,
        cfg: ReportWriterConfig,
        *,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.cfg = cfg
        self._class_names = class_names or {}
        self._event_no = 0

        #create the path of te parents first
        self.cfg.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not self.cfg.path.exists()) or self.cfg.path.stat().st_size == 0

        self._fieldnames = list(self.BASE_COLUMNS)
        if self.cfg.include_extra_cols:
            self._fieldnames.extend(self.EXTRA_COLUMNS)

        self._f = self.cfg.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=self._fieldnames)
        if write_header:
            self._writer.writeheader()
            self._f.flush()


    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass
    

    def record_events(
        self,
        events: Iterable[CrossingEvent],
        *,
        event_records: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> int:
        written = 0
        for idx, ev in enumerate(events):
            self._event_no += 1
            thumb_relpath = self._str_from_records(event_records, idx, "thumb_relpath")
            scene_relpath = self._str_from_records(event_records, idx, "scene_relpath")
            waktu_wib = self._str_from_records(event_records, idx, "occurred_at_local")
            row = self._build_row(
                ev,
                thumb_relpath=thumb_relpath,
                scene_relpath=scene_relpath,
                waktu_wib=waktu_wib,
            )
            self._writer.writerow(row)
            written += 1

        if written:
            self._f.flush()
        return written

    def _build_row(
        self,
        ev: CrossingEvent,
        *,
        thumb_relpath: str,
        scene_relpath: str,
        waktu_wib: str,
    ) -> Dict[str, object]:
        row: Dict[str, object] = {
            "event_no": self._event_no,
            "timestamp_s": self._video_time_s(ev.frame_index),
            "vehicle_type": self._vehicle_type(ev.class_id),
            "direction": ev.direction,
            "notes": self.cfg.notes_default,
            "track_id": int(ev.track_id),
            "frame_index": int(ev.frame_index),
            "class_id": int(ev.class_id) if ev.class_id is not None else "",
            "confidence": float(ev.confidence) if ev.confidence is not None else "",
            "waktu lewat (WIB)": waktu_wib,
            "thumb_relpath": thumb_relpath,
            "scene_relpath": scene_relpath,
            "line_mode": ev.line_mode,
        }
        #set the field names first and then we can build the rows with it
        columns_final = {k: row[k] for k in self._fieldnames}
        return columns_final

    def _str_from_records(
        self,
        event_records: Optional[Sequence[Mapping[str, object]]],
        idx: int,
        key: str,
    ) -> str:
        if event_records is None or idx >= len(event_records):
            return ""
        val = event_records[idx].get(key)
        if val is None:
            return ""
        return str(val)

    def _video_time_s(self, frame_index: int) -> str:
        fps = float(self.cfg.fps)
        if fps <= 0.0:
            return ""
        value = float(frame_index) / fps
        return f"{value:.3f}"

    def _vehicle_type(self, class_id: Optional[int]) -> str:
        if class_id is None:
            return "unknown"
        cid = int(class_id)
        name = self._class_names.get(cid)
        if name is not None and str(name).strip():
            return str(name).strip()
        return f"class_{cid}"
