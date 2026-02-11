from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
import uuid

import cv2
import numpy as np

from .structures import CrossingEvent


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1c = max(0, min(int(x1), w - 1))
    y1c = max(0, min(int(y1), h - 1))
    x2c = max(0, min(int(x2), w - 1))
    y2c = max(0, min(int(y2), h - 1))
    if x2c <= x1c or y2c <= y1c:
        return None
    return x1c, y1c, x2c, y2c


def _resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = float(max_side) / float(m)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


@dataclass
class TrafficSpoolConfig:
    root_dir: Path
    site_id: str
    camera_id: str
    write_thumbnails: bool = True
    thumb_pad: int = 20
    thumb_max_side: int = 320


class TrafficSpoolWriter:
    """
    Filesystem-first spool for traffic counting runs.

    The portal uploader can later read:
    - `run.json`
    - `events.jsonl`
    - thumbnails under `thumbs/`
    """

    def __init__(
        self,
        cfg: TrafficSpoolConfig,
        *,
        source: Dict[str, str],
        model_version: str,
        cfg_version: str,
        line_mode: str,
        line_id: str,
        fps: float,
        frame_size: Tuple[int, int],
        class_names: Optional[Dict[int, str]] = None,
        run_uid: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.run_uid = run_uid or uuid.uuid4().hex
        self.started_at_utc = _iso_utc(time.time())
        self._class_names = class_names or {}
        self._fps = float(fps) if fps else 0.0

        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.run_dir = cfg.root_dir / day / self.run_uid
        self.thumbs_dir = self.run_dir / "thumbs"
        _safe_mkdir(self.thumbs_dir)

        self._events_path = self.run_dir / "events.jsonl"
        self._events_f = self._events_path.open("a", encoding="utf-8")

        run_meta = {
            "run_uid": self.run_uid,
            "site_id": cfg.site_id,
            "camera_id": cfg.camera_id,
            "started_at_utc": self.started_at_utc,
            "source": source,
            "model_version": model_version,
            "cfg_version": cfg_version,
            "line_mode": line_mode,
            "line_id": line_id,
            "fps": self._fps,
            "frame_size": {"width": int(frame_size[0]), "height": int(frame_size[1])},
        }
        self._run_meta_path = self.run_dir / "run.json"
        self._run_meta: Dict[str, Any] = dict(run_meta)
        self._write_run_meta()

    def close(self) -> None:
        try:
            self._events_f.flush()
            self._events_f.close()
        except Exception:
            pass

    def update_run_metadata(self, updates: Mapping[str, Any]) -> None:
        if not isinstance(updates, Mapping):
            raise TypeError("updates must be a mapping")
        self._run_meta.update(dict(updates))
        self._write_run_meta()

    def record_events(
        self,
        events: Iterable[CrossingEvent],
        *,
        frame_bgr: np.ndarray,
        frame_ts: float,
    ) -> int:
        """
        Append events to events.jsonl and optionally write 1 thumbnail per event.
        """

        written = 0
        h, w = frame_bgr.shape[:2]

        for ev in events:
            event_uid = uuid.uuid4().hex

            class_name = None
            if ev.class_id is not None:
                class_name = self._class_names.get(int(ev.class_id))

            thumb_rel = None
            if self.cfg.write_thumbnails and ev.bbox_xyxy is not None:
                x1, y1, x2, y2 = ev.bbox_xyxy
                pad = max(int(self.cfg.thumb_pad), 0)
                clipped = _clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w=w, h=h)
                if clipped is not None:
                    cx1, cy1, cx2, cy2 = clipped
                    crop = frame_bgr[cy1:cy2, cx1:cx2]
                    crop = _resize_max(crop, int(self.cfg.thumb_max_side))
                    thumb_path = self.thumbs_dir / f"{event_uid}.jpg"
                    cv2.imwrite(str(thumb_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    thumb_rel = f"thumbs/{thumb_path.name}"

            rec = {
                "event_uid": event_uid,
                "run_uid": self.run_uid,
                "site_id": self.cfg.site_id,
                "camera_id": self.cfg.camera_id,
                "occurred_at_utc": _iso_utc(frame_ts),
                "frame_index": int(ev.frame_index),
                "video_time_s": (float(ev.frame_index) / self._fps) if self._fps > 0 else None,
                "line_mode": ev.line_mode,
                "direction": ev.direction,
                "track_id": int(ev.track_id),
                "class_id": int(ev.class_id) if ev.class_id is not None else None,
                "class_name": class_name,
                "confidence": float(ev.confidence) if ev.confidence is not None else None,
                "bbox": list(ev.bbox_xyxy) if ev.bbox_xyxy is not None else None,
                "thumb_relpath": thumb_rel,
            }
            self._events_f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            written += 1

        if written:
            self._events_f.flush()
        return written

    def _write_run_meta(self) -> None:
        self._run_meta_path.write_text(json.dumps(self._run_meta, indent=2), encoding="utf-8")
