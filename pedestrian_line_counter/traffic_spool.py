from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
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
    write_scene_thumbnails: bool = True
    thumb_pad: int = 20
    thumb_max_side: int = 320
    scene_thumb_max_side: int = 640
    scene_thumb_quality: int = 85


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
        lines: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
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
        self._lines = lines or []

        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.run_dir = cfg.root_dir / day / self.run_uid
        self.thumbs_dir = self.run_dir / "thumbs"
        self.scene_dir = self.run_dir / "scene"
        _safe_mkdir(self.thumbs_dir)
        _safe_mkdir(self.scene_dir)

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
            "lines": [
                {"p1": [int(p1[0]), int(p1[1])], "p2": [int(p2[0]), int(p2[1])]}
                for (p1, p2) in self._lines
            ],
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
        occurred_at_ts: float,
        occurred_at_utc_source: str,
        capture_records: Optional[List[Dict[str, Any]]] = None,
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

            scene_rel = None
            if self.cfg.write_scene_thumbnails:
                scene_img = self._build_scene_thumbnail(frame_bgr, ev)
                if scene_img is not None:
                    scene_path = self.scene_dir / f"{event_uid}.jpg"
                    q = int(self.cfg.scene_thumb_quality)
                    q = max(10, min(100, q))
                    cv2.imwrite(str(scene_path), scene_img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                    scene_rel = f"scene/{scene_path.name}"

            rec = {
                "event_uid": event_uid,
                "run_uid": self.run_uid,
                "site_id": self.cfg.site_id,
                "camera_id": self.cfg.camera_id,
                "occurred_at_utc": _iso_utc(float(occurred_at_ts)),
                "occurred_at_utc_source": str(occurred_at_utc_source),
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
                "scene_relpath": scene_rel,
            }
            self._events_f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            if capture_records is not None:
                capture_records.append(rec)
            written += 1

        if written:
            self._events_f.flush()
        return written

    def _write_run_meta(self) -> None:
        self._run_meta_path.write_text(json.dumps(self._run_meta, indent=2), encoding="utf-8")

    def _build_scene_thumbnail(self, frame_bgr: np.ndarray, ev: CrossingEvent) -> Optional[np.ndarray]:
        """
        Create a small full-frame evidence thumbnail that shows:
        - counting line(s)
        - event bbox
        - direction + class label
        """

        if frame_bgr is None:
            return None

        max_side = int(self.cfg.scene_thumb_max_side)
        if max_side <= 0:
            return None

        h0, w0 = frame_bgr.shape[:2]
        m = max(h0, w0)
        if m <= 0:
            return None

        scale = 1.0
        out = frame_bgr
        if m > max_side:
            scale = float(max_side) / float(m)
            nh = max(1, int(round(h0 * scale)))
            nw = max(1, int(round(w0 * scale)))
            out = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

        def sc_pt(pt: Tuple[int, int]) -> Tuple[int, int]:
            return (int(round(pt[0] * scale)), int(round(pt[1] * scale)))

        def sc_box(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
            x1, y1, x2, y2 = box
            return (
                int(round(x1 * scale)),
                int(round(y1 * scale)),
                int(round(x2 * scale)),
                int(round(y2 * scale)),
            )

        # Draw lines
        for idx, (p1, p2) in enumerate(self._lines):
            c = (0, 255, 255) if idx == 0 else (255, 255, 0)
            cv2.line(out, sc_pt(p1), sc_pt(p2), c, 2)

        # Draw bbox
        if ev.bbox_xyxy is not None:
            x1, y1, x2, y2 = sc_box(ev.bbox_xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)

        # Label block
        cls = "unknown"
        if ev.class_id is not None:
            cls = self._class_names.get(int(ev.class_id)) or f"class_{int(ev.class_id)}"
        label = f"{cls} | {ev.direction} | t={int(ev.track_id)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x, y = 10, 18
        cv2.rectangle(out, (x - 6, y - th - 6), (x + tw + 6, y + 6), (0, 0, 0), -1)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        return out
