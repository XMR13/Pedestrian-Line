from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class TrackerConfig:
    # Default values tuned for fast vehicles and brief occlusions (matches the main project defaults).
    max_distance: float = 80.0
    max_lost: int = 45
    max_distance_scale_cap: int = 5


@dataclass
class Track:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: Optional[int]
    last_seen_frame: int

    def center(self) -> Tuple[float, float]:
        return (float((self.x1 + self.x2) / 2.0), float((self.y1 + self.y2) / 2.0))


@dataclass
class Tracker:
    """
    Simple greedy multi-object tracker.

    This is intentionally lightweight: it exists to support candidate-gated
    frame sampling for auto-labeling workflows (not to be a SOTA tracker).
    """

    config: TrackerConfig = field(default_factory=TrackerConfig)
    _next_id: int = 1
    _tracks: Dict[int, Track] = field(default_factory=dict)
    _last_center_by_id: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    _velocity_by_id: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    def update(self, detections: Iterable[object], frame_index: int) -> List[Track]:
        dets = list(detections)

        if not dets and not self._tracks:
            return []

        det_centers = np.array([_center(d) for d in dets], dtype=np.float32)
        track_ids = list(self._tracks.keys())

        track_centers_list: List[Tuple[float, float]] = []
        max_dists_list: List[float] = []
        for tid in track_ids:
            track = self._tracks[tid]
            cx, cy = track.center()
            dt = max(int(frame_index - track.last_seen_frame), 0)
            vx, vy = self._velocity_by_id.get(tid, (0.0, 0.0))
            px = float(cx + vx * dt)
            py = float(cy + vy * dt)
            track_centers_list.append((px, py))

            scale = min(dt + 1, max(int(self.config.max_distance_scale_cap), 1))
            max_dists_list.append(float(self.config.max_distance) * float(scale))

        if not track_centers_list:
            for det in dets:
                self._add_track(det, frame_index)
            return list(self._tracks.values())

        track_centers = np.array(track_centers_list, dtype=np.float32)
        max_dists = np.array(max_dists_list, dtype=np.float32)
        dists = _pairwise_distances(det_centers, track_centers)

        matched_dets = set()
        matched_tracks = set()
        flat_indices = np.argsort(dists, axis=None)
        for flat_idx in flat_indices:
            det_idx = int(flat_idx // dists.shape[1])
            track_idx = int(flat_idx % dists.shape[1])
            if det_idx in matched_dets or track_idx in matched_tracks:
                continue
            distance = float(dists[det_idx, track_idx])
            if distance > float(max_dists[track_idx]):
                continue
            det = dets[det_idx]
            track_id = track_ids[track_idx]
            self._update_track(track_id, det, frame_index)
            matched_dets.add(det_idx)
            matched_tracks.add(track_idx)

        for det_idx, det in enumerate(dets):
            if det_idx not in matched_dets:
                self._add_track(det, frame_index)

        self._remove_lost_tracks(frame_index)
        return list(self._tracks.values())

    def _add_track(self, det: object, frame_index: int) -> None:
        x1, y1, x2, y2 = _as_xyxy(det)
        class_id = _class_id(det)
        score = _score(det)
        track = Track(
            track_id=self._next_id,
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            score=float(score),
            class_id=class_id,
            last_seen_frame=int(frame_index),
        )
        self._tracks[self._next_id] = track
        cx, cy = track.center()
        self._last_center_by_id[self._next_id] = (float(cx), float(cy))
        self._velocity_by_id[self._next_id] = (0.0, 0.0)
        self._next_id += 1

    def _update_track(self, track_id: int, det: object, frame_index: int) -> None:
        track = self._tracks[track_id]
        x1, y1, x2, y2 = _as_xyxy(det)
        track.x1 = float(x1)
        track.y1 = float(y1)
        track.x2 = float(x2)
        track.y2 = float(y2)
        track.score = float(_score(det))
        track.class_id = _class_id(det)
        track.last_seen_frame = int(frame_index)

        cx, cy = track.center()
        prev = self._last_center_by_id.get(track_id)
        if prev is not None:
            self._velocity_by_id[track_id] = (float(cx - prev[0]), float(cy - prev[1]))
        self._last_center_by_id[track_id] = (float(cx), float(cy))

    def _remove_lost_tracks(self, frame_index: int) -> None:
        to_remove = []
        for track_id, track in self._tracks.items():
            if frame_index - track.last_seen_frame > int(self.config.max_lost):
                to_remove.append(track_id)
        for track_id in to_remove:
            del self._tracks[track_id]
            self._last_center_by_id.pop(track_id, None)
            self._velocity_by_id.pop(track_id, None)


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _as_xyxy(det: object) -> Tuple[float, float, float, float]:
    if hasattr(det, "as_xyxy") and callable(getattr(det, "as_xyxy")):
        x1, y1, x2, y2 = det.as_xyxy()
        return float(x1), float(y1), float(x2), float(y2)
    # Fallback for Detection-like objects.
    return float(det.x1), float(det.y1), float(det.x2), float(det.y2)


def _center(det: object) -> Tuple[float, float]:
    x1, y1, x2, y2 = _as_xyxy(det)
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def _class_id(det: object) -> Optional[int]:
    cid = getattr(det, "class_id", None)
    if cid is None:
        return None
    try:
        return int(cid)
    except Exception:
        return None


def _score(det: object) -> float:
    s = getattr(det, "score", 0.0)
    try:
        return float(s)
    except Exception:
        return 0.0
