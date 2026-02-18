from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import TrackerConfig
from .structures import Detection, Track


@dataclass
class _ClassVoteState:
    counts: Dict[int, int] = field(default_factory=dict)
    weighted: Dict[int, float] = field(default_factory=dict)
    first_seen_seq: Dict[int, int] = field(default_factory=dict)
    classified_frames: int = 0


@dataclass
class Tracker:
    """
    Multi objek tracker

    Simmple greedy tracker (belum menggunakan tracker SOTA)

    This is a simple greedy tracker:
    - Menhgitung titik tengah untuk deteksi dan track yang ada
    - Secara greedy mencari deteksi ke track terdekat
    - Buat tracking terbaru
    - menghapus track yang ada 
    """

    config: TrackerConfig
    _next_id: int = 1
    _tracks: Dict[int, Track] = field(default_factory=dict)
    _next_display_id_by_class: Dict[int, int] = field(default_factory=dict)
    _last_center_by_id: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    _velocity_by_id: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    _class_vote_by_id: Dict[int, _ClassVoteState] = field(default_factory=dict)
    _class_vote_seq: int = 0

    def update(self, detections: Iterable[Detection], frame_index: int) -> List[Track]:
        detections = list(detections)

        if not detections and not self._tracks:
            return []

        det_centers = np.array([d.center() for d in detections], dtype=np.float32)
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

            scale_cap = max(int(self.config.max_distance_scale_cap), 1)
            scale = min(dt + 1, scale_cap)
            max_dists_list.append(float(self.config.max_distance) * float(scale))

        track_centers = np.array(track_centers_list, dtype=np.float32)
        max_dists = np.array(max_dists_list, dtype=np.float32)

        # If there are no existing tracks, initialize them all
        if track_centers.size == 0:
            for det in detections:
                self._add_track(det, frame_index)
            return list(self._tracks.values())

        # menhgitung jarakk antara deteksi dan tracknya
        
        dists = self._pairwise_distances(det_centers, track_centers)

        # mmatching secara greedy
        matched_dets = set()
        matched_tracks = set()

        flat_indices = np.argsort(dists, axis=None)
        for flat_idx in flat_indices:
            det_idx = flat_idx // dists.shape[1]
            track_idx = flat_idx % dists.shape[1]

            if det_idx in matched_dets or track_idx in matched_tracks:
                continue

            distance = float(dists[det_idx, track_idx])
            if distance > float(max_dists[track_idx]):
                continue

            det = detections[det_idx]
            track_id = track_ids[track_idx]
            self._update_track(track_id, det, frame_index)

            matched_dets.add(det_idx)
            matched_tracks.add(track_idx)

        # match terbaru untuk dedteksi yang error
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                self._add_track(det, frame_index)

        # Menghapus track yang diam
        self._remove_lost_tracks(frame_index)

        return list(self._tracks.values())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Menghitung pairwase euclidian distance dari 2 buah titik.
        """

        if a.size == 0 or b.size == 0:
            return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)

        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    def _add_track(self, det: Detection, frame_index: int) -> None:
        display_class_id = det.class_id
        display_id = None
        display_ids_by_class: Dict[int, int] = {}
        if det.class_id is not None:
            cls_key = int(det.class_id)
            next_disp = self._next_display_id_by_class.get(cls_key, 1)
            self._next_display_id_by_class[cls_key] = next_disp + 1
            display_id = next_disp
            display_ids_by_class[cls_key] = next_disp

        track = Track(
            track_id=self._next_id,
            x1=det.x1,
            y1=det.y1,
            x2=det.x2,
            y2=det.y2,
            score=det.score,
            class_id=det.class_id,
            last_seen_frame=frame_index,
            display_class_id=display_class_id,
            display_id=display_id,
            display_ids_by_class=display_ids_by_class,
        )
        self._update_stable_class(track, det.class_id, det.score)
        self._tracks[self._next_id] = track
        cx, cy = det.center()
        self._last_center_by_id[self._next_id] = (float(cx), float(cy))
        self._velocity_by_id[self._next_id] = (0.0, 0.0)
        self._next_id += 1

    #update track
    def _update_track(self, track_id: int, det: Detection, frame_index: int) -> None:
        """
        Update tracking of an object, track the object by the center of the box itelsf
        and then subtract them by the previous tracking state
        """
        track = self._tracks[track_id]
        track.x1 = det.x1
        track.y1 = det.y1
        track.x2 = det.x2
        track.y2 = det.y2
        track.score = det.score
        track.class_id = det.class_id
        self._update_stable_class(track, det.class_id, det.score)
        track.last_seen_frame = frame_index

        if det.class_id is not None:
            cls_key = int(det.class_id)
            if cls_key not in track.display_ids_by_class:
                next_disp = self._next_display_id_by_class.get(cls_key, 1)
                self._next_display_id_by_class[cls_key] = next_disp + 1
                track.display_ids_by_class[cls_key] = next_disp
            track.display_class_id = det.class_id
            track.display_id = track.display_ids_by_class[cls_key]

        cx, cy = det.center()
        prev = self._last_center_by_id.get(track_id)
        if prev is not None:
            self._velocity_by_id[track_id] = (float(cx - prev[0]), float(cy - prev[1]))
        self._last_center_by_id[track_id] = (float(cx), float(cy))

    def _remove_lost_tracks(self, frame_index: int) -> None:
        to_remove = []
        for track_id, track in self._tracks.items():
            if frame_index - track.last_seen_frame > self.config.max_lost:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self._tracks[track_id]
            self._last_center_by_id.pop(track_id, None)
            self._velocity_by_id.pop(track_id, None)
            self._class_vote_by_id.pop(track_id, None)

    def clear_runtime_state(self) -> None:
        """
        Clear transient tracking buffers after reconnect.

        Intentionally keeps `_next_id` monotonic so track IDs are not reused
        within the same process/session.
        """
        self._tracks.clear()
        self._last_center_by_id.clear()
        self._velocity_by_id.clear()
        self._next_display_id_by_class.clear()
        self._class_vote_by_id.clear()

    def _update_stable_class(
        self,
        track: Track,
        class_id: Optional[int],
        score: Optional[float],
    ) -> None:
        if class_id is None:
            return

        cid = int(class_id)
        mode = self._normalized_class_vote_mode()
        if mode == "instant":
            track.stable_class_id = cid
            return

        state = self._class_vote_by_id.get(track.track_id)
        if state is None:
            state = _ClassVoteState()
            self._class_vote_by_id[track.track_id] = state

        if cid not in state.first_seen_seq:
            state.first_seen_seq[cid] = self._class_vote_seq
            self._class_vote_seq += 1

        state.classified_frames += 1
        state.counts[cid] = state.counts.get(cid, 0) + 1
        weight = float(score) if score is not None else 1.0
        state.weighted[cid] = state.weighted.get(cid, 0.0) + max(weight, 0.0)

        min_frames = max(int(self.config.class_vote_min_frames), 1)
        if state.classified_frames < min_frames:
            track.stable_class_id = track.stable_class_id if track.stable_class_id is not None else cid
            return

        metric = state.weighted if mode == "weighted_majority" else {k: float(v) for k, v in state.counts.items()}
        best, best_value, second_value = self._pick_vote_winner(metric, state.first_seen_seq)
        if best is None:
            track.stable_class_id = track.stable_class_id if track.stable_class_id is not None else cid
            return

        ratio = float(self.config.class_vote_ambiguity_ratio)
        is_ambiguous = (
            ratio > 1.0
            and second_value > 0.0
            and (best_value / second_value) < ratio
        )
        if is_ambiguous and track.stable_class_id is not None:
            return

        track.stable_class_id = best

    @staticmethod
    def _pick_vote_winner(
        metric: Dict[int, float],
        first_seen_seq: Dict[int, int],
    ) -> Tuple[Optional[int], float, float]:
        if not metric:
            return None, 0.0, 0.0

        ranked = sorted(
            ((int(cid), float(val)) for cid, val in metric.items()),
            key=lambda item: (-item[1], first_seen_seq.get(item[0], 10**9), item[0]),
        )
        best_cid, best_value = ranked[0]
        second_value = ranked[1][1] if len(ranked) > 1 else 0.0
        return int(best_cid), float(best_value), float(second_value)

    def _normalized_class_vote_mode(self) -> str:
        mode = str(self.config.class_vote_mode).strip().lower()
        if mode in {"instant", "majority", "weighted_majority"}:
            return mode
        return "weighted_majority"
