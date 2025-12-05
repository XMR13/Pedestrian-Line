from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np

from .config import TrackerConfig
from .structures import Detection, Track


@dataclass
class Tracker:
    """
    Lightweight multi-object tracker.

    This is a simple greedy tracker:
    - Computes center points for detections and existing tracks.
    - Greedily matches detections to the nearest track within a distance
      threshold.
    - Creates new tracks for unmatched detections.
    - Removes tracks that have been lost for too many frames.
    """

    config: TrackerConfig
    _next_id: int = 1
    _tracks: Dict[int, Track] = field(default_factory=dict)

    def update(self, detections: Iterable[Detection], frame_index: int) -> List[Track]:
        detections = list(detections)

        if not detections and not self._tracks:
            return []

        det_centers = np.array([d.center() for d in detections], dtype=np.float32)
        track_ids = list(self._tracks.keys())
        track_centers = np.array(
            [self._tracks[tid].center() for tid in track_ids], dtype=np.float32
        )

        # If there are no existing tracks, initialize them all
        if track_centers.size == 0:
            for det in detections:
                self._add_track(det, frame_index)
            return list(self._tracks.values())

        # Compute pairwise distances between detections and tracks
        dists = self._pairwise_distances(det_centers, track_centers)

        # Greedy matching
        matched_dets = set()
        matched_tracks = set()

        flat_indices = np.argsort(dists, axis=None)
        for flat_idx in flat_indices:
            det_idx = flat_idx // dists.shape[1]
            track_idx = flat_idx % dists.shape[1]

            if det_idx in matched_dets or track_idx in matched_tracks:
                continue

            distance = dists[det_idx, track_idx]
            if distance > self.config.max_distance:
                continue

            det = detections[det_idx]
            track_id = track_ids[track_idx]
            self._update_track(track_id, det, frame_index)

            matched_dets.add(det_idx)
            matched_tracks.add(track_idx)

        # New tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                self._add_track(det, frame_index)

        # Remove stale tracks
        self._remove_lost_tracks(frame_index)

        return list(self._tracks.values())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between two sets of points.
        """

        if a.size == 0 or b.size == 0:
            return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)

        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    def _add_track(self, det: Detection, frame_index: int) -> None:
        track = Track(
            track_id=self._next_id,
            x1=det.x1,
            y1=det.y1,
            x2=det.x2,
            y2=det.y2,
            score=det.score,
            class_id=det.class_id,
            last_seen_frame=frame_index,
        )
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _update_track(self, track_id: int, det: Detection, frame_index: int) -> None:
        track = self._tracks[track_id]
        track.x1 = det.x1
        track.y1 = det.y1
        track.x2 = det.x2
        track.y2 = det.y2
        track.score = det.score
        track.class_id = det.class_id
        track.last_seen_frame = frame_index

    def _remove_lost_tracks(self, frame_index: int) -> None:
        to_remove = []
        for track_id, track in self._tracks.items():
            if frame_index - track.last_seen_frame > self.config.max_lost:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self._tracks[track_id]
