from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, Optional, List

import numpy as np

from .structures import Track


@dataclass
class _TrackState:
    """
    Internal per-track state used by LineCounter to make direction decisions
    more robust over a short time window.
    """

    last_side: int = 0
    last_point: Optional[Tuple[float, float]] = None
    history: List[Tuple[float, float, int]] = field(default_factory=list)
    has_counted: bool = False

    # Crossing candidate
    crossing_active: bool = False
    crossing_prev_side: int = 0
    crossing_new_side: int = 0
    crossing_pre_points: List[Tuple[float, float]] = field(default_factory=list)
    crossing_post_points: List[Tuple[float, float]] = field(default_factory=list)
    frames_on_new_side: int = 0


@dataclass
class LineCounter:
    """
    Line virtual yang menghitung dan mentrack objek yang melewati garis tersebut
    dari dua arah sekaligus.

    Semantik area
    - Ketika sebuah objek bergerak dari area 'negatif' garis ke area 'positif' garis, maka naikkan variabel a_to_b
    - jika sebaliknya, bergerak dari bagian 'positif' ke arae negatif' maka' maka tambahkan variabel a_to_b
    
    Virtual line that counts tracked objects crossing from one side to the other.

    Areanya ditentukan dengan perkalian 2d cross product oleh garis p1->-2 dengan vektor dari 
    titik p1 ke point

    """

    #tiik koordinat
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    count_a_to_b: int = 0
    count_b_to_a: int = 0
    count_by_class_dir: Dict[str, Dict[int, int]] = field(
        default_factory=lambda: {"a_to_b": {}, "b_to_a": {}}
    )

    # Hyperparameters for robustness
    history_window: int = 12
    pre_window: int = 4
    min_pre_points: int = 1
    min_post_frames: int = 3
    min_along_distance: float = 8.0

    _tracks: Dict[int, _TrackState] = field(default_factory=dict)

    def update(self, tracks: Iterable[Track]) -> None:
        """
        Update angkanya berdasarkan jumlah hasil tracking yanga ada.
        """

        current_ids = set()

        line_vec = np.array([float(self.p2[0] - self.p1[0]), float(self.p2[1] - self.p1[1])], dtype=np.float32)
        norm = float(np.linalg.norm(line_vec))
        if norm > 0:
            line_dir = line_vec / norm
        else:
            line_dir = np.array([1.0, 0.0], dtype=np.float32)

        for track in tracks:
            tid = track.track_id
            current_ids.add(tid)

            px, py = track.bottom_center()
            side = self._point_side(px, py)

            state = self._tracks.get(tid)
            if state is None:
                state = _TrackState()
                self._tracks[tid] = state

            # Append to history (x, y, side)
            state.history.append((px, py, side))
            if len(state.history) > self.history_window:
                state.history.pop(0)

            prev_side = state.last_side
            state.last_side = side
            state.last_point = (px, py)

            if state.has_counted:
                # sudah menghttung sekali, meke sure untuk skup perhitunan sebelumnya
                continue

            if side == 0:
                # jika sudah di garis, jangan trigger crossing terlebih dahyly
                continue

            if prev_side == 0:
                # First time we have a non-zero side for this track.
                continue
            
            #pengecekan sign flip, atau mulaui mencari candidate crossignya
            if side != prev_side:
                self._start_crossing_candidate(state, prev_side, side, px, py)
            elif state.crossing_active and side == state.crossing_new_side:
                # Lanjut menambahkan cross relation points.
                state.crossing_post_points.append((px, py))
                state.frames_on_new_side += 1
                if state.frames_on_new_side >= self.min_post_frames:
                    self._finalise_crossing(state, line_dir, track.class_id)
            elif state.crossing_active and side == state.crossing_prev_side:
                # Kembali ke original statenya.
                state.crossing_active = False
                state.crossing_pre_points.clear()
                state.crossing_post_points.clear()
                state.frames_on_new_side = 0

        # Bersihkan state untuk track yang menghilang
        for tid in list(self._tracks.keys()):
            if tid not in current_ids:
                self._tracks.pop(tid, None)

    def _start_crossing_candidate(
        self,
        state: _TrackState,
        prev_side: int,
        new_side: int,
        px: float,
        py: float,
    ) -> None:
        """
        Initialize or refresh a crossing candidate when a sign flip is observed.
        """

        state.crossing_active = True
        state.crossing_prev_side = prev_side
        state.crossing_new_side = new_side

        # Gunakan last point di bagian sebelumnya sebagai pre _cross history.
        pre_points: List[Tuple[float, float]] = [
            (x, y) for (x, y, s) in state.history if s == prev_side
        ]
        if pre_points:
            state.crossing_pre_points = pre_points[-self.pre_window :]
        else:
            state.crossing_pre_points = []

        state.crossing_post_points = [(px, py)]
        state.frames_on_new_side = 1

    def _finalise_crossing(
        self,
        state: _TrackState,
        line_dir: np.ndarray,
        class_id: Optional[int],
    ) -> None:
        """
        Menentukan arah lewatnya berdasarkan pergerakan di garis pembatas.
        """

        if state.has_counted:
            state.crossing_active = False
            return

        if len(state.crossing_pre_points) < self.min_pre_points:
            # Not enough run-up on the entry side; treat as ambiguous.
            state.crossing_active = False
            state.crossing_pre_points.clear()
            state.crossing_post_points.clear()
            state.frames_on_new_side = 0
            return

        if not state.crossing_post_points:
            state.crossing_active = False
            return

        pre_avg = np.mean(np.array(state.crossing_pre_points, dtype=np.float32), axis=0)
        post_avg = np.mean(np.array(state.crossing_post_points, dtype=np.float32), axis=0)
        delta = post_avg - pre_avg
        along = float(delta[0] * line_dir[0] + delta[1] * line_dir[1])

        if abs(along) < self.min_along_distance:
            # Pergerakan yagn noi di igonre saja
            state.crossing_active = False
            state.crossing_pre_points.clear()
            state.crossing_post_points.clear()
            state.frames_on_new_side = 0
            return

        if along > 0:
            self.count_a_to_b += 1
            self._bump_class_count("a_to_b", class_id)
        else:
            self.count_b_to_a += 1
            self._bump_class_count("b_to_a", class_id)

        state.has_counted = True
        state.crossing_active = False
        state.crossing_pre_points.clear()
        state.crossing_post_points.clear()
        state.frames_on_new_side = 0

    def _point_side(self, px: float, py: float) -> int:
        """
        Mengembalikan -1, 0 atau +1 tergantung dari mana sudut garisnya berada.
        """

        x1, y1 = self.p1
        x2, y2 = self.p2

        vx1 = x2 - x1
        vy1 = y2 - y1
        vx2 = px - x1
        vy2 = py - y1

        cross = vx1 * vy2 - vy1 * vx2
        if cross > 0:
            return 1
        if cross < 0:
            return -1
        return 0

    def _bump_class_count(self, direction: str, class_id: Optional[int]) -> None:
        if class_id is None:
            return
        if direction not in self.count_by_class_dir:
            self.count_by_class_dir[direction] = {}
        counts = self.count_by_class_dir[direction]
        counts[class_id] = counts.get(class_id, 0) + 1
