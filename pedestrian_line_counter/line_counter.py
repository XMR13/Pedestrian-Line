from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np

from .structures import Track


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
    _track_sides: Dict[int, int] = field(default_factory=dict)

    def update(self, tracks: Iterable[Track]) -> None:
        """
        Update angkanya berdasarkan jumlah hasil tracking yanga ada.
        """

        current_ids = set()

        for track in tracks:
            current_ids.add(track.track_id)

            px, py = track.bottom_center()
            side = self._point_side(px, py)
            if side == 0:
                continue

            prev_side = self._track_sides.get(track.track_id)
            if prev_side is not None and prev_side != 0 and side != prev_side:
                # Crossing detected
                if prev_side < 0 < side:
                    self.count_a_to_b += 1
                elif prev_side > 0 > side:
                    self.count_b_to_a += 1

            self._track_sides[track.track_id] = side

        # Bersihkan state untuk track yang menghilang
        for tid in list(self._track_sides.keys()):
            if tid not in current_ids:
                del self._track_sides[tid]

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
