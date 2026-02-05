# Multi‑Direction Line Counting – How It Works

This document explains how the virtual line counting logic works in this
project and how the system distinguishes objects moving in opposite
directions (e.g. left → right vs right → left along the road).

The implementation lives in:

- `tracker.py` – maintains stable track IDs over time.
- `structures.py` – data classes for `Detection` and `Track`.
- `line_counter.py` – core direction and per‑class logic.
- `draw_utils.py` – draws the line and live counts.
- `main.py` – wires detector, tracker, and line counter into the per‑frame loop.

The key ideas:

- The detector outputs bounding boxes for the **target vehicle subclasses** defined by your model (custom taxonomy).
- The tracker keeps a **stable `track_id`** for each object as it moves.
- The line is treated as an **oriented segment**; we use the sign of a 2D cross
  product to know on which side of the line an object is.
- When the side for a given track changes from negative → positive or
  positive → negative, we count a crossing in a specific direction and update
  per‑class (vehicle subclass) stats.

---

## 1. High‑Level Pipeline

On each video frame, the main loop does roughly:

1. **Detection** – `Detector.detect(frame)` returns a list of `Detection` objects.
2. **Tracking** – `Tracker.update(detections, frame_index)` associates detections
   to existing tracks (and spawns/removes tracks as needed), returning a list of
   `Track` objects with stable `track_id`s.
3. **Line counting** – `LineCounter.update(tracks)` inspects each track’s
   bottom‑center point relative to the oriented line and updates:
   - total direction counts (`count_a_to_b`, `count_b_to_a`),
   - per‑direction, per‑class counts (`count_by_class_dir`).
4. **Drawing** – `draw_tracks` and `draw_line_and_counts` overlay boxes, the
   line, and totals on the frame.

Everything in this document focuses on steps 2–3.

---

## 2. Oriented Line Definition

The line is defined by two pixel points:

- `p1 = (x1, y1)` – start point
- `p2 = (x2, y2)` – end point

The orientation is from `p1 → p2`. This orientation defines which way is
considered **A→B** vs **B→A** in the counters.

In code (`LineCounter`):

- `p1` and `p2` are stored in `line_counter.p1` / `line_counter.p2`.
- The counts are:
- `count_a_to_b` – crossings from the **A side** of the line to the **B side**.
- `count_b_to_a` – crossings from the **B side** of the line back to the
  **A side**.

The A/B *direction* is defined by motion along the oriented line:

- A→B means the object moves roughly from `p1` towards `p2` (e.g. left → right
  in your annotated camera).
- B→A means the object moves roughly from `p2` back towards `p1` (right → left).

We still use a cross‑product sign change to detect “has crossed the line?”,
but the **direction** of that crossing (A→B vs B→A) is decided by projecting
the track’s motion onto the line direction (dot product with `p2 - p1`).

The actual mapping to “left” vs “right” on the screen depends on how you
place `p1` and `p2` (explained later).

---

## 3. Stable Track IDs from the Tracker

Each raw detection is a `Detection(x1, y1, x2, y2, score, class_id)`. The
tracker turns these into persistent tracks:

- Each active object is represented as a `Track` with:
  - `track_id` – stable integer ID,
  - bounding box `(x1, y1, x2, y2)`,
  - `score`, `class_id`,
  - `last_seen_frame`.
- The tracker computes the center of each detection and each existing track.
- It builds a distance matrix of detection‑to‑track distances and greedily matches
  the closest pairs under a configurable `max_distance`.
- Matched tracks update their box, class, and timestamp.
- Unmatched detections start new tracks with new IDs.
- Tracks that are not matched for more than `max_lost` frames are removed.

Because the line counter works on **tracks**, not raw detections:

- An object’s motion is described as a sequence of positions for a single `track_id`.
- The counting logic sees a smooth path for each object over time, which makes
  crossing detection robust to single‑frame noise.

---

## 4. Which Point of the Object Is Used?

Each tracked object is represented by a bounding box `(x1, y1, x2, y2)`.

To approximate where the object touches the ground, `LineCounter` uses the
**bottom center** of that box:

```python
px, py = track.bottom_center()
```

This makes direction detection more stable on a sloped road: as vehicles
move along the road, their bottom center crosses the line roughly where
their wheels touch the ground.

---

## 5. Side Test Using Cross Product

For any point `P = (px, py)` (the point we use for an object), we compute
which side of the line it lies on using a 2D cross product:

```text
v_line  = p2 - p1 = (x2 - x1, y2 - y1)
v_point = P  - p1 = (px - x1, py - y1)

cross = v_line.x * v_point.y - v_line.y * v_point.x
```

In `line_counter.py` this is implemented in `_point_side` and returns:

- `+1` → point is on the **positive** side of the line.
- `-1` → point is on the **negative** side.
- `0` → point is exactly on the line (or numerically very close).

The coordinate system is the usual image one:

- `x` increases to the right.
- `y` increases downward.

So the “positive” vs “negative” side is purely a geometric property of the
oriented line and the point location.

---

## 6. Detecting Crossings and Direction

For each frame:

1. The tracker updates all active tracks (`Tracker.update`).
2. `LineCounter.update(tracks)` iterates over all current tracks.
3. For each track:
   - Compute its bottom‑center point `P`.
   - Compute `side = _point_side(P)`, giving `-1`, `0`, or `+1`.
   - Look up the previous side for this track ID in the dictionary
     `_track_sides`.
   - If the side changed sign (from `-1` to `+1` or `+1` to `-1`), count
     a crossing in the appropriate direction.

In code (simplified):

```python
prev_side = self._track_sides.get(track_id)
if prev_side is not None and prev_side != 0 and side != prev_side:
    if prev_side < 0 < side:
        self.count_a_to_b += 1
        self._bump_class_count("a_to_b", track.class_id)
    elif prev_side > 0 > side:
        self.count_b_to_a += 1
        self._bump_class_count("b_to_a", track.class_id)

self._track_sides[track_id] = side
```

Interpretation:

- Moving from negative side → positive side:
  - Increment `count_a_to_b` (A→B).
- Moving from positive side → negative side:
  - Increment `count_b_to_a` (B→A).

Because this is based on the **sign change**, an object is only counted
once when it crosses the line, even if it jitters back and forth slightly
due to detection noise.

---

## 7. Per‑Direction, Per‑Class Counts

In addition to global direction totals, the counter tracks **which vehicle subclasses**
cross in each direction (based on your model’s `class_id` taxonomy).

Internal structure:

- `count_by_class_dir: Dict[str, Dict[int, int]]`
  - `count_by_class_dir["a_to_b"][class_id]` → how many times this class
    crossed from A to B.
  - `count_by_class_dir["b_to_a"][class_id]` → how many times this class
    crossed from B to A.

When a crossing is detected:

- `_bump_class_count("a_to_b", track.class_id)` or
  `_bump_class_count("b_to_a", track.class_id)` increments the appropriate
  bucket, as long as `class_id` is not `None`.

The drawing code uses COCO class names (`COCO_NAMES`) to render a compact
summary in the overlay:

- Top N classes for A→B:
  - e.g. `A->B top: person 10 | car 8 | motorcycle 3`
- Top N classes for B→A:
  - e.g. `B->A top: person 7 | car 5 | bus 2`

This makes it easy to see not only **how many** objects cross in each
direction, but also **what kind** of objects dominate.

---

## 8. Relating A→B / B→A to “Left” vs “Right”

The counters themselves are abstract (“negative side” vs “positive side”).
To make them meaningful (e.g. “left → right” vs “right → left”), you
control the mapping when you place the line:

- When picking the line (either via `line_picker.py` or the interactive
  picker in `main.py`):
  - Choose `p1` and `p2` so that the direction from `p1` to `p2` matches
    the direction you want to call **A→B**.
  - Example for your camera:
    - Click `p1` near the lower‑left side of the road.
    - Click `p2` further along the road towards the upper‑right.
    - Then:
      - Vehicles moving from left/bottom → right/top will primarily go from
        the negative side to the positive side ⇒ counted as A→B.
      - Vehicles moving in the opposite direction will primarily go from
        positive to negative ⇒ counted as B→A.

If you ever see the counts “flipped” relative to what you expect, you can
just swap the order of the points (click the line in the opposite order),
or conceptually swap how you interpret A/B in your UI.

---

## 9. Handling Occlusion and Stuck Boxes

The tracker may keep a track alive for a short time when an object is
occluded (e.g. behind banana leaves). To avoid “ghost” boxes and spurious
crossings:

- Each track stores `last_seen_frame`.
- Drawing (`draw_utils.draw_tracks`) only shows tracks that were updated on
  the current frame, so boxes do not remain visually stuck when the object
  is gone.
- `LineCounter.update` only considers the current active tracks, and it
  cleans up state for tracks that disappear.

Because crossings are based on sign changes across time, an object that is
temporarily hidden but stays on the same side of the line will not create
extra counts.

---

## 10. Summary

- Detector → tracker → line counter is the pipeline:
  - detections become stable `Track` objects,
  - tracks feed into geometric side tests around an oriented line.
- The line is an oriented segment `p1 → p2`.
- Each tracked object is reduced to a bottom‑center point.
- A 2D cross product decides whether the point is on the negative or
  positive side of the line.
- When a track’s side changes sign, a single crossing is counted:
  - negative → positive ⇒ `A→B`
  - positive → negative ⇒ `B→A`
- Per‑direction, per‑class counters capture which categories (person, car,
  motorcycle, bus, truck…) dominate each direction.
- Which physical direction (“left/right” or “in/out”) corresponds to A→B
  vs B→A is fully controlled by how you place and orient the line when you
  click `p1` and `p2`.
