Pedestrian / Vehicle Line Counter
================================

This project processes a video of a road and counts how many
**people/vehicles** cross a virtual line in each direction. It is designed
to be:

- Simple and reusable.
- Configurable (no hard‑coded absolute paths).
- Friendly to commercial use (permissive dependencies).

The core pipeline is:

> video → detect objects → track objects → count line crossings →
> write annotated output video + print counts


Architecture (High Level)
-------------------------

- `config.py` – configuration dataclasses for the whole app.
- `detector.py` – ONNXRuntime‑based YOLO‑style detector wrapper, plus a
  motion‑based fallback.
- `tracker.py` – lightweight greedy multi‑object tracker.
- `line_counter.py` – virtual line + direction logic (A→B and B→A).
- `draw_utils.py` – drawing helpers for tracks and counters.
- Ignore regions and noise filtering are built into the detector:
  - `ModelConfig.ignore_regions` (normalized rectangles) drops detections whose
    center lies in static foliage/occlusion zones (defaults target the left
    banana leaves and top‑right leaf).
  - `ModelConfig.min_box_area_ratio` filters out tiny boxes.
- `line_picker.py` – interactive GUI to pick line(s) and save them to JSON.
- `main.py` – CLI entry point that ties everything together.
- `structures.py` – small data classes for `Detection` and `Track`.
- `media/` – local input/output videos (ignored by Git).
- `Models/` – local ONNX model weights (ignored by Git).
- `Progress/` – session logs that track work against `plan.md`.


Dependencies & Setup
--------------------

Requirements:

- Python 3.8+.
- `uv` as the Python package manager.

Install dependencies:

```bash
uv sync
```

Key runtime dependencies (from `pyproject.toml`):

- `opencv-python`
- `numpy`
- `onnxruntime` (CPU)
- `onnxruntime-gpu` (optional, if you have a compatible GPU)


Model & Data Layout
-------------------

The repository does **not** store large binaries (models or videos) in Git.
You must place them locally:

- ONNX model file:
  - Expected: `Models/yolov9-c.onnx`
  - You can change the path via `--model` or by editing `ModelConfig` in
    `config.py`.
- Input videos:
  - Default: a sample file under `media/` (see `IOConfig` in `config.py`).
  - You can override via `--input`.
- Output videos:
  - Default: `output.mp4` in the project root (see `IOConfig`).
  - You can override via `--output`.


Basic Usage
-----------

From WSL or a shell:

```bash
cd "/mnt/d/RZQ/Coding/Python/Projects/Pedestrian Line"
```

Run with ONNX detector and explicit model path:

```bash
uv run python main.py \
  --backend onnx \
  --model Models/yolov9-c.onnx \
  --input media/input.mp4 \
  --output media/output_test.mp4 \
  --show
```

What this does:

- Loads the ONNX model with `onnxruntime` (GPU if available, else CPU).
- Runs detection → tracking → line counting for each frame.
- Draws bounding boxes, the virtual line, and live A→B / B→A counts.
- Saves an annotated video to `media/output_test.mp4`.
- Prints final totals when finished.


Picking the Counting Line Interactively
---------------------------------------

You can define the line once using `line_picker.py` and then reuse it via
JSON.

1. Pick the line and save it:

```bash
uv run python line_picker.py \
  --input media/input.mp4 \
  --lines 1 \
  --save line.json
```

Controls:

- Left click: add points.
- `R` or `C`: reset points.
- `Enter` / `Space`: accept when required points are placed.
- `Esc` / `Q`: cancel.

2. Run the main app with that line:

```bash
uv run python main.py \
  --backend onnx \
  --model Models/yolov9-c.onnx \
  --input media/input.mp4 \
  --output media/output_with_line.mp4 \
  --line-json line.json \
  --show
```


Motion‑Only Backend (No Model)
------------------------------

For quick debugging or if you don’t have an ONNX model yet, you can use
the motion‑based backend:

```bash
uv run python main.py \
  --backend motion \
  --input media/input.mp4 \
  --output media/output_motion.mp4 \
  --show
```

This uses background subtraction instead of a learned detector, so it
won’t classify objects but still gives you a feel for the tracking and
counting behaviour.


Progress Tracking
-----------------

Development progress is tracked in:

- `plan.md` – high‑level architecture and roadmap (kept in sync with the
  actual implementation).
- `Progress/` – per‑session Markdown logs that record:
  - What changed (files added/modified/deleted).
  - How it maps to the plan.
  - Next steps.

When you make significant changes, add or update a session file in
`Progress/` following the instructions in `Progress/README.md`.
