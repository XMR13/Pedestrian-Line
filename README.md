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

Core application code now lives under the `pedestrian_line_counter/`
package:

- `pedestrian_line_counter/config.py` – configuration dataclasses for the whole app.
- `pedestrian_line_counter/detector.py` – ONNXRuntime‑based YOLO‑style detector wrapper, plus a
  motion‑based fallback.
- `pedestrian_line_counter/tracker.py` – lightweight greedy multi‑object tracker.
- `pedestrian_line_counter/line_counter.py` – virtual line + direction logic (A→B and B→A).
- `pedestrian_line_counter/draw_utils.py` – drawing helpers for tracks and counters.
- Ignore regions and noise filtering are built into the detector:
  - `ModelConfig.ignore_regions` (normalized rectangles) drops detections whose
    center lies in static foliage/occlusion zones (defaults target the left
    banana leaves and top‑right leaf).
  - `ModelConfig.min_box_area_ratio` filters out tiny boxes.
- `pedestrian_line_counter/line_picker.py` – interactive GUI to pick line(s) and save them to JSON.
- `pedestrian_line_counter/structures.py` – small data classes for `Detection` and `Track`.
- `pedestrian_line_counter/main.py` – main CLI implementation.
- `main.py` – thin wrapper that calls into `pedestrian_line_counter.main`
  so `python main.py` keeps working.
- `media/` – local input/output videos (ignored by Git).
- `Models/` – local ONNX model weights (ignored by Git).
- `Progress/` – session logs that track work against `plan.md`.
- `scripts/` – helper/debug scripts (e.g. ONNX inspection); not required
  for normal use.


Dependencies & Setup
--------------------

Requirements:

- Python 3.10+.
- `uv` as the Python package manager.

Install dependencies:

```bash
uv sync
```

Key runtime dependencies (from `pyproject.toml`):

- `opencv-python`
- `numpy`
- `onnxruntime` / `onnxruntime-gpu` (ONNX backend, default)
- (optional) `torch` if you want to use the Torch backend.


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
uv run python pedestrian_line_counter/line_picker.py \
  --input media/input.mp4 \
  --lines 1 \
  --save config/line.json
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
  --line-json config/line.json \
  --show
```


Optional Torch Backend
----------------------

If you prefer to experiment with a PyTorch model instead of ONNX:

- Install PyTorch in your environment (for example, following the official
  PyTorch instructions for your OS / CUDA setup).
- Place your Torch model weights (e.g. `model.pt`) somewhere accessible,
  typically under `Models/`.
- Run the app with `--backend torch` and point `--model` to the `.pt` file:

```bash
uv run python main.py \
  --backend torch \
  --model Models/your_model.pt \
  --input media/input.mp4 \
  --output media/output_torch.mp4 \
  --line-json config/line.json \
  --show
```

The Torch backend expects the loaded model to accept a `(1, 3, H, W)` float
tensor in `[0, 1]` and return detections shaped roughly like `(N, 6)` with
`[x1, y1, x2, y2, score, class_id]`. If your model uses a different
format, you can adapt `pedestrian_line_counter/torch_detector.py` as
needed.


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
