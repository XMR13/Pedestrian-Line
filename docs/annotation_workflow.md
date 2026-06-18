# Dataset Annotation Workflow (Vehicle Subclasses)

This project’s counting/tracking pipeline depends on a detector that can recognize
your **custom vehicle subclasses** (e.g. truck, trailer, pickup, etc). The immediate
next step is therefore: **collect → sample frames → annotate → train → export**.

This document focuses on the **annotation** portion.

---

## 1. Decide the taxonomy (class list)

Before labeling, write down the class list and definitions:

- What is the difference between *truck* vs *pickup*?
- How to label *truck+trailer* (one box vs two boxes)?
- How to label partially visible vehicles?
- What to do with “unknown / ambiguous” cases (ignore vs pick closest class)?

Keep this document versioned with the dataset (even a simple `TAXONOMY.md`).

---

## 2. Recommended label format

Recommended training label format: **YOLO detection**.

Typical layout:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml   # includes `names: [...]`
```

The `data.yaml` should include your class list under `names:` (list or dict).
This same file can later be reused by this repo as a **class-name map** for overlays.

---

## 3. Frame sampling (what to label)

Try to avoid labeling only “easy” frames. Mix:

- day / night / rain,
- glare / shadows,
- dense traffic,
- small/distant vehicles,
- occlusions (e.g. poles, trees/leaves),
- rare classes (trailers, special trucks).

Practical approach:

- Sample at a fixed interval (e.g. 1 fps) for baseline coverage.
- Add extra frames around “hard moments” (crowds, fast motion, occlusion).

If you already have a model, use `python -m yolo_kitv2 label run` with `--mode coco` and
`--fps` / `--every-n` to sample frames while also generating a COCO JSON for CVAT.

If you only want to **extract frames** without running the model, use `--mode frames`.

---

## 4. Detector-assisted candidate extraction (optional)

To reduce manual scrubbing, you can let a **basic vehicle detector** save
candidate frames, then you annotate those images yourself.

Example (track-based auto-label, similar to extract_candidates):

```bash
uv run python -m yolo_kitv2 label run \
  --mode candidates \
  --input media/input.mp4 \
  --output-dir data/candidates/input_mp4 \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --max-per-track 3 \
  --warmup-frames 5 \
  --min-seconds-between 1.0
```

Notes:

- This is **not** ground truth. It only decides *which frames to save*.
- Keep a baseline sample too (e.g. 1 fps) to avoid bias toward easy cases.

---

## 4.1 Auto-label video → COCO for CVAT (bootstrap)

If you already have a first model, you can **auto-label** a video and export
COCO JSON so you can fix labels in CVAT and retrain.

This script will:

- sample frames from a video,
- run your YOLO model,
- save frames to `images/`,
- write `annotations.json` in **COCO dataset** format (CVAT compatible).

Example:

```bash
uv run python -m yolo_kitv2 label run \
  --mode coco \
  --input media/input.mp4 \
  --output-dir data/auto_labels/input_mp4 \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --class-names Models/metadata.yaml \
  --min-seconds-between 1.0 \
  --max-per-track 3 \
  --warmup-frames 5
```

You can also auto-label a **folder of images** by pointing `--input` to a directory:

```bash
uv run python -m yolo_kitv2 label run \
  --mode coco \
  --input data/images_raw \
  --output-dir data/auto_labels/images_raw \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --class-names Models/metadata.yaml \
  --every-n 2
```

To **merge multiple auto-label outputs** into one COCO dataset for CVAT:

```bash
uv run python -m yolo_kitv2 coco merge \
  --inputs data/auto_labels/set_a data/auto_labels/set_b \
  --output-dir data/auto_labels/merged
```

If you delete images during QA, prune the COCO JSON:

```bash
uv run python -m yolo_kitv2 coco prune \
  --dataset-dir data/auto_labels/merged \
  --in-place
```

If CVAT cannot find images during COCO import because `images[].file_name`
contains a nested path such as `images/<name>.jpg` while the uploaded folder is
flat, normalize the JSON for CVAT:

```bash
uv run python -m yolo_kitv2 coco cvat-fix \
  --dataset-dir data/auto_labels/merged \
  --in-place \
  --basename-file-names
```

If you only have an `annotations.json` file and want to inspect label
distribution:

```bash
uv run python -m yolo_kitv2 dataset viz \
  --annotations data/auto_labels/merged/annotations.json \
  --distribution-only \
  --output-dir data/auto_labels/merged_viz_dist
```

Tips:

- Use `--min-seconds-between` or `--min-frames-between` to control spacing.
- Set `--skip-empty` if you only want frames that contain detections.
- The COCO JSON can be imported directly into CVAT with the saved images folder.
- For QA/visual inspection, use `scripts/preview_video.py` (see below).

---

## 4.2 Preview/QA (visual check)

Use this when you want to **inspect detections** or save annotated previews
without generating COCO files.

Example (preview only):

```bash
uv run python scripts/preview_video.py \
  --input media/input.mp4 \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --max-seconds 10 \
  --show
```

Example (save annotated frames + video):

```bash
uv run python scripts/preview_video.py \
  --input media/input.mp4 \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --fps 1 \
  --save-frames \
  --save-video \
  --output-dir data/preview_outputs 
```

---

## 5. Annotation tools

Any bounding-box labeling tool is fine; prefer tools that can export YOLO.

Common options:

- CVAT (self-hostable)
- Label Studio (self-hostable)
- labelme (local)

Pick one that fits your team workflow and data privacy constraints.

### IF YOU USE CVATT
If you gonna use the cvat app, and wants to import annotations with coco format, what you need to do first is fix the auto labeling format because the scripts `yolo_kitv2/datasets/label.py` index the class starting from 0, but cvat expects index 1. what you need is to run the script `yolo_kitv2/datasets/coco_cvat_fix.py`

---

## 6. Labeling guidelines (consistency > perfection)

Suggested rules:

- Use **tight boxes** around the vehicle body (consistent definition).
- If a vehicle is truncated by the image border, still label the visible part.
- For heavy occlusion: label if you are confident it is the target class; otherwise ignore.
- Keep class usage consistent across all labelers (do quick review early).

---

## 7. Next steps after annotation (high level)

After you have an initial dataset:

- Train the detector (separate repo/environment is OK).
- Export to ONNX for this project’s default runtime.
- Generate a class-name map file (YAML/JSON) for overlays and reporting.

Once the model exists, this repo can be configured to count **only** the target classes
via `--class-ids` and (optionally) label overlays via `--class-names`.
