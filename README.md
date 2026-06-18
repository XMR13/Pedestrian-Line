# Vehicle Subclass Line Counter

SIstem edge AI untuk menghitung banyaknya vehicle yang melewati road area virtual
Program ini akan mendeteksi vehicle, kemudian track, dan menghitung vehcile berdasarkan arah
kemudian menggunakan FastAPI service untuk review operator dan backend.


This repo owns the AI-box side:

- video/RTSP input
- detection, tracking, line counting
- event spool with thumbnails and `report.csv`
- local FastAPI operator UI/API
- delivery worker for IT-managed backend endpoints

Large files are intentionally not committed. Put models in `Models/`, videos in
`media/`, and runtime event data in `spool/` or another local spool directory.

## How It Works

![Project workflow](docs/readme_png/cara_kerja.png)

## Architecture

![High-level architecture](docs/readme_png/arch_new_modified.png)

## Requirements

Development:

- Python 3.10+
- `uv`

Jetson deployment:
- Jetson Orin / Jetpack Runtime
- Menggunakkan wrapper script dan runbook untuk repo ini. Jetson sekarang 
  sudah divalidasi di environment python 3.8

Install for local development:

```bash
uv sync
```

Install for Jetson TensorRT runtime:

```bash
bash scripts/bootstrap_jetson.sh
```

This creates `.venv` with `--system-site-packages`, installs
`requirements-jetson.txt`, and installs this repo with `pip install -e .
--no-deps`. Use this path on Jetson so Python can see JetPack-provided
`tensorrt` / `cv2` packages without pulling the desktop ONNXRuntime dependency
set. TensorRT runtime depends on JetPack TensorRT plus `cuda-python`; it does
not require PyTorch.

## Prepare Files

Local file harus memiliki bentuk seperti ini:

```text
Models/
  vehicle_subclasses.onnx      # ONNX model, or adjust --model
  model_fp16.engine            # TensorRT engine on Jetson, optional
  metadata_vehicle.yaml        # class_id -> class name map, optional
media/
  input.mp4                    # local test video
config/cameras/
  camera_subang.json           # camera/line config, if used
```

PENTING: selalu set `--class-ids` untuk backend model agar app hanya menghitung
kelas vehicle yang sesuai dengan project ini.

Quick TensorRT sanity check:

```bash
source .venv/bin/activate
python -m pedestrian_line_counter.main --help
python scripts/preview_video.py \
  --input media/test.mp4 \
  --model Models/model_fp16.engine \
  --backend tensorrt \
  --allow-all-classes \
  --max-frames 30 \
  --save-frames \
  --frames-dir preview_outputs/trt_test \
  --no-progress
```

`scripts/preview_video.py --backend tensorrt` uses the same production
`Detector` / `TensorRTRunner` path as the counter app, so a preview run is a
valid detector smoke test before switching to the full counting loop.

## Fast Start: Local Video

Jalnkan vidoe lokal dan write ouputnya sebagai berikut:

```bash
uv run python -m pedestrian_line_counter.main \
  --backend onnx \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --input media/input.mp4 \
  --output media/output_test.mp4 \
  --output-encoder ffmpeg \
  --output-crf 28 \
  --output-preset slow \
  --show
```

Jika `ffmpeg` tidak tesedia, maka gunakan `--output-encoder auto` sehingga apliaksi bisa 
menggunakan default OpenCV untuk writing video

Langsung pick a new counting line:

```bash
uv run python -m pedestrian_line_counter.line_picker \
  --input media/input.mp4 \
  --lines 1 \
  --save config/cameras/camera_subang.json
```

Menggunakan konfigurasi untuk kamera/line


```bash
uv run python -m pedestrian_line_counter.main \
  --backend onnx \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --input media/input.mp4 \
  --camera camera_subang \
  --output media/output_test.mp4
```

## Event untuk dashborad/review
Gunakan mode spool jika kamu ingin menggunakan record per config


```bash
uv run python -m pedestrian_line_counter.main \
  --backend onnx \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --input media/input.mp4 \
  --spool-dir data/traffic_runs \
  --site-id subang \
  --camera-id cam_01 \
  --report-csv \
  --report-name report.csv
```

Output:

```text
data/traffic_runs/YYYY-MM-DD/<run_uid>/run.json
data/traffic_runs/YYYY-MM-DD/<run_uid>/events.jsonl
data/traffic_runs/YYYY-MM-DD/<run_uid>/report.csv
data/traffic_runs/YYYY-MM-DD/<run_uid>/thumbs/<event_uid>.jpg
```

`run.json` terdiri dari health summary. `report.csv` memiliki satu baris per satu crossign vehicle, timestamp, class, direction, notes, dan field lainnya.

## Operator UI / Local API
Start the FastAPi edge service on loopback:

```bash
uv run python -m pedestrian_line_counter.service \
  --spool-dir data/traffic_runs \
  --host 127.0.0.1 \
  --port 8080 \
  --ui-password "replace-me" \
  --mutation-api-key "replace-me"
```

buka link ini

- `http://127.0.0.1:8080/ui/dashboard`
- `http://127.0.0.1:8080/ui/review`
- `http://127.0.0.1:8080/healthz`

Untuk akses sementara dengan menggunakan LAN/IP Access

```bash
uv run python -m pedestrian_line_counter.service \
  --spool-dir data/traffic_runs \
  --host 0.0.0.0 \
  --port 8080 \
  --service-exposure lan \
  --no-service-docs \
  --service-trusted-host 127.0.0.1 \
  --service-trusted-host localhost \
  --service-trusted-host 192.168.1.50 \
  --ui-password "replace-me" \
  --mutation-api-key "replace-me"
```

Mode LAN membutuhkan autentikasi UI, API key, docs disabled, dan host yang terpecaya 

## Jetson Production Shape

Deploy to production memiliki 2 servis yang membagi direktori pool yang sama

1. `scripts/run_single_loop_live.sh`
   - reads local video or RTSP
   - runs detector/tracker/counter
   - writes event spool
   - can upload events if configured
2. `scripts/run_edge_service.sh`
   - serves local UI/API
   - exposes health/status/review/sync controls
   - runs retention cleanup

Full guide untuk menerapkan deployment:

- `docs/jetson_deployment_runbook.md`
- `docs/tensorrt_engine_bringup.md`
- `docs/security_review.md`

Minimal local-file validation with the wrapper:

```bash
PLC_INPUT_PATH=media/input.mp4 \
PLC_BACKEND=onnx \
PLC_MODEL_PATH=Models/vehicle_subclasses.onnx \
PLC_CLASS_IDS=0,1,2 \
PLC_SPOOL_DIR=data/traffic_runs \
PLC_PORTAL_UPLOAD_ENABLED=0 \
bash scripts/run_single_loop_live.sh
```

TensorRT local-file validation:

```bash
PLC_INPUT_PATH=media/input.mp4 \
PLC_BACKEND=tensorrt \
PLC_MODEL_PATH=Models/model_fp16.engine \
PLC_CLASS_NAMES=Models/metadata_vehicle.yaml \
PLC_CAMERA=camera_subang \
PLC_FRAME_STRIDE=2 \
PLC_SPOOL_DIR=data/traffic_runs \
PLC_PORTAL_UPLOAD_ENABLED=0 \
bash scripts/run_single_loop_live.sh
```

Jetson decoding, CLI juga support `--input-capture-backend gstreamer` dan `--input-get-pipeline`. Check full command cli dengan:

```bash
uv run python -m pedestrian_line_counter.main --help
```

## RTSP Live Mode

Contoh:

```bash
uv run python -m pedestrian_line_counter.main \
  --rtsp-url "rtsp://user:pass@camera-host:554/stream" \
  --camera camera_subang \
  --backend onnx \
  --model Models/vehicle_subclasses.onnx \
  --class-ids 0,1,2 \
  --rtsp-capture-backend gstreamer \
  --rtsp-transport tcp \
  --rtsp-codec h264 \
  --rtsp-latency-ms 200 \
  --queue-policy drop_oldest \
  --target-fps 12 \
  --log-every-seconds 10 \
  --spool-dir data/traffic_runs \
  --no-write
```

Mode live tidak write video output 

Untuk deployment live dengan windows/EZVIS, baca bagian berikut:

- `docs/ezviz_windows_bridge_rtsp_runbook.md`

## Backend Delivery

Ketika IT backend belum selesai, gunakan delivery untuk mock backend

```bash
uv run python scripts/mock_delivery_backend.py \
  --port 18080 \
  --state-dir tmp/mock_delivery_backend
```

Mengupload run:

```bash
uv run python -m pedestrian_line_counter.event_uploader \
  --spool-dir data/traffic_runs \
  --api-base-url http://127.0.0.1:18080 \
  --api-key local-dev
```

Backend payload dapat dilihat pada direktori berikut ini `tmp/mock_delivery_backend?`.


## Useful Docs

- `plan.md` - current project status and execution order
- `docs/README.md` - documentation map
- `docs/jetson_deployment_runbook.md` - main Jetson deployment guide
- `docs/tensorrt_engine_bringup.md` - TensorRT bring-up notes
- `docs/remote_ai_box_public_hosting_guide.md` - reverse proxy / public access
- `docs/security_review.md` - deployment security notes
- `docs/line_direction.md` - direction/counting geometry

## Main Source Layout

```text
pedestrian_line_counter/
  main.py              # detection/tracking/counting loop
  detector.py          # motion/ONNX/TensorRT/Torch detector wrapper
  tracker.py           # lightweight tracker
  line_counter.py      # line and two-line gate counting logic
  traffic_spool.py     # local run/event/evidence persistence
  event_uploader.py    # backend delivery worker
  api.py               # FastAPI app factory
  service.py           # FastAPI service runner
  ui_templates/        # operator UI templates
  ui_static/           # operator UI CSS/JS/assets
yolo_kitv2/            # YOLO postprocess/runtime/dataset helper utilities
scripts/               # deployment and helper scripts
deploy/                # systemd, Caddy, Windows bridge examples
docs/                  # runbooks and reference docs
```
