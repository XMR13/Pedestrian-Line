# Single-Loop Production Runbook (Edge Detect + Spool + Portal Upload)

This runbook is for production operation in **one process** using:

- `pedestrian_line_counter.main`
- `--portal-upload` (integrated uploader)
- RTSP live source

Target: low-latency, stable restart behavior, and predictable throughput.

## 1) Scope and prerequisites

- Python `3.10` environment managed by `uv`.
- Portal API is already running and reachable.
- Camera line config is already prepared (`--camera` or `--line-json`).
- Secrets are provided from environment (not tracked files):
  - `PLC_RTSP_URL`
  - `PORTAL_API_KEY`
- Camera connectivity is validated at startup: process exits if first frame cannot be read.

Install/update dependencies:

```bash
uv sync --frozen
```

## 2) Fast baseline profile (recommended)

Use the provided launcher script:

```bash
scripts/run_single_loop_live.sh
```

Default profile used by this script:

- Live RTSP with reconnect and bounded queue.
- `drop_oldest` queue policy (bounded latency).
- `--target-fps 12` (good starting point for edge stability).
- Output video disabled (`--no-write`) to save CPU and disk I/O.
- Scene thumbnails disabled (keeps uploader lighter).
- Integrated portal upload enabled with periodic pass.
- Headless status snapshots written periodically (`run.json` + `status.json`) for portal health visibility.

Run duration:

- By default it runs continuously (no stop limit).
- Optional smoke limits:
  - `PLC_MAX_SECONDS=30`
  - `PLC_MAX_FRAMES=300`

Important environment variables:

```bash
export PLC_RTSP_URL="rtsp://user:pass@camera-host:554/stream"
export PORTAL_API_KEY="replace-me"
export PLC_PORTAL_API_BASE_URL="http://127.0.0.1:5000"
export PLC_CAMERA="camera_subang"
export PLC_SITE_ID="subang"
export PLC_CAMERA_ID="cam_01"
```

Optional performance overrides:

```bash
export PLC_BACKEND="onnx"                       # or "tensorrt" when .engine is ready
export PLC_MODEL_PATH="Models/vehicle_subclasses.onnx"
export PLC_CLASS_IDS="0,1,2"
export PLC_TARGET_FPS="12"
export PLC_FRAME_STRIDE="1"
export PLC_QUEUE_SIZE="3"
export PLC_RTSP_CAPTURE_BACKEND="gstreamer"     # Jetson recommended
export PLC_RTSP_TRANSPORT="tcp"
export PLC_RTSP_CODEC="h264"
export PLC_RTSP_LATENCY_MS="120"
export PLC_SPOOL_DIR="/var/lib/pedline/traffic_runs"
export PLC_PORTAL_UPLOAD_INTERVAL_S="10"
export PLC_PORTAL_UPLOAD_MAX_RUNS_PER_PASS="2"
export PLC_PORTAL_UPLOAD_EVENTS_BATCH_SIZE="200"
export PLC_HEADLESS_STATUS_EVERY_S="10"
# export PLC_HEADLESS_STATUS_JSON="/var/lib/pedline/status/cam_01.json"
# export PLC_MAX_SECONDS="30"
# export PLC_MAX_FRAMES="300"
```

## 3) Throughput tuning checklist

Use this order so tuning stays predictable:

1. Set `PLC_TARGET_FPS` to a value the device can sustain continuously.
2. Keep `PLC_QUEUE_POLICY=drop_oldest` and `PLC_QUEUE_SIZE` small (`3` or `4`) for low lag.
3. Disable expensive outputs first:
   - keep `--no-write`
   - keep scene thumbnails disabled
4. If still overloaded, raise `PLC_FRAME_STRIDE` (`2` before `3`) and validate counting accuracy on that camera.
5. Prefer `tensorrt` + `.engine` on Jetson when available; otherwise keep `onnx`.

## 4) Service manager setup (systemd)

### 4.1 Copy templates

```bash
sudo mkdir -p /etc/pedline
sudo cp deploy/systemd/pedestrian-single-loop.service.example /etc/systemd/system/pedestrian-single-loop.service
sudo cp deploy/systemd/pedestrian-single-loop.env.example /etc/pedline/single-loop.env
sudo nano /etc/pedline/single-loop.env
```

Set real values in `/etc/pedline/single-loop.env` (RTSP URL, API key, camera IDs).

### 4.2 Enable service

```bash
sudo systemctl daemon-reload
sudo systemctl enable pedestrian-single-loop.service
sudo systemctl start pedestrian-single-loop.service
```

### 4.3 Operate and verify

```bash
sudo systemctl status pedestrian-single-loop.service
journalctl -u pedestrian-single-loop.service -f
```

## 5) Health checks during operation

Check these continuously:

- Service uptime and restart count via `systemctl status`.
- Log lines from main/uploader for:
  - reconnect loops,
  - uploader pass failures,
  - unusual frame drop growth.
- Spool output exists and keeps growing under:
  - `<spool_dir>/<yyyy-mm-dd>/<run_uid>/run.json`
  - `<spool_dir>/<yyyy-mm-dd>/<run_uid>/events.jsonl`
  - `<spool_dir>/<yyyy-mm-dd>/<run_uid>/status.json`
- Portal dashboard "Headless runtime status" card updates for active run health.

After a graceful stop, confirm `run.json` has:

- `ended_at_utc`
- `health_summary`

This confirms final uploader pass can mark run complete.

## 6) Failure playbook

- Portal temporarily down:
  - main loop continues writing spool,
  - integrated uploader retries/backoffs and syncs when portal returns.
- RTSP source drop:
  - reconnect policy retries automatically with backoff.
- Process crash:
  - systemd `Restart=always` brings process back.

## 7) Notes for Jetson rollout

- Keep spool on fast local storage (not slow SD/network mount).
- Prefer `PLC_RTSP_CAPTURE_BACKEND=gstreamer` on Jetson.
- Move to `tensorrt` backend after `.engine` is validated for your camera/model pair.
