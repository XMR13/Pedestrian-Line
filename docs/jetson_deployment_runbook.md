# Jetson Deployment Runbook

This document is the single source of truth for deploying the project on the
Jetson.

It replaces the previous split Jetson deployment notes with one concrete guide.

Recommended deployment shape:

- `single_loop.service`
  - runs `pedestrian_line_counter.main`
  - reads local video or RTSP
  - detects, tracks, counts
  - writes spool output
- `edge_service.service`
  - runs `pedestrian_line_counter.service`
  - reads the same spool
  - serves local dashboard, review UI, and API

This is the most practical current setup because:

- it works before the final RTSP URL is available,
- it works before backend auth and domain decisions are finalized,
- it keeps inference and UI/API concerns separated,
- it lets the FastAPI service stay up even when the detector loop finishes a local test clip.

## 1. What this setup proves

When both services work together, the local edge stack is working end to end:

1. `main.py` reads input
2. detections and crossing events are written into the spool
3. the FastAPI service reads the same spool
4. dashboard, review queue, and event detail pages show live local data

Current limitation:

- If you use a local file, this validates local integration but not final RTSP behavior.
- When RTSP is available later, only the detector service input settings need to change.

## 2. Recommended deployment layout

Use this layout on Jetson:

```text
<repo-root>/                                # example: /home/iks-ai3/Development/Line_counting
<repo-root>/scripts/run_single_loop_live.sh
<repo-root>/scripts/run_edge_service.sh
/etc/vehicle_count/single_loop.env
/etc/vehicle_count/edge_service.env
/etc/systemd/system/single_loop.service
/etc/systemd/system/edge_service.service
/var/lib/pedline/traffic_runs
```

Important:

- `.service` files belong in `/etc/systemd/system/`
- `.env` files belong in `/etc/vehicle_count/`
- `WorkingDirectory=` and `ExecStart=` must match the real repo path on the Jetson
- `User=` and `Group=` must match a real Linux account on that machine

## 3. Prerequisites

- Python environment installed on the Jetson
- dependencies installed for the repo
- model file present locally
- camera line config already prepared
- repo path known and stable
- scripts are executable

Fast dependency bootstrap from the repo root:

```bash
bash scripts/bootstrap_jetson.sh
```

This script creates `.venv` with `--system-site-packages`, installs
`requirements-jetson.txt`, then installs the repo editable with `--no-deps`.
That keeps the TensorRT deployment path from trying to install the desktop
ONNXRuntime dependency set while still making `pedestrian_line_counter`
importable from anywhere inside the venv. The TensorRT runtime path uses
JetPack TensorRT plus `cuda-python`; it does not require PyTorch.

Make the launcher scripts executable:

```bash
chmod +x <repo-root>/scripts/run_single_loop_live.sh
chmod +x <repo-root>/scripts/run_edge_service.sh
```

## 4. Important rule: shared spool directory

Both services must point to the same spool directory:

```env
PLC_SPOOL_DIR=/var/lib/pedline/traffic_runs
```

If these paths differ, the UI service may start successfully but show no runs or events from the detector service.

Create the spool directory:

```bash
sudo mkdir -p /var/lib/pedline/traffic_runs
sudo chown -R <service-user>:<service-group> /var/lib/pedline
```

## 5. Detector service configuration

Create `/etc/vehicle_count/single_loop.env`.

### 5.1 Local file validation

Use this while RTSP is not available yet:

```env
PLC_INPUT_PATH=<repo-root>/media/test.mp4
PLC_RTSP_URL=

PLC_BACKEND=onnx
PLC_MODEL_PATH=Models/vehicle_subclasses.onnx
PLC_CLASS_NAMES=Models/metadata_vehicle.yaml
PLC_CONFIDENCE=0.65

PLC_SITE_ID=subang
PLC_CAMERA_ID=cam_01
PLC_CAMERA=camera_subang

PLC_SPOOL_DIR=/var/lib/pedline/traffic_runs
PLC_PORTAL_UPLOAD_ENABLED=0

PLC_TARGET_FPS=12
PLC_FRAME_STRIDE=1
PLC_QUEUE_SIZE=3
PLC_QUEUE_POLICY=drop_oldest
PLC_HEADLESS_STATUS_EVERY_S=10
```

Notes:

- `PLC_INPUT_PATH` is the switch for local-file validation.
- `PLC_PORTAL_UPLOAD_ENABLED=0` keeps the run local and avoids backend dependencies.
- In local-file mode, the detector is expected to exit cleanly at EOF.
- If the machine reboots with the same `PLC_INPUT_PATH` + `PLC_VIDEO_START`, the service now skips reprocessing by default once that exact local-file run has already completed.
- Set `PLC_ALLOW_DUPLICATE_LOCAL_INPUT=1` only when you intentionally want to rerun the same local validation clip.

### 5.2 RTSP mode

When RTSP is ready, switch to:

```env
PLC_INPUT_PATH=
PLC_RTSP_URL=rtsp://user:pass@camera-host:554/stream

PLC_BACKEND=onnx
PLC_MODEL_PATH=Models/vehicle_subclasses.onnx
PLC_CLASS_NAMES=Models/metadata_vehicle.yaml
PLC_CONFIDENCE=0.65

PLC_SITE_ID=subang
PLC_CAMERA_ID=cam_01
PLC_CAMERA=camera_subang

PLC_SPOOL_DIR=/var/lib/pedline/traffic_runs
PLC_PORTAL_UPLOAD_ENABLED=0

PLC_RTSP_CAPTURE_BACKEND=gstreamer
PLC_RTSP_TRANSPORT=tcp
PLC_RTSP_CODEC=h264
PLC_RTSP_LATENCY_MS=120

PLC_TARGET_FPS=12
PLC_FRAME_STRIDE=1
PLC_QUEUE_SIZE=3
PLC_QUEUE_POLICY=drop_oldest
PLC_HEADLESS_STATUS_EVERY_S=10
```

If backend upload is ready later:

```env
PLC_PORTAL_UPLOAD_ENABLED=1
PORTAL_API_KEY=replace-me
PLC_PORTAL_API_BASE_URL=http://127.0.0.1:5000
```

## 6. FastAPI edge service configuration

Create `/etc/vehicle_count/edge_service.env`.

Recommended first setup:

```env
PLC_SPOOL_DIR=/var/lib/pedline/traffic_runs

PLC_SERVICE_HOST=127.0.0.1
PLC_SERVICE_PORT=8080
PLC_SERVICE_EXPOSURE=loopback

EDGE_UI_USERNAME=admin
EDGE_UI_PASSWORD=
EDGE_SERVICE_API_KEY=

PLC_SERVICE_RETENTION_ENABLED=1
PLC_SERVICE_RETENTION_MAX_AGE_DAYS=90
PLC_SERVICE_RETENTION_AUTO_INTERVAL_S=3600
```

Recommended current rule:

- set real, non-placeholder values for `EDGE_UI_PASSWORD` and `EDGE_SERVICE_API_KEY`
  before enabling LAN/IP access
- leave `PLC_PORTAL_API_BASE_URL` unset here

Why:

- let `single_loop.service` own automatic upload later if needed
- let `edge_service.service` focus on UI/API
- avoid overlapping uploader ownership

### 6.1 Manual FastAPI bring-up without systemd

Use this when you want to validate the UI/API manually before enabling
`edge_service.service`, or when you are debugging a deployment issue.

Run from the repo root:

```bash
uv run python -m pedestrian_line_counter.service \
  --spool-dir data/traffic_runs \
  --host 127.0.0.1 \
  --port 8080
```

Open:

- Login: `http://127.0.0.1:8080/ui/login`
- Dashboard: `http://127.0.0.1:8080/ui/dashboard`
- Review Queue: `http://127.0.0.1:8080/ui/review`

Useful notes:

- If the spool is empty, the service still starts, but the dashboard and review
  pages will be sparse.
- If `EDGE_UI_PASSWORD` is set, the UI requires login.
- If `EDGE_UI_PASSWORD` is not set, login gating is disabled for the UI.
- Review decisions are stored locally under:

```text
<spool-dir>/.edge_ui_reviews.sqlite3
```

To save logs while running manually:

```bash
mkdir -p logs
uv run python -m pedestrian_line_counter.service \
  --spool-dir data/traffic_runs \
  --host 127.0.0.1 \
  --port 8080 \
  2>&1 | tee "logs/mvp-ui-$(date +%Y%m%d-%H%M%S).log"
```

To stop the service manually:

```bash
Ctrl+C
```

## 7. Service unit files

Create the service files under `/etc/systemd/system/`.

Tracked examples live under `deploy/systemd/`:

- `deploy/systemd/pedestrian-single-loop.env.example`
- `deploy/systemd/pedestrian-single-loop.service.example`
- `deploy/systemd/pedestrian-edge-service.env.example`
- `deploy/systemd/pedestrian-edge-service.service.example`

Keep real camera URLs, local IPs, passwords, and API keys in the Jetson-local
files under `/etc/vehicle_count/`; do not hard-code them into tracked launcher
scripts.

### 7.1 `single_loop.service`

```ini
[Unit]
Description=Pedestrian Line Counter (single-loop live)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=<service-user>
Group=<service-group>
WorkingDirectory=<repo-root>
EnvironmentFile=/etc/vehicle_count/single_loop.env
ExecStart=<repo-root>/scripts/run_single_loop_live.sh
Restart=on-failure
RestartSec=5
StartLimitIntervalSec=60
StartLimitBurst=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=pedline-single-loop
LimitNOFILE=65536
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

### 7.2 `edge_service.service`

```ini
[Unit]
Description=Pedestrian Line Counter (edge FastAPI service)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=<service-user>
Group=<service-group>
WorkingDirectory=<repo-root>
EnvironmentFile=/etc/vehicle_count/edge_service.env
ExecStart=<repo-root>/scripts/run_edge_service.sh
Restart=always
RestartSec=5
StartLimitIntervalSec=60
StartLimitBurst=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=pedline-edge-service
LimitNOFILE=65536
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

## 8. Enable and start

```bash
sudo systemctl daemon-reload

sudo systemctl enable single_loop.service
sudo systemctl enable edge_service.service

sudo systemctl start edge_service.service
sudo systemctl start single_loop.service
```

Check status:

```bash
systemctl status edge_service.service
systemctl status single_loop.service
```

Follow logs:

```bash
journalctl -u edge_service.service -f
journalctl -u single_loop.service -f
```

## 9. Expected behavior

`edge_service.service`:

- should stay up continuously
- should serve local dashboard, review queue, and detail pages

`single_loop.service`:

- local-file mode: should process the file and exit cleanly at EOF
- RTSP mode: should stay up and reconnect when the stream drops

Spool output should appear under:

```text
/var/lib/pedline/traffic_runs/<yyyy-mm-dd>/<run_uid>/
```

Common files:

- `run.json`
- `events.jsonl`
- `status.json`
- `thumbs/`
- `scene/`

## 10. Throughput tuning checklist

Use this order:

1. Set `PLC_TARGET_FPS` to a rate the device can sustain.
2. Keep `PLC_QUEUE_POLICY=drop_oldest` and `PLC_QUEUE_SIZE` small.
3. Keep output video disabled unless specifically needed.
4. If still overloaded, increase `PLC_FRAME_STRIDE`.
5. Prefer `tensorrt` on Jetson once the engine is validated.

## 11. Failure guide

### 11.1 `Failed to load environment files`

Cause:

- `EnvironmentFile=` path does not match the real `.env` filename

Check:

```bash
systemctl cat edge_service.service
systemctl cat single_loop.service
ls -la /etc/vehicle_count
```

### 11.2 `status=217/USER`

Cause:

- `User=` or `Group=` does not exist on the Jetson

Check:

```bash
id <service-user>
getent passwd <service-user>
getent group <service-group>
```

### 11.3 `status=200/CHDIR`

Cause:

- `WorkingDirectory=` points to a repo path that does not exist

Check:

```bash
pwd
ls -ld <repo-root>
```

### 11.4 `status=203/EXEC` or `Permission denied`

Cause:

- launcher script exists but is not executable

Fix:

```bash
chmod +x <repo-root>/scripts/run_edge_service.sh
chmod +x <repo-root>/scripts/run_single_loop_live.sh
```

### 11.5 Service file exists but `systemctl` cannot start it

Cause:

- `.service` file was placed in `/etc/vehicle_count/` instead of `/etc/systemd/system/`

Rule:

- install `.service` files in `/etc/systemd/system/`
- keep only `.env` files in `/etc/vehicle_count/`

### 11.6 Verify the effective unit

Whenever startup behavior is confusing, inspect the live unit:

```bash
sudo systemctl cat edge_service.service | cat -vet
sudo systemctl cat single_loop.service | cat -vet
```

This shows the actual paths systemd is using.

## 12. Current recommended validation order

1. Start `edge_service.service` and confirm the UI is reachable.
2. Start `single_loop.service` against a short local file.
3. Confirm spool output is created.
4. Confirm the FastAPI UI reads those events.
5. After that works, switch the detector service to RTSP when the stream is available.

## 13. Notes

- This runbook intentionally assumes local-first deployment and unresolved external dependencies.
- Backend auth, public domain setup, and final RTSP hookup should be treated as later integration steps.
- If the repo path on the Jetson is not stable yet, update the unit files first before debugging Python-level issues.
