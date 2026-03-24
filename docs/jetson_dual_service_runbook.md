# Jetson Dual-Service Runbook

This runbook explains how to run the project on the Jetson using **two services**:

- `pedestrian-single-loop.service`
  - runs `pedestrian_line_counter.main`
  - reads video/RTSP
  - detects, tracks, counts
  - writes spool data
- `pedestrian-edge-service.service`
  - runs `pedestrian_line_counter.service`
  - reads the same spool data
  - serves dashboard/UI/API

This is the current recommended deployment shape for Jetson.

## 1) What this setup proves

When both services are running correctly, you have a working local edge stack:

1. input enters `main.py`
2. counts/events are written into the spool
3. the FastAPI service reads that spool
4. dashboard/UI/API shows the result

Current limitation:

- If you use a **local video file**, this validates the full local service interaction, but not real RTSP/live behavior.
- When RTSP is available later, you only need to switch the detector service input mode and re-validate live behavior.
- In local-file mode, the detector process is expected to exit cleanly at EOF; the systemd template uses `Restart=on-failure` so a finished test clip does not create an artificial restart loop.

## 2) Recommended directory layout on Jetson

Use this layout:

```text
/opt/pedestrian-line                    # repo root
/etc/pedline/single-loop.env            # detector service env
/etc/pedline/edge-service.env           # FastAPI service env
/etc/systemd/system/pedestrian-single-loop.service
/etc/systemd/system/pedestrian-edge-service.service
/var/lib/pedline/traffic_runs           # shared spool directory
```

## 3) Prerequisites

- Repo copied to `/opt/pedestrian-line`
- Python environment installed
- dependencies installed (`uv sync --frozen` or your chosen equivalent)
- model file present locally
- line config prepared (`PLC_CAMERA` or `PLC_LINE_JSON`)

Both services run as the same user in the templates:

- `pedline`

So that user must be able to:

- read the repo
- execute the scripts
- read the model file
- read the input file or RTSP source
- write to the spool directory

## 4) Copy the systemd files

```bash
sudo mkdir -p /etc/pedline

sudo cp /opt/pedestrian-line/deploy/systemd/pedestrian-single-loop.service.example \
  /etc/systemd/system/pedestrian-single-loop.service

sudo cp /opt/pedestrian-line/deploy/systemd/pedestrian-edge-service.service.example \
  /etc/systemd/system/pedestrian-edge-service.service

sudo cp /opt/pedestrian-line/deploy/systemd/pedestrian-single-loop.env.example \
  /etc/pedline/single-loop.env

sudo cp /opt/pedestrian-line/deploy/systemd/pedestrian-edge-service.env.example \
  /etc/pedline/edge-service.env

sudo chmod +x /opt/pedestrian-line/scripts/run_single_loop_live.sh
sudo chmod +x /opt/pedestrian-line/scripts/run_edge_service.sh
```

## 5) Important rule: shared spool directory

Both services must use the **same** spool directory:

```env
PLC_SPOOL_DIR=/var/lib/pedline/traffic_runs
```

That is the integration point between the two processes.

If these do not match, the FastAPI UI will start but it will not show the runs/events written by `main.py`.

## 6) Configure the detector service

Edit:

```bash
sudo nano /etc/pedline/single-loop.env
```

### 6.1 Local file mode (recommended current validation path)

Use this while RTSP is not available yet:

```env
PLC_INPUT_PATH=/opt/pedestrian-line/media/test.mp4
PLC_RTSP_URL=

PLC_BACKEND=onnx
PLC_MODEL_PATH=Models/vehicle_subclasses.onnx
PLC_CLASS_IDS=0,1,2

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

- `PLC_INPUT_PATH` is the important switch for local-file mode.
- `PLC_PORTAL_UPLOAD_ENABLED=0` means:
  - no backend upload is attempted,
  - spool is still written normally,
  - FastAPI service can still show the data.

### 6.2 RTSP mode (for later)

When RTSP is available, switch to:

```env
PLC_INPUT_PATH=
PLC_RTSP_URL=rtsp://user:pass@camera-host:554/stream

PLC_BACKEND=onnx
PLC_MODEL_PATH=Models/vehicle_subclasses.onnx
PLC_CLASS_IDS=0,1,2

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

If backend upload is ready later, set:

```env
PLC_PORTAL_UPLOAD_ENABLED=1
PORTAL_API_KEY=replace-me
PLC_PORTAL_API_BASE_URL=http://127.0.0.1:5000
```

## 7) Configure the FastAPI service

Edit:

```bash
sudo nano /etc/pedline/edge-service.env
```

Recommended current setup:

```env
PLC_SPOOL_DIR=/var/lib/pedline/traffic_runs

PLC_SERVICE_HOST=127.0.0.1
PLC_SERVICE_PORT=8080
PLC_SERVICE_EXPOSURE=loopback

EDGE_UI_USERNAME=admin
EDGE_UI_PASSWORD=replace-me
EDGE_SERVICE_API_KEY=replace-me

PLC_SERVICE_RETENTION_ENABLED=1
PLC_SERVICE_RETENTION_MAX_AGE_DAYS=90
PLC_SERVICE_RETENTION_AUTO_INTERVAL_S=3600
```

Recommended current rule:

- leave `PLC_PORTAL_API_BASE_URL` unset in `edge-service.env`

Why:

- let `pedestrian-single-loop.service` own automatic uploading later
- let `pedestrian-edge-service.service` focus on UI/API
- avoid overlapping uploader ownership

## 8) Create and verify the spool directory

```bash
sudo mkdir -p /var/lib/pedline/traffic_runs
sudo chown -R pedline:pedline /var/lib/pedline
```

Also ensure the repo is readable by `pedline`:

```bash
sudo chown -R pedline:pedline /opt/pedestrian-line
```

Adjust this if your deployment uses a different ownership model.

## 9) Start both services

```bash
sudo systemctl daemon-reload

sudo systemctl enable pedestrian-single-loop.service
sudo systemctl enable pedestrian-edge-service.service

sudo systemctl start pedestrian-single-loop.service
sudo systemctl start pedestrian-edge-service.service
```

## 10) Verify the detector service

```bash
sudo systemctl status pedestrian-single-loop.service
journalctl -u pedestrian-single-loop.service -f
```

You want to see:

- the service processes the selected file or stream without repeated crash/restart loops
- spool directories are created
- `run.json`, `events.jsonl`, `status.json` appear

Check the spool manually:

```bash
find /var/lib/pedline/traffic_runs -maxdepth 3 -type f
```

Expected examples:

- `/var/lib/pedline/traffic_runs/YYYY-MM-DD/<run_uid>/run.json`
- `/var/lib/pedline/traffic_runs/YYYY-MM-DD/<run_uid>/events.jsonl`
- `/var/lib/pedline/traffic_runs/YYYY-MM-DD/<run_uid>/status.json`

## 11) Verify the FastAPI service

```bash
sudo systemctl status pedestrian-edge-service.service
journalctl -u pedestrian-edge-service.service -f
```

Then open:

- `http://127.0.0.1:8080/ui/login`

After login, check:

- dashboard loads
- recent runs appear
- recent events appear
- status/metrics endpoints respond

Quick API checks:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/status
curl http://127.0.0.1:8080/metrics
```

## 12) What “success” looks like

Your current Jetson-local validation is successful if:

1. `pedestrian-single-loop.service` processes the selected input without restart loops
2. spool files are created under the shared spool directory
3. `pedestrian-edge-service.service` stays up
4. FastAPI UI shows runs/events from the same spool
5. local review/status pages work

At that point, the local end-to-end deployment shape is validated.

## 13) What this does not prove yet

If you are using a local video file, this still does **not** prove:

- real RTSP reconnect behavior
- real live throughput behavior
- camera/network instability handling
- final backend delivery contract behavior
- final auth/domain exposure behavior

Those are later integration validations.

## 14) Common problems

### 14.1 FastAPI UI starts, but dashboard is empty

Usually one of these:

- `PLC_SPOOL_DIR` differs between the two env files
- detector service is not writing spool data
- local input/video is wrong and `main.py` exits early

### 14.2 Detector service keeps restarting

Usually one of these:

- missing model file
- bad `PLC_INPUT_PATH`
- bad `PLC_RTSP_URL`
- `pedline` cannot read repo/model/input
- `pedline` cannot write spool directory

### 14.3 FastAPI service keeps restarting

Usually one of these:

- bad host/exposure config
- missing `EDGE_UI_PASSWORD`
- missing `EDGE_SERVICE_API_KEY` when LAN mode is configured
- trusted hosts missing in LAN mode

### 14.4 I want local-only validation first

Do this:

- use `PLC_INPUT_PATH`
- set `PLC_PORTAL_UPLOAD_ENABLED=0`
- keep `PLC_SERVICE_HOST=127.0.0.1`
- keep `PLC_SERVICE_EXPOSURE=loopback`

That is the safest current path.

## 15) Stop / restart commands

Stop:

```bash
sudo systemctl stop pedestrian-single-loop.service
sudo systemctl stop pedestrian-edge-service.service
```

Restart after env changes:

```bash
sudo systemctl restart pedestrian-single-loop.service
sudo systemctl restart pedestrian-edge-service.service
```

## 16) Recommended rollout order

Use this order:

1. validate both services with a local video file
2. confirm spool + UI interaction works
3. switch detector service to RTSP when camera access is available
4. enable backend upload when the delivery contract is ready
5. later add final domain/auth/reverse-proxy integration
