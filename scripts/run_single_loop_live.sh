#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-uv}"

: "${PLC_RTSP_URL:?Set PLC_RTSP_URL (example: rtsp://user:pass@camera-host:554/stream)}"
: "${PORTAL_API_KEY:?Set PORTAL_API_KEY}"

PLC_BACKEND="${PLC_BACKEND:-onnx}"
PLC_MODEL_PATH="${PLC_MODEL_PATH:-Models/vehicle_subclasses.onnx}"
PLC_CLASS_IDS="${PLC_CLASS_IDS:-0,1,2}"
PLC_CAMERA="${PLC_CAMERA:-camera_subang}"
PLC_LINE_JSON="${PLC_LINE_JSON:-}"
PLC_SITE_ID="${PLC_SITE_ID:-subang}"
PLC_CAMERA_ID="${PLC_CAMERA_ID:-cam_01}"
PLC_SPOOL_DIR="${PLC_SPOOL_DIR:-/var/lib/pedline/traffic_runs}"
PLC_TARGET_FPS="${PLC_TARGET_FPS:-12}"
PLC_FRAME_STRIDE="${PLC_FRAME_STRIDE:-1}"
PLC_QUEUE_SIZE="${PLC_QUEUE_SIZE:-3}"
PLC_QUEUE_POLICY="${PLC_QUEUE_POLICY:-drop_oldest}"
PLC_RTSP_CAPTURE_BACKEND="${PLC_RTSP_CAPTURE_BACKEND:-gstreamer}"
PLC_RTSP_TRANSPORT="${PLC_RTSP_TRANSPORT:-tcp}"
PLC_RTSP_CODEC="${PLC_RTSP_CODEC:-h264}"
PLC_RTSP_LATENCY_MS="${PLC_RTSP_LATENCY_MS:-120}"
PLC_PORTAL_API_BASE_URL="${PLC_PORTAL_API_BASE_URL:-http://127.0.0.1:5000}"
PLC_PORTAL_UPLOAD_INTERVAL_S="${PLC_PORTAL_UPLOAD_INTERVAL_S:-10}"
PLC_PORTAL_UPLOAD_MAX_RUNS_PER_PASS="${PLC_PORTAL_UPLOAD_MAX_RUNS_PER_PASS:-2}"
PLC_PORTAL_UPLOAD_EVENTS_BATCH_SIZE="${PLC_PORTAL_UPLOAD_EVENTS_BATCH_SIZE:-200}"
PLC_MAX_SECONDS="${PLC_MAX_SECONDS:-}"
PLC_MAX_FRAMES="${PLC_MAX_FRAMES:-}"

mkdir -p "$PLC_SPOOL_DIR"

args=(
  -m pedestrian_line_counter.main
  --rtsp-url-env PLC_RTSP_URL
  --backend "$PLC_BACKEND"
  --model "$PLC_MODEL_PATH"
  --class-ids "$PLC_CLASS_IDS"
  --rtsp-capture-backend "$PLC_RTSP_CAPTURE_BACKEND"
  --rtsp-transport "$PLC_RTSP_TRANSPORT"
  --rtsp-codec "$PLC_RTSP_CODEC"
  --rtsp-latency-ms "$PLC_RTSP_LATENCY_MS"
  --queue-size "$PLC_QUEUE_SIZE"
  --queue-policy "$PLC_QUEUE_POLICY"
  --target-fps "$PLC_TARGET_FPS"
  --frame-stride "$PLC_FRAME_STRIDE"
  --log-every-seconds 10
  --rtsp-reconnect
  --rtsp-reconnect-max-attempts 0
  --rtsp-reconnect-initial-delay 1.0
  --rtsp-reconnect-max-delay 30.0
  --rtsp-reconnect-backoff 2.0
  --rtsp-stall-timeout 5.0
  --spool-dir "$PLC_SPOOL_DIR"
  --site-id "$PLC_SITE_ID"
  --camera-id "$PLC_CAMERA_ID"
  --spool-thumbnails
  --no-spool-scene-thumbnails
  --spool-thumb-max-side 256
  --portal-upload
  --portal-api-base-url "$PLC_PORTAL_API_BASE_URL"
  --portal-api-key-env PORTAL_API_KEY
  --portal-upload-interval-s "$PLC_PORTAL_UPLOAD_INTERVAL_S"
  --portal-upload-max-runs-per-pass "$PLC_PORTAL_UPLOAD_MAX_RUNS_PER_PASS"
  --portal-upload-events-batch-size "$PLC_PORTAL_UPLOAD_EVENTS_BATCH_SIZE"
  --portal-upload-thumbnails
  --no-portal-upload-scene-thumbnails
  --no-write
)

if [[ -n "$PLC_LINE_JSON" ]]; then
  args+=(--line-json "$PLC_LINE_JSON")
else
  args+=(--camera "$PLC_CAMERA")
fi

if [[ -n "$PLC_MAX_SECONDS" ]]; then
  args+=(--max-seconds "$PLC_MAX_SECONDS")
fi

if [[ -n "$PLC_MAX_FRAMES" ]]; then
  args+=(--max-frames "$PLC_MAX_FRAMES")
fi

exec "$UV_BIN" run python "${args[@]}"
