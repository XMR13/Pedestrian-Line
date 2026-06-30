#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
UV_BIN="${UV_BIN:-uv}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  fi
fi

PLC_INPUT_PATH="${PLC_INPUT_PATH:-}"
PLC_VIDEO_START="${PLC_VIDEO_START:-}"
PLC_PROMPT_VIDEO_START="${PLC_PROMPT_VIDEO_START:-auto}"
PLC_RTSP_URL="${PLC_RTSP_URL:-}"

if [[ -z "$PLC_INPUT_PATH" && -z "$PLC_RTSP_URL" ]]; then
  echo "Set PLC_INPUT_PATH for local file mode or PLC_RTSP_URL for RTSP live mode." >&2
  exit 1
fi

PLC_BACKEND="${PLC_BACKEND:-onnx}"
PLC_MODEL_PATH="${PLC_MODEL_PATH:-Models/models.engine}"
PLC_CLASS_IDS="${PLC_CLASS_IDS:-}"
PLC_CLASS_NAMES="${PLC_CLASS_NAMES:-Models/metadata_vehicle.yaml}"
PLC_DEFAULT_CLASS_IDS="${PLC_DEFAULT_CLASS_IDS:-0,1,2}"
PLC_ALLOW_ALL_CLASSES="${PLC_ALLOW_ALL_CLASSES:-0}"
PLC_CAMERA="${PLC_CAMERA:-camera_subang}"
PLC_LINE_JSON="${PLC_LINE_JSON:-config/line5.json}"
PLC_SITE_ID="${PLC_SITE_ID:-subang}"
PLC_CAMERA_ID="${PLC_CAMERA_ID:-cam_01}"
PLC_SPOOL_DIR="${PLC_SPOOL_DIR:-/var/lib/pedline/traffic_runs}"
PLC_SPOOL_THUMBNAILS="${PLC_SPOOL_THUMBNAILS:-1}"
PLC_SPOOL_SCENE_THUMBNAILS="${PLC_SPOOL_SCENE_THUMBNAILS:-1}"
PLC_SPOOL_THUMB_MAX_SIDE="${PLC_SPOOL_THUMB_MAX_SIDE:-256}"
PLC_SPOOL_SCENE_THUMB_MAX_SIDE="${PLC_SPOOL_SCENE_THUMB_MAX_SIDE:-}"
PLC_SPOOL_SCENE_THUMB_QUALITY="${PLC_SPOOL_SCENE_THUMB_QUALITY:-}"
PLC_ALLOW_DUPLICATE_LOCAL_INPUT="${PLC_ALLOW_DUPLICATE_LOCAL_INPUT:-0}"
PLC_TARGET_FPS="${PLC_TARGET_FPS:-}"
PLC_FRAME_STRIDE="${PLC_FRAME_STRIDE:-1}"
PLC_QUEUE_SIZE="${PLC_QUEUE_SIZE:-3}"
PLC_QUEUE_POLICY="${PLC_QUEUE_POLICY:-drop_oldest}"
PLC_RTSP_CAPTURE_BACKEND="${PLC_RTSP_CAPTURE_BACKEND:-gstreamer}"
PLC_RTSP_TRANSPORT="${PLC_RTSP_TRANSPORT:-tcp}"
PLC_RTSP_CODEC="${PLC_RTSP_CODEC:-h264}"
PLC_RTSP_LATENCY_MS="${PLC_RTSP_LATENCY_MS:-120}"
PLC_PORTAL_UPLOAD_ENABLED="${PLC_PORTAL_UPLOAD_ENABLED:-1}"
PLC_PORTAL_API_BASE_URL="${PLC_PORTAL_API_BASE_URL:-http://127.0.0.1:5000}"
PLC_PORTAL_UPLOAD_INTERVAL_S="${PLC_PORTAL_UPLOAD_INTERVAL_S:-10}"
PLC_PORTAL_UPLOAD_MAX_RUNS_PER_PASS="${PLC_PORTAL_UPLOAD_MAX_RUNS_PER_PASS:-2}"
PLC_PORTAL_UPLOAD_EVENTS_BATCH_SIZE="${PLC_PORTAL_UPLOAD_EVENTS_BATCH_SIZE:-200}"
PLC_PORTAL_UPLOAD_THUMBNAILS="${PLC_PORTAL_UPLOAD_THUMBNAILS:-1}"
PLC_PORTAL_UPLOAD_SCENE_THUMBNAILS="${PLC_PORTAL_UPLOAD_SCENE_THUMBNAILS:-0}"
PLC_HEADLESS_STATUS_EVERY_S="${PLC_HEADLESS_STATUS_EVERY_S:-10}"
PLC_HEADLESS_STATUS_JSON="${PLC_HEADLESS_STATUS_JSON:-}"
PLC_CONFIDENCE="${PLC_CONFIDENCE:-}"
PLC_MAX_SECONDS="${PLC_MAX_SECONDS:-}"
PLC_MAX_FRAMES="${PLC_MAX_FRAMES:-}"

mkdir -p "$PLC_SPOOL_DIR"

args=(
  -m pedestrian_line_counter.main
  --backend "$PLC_BACKEND"
  --model "$PLC_MODEL_PATH"
  --queue-size "$PLC_QUEUE_SIZE"
  --queue-policy "$PLC_QUEUE_POLICY"
  --frame-stride "$PLC_FRAME_STRIDE"
  --log-every-seconds 10
  --spool-dir "$PLC_SPOOL_DIR"
  --site-id "$PLC_SITE_ID"
  --camera-id "$PLC_CAMERA_ID"
  --headless-status-every-seconds "$PLC_HEADLESS_STATUS_EVERY_S"
  --no-write
)

if [[ -n "$PLC_TARGET_FPS" ]]; then
  args+=(--target-fps "$PLC_TARGET_FPS")
fi

if [[ -n "$PLC_CONFIDENCE" ]]; then
  args+=(--confidence "$PLC_CONFIDENCE")
fi

case "${PLC_SPOOL_THUMBNAILS}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--spool-thumbnails)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    args+=(--no-spool-thumbnails)
    ;;
  *)
    echo "Unsupported PLC_SPOOL_THUMBNAILS value: $PLC_SPOOL_THUMBNAILS" >&2
    exit 1
    ;;
esac

case "${PLC_SPOOL_SCENE_THUMBNAILS}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--spool-scene-thumbnails)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    args+=(--no-spool-scene-thumbnails)
    ;;
  *)
    echo "Unsupported PLC_SPOOL_SCENE_THUMBNAILS value: $PLC_SPOOL_SCENE_THUMBNAILS" >&2
    exit 1
    ;;
esac

if [[ -n "$PLC_SPOOL_THUMB_MAX_SIDE" ]]; then
  args+=(--spool-thumb-max-side "$PLC_SPOOL_THUMB_MAX_SIDE")
fi

if [[ -n "$PLC_SPOOL_SCENE_THUMB_MAX_SIDE" ]]; then
  args+=(--spool-scene-thumb-max-side "$PLC_SPOOL_SCENE_THUMB_MAX_SIDE")
fi

if [[ -n "$PLC_SPOOL_SCENE_THUMB_QUALITY" ]]; then
  args+=(--spool-scene-thumb-quality "$PLC_SPOOL_SCENE_THUMB_QUALITY")
fi

case "${PLC_ALLOW_ALL_CLASSES}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--allow-all-classes)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    if [[ -n "$PLC_CLASS_IDS" ]]; then
      args+=(--class-ids "$PLC_CLASS_IDS")
    elif [[ -z "$PLC_CLASS_NAMES" ]]; then
      case "$PLC_BACKEND" in
        onnx|tensorrt|torchscript)
          args+=(--class-ids "$PLC_DEFAULT_CLASS_IDS")
          ;;
      esac
    fi
    ;;
  *)
    echo "Unsupported PLC_ALLOW_ALL_CLASSES value: $PLC_ALLOW_ALL_CLASSES" >&2
    exit 1
    ;;
esac

if [[ -n "$PLC_CLASS_NAMES" ]]; then
  args+=(--class-names "$PLC_CLASS_NAMES")
fi

if [[ -n "$PLC_INPUT_PATH" ]]; then
  args+=(--input "$PLC_INPUT_PATH")
  if [[ -n "$PLC_VIDEO_START" ]]; then
    args+=(--video-start "$PLC_VIDEO_START")
  elif [[ "$PLC_PROMPT_VIDEO_START" == "1" || "$PLC_PROMPT_VIDEO_START" == "true" || "$PLC_PROMPT_VIDEO_START" == "TRUE" || "$PLC_PROMPT_VIDEO_START" == "yes" || "$PLC_PROMPT_VIDEO_START" == "YES" || "$PLC_PROMPT_VIDEO_START" == "on" || "$PLC_PROMPT_VIDEO_START" == "ON" || ( "$PLC_PROMPT_VIDEO_START" == "auto" && -t 0 ) ]]; then
    args+=(--prompt-video-start)
  fi
else
  args+=(
    --rtsp-url-env PLC_RTSP_URL
    --rtsp-capture-backend "$PLC_RTSP_CAPTURE_BACKEND"
    --rtsp-transport "$PLC_RTSP_TRANSPORT"
    --rtsp-codec "$PLC_RTSP_CODEC"
    --rtsp-latency-ms "$PLC_RTSP_LATENCY_MS"
    --rtsp-reconnect
    --rtsp-reconnect-max-attempts 0
    --rtsp-reconnect-initial-delay 1.0
    --rtsp-reconnect-max-delay 30.0
    --rtsp-reconnect-backoff 2.0
    --rtsp-stall-timeout 5.0
  )
fi

case "${PLC_PORTAL_UPLOAD_ENABLED}" in
  1|true|TRUE|yes|YES|on|ON)
    : "${PORTAL_API_KEY:?Set PORTAL_API_KEY when PLC_PORTAL_UPLOAD_ENABLED=1}"
    args+=(
      --portal-upload
      --portal-api-base-url "$PLC_PORTAL_API_BASE_URL"
      --portal-api-key-env PORTAL_API_KEY
      --portal-upload-interval-s "$PLC_PORTAL_UPLOAD_INTERVAL_S"
      --portal-upload-max-runs-per-pass "$PLC_PORTAL_UPLOAD_MAX_RUNS_PER_PASS"
      --portal-upload-events-batch-size "$PLC_PORTAL_UPLOAD_EVENTS_BATCH_SIZE"
    )
    case "${PLC_PORTAL_UPLOAD_THUMBNAILS}" in
      1|true|TRUE|yes|YES|on|ON)
        args+=(--portal-upload-thumbnails)
        ;;
      0|false|FALSE|no|NO|off|OFF)
        args+=(--no-portal-upload-thumbnails)
        ;;
      *)
        echo "Unsupported PLC_PORTAL_UPLOAD_THUMBNAILS value: $PLC_PORTAL_UPLOAD_THUMBNAILS" >&2
        exit 1
        ;;
    esac
    case "${PLC_PORTAL_UPLOAD_SCENE_THUMBNAILS}" in
      1|true|TRUE|yes|YES|on|ON)
        args+=(--portal-upload-scene-thumbnails)
        ;;
      0|false|FALSE|no|NO|off|OFF)
        args+=(--no-portal-upload-scene-thumbnails)
        ;;
      *)
        echo "Unsupported PLC_PORTAL_UPLOAD_SCENE_THUMBNAILS value: $PLC_PORTAL_UPLOAD_SCENE_THUMBNAILS" >&2
        exit 1
        ;;
    esac
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported PLC_PORTAL_UPLOAD_ENABLED value: $PLC_PORTAL_UPLOAD_ENABLED" >&2
    exit 1
    ;;
esac

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

case "${PLC_ALLOW_DUPLICATE_LOCAL_INPUT}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--allow-duplicate-local-input)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    echo "Unsupported PLC_ALLOW_DUPLICATE_LOCAL_INPUT value: $PLC_ALLOW_DUPLICATE_LOCAL_INPUT" >&2
    exit 1
    ;;
esac

if [[ -n "$PLC_HEADLESS_STATUS_JSON" ]]; then
  args+=(--headless-status-json "$PLC_HEADLESS_STATUS_JSON")
fi

if [[ -n "$PYTHON_BIN" ]]; then
  exec "$PYTHON_BIN" "${args[@]}"
fi

exec "$UV_BIN" run python "${args[@]}"
