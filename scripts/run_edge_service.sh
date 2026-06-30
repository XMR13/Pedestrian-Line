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

PLC_SPOOL_DIR="${PLC_SPOOL_DIR:-/var/lib/pedline/traffic_runs}"
PLC_SERVICE_HOST="${PLC_SERVICE_HOST:-127.0.0.1}"
PLC_SERVICE_PORT="${PLC_SERVICE_PORT:-8080}"
PLC_SERVICE_EXPOSURE="${PLC_SERVICE_EXPOSURE:-loopback}"
PLC_SERVICE_DOCS="${PLC_SERVICE_DOCS:-}"
PLC_SERVICE_TRUSTED_HOSTS="${PLC_SERVICE_TRUSTED_HOSTS:-}"
PLC_SERVICE_TITLE="${PLC_SERVICE_TITLE:-Pedestrian Line Edge Service}"
PLC_SERVICE_RETENTION_ENABLED="${PLC_SERVICE_RETENTION_ENABLED:-1}"
PLC_SERVICE_RETENTION_MAX_AGE_DAYS="${PLC_SERVICE_RETENTION_MAX_AGE_DAYS:-90}"
PLC_SERVICE_RETENTION_AUTO_INTERVAL_S="${PLC_SERVICE_RETENTION_AUTO_INTERVAL_S:-3600}"
PLC_PORTAL_API_BASE_URL="${PLC_PORTAL_API_BASE_URL:-}"

mkdir -p "$PLC_SPOOL_DIR"

reject_placeholder_secret() {
  local name="$1"
  local value="$2"
  local lowered

  lowered="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  case "$lowered" in
    replace-me|replace_me|changeme|change-me|password|admin|secret|test|demo)
      echo "Refusing placeholder value for $name. Set a real secret in /etc/vehicle_count/edge_service.env." >&2
      exit 1
      ;;
  esac
}

if [[ "$PLC_SERVICE_EXPOSURE" == "lan" ]]; then
  : "${EDGE_UI_PASSWORD:?Set EDGE_UI_PASSWORD for LAN/IP service exposure}"
  : "${EDGE_SERVICE_API_KEY:?Set EDGE_SERVICE_API_KEY for LAN/IP service exposure}"
  : "${PLC_SERVICE_TRUSTED_HOSTS:?Set PLC_SERVICE_TRUSTED_HOSTS for LAN/IP service exposure}"
  reject_placeholder_secret "EDGE_UI_PASSWORD" "$EDGE_UI_PASSWORD"
  reject_placeholder_secret "EDGE_SERVICE_API_KEY" "$EDGE_SERVICE_API_KEY"
fi

args=(
  -m pedestrian_line_counter.service
  --spool-dir "$PLC_SPOOL_DIR"
  --host "$PLC_SERVICE_HOST"
  --port "$PLC_SERVICE_PORT"
  --title "$PLC_SERVICE_TITLE"
  --service-exposure "$PLC_SERVICE_EXPOSURE"
  --spool-retention-max-age-days "$PLC_SERVICE_RETENTION_MAX_AGE_DAYS"
  --spool-retention-auto-interval-s "$PLC_SERVICE_RETENTION_AUTO_INTERVAL_S"
)

case "${PLC_SERVICE_RETENTION_ENABLED}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--spool-retention-enabled)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    args+=(--no-spool-retention-enabled)
    ;;
  *)
    echo "Unsupported PLC_SERVICE_RETENTION_ENABLED value: $PLC_SERVICE_RETENTION_ENABLED" >&2
    exit 1
    ;;
esac

docs_enabled="$PLC_SERVICE_DOCS"
if [[ -z "$docs_enabled" ]]; then
  if [[ "$PLC_SERVICE_EXPOSURE" == "lan" ]]; then
    docs_enabled="0"
  else
    docs_enabled="1"
  fi
fi

case "$docs_enabled" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--service-docs)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    args+=(--no-service-docs)
    ;;
  *)
    echo "Unsupported PLC_SERVICE_DOCS value: $docs_enabled" >&2
    exit 1
    ;;
esac

if [[ -n "$PLC_SERVICE_TRUSTED_HOSTS" ]]; then
  IFS=',' read -r -a trusted_hosts <<< "$PLC_SERVICE_TRUSTED_HOSTS"
  for host in "${trusted_hosts[@]}"; do
    trimmed="${host#"${host%%[![:space:]]*}"}"
    trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
    if [[ -n "$trimmed" ]]; then
      args+=(--service-trusted-host "$trimmed")
    fi
  done
fi

if [[ -n "$PLC_PORTAL_API_BASE_URL" ]]; then
  args+=(--api-base-url "$PLC_PORTAL_API_BASE_URL")
fi

if [[ -n "$PYTHON_BIN" ]]; then
  exec "$PYTHON_BIN" "${args[@]}"
fi

exec "$UV_BIN" run python "${args[@]}"
