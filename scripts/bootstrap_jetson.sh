#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-jetson.txt

# Install the local package without resolving pyproject dependencies. Jetson
# TensorRT runtime uses system TensorRT/OpenCV and the runtime requirements
# above, while pyproject.toml remains the desktop/dev source of truth.
python -m pip install -e . --no-deps

python - <<'PY'
import importlib

checks = [
    ("pedestrian_line_counter", "local package"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("fastapi", "FastAPI"),
    ("tensorrt", "TensorRT"),
]

missing = []
for module, label in checks:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{label} ({module}): {exc}")

try:
    from cuda.bindings import runtime as cudart  # noqa: F401
except Exception:
    try:
        from cuda import cudart  # noqa: F401
    except Exception as exc:
        missing.append(f"CUDA Python runtime (cuda): {exc}")

if missing:
    print("[bootstrap_jetson] installed, but these imports still need attention:")
    for item in missing:
        print(f"  - {item}")
    raise SystemExit(1)

print("[bootstrap_jetson] ok: Jetson runtime imports are available")
PY

cat <<'EOF'

Next smoke test:

  python -m pedestrian_line_counter.main --help

TensorRT preview:

  python scripts/preview_video.py \
    --input media/test.mp4 \
    --model Models/model_fp16.engine \
    --backend tensorrt \
    --allow-all-classes \
    --max-frames 30 \
    --save-frames \
    --frames-dir preview_outputs/trt_test \
    --no-progress
EOF
