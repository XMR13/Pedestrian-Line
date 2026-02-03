from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _coerce_names(obj: Any) -> Dict[int, str]:
    # Accept:
    # - {"names": {0: "car", 1: "truck"}}
    # - {"names": ["car", "truck"]}
    # - {0: "car", 1: "truck"}
    # - ["car", "truck"]
    if isinstance(obj, dict) and "names" in obj:
        obj = obj["names"]

    if isinstance(obj, list):
        out: Dict[int, str] = {}
        for idx, name in enumerate(obj):
            out[int(idx)] = str(name)
        return out

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out

    return {}


def _try_load_yaml(path: Path) -> Optional[Dict[int, str]]:
    try:
        import yaml  # type: ignore
    except Exception:
        return None

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    names = _coerce_names(data)
    return names or None


def _load_minimal_names_yaml(path: Path) -> Dict[int, str]:
    """
    Minimal parser for the common pattern:

        names:
          0: class_a
          1: class_b
    """
    names: Dict[int, str] = {}
    in_names = False

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "names:":
            in_names = True
            continue
        if not in_names:
            continue
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        left = left.strip()
        right = right.strip().strip("'").strip('"')
        if not left.isdigit():
            continue
        names[int(left)] = right

    return names


def load_class_names(metadata_path: str) -> Dict[int, str]:
    """
    Load class names from YAML/JSON.

    Supported inputs:
    - Ultralytics-style YAML: `names: {0: car, 1: truck}` or `names: [car, truck]`
    - Simple mapping JSON: same shapes as above
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {metadata_path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        names = _coerce_names(data)
        if names:
            return names
        return {}

    yaml_names = _try_load_yaml(path)
    if yaml_names is not None:
        return yaml_names

    return _load_minimal_names_yaml(path)
