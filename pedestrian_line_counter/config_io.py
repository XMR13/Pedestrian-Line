from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Type, Union, get_args, get_origin, get_type_hints

from .config import AppConfig, ROOT_DIR


def load_config_overrides(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file containing partial config overrides.

    Example shape:
    {
      "model": {"backend": "onnx", "confidence_threshold": 0.45, "track_class_ids": [0,1]},
      "tracker": {"max_distance": 100.0},
      "line": {"start_norm": [0.35, 0.45], "end_norm": [0.85, 0.75]},
      "io": {"input_path": "media/input.mp4", "output_path": "media/output.mp4"}
    }
    """

    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config override JSON must be a JSON object at the top level.")
    return data  # type: ignore[return-value]

def split_overrides(data: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split a config override mapping into:
    - app overrides (applied to AppConfig dataclasses)
    - extra overrides (reserved for future non-AppConfig keys)

    Current behavior:
    - If top-level key "app" exists, it is used as the AppConfig override block.
    - Otherwise, the entire mapping is treated as AppConfig overrides (backwards compatible),
      except for top-level reserved keys.
    """

    if not isinstance(data, Mapping):
        raise TypeError("Config override data must be a mapping.")

    reserved = {"run", "runtime"}  # not implemented yet, but keep out of AppConfig.
    if "app" in data:
        app = data.get("app")
        if not isinstance(app, Mapping):
            raise TypeError("Config override key 'app' must be an object.")
        extra = {k: v for k, v in data.items() if k != "app"}
        return dict(app), dict(extra)

    app2 = {k: v for k, v in data.items() if k not in reserved}
    extra2 = {k: v for k, v in data.items() if k in reserved}
    return dict(app2), dict(extra2)


def apply_config_overrides(
    cfg: AppConfig,
    overrides: Mapping[str, Any],
    *,
    path_base_dir: Path = ROOT_DIR,
) -> None:
    """
    Apply nested overrides onto an AppConfig in-place.

    Notes:
    - Unknown keys raise to catch typos early.
    - Relative paths are resolved against `path_base_dir` (defaults to repo root).
    - Precedence is controlled by the caller (typically: defaults < overrides < CLI flags).
    """

    if not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping")
    _apply_dataclass_overrides(cfg, overrides, path_base_dir=path_base_dir, prefix="")  # type: ignore[arg-type]


def app_config_to_dict(cfg: AppConfig) -> Dict[str, Any]:
    """
    Convert the resolved config into a JSON-serializable dict.
    """

    return _dataclass_to_jsonable(cfg)  # type: ignore[return-value]


def write_app_config_json(path: Path, cfg: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(app_config_to_dict(cfg), indent=2, sort_keys=True) + "\n")


def _dataclass_to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        out: Dict[str, Any] = {}
        for f in fields(obj):
            out[f.name] = _dataclass_to_jsonable(getattr(obj, f.name))
        return out
    if isinstance(obj, Path):
        try:
            return obj.relative_to(ROOT_DIR).as_posix()
        except Exception:
            return str(obj)
    if isinstance(obj, tuple):
        return [_dataclass_to_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_dataclass_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _dataclass_to_jsonable(v) for k, v in obj.items()}
    return obj


def _apply_dataclass_overrides(
    obj: Any,
    overrides: Mapping[str, Any],
    *,
    path_base_dir: Path,
    prefix: str,
) -> None:
    if not is_dataclass(obj):
        raise TypeError(f"Target for overrides is not a dataclass at '{prefix or '<root>'}'.")
    type_hints = _get_type_hints(type(obj))

    known = {f.name for f in fields(obj)}
    for key in overrides.keys():
        if key not in known:
            raise KeyError(f"Unknown config key: {prefix}{key}")

    for key, value in overrides.items():
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise TypeError(f"Expected object for '{prefix}{key}', got {type(value).__name__}")
            _apply_dataclass_overrides(current, value, path_base_dir=path_base_dir, prefix=f"{prefix}{key}.")
            continue

        annotation = type_hints.get(key, Any)
        converted = _convert_value(annotation, value, path_base_dir=path_base_dir, key_path=f"{prefix}{key}")
        setattr(obj, key, converted)


_TYPE_HINTS_CACHE: Dict[Type[Any], Dict[str, Any]] = {}


def _get_type_hints(cls: Type[Any]) -> Dict[str, Any]:
    cached = _TYPE_HINTS_CACHE.get(cls)
    if cached is not None:
        return cached
    module = sys.modules.get(cls.__module__)
    globalns = vars(module) if module is not None else None
    hints = get_type_hints(cls, globalns=globalns)
    _TYPE_HINTS_CACHE[cls] = hints
    return hints


def _convert_value(annotation: Any, value: Any, *, path_base_dir: Path, key_path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[T]
    if origin is Union and type(None) in args:
        inner = [a for a in args if a is not type(None)][0]
        if value is None:
            return None
        return _convert_value(inner, value, path_base_dir=path_base_dir, key_path=key_path)

    if annotation is Path:
        return _coerce_path(value, path_base_dir=path_base_dir, key_path=key_path)

    if origin is list:
        inner = args[0] if args else Any
        if isinstance(value, str) and key_path.endswith("track_class_ids"):
            items = [x.strip() for x in value.split(",") if x.strip()]
            return [int(x) for x in items]
        if not isinstance(value, list):
            raise TypeError(f"Expected list for '{key_path}', got {type(value).__name__}")
        return [_convert_value(inner, v, path_base_dir=path_base_dir, key_path=f"{key_path}[]") for v in value]

    if origin is tuple:
        # Support fixed-length tuples used in this project (e.g. (float,float)).
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected tuple/list for '{key_path}', got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            inner = args[0]
            return tuple(_convert_value(inner, v, path_base_dir=path_base_dir, key_path=f"{key_path}[]") for v in value)
        if args and len(value) != len(args):
            raise ValueError(f"Expected {len(args)} items for '{key_path}', got {len(value)}")
        out_items = []
        for i, v in enumerate(value):
            inner = args[i] if args else Any
            out_items.append(_convert_value(inner, v, path_base_dir=path_base_dir, key_path=f"{key_path}[{i}]"))
        return tuple(out_items)

    # Primitive-ish types
    if annotation is bool:
        return bool(value)
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            return str(value)
        return _expand_env_refs(value, key_path=key_path)

    # Fallback: leave as-is.
    return value


def _coerce_path(value: Any, *, path_base_dir: Path, key_path: str) -> Path:
    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected string path for '{key_path}', got {type(value).__name__}")
    p = Path(value)
    if p.is_absolute():
        return p
    return (path_base_dir / p)


def _expand_env_refs(value: str, *, key_path: str) -> str:
    """
    Expand env var references in string values.

    Supported forms:
    - "env:VAR_NAME"
    - "${VAR_NAME}"  (exact match)

    If the env var is missing, raises a ValueError with a clear message.
    """

    s = (value or "").strip()
    var: Optional[str] = None
    if s.startswith("env:") and len(s) > 4:
        var = s[4:].strip()
    elif s.startswith("${") and s.endswith("}") and s.count("${") == 1:
        inner = s[2:-1].strip()
        if inner:
            var = inner

    if not var:
        return value

    resolved = os.environ.get(var)
    if resolved is None or resolved == "":
        raise ValueError(f"Missing environment variable '{var}' required by config key '{key_path}'.")
    return resolved
