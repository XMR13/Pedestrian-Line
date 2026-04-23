from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

def is_sensitive_source_type(source_type: Any) -> bool:
    normalized = _normalize_source_type(source_type)
    return normalized in {"rtsp", "rtsps"}

def sanitize_source_value(
    source_value: Any,
    *,
    source_type: Any = None,
    camera_id: Any = None,
) -> Optional[str]:
    value = _optional_text(source_value)
    if value is None:
        return None
    if not is_sensitive_source_type(source_type):
        return value
    camera = _optional_text(camera_id)
    if camera is not None:
        return f"camera:{camera}"
    return "live_stream"


def sanitize_source_record(
    source: Mapping[str, Any],
    *,
    camera_id: Any = None,
) -> Dict[str, Optional[str]]:
    source_type = _optional_text(source.get("type")) if isinstance(source, Mapping) else None
    source_value = _optional_text(source.get("value")) if isinstance(source, Mapping) else None
    return {
        "type": source_type,
        "value": sanitize_source_value(source_value, source_type=source_type, camera_id=camera_id),
    }

def safe_source_label(
    source_value: Any,
    *,
    source_type: Any = None,
    camera_id: Any = None,
) -> str:
    return sanitize_source_value(source_value, source_type=source_type, camera_id=camera_id) or "source"

#just the same as the optional text but we need to return is as a lower case letters
def _normalize_source_type(value: Any) -> Optional[str]:
    text = _optional_text(value)
    if text is None:
        return None
    return text.lower()

#semus source of value harus dibuat dan diasumsikan sensitiv, jadi akan lebih baik jika ada helper ucntion 
#yang dapat menconvert menjadi single optional text
def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    return text if text else None


__all__ = [
    "is_sensitive_source_type",
    "safe_source_label",
    "sanitize_source_record",
    "sanitize_source_value",
]
