from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Optional

WIB = timezone(timedelta(hours=7), name="WIB")
WIB_NAME = "WIB"


def local_timezone() -> tzinfo:
    return WIB


def local_timezone_name() -> str:
    return WIB_NAME


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def iso_local(ts: float, tz: Optional[tzinfo] = None) -> str:
    return datetime.fromtimestamp(ts, tz=tz or WIB).isoformat()


def day_string_local(ts: float, tz: Optional[tzinfo] = None) -> str:
    return datetime.fromtimestamp(ts, tz=tz or WIB).strftime("%Y-%m-%d")


def utc_iso_to_local_text(value: Any, tz: Optional[tzinfo] = None) -> Optional[str]:
    text = _optional_text(value)
    if text is None:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(tz or WIB).isoformat()


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


__all__ = [
    "WIB",
    "WIB_NAME",
    "day_string_local",
    "iso_local",
    "iso_utc",
    "local_timezone",
    "local_timezone_name",
    "utc_iso_to_local_text",
]
