from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

PORTAL_CONTRACT_VERSION = "v1"


class PortalContractError(ValueError):
    """Raised when spool data cannot be mapped to portal contract payloads."""


def load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    path = Path(run_dir) / "run.json"
    if not path.exists():
        raise PortalContractError(f"run metadata not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PortalContractError(f"invalid run.json in {path}: {exc}") from exc


def iter_event_records(run_dir: Path) -> Iterator[Dict[str, Any]]:
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        raise PortalContractError(f"event stream not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            try:
                obj = json.loads(row)
            except json.JSONDecodeError as exc:
                # In live mode, uploader may read while writer is appending the last line.
                # If the final line is not newline-terminated, treat it as an in-progress
                # partial write and ignore for this pass.
                if not line.endswith("\n"):
                    break
                raise PortalContractError(f"invalid JSON on {path}:{i}: {exc}") from exc
            if not isinstance(obj, dict):
                raise PortalContractError(f"invalid event object on {path}:{i}: expected JSON object")
            yield obj


def load_event_records(run_dir: Path) -> List[Dict[str, Any]]:
    return list(iter_event_records(run_dir))


def build_run_upsert_payload(run_meta: Mapping[str, Any]) -> Dict[str, Any]:
    run_uid = _required_text(run_meta, "run_uid")
    site_id = _required_text(run_meta, "site_id")
    camera_id = _required_text(run_meta, "camera_id")

    source = run_meta.get("source") if isinstance(run_meta.get("source"), Mapping) else {}
    source_type = _coalesce_text(run_meta.get("source_type"), source.get("type"))
    source_value = _coalesce_text(run_meta.get("source_value"), source.get("value"))

    frame_size = run_meta.get("frame_size") if isinstance(run_meta.get("frame_size"), Mapping) else {}
    frame_width = _coalesce_int(run_meta.get("frame_width"), frame_size.get("width"))
    frame_height = _coalesce_int(run_meta.get("frame_height"), frame_size.get("height"))

    health_summary = run_meta.get("health_summary")
    if not isinstance(health_summary, Mapping):
        health_summary = None

    ended_at_utc = _coalesce_text(run_meta.get("ended_at_utc"), None)
    if ended_at_utc is None and health_summary is not None:
        ended_at_utc = _coalesce_text(health_summary.get("ended_at_utc"), None)

    payload: Dict[str, Any] = {
        "contract_version": PORTAL_CONTRACT_VERSION,
        "run_uid": run_uid,
        "site_id": site_id,
        "camera_id": camera_id,
        "started_at_utc": _optional_text(run_meta.get("started_at_utc")),
        "ended_at_utc": ended_at_utc,
        "source_type": source_type,
        "source_value": source_value,
        "model_version": _optional_text(run_meta.get("model_version")),
        "cfg_version": _optional_text(run_meta.get("cfg_version")),
        "line_mode": _optional_text(run_meta.get("line_mode")),
        "line_id": _optional_text(run_meta.get("line_id")),
        "fps": _optional_float(run_meta.get("fps")),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "health_summary_json": health_summary,
        "report_csv_relpath": _optional_text(run_meta.get("report_csv_relpath")),
    }
    return payload


def build_event_upsert_payload(event: Mapping[str, Any]) -> Dict[str, Any]:
    event_uid = _required_text(event, "event_uid")
    run_uid = _required_text(event, "run_uid")
    site_id = _required_text(event, "site_id")
    camera_id = _required_text(event, "camera_id")

    bbox = event.get("bbox_xyxy")
    if bbox is None:
        bbox = event.get("bbox")
    bbox_xyxy = _optional_bbox_xyxy(bbox)

    payload: Dict[str, Any] = {
        "contract_version": PORTAL_CONTRACT_VERSION,
        "event_uid": event_uid,
        "run_uid": run_uid,
        "site_id": site_id,
        "camera_id": camera_id,
        "occurred_at_utc": _optional_text(event.get("occurred_at_utc")),
        "frame_index": _optional_int(event.get("frame_index")),
        "video_time_s": _optional_float(event.get("video_time_s")),
        "direction": _optional_text(event.get("direction")),
        "track_id": _optional_int(event.get("track_id")),
        "class_id": _optional_int(event.get("class_id")),
        "class_name": _optional_text(event.get("class_name")),
        "confidence": _optional_float(event.get("confidence")),
        "bbox_xyxy": bbox_xyxy,
        "line_mode": _optional_text(event.get("line_mode")),
        "occurred_at_utc_source": _optional_text(event.get("occurred_at_utc_source")),
        "thumb_relpath": _optional_text(event.get("thumb_relpath")),
        "scene_relpath": _optional_text(event.get("scene_relpath")),
    }
    return payload


def build_events_batch_payload(events: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [build_event_upsert_payload(e) for e in events]
    return {
        "contract_version": PORTAL_CONTRACT_VERSION,
        "events": rows,
    }


def split_batches(rows: List[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    size = max(int(batch_size), 1)
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = _optional_text(row.get(key))
    if value is None:
        raise PortalContractError(f"missing required field '{key}'")
    return value


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _coalesce_text(*values: Any) -> Optional[str]:
    for value in values:
        out = _optional_text(value)
        if out is not None:
            return out
    return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PortalContractError(f"invalid int value: {value!r}") from exc


def _coalesce_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None or value == "":
            continue
        return _optional_int(value)
    return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PortalContractError(f"invalid float value: {value!r}") from exc


def _optional_bbox_xyxy(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise PortalContractError("bbox must be a list/tuple of four numbers")
    out: List[int] = []
    for item in value:
        out.append(int(item))
    return out
