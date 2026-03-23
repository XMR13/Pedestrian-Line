from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlencode

from ._api_common import (
    DEFAULT_REVIEW_PAGE_SIZE,
    DEFAULT_STATE_FILENAME,
    REVIEW_PAGE_SIZE_OPTIONS,
    REVIEW_STATUS_ALL,
    REVIEW_STATUS_PENDING,
    TREND_BUCKET_HOURS,
    TREND_MAX_BUCKETS,
    UI_BASE_PATH,
    UI_STATIC_VERSION,
    UiDateRange,
)
from .review_store import DECISION_NO, DECISION_YES


def _load_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(raw, dict):
        return raw
    return None


def _iter_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text == "":
                continue
            record = json.loads(text)
            if isinstance(record, dict):
                rows.append(record)
    except Exception:
        return []
    return rows


def _build_run_summary(
    run_dir: Path,
    run_meta: Mapping[str, Any],
    status_meta: Optional[Mapping[str, Any]],
    state_meta: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    health = status_meta.get("health_summary") if isinstance(status_meta, Mapping) else None
    if not isinstance(health, Mapping):
        health = run_meta.get("health_summary")
    if not isinstance(health, Mapping):
        health = {}

    delivery_state = _delivery_state_name(state_meta)
    last_sync_at_utc = _latest_delivery_timestamp(
        state_meta,
        "completed_at_utc",
        "in_progress_last_sync_at_utc",
        "scene_upserted_at_utc",
        "thumbs_upserted_at_utc",
        "events_upserted_at_utc",
        "run_finalized_at_utc",
        "run_upserted_at_utc",
    )
    last_error = _mapping_get_text(state_meta, "last_error")

    return {
        "run_uid": _text(run_meta.get("run_uid")),
        "site_id": _text(run_meta.get("site_id")),
        "camera_id": _text(run_meta.get("camera_id")),
        "started_at_utc": _text(run_meta.get("started_at_utc")),
        "updated_at_utc": _coalesce_text(
            (status_meta or {}).get("updated_at_utc") if isinstance(status_meta, Mapping) else None,
            run_meta.get("updated_at_utc"),
            run_meta.get("ended_at_utc"),
        ),
        "ended_at_utc": _text(run_meta.get("ended_at_utc")),
        "source_type": _mapping_get_text(run_meta.get("source"), "type"),
        "source_value": _mapping_get_text(run_meta.get("source"), "value"),
        "line_mode": _text(run_meta.get("line_mode")),
        "delivery_state": delivery_state,
        "delivery_state_label": _delivery_state_label(delivery_state),
        "delivery_state_pill_class": _delivery_state_pill_class(delivery_state),
        "last_sync_at_utc": last_sync_at_utc,
        "last_error": last_error,
        "last_error_at_utc": _mapping_get_text(state_meta, "last_error_at_utc"),
        "last_error_short": _friendly_delivery_error(last_error),
        "retry_recommended": delivery_state in {"pending", "failed"},
        "lifecycle_status": _mapping_get_text(health, "lifecycle_status"),
        "frames_total": _mapping_get_int(health, "frames_total"),
        "frames_processed": _mapping_get_int(health, "frames_processed"),
        "events_emitted_total": _mapping_get_int(health, "events_emitted_total"),
        "count_a_to_b": _mapping_get_int(health, "count_a_to_b"),
        "count_b_to_a": _mapping_get_int(health, "count_b_to_a"),
        "effective_fps": _mapping_get_float(health, "effective_fps"),
        "processed_fps": _mapping_get_float(health, "processed_fps"),
        "run_dir": str(run_dir),
        "report_csv_path": str(run_dir / "report.csv"),
        "report_csv_relpath": _text(run_meta.get("report_csv_relpath")),
        "state_path": str(run_dir / DEFAULT_STATE_FILENAME),
    }


def _build_event_summary(
    run_dir: Path,
    run_meta: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    spool_dir: Path,
) -> Dict[str, Any]:
    thumb_relpath = _text(event.get("thumb_relpath"))
    scene_relpath = _text(event.get("scene_relpath"))
    thumb_path = (run_dir / thumb_relpath) if thumb_relpath else None
    scene_path = (run_dir / scene_relpath) if scene_relpath else None
    return {
        "event_uid": _text(event.get("event_uid")),
        "run_uid": _text(event.get("run_uid")),
        "site_id": _coalesce_text(event.get("site_id"), run_meta.get("site_id")),
        "camera_id": _coalesce_text(event.get("camera_id"), run_meta.get("camera_id")),
        "occurred_at_utc": _text(event.get("occurred_at_utc")),
        "occurred_at_local": _text(event.get("occurred_at_local")),
        "frame_index": _mapping_get_int(event, "frame_index"),
        "video_time_s": _mapping_get_float(event, "video_time_s"),
        "direction": _text(event.get("direction")),
        "track_id": _mapping_get_int(event, "track_id"),
        "class_id": _mapping_get_int(event, "class_id"),
        "class_name": _text(event.get("class_name")),
        "confidence": _mapping_get_float(event, "confidence"),
        "thumb_path": thumb_path.as_posix() if thumb_path is not None else None,
        "scene_path": scene_path.as_posix() if scene_path is not None else None,
        "thumb_relpath": thumb_relpath,
        "scene_relpath": scene_relpath,
        "run_dir": str(run_dir),
        "spool_dir": str(spool_dir),
    }


#The delivery state section 

def _delivery_state_name(state_meta: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(state_meta, Mapping):
        return "pending"
    if _text(state_meta.get("completed_at_utc")):
        return "completed"
    if _text(state_meta.get("in_progress_last_sync_at_utc")):
        return "in_progress"
    if _text(state_meta.get("last_error")):
        return "failed"
    return "pending"


def _delivery_state_label(state_name: Optional[str]) -> str:
    mapping = {
        "pending": "Pending",
        "in_progress": "Sedang dikirim",
        "failed": "Gagal",
        "completed": "Selesai",
    }
    key = _text(state_name) or ""
    return mapping.get(key, "Unknown")


def _delivery_state_pill_class(state_name: Optional[str]) -> str:
    mapping = {
        "pending": "ink",
        "in_progress": "brand",
        "failed": "no",
        "completed": "yes",
    }
    key = _text(state_name) or ""
    return mapping.get(key, "")


def _latest_delivery_timestamp(state_meta: Optional[Mapping[str, Any]], *keys: str) -> Optional[str]:
    if not isinstance(state_meta, Mapping):
        return None

    latest_value: Optional[str] = None
    latest_dt: Optional[datetime] = None
    for key in keys:
        value = _mapping_get_text(state_meta, key)
        parsed = _parse_iso_datetime(value)
        if value is None or parsed is None:
            continue
        if latest_dt is None or parsed > latest_dt:
            latest_dt = parsed
            latest_value = value
    return latest_value


def _friendly_delivery_error(value: Any) -> Optional[str]:
    text = _text(value)
    if text is None:
        return None

    normalized = text.lower()
    if "thumbnail" in normalized or "thumb" in normalized:
        return "Upload thumbnail gagal."
    if "network error" in normalized:
        return "Koneksi backend gagal."
    if "timeout" in normalized:
        return "Koneksi backend timeout."
    if "http 401" in normalized or "http 403" in normalized:
        return "Akses backend ditolak."
    if "http 5" in normalized:
        return "Backend sync sedang bermasalah."
    if "upsert_events" in normalized:
        return "Kirim event gagal."
    if "upsert_run" in normalized:
        return "Sync run gagal."
    return "Gagal kirim data. coba sync ulang."


def _run_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _coalesce_text(row.get("updated_at_utc"), row.get("started_at_utc"), row.get("run_uid")) or "",
        _text(row.get("run_uid")) or "",
    )


def _event_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _coalesce_text(row.get("occurred_at_utc"), row.get("event_uid")) or "",
        _text(row.get("event_uid")) or "",
    )


def _mapping_get_text(mapping: Any, key: str) -> Optional[str]:
    if not isinstance(mapping, Mapping):
        return None
    return _text(mapping.get(key))


def _mapping_get_int(mapping: Any, key: str) -> Optional[int]:
    if not isinstance(mapping, Mapping):
        return None
    try:
        value = mapping.get(key)
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _mapping_get_float(mapping: Any, key: str) -> Optional[float]:
    if not isinstance(mapping, Mapping):
        return None
    try:
        value = mapping.get(key)
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _coalesce_text(*values: Any) -> Optional[str]:
    for value in values:
        text = _text(value)
        if text is not None:
            return text
    return None


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text != "" else None


def _clamp_limit(value: int, max_value: int) -> int:
    return max(1, min(int(value), int(max_value)))


def _serialize_retention_summary(summary: Any) -> Dict[str, Any]:
    return {
        "ok": True,
        "root_dir": str(summary.root_dir),
        "now_utc": summary.now_utc,
        "dry_run": bool(summary.dry_run),
        "max_age_days": int(summary.max_age_days),
        "max_total_bytes": int(summary.max_total_bytes) if summary.max_total_bytes is not None else None,
        "min_free_bytes": int(summary.min_free_bytes) if summary.min_free_bytes is not None else None,
        "state_filename": str(summary.state_filename),
        "scanned_runs": int(summary.scanned_runs),
        "eligible_runs": int(summary.eligible_runs),
        "deleted_runs": int(summary.deleted_runs),
        "protected_runs": int(summary.protected_runs),
        "retained_recent_runs": int(summary.retained_recent_runs),
        "bytes_reclaimable": int(summary.bytes_reclaimable),
        "bytes_deleted": int(summary.bytes_deleted),
        "total_runs_bytes": int(summary.total_runs_bytes),
        "disk_total_bytes": int(summary.disk_total_bytes) if summary.disk_total_bytes is not None else None,
        "disk_used_bytes": int(summary.disk_used_bytes) if summary.disk_used_bytes is not None else None,
        "disk_free_bytes_before": int(summary.disk_free_bytes_before) if summary.disk_free_bytes_before is not None else None,
        "projected_runs_bytes_after": int(summary.projected_runs_bytes_after),
        "projected_disk_free_bytes_after": (
            int(summary.projected_disk_free_bytes_after) if summary.projected_disk_free_bytes_after is not None else None
        ),
        "pressure_bytes_target": int(summary.pressure_bytes_target),
        "pressure_bytes_remaining_after": int(summary.pressure_bytes_remaining_after),
        "items": [_serialize_retention_run_info(item) for item in summary.runs],
    }


def _serialize_retention_run_info(info: Any) -> Dict[str, Any]:
    return {
        "run_uid": info.run_uid,
        "run_dir": str(info.run_dir),
        "size_bytes": int(info.size_bytes),
        "status": str(info.status),
        "reason": str(info.reason),
        "ended_at_utc": info.ended_at_utc,
        "age_days": float(info.age_days) if info.age_days is not None else None,
        "state_path": str(info.state_path) if info.state_path is not None else None,
        "eligible_by_age": bool(info.eligible_by_age),
        "selected_for_deletion": bool(info.selected_for_deletion),
        "deletion_basis": info.deletion_basis,
    }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_ui_context(
    *,
    runtime: Any,
    request: Any,
    page_name: str,
    page_title: str,
    page_subtitle: str,
) -> Dict[str, Any]:
    return {
        "request": request,
        "runtime": runtime,
        "page_name": page_name,
        "page_title": page_title,
        "page_subtitle": page_subtitle,
        "ui_base_path": UI_BASE_PATH,
        "static_base": "/ui-static",
        "ui_static_version": UI_STATIC_VERSION,
        "decision_yes": DECISION_YES,
        "decision_no": DECISION_NO,
        "format_count": _format_count,
        "format_float": _format_float,
        "format_date": _format_date,
        "format_time": _format_time,
        "format_datetime": _format_datetime,
        "display_event_timestamp": _display_event_timestamp,
        "short_event_uid": _short_event_uid,
        "compact_path": _compact_path,
        "review_pill_class": _review_pill_class,
        "review_label": _review_label,
        "status_filter_pending": REVIEW_STATUS_PENDING,
        "status_filter_all": REVIEW_STATUS_ALL,
    }


def _normalize_review_filter(value: Optional[str]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {DECISION_YES, DECISION_NO, REVIEW_STATUS_PENDING, REVIEW_STATUS_ALL}:
        return normalized
    return REVIEW_STATUS_PENDING


def _normalize_review_page_size(value: Any) -> int:
    try:
        page_size = int(value)
    except Exception:
        return DEFAULT_REVIEW_PAGE_SIZE
    if page_size in REVIEW_PAGE_SIZE_OPTIONS:
        return page_size
    return DEFAULT_REVIEW_PAGE_SIZE


def _parse_iso_date(value: Any) -> Optional[date]:
    text = _text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except Exception:
        return None


def _normalize_ui_date_range(*, date_from: Optional[str], date_to: Optional[str]) -> UiDateRange:
    start_date = _parse_iso_date(date_from)
    end_date = _parse_iso_date(date_to)
    if start_date is not None and end_date is not None and start_date > end_date:
        start_date, end_date = end_date, start_date
    start_utc = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc) if start_date is not None else None
    end_utc_exclusive = (
        datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        if end_date is not None
        else None
    )
    return UiDateRange(
        date_from=start_date.isoformat() if start_date is not None else None,
        date_to=end_date.isoformat() if end_date is not None else None,
        start_utc=start_utc,
        end_utc_exclusive=end_utc_exclusive,
    )


def _ui_date_range_context(date_range: UiDateRange) -> Dict[str, Any]:
    if not date_range.active:
        return {
            "active": False,
            "date_from": "",
            "date_to": "",
            "label": "All dates",
            "summary": "All UTC dates",
        }
    if date_range.date_from and date_range.date_to:
        if date_range.date_from == date_range.date_to:
            label = date_range.date_from
            summary = f"UTC day {date_range.date_from}"
        else:
            label = f"{date_range.date_from} to {date_range.date_to}"
            summary = f"UTC range {date_range.date_from} to {date_range.date_to}"
    elif date_range.date_from:
        label = f"From {date_range.date_from}"
        summary = f"UTC from {date_range.date_from}"
    else:
        label = f"Until {date_range.date_to}"
        summary = f"UTC until {date_range.date_to}"
    return {
        "active": True,
        "date_from": date_range.date_from or "",
        "date_to": date_range.date_to or "",
        "label": label,
        "summary": summary,
    }


def _datetime_in_date_range(value: Optional[datetime], date_range: UiDateRange) -> bool:
    if value is None:
        return False
    normalized = value.astimezone(timezone.utc)
    if date_range.start_utc is not None and normalized < date_range.start_utc:
        return False
    if date_range.end_utc_exclusive is not None and normalized >= date_range.end_utc_exclusive:
        return False
    return True


def _ui_query_string(
    *,
    camera_id: Optional[str] = None,
    status: Optional[str] = None,
    event_uid: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    params: Dict[str, str] = {}
    if camera_id:
        params["camera_id"] = str(camera_id).strip()
    normalized_status = _normalize_review_filter(status)
    if normalized_status:
        params["status"] = normalized_status
    if event_uid:
        params["event_uid"] = str(event_uid).strip()
    if page is not None:
        params["page"] = str(max(1, int(page)))
    if page_size is not None:
        params["page_size"] = str(_normalize_review_page_size(page_size))
    normalized_date_range = _normalize_ui_date_range(date_from=date_from, date_to=date_to)
    if normalized_date_range.date_from:
        params["date_from"] = normalized_date_range.date_from
    if normalized_date_range.date_to:
        params["date_to"] = normalized_date_range.date_to
    return urlencode(params)


def _ui_review_queue_url(
    *,
    camera_id: Optional[str] = None,
    status: Optional[str] = None,
    event_uid: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    query = _ui_query_string(
        camera_id=camera_id,
        status=status,
        event_uid=event_uid,
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
    )
    if query:
        return f"{UI_BASE_PATH}/review?{query}"
    return f"{UI_BASE_PATH}/review"


def _ui_event_detail_url(
    event_uid: str,
    *,
    camera_id: Optional[str] = None,
    status: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    query = _ui_query_string(
        camera_id=camera_id,
        status=status,
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
    )
    if query:
        return f"{UI_BASE_PATH}/events/{event_uid}?{query}"
    return f"{UI_BASE_PATH}/events/{event_uid}"


def _review_label(value: Optional[str]) -> str:
    if value == DECISION_YES:
        return "Diterima"
    if value == DECISION_NO:
        return "Ditolak"
    return "Pending"


def _review_pill_class(value: Optional[str]) -> str:
    if value == DECISION_YES:
        return "yes"
    if value == DECISION_NO:
        return "no"
    return ""


def _path_to_public_url(spool_dir: Path, value: Any) -> Optional[str]:
    text = _text(value)
    if text is None:
        return None
    return _relpath_to_public_url(spool_dir, Path(text))


def _relpath_to_public_url(spool_dir: Path, path: Path) -> Optional[str]:
    try:
        relpath = path.resolve().relative_to(spool_dir.resolve())
    except Exception:
        return None
    return "/evidence/" + relpath.as_posix()


def _format_count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def _format_float(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{int(digits)}f}"
    except Exception:
        return "0.0"


def _format_datetime(value: Any) -> str:
    text = _text(value)
    if text is None:
        return "Unavailable"
    if "T" in text:
        return text.replace("T", " ").replace("Z", " UTC")
    return text


def _display_event_timestamp(row: Any) -> Optional[str]:
    if not isinstance(row, Mapping):
        return _text(row)
    return _coalesce_text(row.get("occurred_at_local"), row.get("occurred_at_utc"))


def _format_date(value: Any) -> str:
    text = _text(value)
    if text is None:
        return "Unavailable"
    if "T" in text:
        return text.split("T", 1)[0]
    if " " in text:
        return text.split(" ", 1)[0]
    return text


def _format_time(value: Any) -> str:
    text = _text(value)
    if text is None:
        return "Unavailable"
    if "T" in text:
        time_text = text.split("T", 1)[1]
    elif " " in text:
        time_text = text.split(" ", 1)[1]
    else:
        return "Unavailable"
    return time_text.replace("Z", " UTC")


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    text = _text(value)
    if text is None:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _event_occurrence_utc(event: Mapping[str, Any]) -> Optional[datetime]:
    parsed = _parse_iso_datetime(event.get("occurred_at_utc")) or _parse_iso_datetime(event.get("occurred_at_local"))
    if parsed is None:
        return None
    return parsed.astimezone(timezone.utc)


def _run_occurrence_utc(run: Mapping[str, Any]) -> Optional[datetime]:
    parsed = (
        _parse_iso_datetime(run.get("updated_at_utc"))
        or _parse_iso_datetime(run.get("ended_at_utc"))
        or _parse_iso_datetime(run.get("started_at_utc"))
    )
    if parsed is None:
        return None
    return parsed.astimezone(timezone.utc)


def _empty_dashboard_trend(date_range: UiDateRange) -> Dict[str, Any]:
    range_start, range_end_exclusive = _empty_trend_window_bounds(date_range=date_range)
    bucket_mode = _select_trend_bucket_mode(range_start=range_start, range_end_exclusive=range_end_exclusive)
    buckets = _build_trend_buckets(
        range_start=range_start,
        range_end_exclusive=range_end_exclusive,
        bucket_mode=bucket_mode,
    )
    bucket_count = len(buckets)
    x_positions = _trend_x_positions(bucket_count)
    empty_points = [
        {"x": x_positions[index], "y": _trend_y_position(0, 1), "count": 0, "label": buckets[index]["label"]}
        for index in range(bucket_count)
    ]
    return {
        "empty": True,
        "bucket_hours": bucket_count if bucket_mode == "hour" else None,
        "bucket_mode": bucket_mode,
        "time_basis_label": "Time (UTC)",
        "window_label": _trend_window_label(date_range=date_range, range_start=range_start, range_end_exclusive=range_end_exclusive),
        "buckets": buckets,
        "series": [
            {
                "key": key,
                "label": label,
                "css_class": css_class,
                "path": _trend_svg_path(empty_points),
                "points": list(empty_points),
                "window_total": 0,
                "latest_value": 0,
            }
            for key, label, css_class in (
                ("a_to_b", "A_TO_B", "dir-a"),
                ("b_to_a", "B_TO_A", "dir-b"),
                ("pending", "Pending", "no"),
            )
        ],
        "grid_lines": _trend_grid_lines(1),
        "window_start_label": _format_datetime(range_start.isoformat().replace("+00:00", "Z")),
        "window_end_label": _format_datetime((range_end_exclusive - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")),
        "window_totals": {"a_to_b": 0, "b_to_a": 0, "pending": 0},
    }


def _empty_trend_window_bounds(*, date_range: UiDateRange) -> tuple[datetime, datetime]:
    if date_range.active:
        if date_range.start_utc is not None and date_range.end_utc_exclusive is not None:
            return date_range.start_utc, date_range.end_utc_exclusive
        if date_range.start_utc is not None:
            return date_range.start_utc, date_range.start_utc + timedelta(days=1)
        if date_range.end_utc_exclusive is not None:
            return date_range.end_utc_exclusive - timedelta(days=1), date_range.end_utc_exclusive
    latest_bucket = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return latest_bucket - timedelta(hours=TREND_BUCKET_HOURS - 1), latest_bucket + timedelta(hours=1)


def _trend_window_bounds(
    *,
    date_range: UiDateRange,
    points: List[tuple[datetime, Dict[str, Any]]],
) -> tuple[datetime, datetime]:
    if date_range.start_utc is not None:
        range_start = date_range.start_utc
    else:
        range_start = min(item[0] for item in points)
    if date_range.end_utc_exclusive is not None:
        range_end_exclusive = date_range.end_utc_exclusive
    else:
        latest_point = max(item[0] for item in points)
        range_end_exclusive = latest_point + timedelta(seconds=1)
    return range_start, range_end_exclusive


def _select_trend_bucket_mode(*, range_start: datetime, range_end_exclusive: datetime) -> str:
    total_days = max(1.0, (range_end_exclusive - range_start).total_seconds() / 86400.0)
    if total_days <= 1.5:
        return "hour"
    if total_days <= 31:
        return "day"
    return "month"


def _align_datetime_to_bucket(value: datetime, *, bucket_mode: str) -> datetime:
    normalized = value.astimezone(timezone.utc)
    if bucket_mode == "hour":
        return normalized.replace(minute=0, second=0, microsecond=0)
    if bucket_mode == "day":
        return normalized.replace(hour=0, minute=0, second=0, microsecond=0)
    return normalized.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _bucket_step(bucket_mode: str) -> timedelta:
    if bucket_mode == "hour":
        return timedelta(hours=1)
    return timedelta(days=1)


def _build_trend_buckets(
    *,
    range_start: datetime,
    range_end_exclusive: datetime,
    bucket_mode: str,
) -> List[Dict[str, Any]]:
    aligned_start = _align_datetime_to_bucket(range_start, bucket_mode=bucket_mode)
    buckets: List[Dict[str, Any]] = []
    current = aligned_start
    while current < range_end_exclusive and len(buckets) < TREND_MAX_BUCKETS:
        if bucket_mode == "month":
            next_bucket = _month_start(current, 1)
            label = current.strftime("%Y-%m")
        elif bucket_mode == "day":
            next_bucket = current + timedelta(days=1)
            label = current.strftime("%m-%d")
        else:
            next_bucket = current + timedelta(hours=1)
            label = current.strftime("%H:%M")
        buckets.append(
            {
                "start": current,
                "end": next_bucket,
                "label": label,
                "a_to_b": 0,
                "b_to_a": 0,
                "pending": 0,
            }
        )
        current = next_bucket
    if not buckets:
        buckets.append(
            {
                "start": aligned_start,
                "end": aligned_start + _bucket_step(bucket_mode),
                "label": aligned_start.strftime("%H:%M" if bucket_mode == "hour" else "%m-%d"),
                "a_to_b": 0,
                "b_to_a": 0,
                "pending": 0,
            }
        )
    return buckets


def _month_start(value: datetime, offset: int = 0) -> datetime:
    base = value.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_index = (base.year * 12 + (base.month - 1)) + offset
    year = month_index // 12
    month = (month_index % 12) + 1
    return datetime(year, month, 1, tzinfo=timezone.utc)


def _trend_window_label(
    *,
    date_range: UiDateRange,
    range_start: datetime,
    range_end_exclusive: datetime,
) -> str:
    if date_range.date_from and date_range.date_to:
        if date_range.date_from == date_range.date_to:
            return f"UTC day {date_range.date_from}"
        return f"UTC range {date_range.date_from} to {date_range.date_to}"
    if date_range.date_from:
        return f"UTC from {date_range.date_from}"
    if date_range.date_to:
        return f"UTC until {date_range.date_to}"
    day_span = max(1.0, (range_end_exclusive - range_start).total_seconds() / 86400.0)
    if day_span <= 1.5:
        return f"Last {TREND_BUCKET_HOURS} hours"
    return f"Observed range ({range_start.date().isoformat()} to {(range_end_exclusive - timedelta(days=1)).date().isoformat()})"


def _trend_x_positions(bucket_count: int) -> List[int]:
    chart_left = 60
    chart_right = 720
    if bucket_count <= 1:
        return [chart_right]
    span = chart_right - chart_left
    return [
        chart_left + round(span * index / (bucket_count - 1))
        for index in range(bucket_count)
    ]


def _trend_y_position(value: int, y_max: int) -> int:
    top = 26
    bottom = 154
    if y_max <= 0:
        return bottom
    ratio = max(0.0, min(1.0, float(value) / float(y_max)))
    return round(bottom - ((bottom - top) * ratio))


def _trend_svg_path(points: List[Mapping[str, Any]]) -> str:
    if not points:
        return ""
    commands = [f"M{int(points[0]['x'])} {int(points[0]['y'])}"]
    commands.extend(f"L{int(point['x'])} {int(point['y'])}" for point in points[1:])
    return " ".join(commands)


def _trend_grid_lines(y_max: int) -> List[Dict[str, int]]:
    values = sorted({y_max, max(0, round(y_max * 0.6)), max(0, round(y_max * 0.2))}, reverse=True)
    return [
        {"value": int(value), "y": _trend_y_position(int(value), y_max)}
        for value in values
    ]


def _short_event_uid(value: Any, head: int = 10, tail: int = 8) -> str:
    text = _text(value)
    if text is None:
        return "Unavailable"
    if len(text) <= int(head) + int(tail) + 3:
        return text
    return f"{text[:int(head)]}...{text[-int(tail):]}"


def _compact_path(value: Any, parts: int = 2) -> str:
    text = _text(value)
    if text is None:
        return "Unavailable"
    normalized = text.replace("\\", "/").rstrip("/")
    if normalized == "":
        return text
    path_parts = [part for part in normalized.split("/") if part]
    if len(path_parts) <= max(1, int(parts)):
        return text
    return ".../" + "/".join(path_parts[-max(1, int(parts)):])
