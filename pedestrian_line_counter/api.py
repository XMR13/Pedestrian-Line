from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import secrets
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import parse_qs, urlencode

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import ROOT_DIR, ServiceConfig, SpoolRetentionConfig
from .event_uploader import (
    DeliveryApiClient,
    UploaderConfig,
    iter_spool_runs,
    process_pending_runs,
    process_single_run,
)
from .review_store import DECISION_NO, DECISION_YES, ReviewStore
from .spool_retention import apply_retention_policy
from .ui_auth import UiAuthConfig, issue_session_token, validate_session_token

## Import all the necessary helpers and common definitions for the API implementation
from ._api_common import (
    DEFAULT_MUTATION_API_KEY_HEADER,
    DEFAULT_REVIEW_DB_FILENAME,
    DEFAULT_REVIEW_PAGE_SIZE,
    DEFAULT_STATE_FILENAME,
    LoginRequest,
    MAX_EVENTS_LIMIT,
    MAX_RUNS_LIMIT,
    MutationAuthConfig,
    REVIEW_PAGE_SIZE_OPTIONS,
    REVIEW_STATUS_ALL,
    REVIEW_STATUS_PENDING,
    RetentionRequest,
    ReviewUpdateRequest,
    SingleRunSyncRequest,
    SyncRequest,
    UI_ASSET_DIR,
    UI_BASE_PATH,
    UI_STATIC_DIR,
    UI_TEMPLATE_DIR,
    UiDateRange,
)
from ._api_helpers import (
    _build_event_summary,
    _build_run_summary,
    _build_ui_context,
    _clamp_limit,
    _coalesce_text,
    _compact_path,
    _datetime_in_date_range,
    _delivery_state_label,
    _delivery_state_pill_class,
    _display_event_timestamp,
    _empty_dashboard_trend,
    _event_occurrence_utc,
    _event_sort_key,
    _format_count,
    _format_date,
    _format_datetime,
    _format_float,
    _format_time,
    _load_json_dict,
    _mapping_get_float,
    _mapping_get_int,
    _month_start,
    _normalize_review_filter,
    _normalize_review_page_size,
    _normalize_ui_date_range,
    _parse_iso_datetime,
    _path_to_public_url,
    _relpath_to_public_url,
    _review_label,
    _review_pill_class,
    _run_occurrence_utc,
    _run_sort_key,
    _serialize_retention_summary,
    _serialize_retention_run_info,
    _select_trend_bucket_mode,
    _short_event_uid,
    _text,
    _trend_grid_lines,
    _trend_svg_path,
    _trend_window_bounds,
    _trend_window_label,
    _trend_x_positions,
    _trend_y_position,
    _ui_date_range_context,
    _ui_event_detail_url,
    _ui_query_string,
    _ui_review_queue_url,
    _utcnow_iso,
    _iter_jsonl_records,
    _align_datetime_to_bucket,
    _build_trend_buckets,
)


@dataclass
class EdgeApiRuntime:
    spool_dir: Path
    uploader_cfg: Optional[UploaderConfig] = None
    retention_cfg: SpoolRetentionConfig = field(default_factory=SpoolRetentionConfig)
    service_cfg: ServiceConfig = field(default_factory=ServiceConfig)
    mutation_auth_cfg: MutationAuthConfig = field(default_factory=MutationAuthConfig)
    ui_auth_cfg: UiAuthConfig = field(default_factory=UiAuthConfig)
    review_store: ReviewStore = field(default_factory=lambda: ReviewStore(ROOT_DIR / DEFAULT_REVIEW_DB_FILENAME))
    service_started_at_utc: str = field(default_factory=lambda: _utcnow_iso())

    def health_payload(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "service_started_at_utc": self.service_started_at_utc,
            "spool_dir": str(self.spool_dir),
            "spool_exists": self.spool_dir.exists(),
            "uploader_enabled": self.uploader_cfg is not None,
            "retention_enabled": bool(self.retention_cfg.enabled),
            "service_exposure_mode": str(self.service_cfg.exposure_mode),
            "docs_enabled": bool(self.service_cfg.enable_docs),
            "mutation_auth_enabled": self.mutation_auth_cfg.enabled(),
            "ui_auth_enabled": self.ui_auth_cfg.enabled(),
        }

    def status_payload(self) -> Dict[str, Any]:
        run_summaries = list(self._iter_all_runs())
        run_summaries.sort(key=_run_sort_key, reverse=True)
        counts = self._collect_delivery_counts(run_summaries=run_summaries)
        review_counts = self.review_store.summary()
        return {
            "ok": True,
            "service_started_at_utc": self.service_started_at_utc,
            "spool_dir": str(self.spool_dir),
            "spool_exists": self.spool_dir.exists(),
            "uploader_enabled": self.uploader_cfg is not None,
            "retention_enabled": bool(self.retention_cfg.enabled),
            "service_exposure_mode": str(self.service_cfg.exposure_mode),
            "docs_enabled": bool(self.service_cfg.enable_docs),
            "mutation_auth_enabled": self.mutation_auth_cfg.enabled(),
            "ui_auth_enabled": self.ui_auth_cfg.enabled(),
            "runs_total": counts["runs_total"],
            "delivery_state_counts": {
                "completed": counts["completed"],
                "pending": counts["pending"],
                "failed": counts["failed"],
                "in_progress": counts["in_progress"],
                "unknown": counts["unknown"],
            },
            "sync_overview": self._build_sync_overview(run_summaries=run_summaries, counts=counts),
            "review_counts": {
                "qualified_yes": review_counts[DECISION_YES],
                "qualified_no": review_counts[DECISION_NO],
                "reviewed_total": review_counts["reviewed_total"],
            },
            "latest_run": run_summaries[0] if run_summaries else None,
        }

    def metrics_payload(
        self,
        *,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        date_range = _normalize_ui_date_range(date_from=date_from, date_to=date_to)
        run_summaries = list(self._iter_all_runs(date_range=date_range))
        filtered_events = self._attach_review_data(list(self._iter_all_events(date_range=date_range)))
        delivery_counts = self._collect_delivery_counts(run_summaries=run_summaries)
        review_counts = self._review_counts_for_items(filtered_events)
        lifecycle_status_counts: Dict[str, int] = {}
        totals = {
            "events_total": len(filtered_events),
            "frames_total": 0,
            "frames_processed": 0,
            "events_emitted_total": 0,
            "count_a_to_b_total": sum(1 for item in filtered_events if _text(item.get("direction")) == "A_TO_B"),
            "count_b_to_a_total": sum(1 for item in filtered_events if _text(item.get("direction")) == "B_TO_A"),
        }
        latest_run: Optional[Dict[str, Any]] = None
        latest_key: tuple[str, str] = ("", "")

        for summary in run_summaries:
            for key in (
                "frames_total",
                "frames_processed",
                "events_emitted_total",
            ):
                value = _mapping_get_int(summary, key)
                if value is not None:
                    totals[key] += value

            lifecycle_status = _text(summary.get("lifecycle_status"))
            if lifecycle_status is not None:
                lifecycle_status_counts[lifecycle_status] = lifecycle_status_counts.get(lifecycle_status, 0) + 1

            summary_key = _run_sort_key(summary)
            if summary_key >= latest_key:
                latest_key = summary_key
                latest_run = {
                    "run_uid": summary.get("run_uid"),
                    "updated_at_utc": summary.get("updated_at_utc"),
                    "lifecycle_status": summary.get("lifecycle_status"),
                    "effective_fps": summary.get("effective_fps"),
                    "processed_fps": summary.get("processed_fps"),
                    "count_a_to_b": summary.get("count_a_to_b"),
                    "count_b_to_a": summary.get("count_b_to_a"),
                    "events_emitted_total": summary.get("events_emitted_total"),
                }

        return {
            "ok": True,
            "service_started_at_utc": self.service_started_at_utc,
            "spool_dir": str(self.spool_dir),
            "spool_exists": self.spool_dir.exists(),
            "uploader_enabled": self.uploader_cfg is not None,
            "retention_enabled": bool(self.retention_cfg.enabled),
            "service_exposure_mode": str(self.service_cfg.exposure_mode),
            "docs_enabled": bool(self.service_cfg.enable_docs),
            "mutation_auth_enabled": self.mutation_auth_cfg.enabled(),
            "ui_auth_enabled": self.ui_auth_cfg.enabled(),
            "runs_total": delivery_counts["runs_total"],
            "events_total": totals["events_total"],
            "frames_total": totals["frames_total"],
            "frames_processed": totals["frames_processed"],
            "events_emitted_total": totals["events_emitted_total"],
            "count_a_to_b_total": totals["count_a_to_b_total"],
            "count_b_to_a_total": totals["count_b_to_a_total"],
            "review_counts": {
                "qualified_yes": review_counts[DECISION_YES],
                "qualified_no": review_counts[DECISION_NO],
                "reviewed_total": review_counts["reviewed_total"],
            },
            "delivery_state_counts": {
                "completed": delivery_counts["completed"],
                "pending": delivery_counts["pending"],
                "failed": delivery_counts["failed"],
                "in_progress": delivery_counts["in_progress"],
                "unknown": delivery_counts["unknown"],
            },
            "sync_overview": self._build_sync_overview(run_summaries=run_summaries, counts=delivery_counts),
            "lifecycle_status_counts": lifecycle_status_counts,
            "latest_run": latest_run,
        }

    def config_payload(self) -> Dict[str, Any]:
        uploader_payload: Dict[str, Any] = {
            "enabled": self.uploader_cfg is not None,
        }
        if self.uploader_cfg is not None:
            uploader_payload.update(
                {
                    "api_base_url": self.uploader_cfg.api_base_url,
                    "timeout_s": float(self.uploader_cfg.timeout_s),
                    "events_batch_size": int(self.uploader_cfg.events_batch_size),
                    "state_filename": str(self.uploader_cfg.state_filename),
                    "upload_thumbnails": bool(self.uploader_cfg.upload_thumbnails),
                    "upload_scene_thumbnails": bool(self.uploader_cfg.upload_scene_thumbnails),
                    "api_key_configured": bool(str(self.uploader_cfg.api_key).strip()),
                    "retry": {
                        "max_attempts": int(self.uploader_cfg.retry.max_attempts),
                        "initial_delay_s": float(self.uploader_cfg.retry.initial_delay_s),
                        "max_delay_s": float(self.uploader_cfg.retry.max_delay_s),
                        "backoff_factor": float(self.uploader_cfg.retry.backoff_factor),
                    },
                }
            )

        return {
            "ok": True,
            "service_started_at_utc": self.service_started_at_utc,
            "spool_dir": str(self.spool_dir),
            "spool_exists": self.spool_dir.exists(),
            "uploader": uploader_payload,
            "service": {
                "exposure_mode": str(self.service_cfg.exposure_mode),
                "docs_enabled": bool(self.service_cfg.enable_docs),
                "trusted_hosts": list(self.service_cfg.trusted_hosts),
            },
            "mutation_auth": {
                "enabled": self.mutation_auth_cfg.enabled(),
                "header_name": str(self.mutation_auth_cfg.header_name or DEFAULT_MUTATION_API_KEY_HEADER),
            },
            "ui_auth": {
                "enabled": self.ui_auth_cfg.enabled(),
                "username": self.ui_auth_cfg.username,
                "cookie_name": self.ui_auth_cfg.cookie_name,
            },
            "retention": {
                "enabled": bool(self.retention_cfg.enabled),
                "max_age_days": int(self.retention_cfg.max_age_days),
                "max_total_bytes": (
                    int(self.retention_cfg.max_total_bytes) if self.retention_cfg.max_total_bytes is not None else None
                ),
                "min_free_bytes": (
                    int(self.retention_cfg.min_free_bytes) if self.retention_cfg.min_free_bytes is not None else None
                ),
                "protect_incomplete_runs": bool(self.retention_cfg.protect_incomplete_runs),
                "state_filename": str(self.retention_cfg.state_filename),
                "auto_run_interval_s": float(self.retention_cfg.auto_run_interval_s),
            },
        }

    def retention_preview_payload(self) -> Dict[str, Any]:
        return self._retention_payload(dry_run=True)

    def run_retention(
        self,
        *,
        dry_run: bool = True,
        max_age_days: Optional[int] = None,
        max_total_bytes: Optional[int] = None,
        min_free_bytes: Optional[int] = None,
        state_filename: Optional[str] = None,
        protect_incomplete_runs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return self._retention_payload(
            dry_run=bool(dry_run),
            max_age_days=max_age_days,
            max_total_bytes=max_total_bytes,
            min_free_bytes=min_free_bytes,
            state_filename=state_filename,
            protect_incomplete_runs=protect_incomplete_runs,
        )

    def list_recent_runs(
        self,
        *,
        limit: int = 20,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        date_range = _normalize_ui_date_range(date_from=date_from, date_to=date_to)
        rows = list(self._iter_all_runs(date_range=date_range))
        rows.sort(key=_run_sort_key, reverse=True)
        return rows[: _clamp_limit(limit, MAX_RUNS_LIMIT)]

    def list_recent_events(
        self,
        *,
        limit: int = 50,
        run_uid: Optional[str] = None,
        camera_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        target_run_uid = str(run_uid).strip() if run_uid else None
        date_range = _normalize_ui_date_range(date_from=date_from, date_to=date_to)

        for summary in self._iter_all_events(camera_id=camera_id, date_range=date_range):
            if target_run_uid is not None and _text(summary.get("run_uid")) != target_run_uid:
                continue
            items.append(summary)

        items.sort(key=_event_sort_key, reverse=True)
        merged = self._attach_review_data(items[: _clamp_limit(limit, MAX_EVENTS_LIMIT)])
        return merged

    def get_event(self, event_uid: str) -> Optional[Dict[str, Any]]:
        target_uid = str(event_uid).strip()
        if target_uid == "":
            return None

        for run_dir in iter_spool_runs(self.spool_dir):
            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue
            status_meta = _load_json_dict(run_dir / "status.json")
            state_meta = _load_json_dict(run_dir / self._state_filename())
            run_summary = _build_run_summary(run_dir, run_meta, status_meta, state_meta)
            for event in _iter_jsonl_records(run_dir / "events.jsonl"):
                if _text(event.get("event_uid")) != target_uid:
                    continue
                event_summary = _build_event_summary(run_dir, run_meta, event, spool_dir=self.spool_dir)
                event_summary["run"] = run_summary
                event_summary = self._attach_review_data([event_summary])[0]
                event_summary["timeline"] = self._build_event_timeline(event_summary)
                return event_summary
        return None

    def list_review_queue(
        self,
        *,
        limit: int = 100,
        camera_id: Optional[str] = None,
        status_filter: str = REVIEW_STATUS_PENDING,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        date_range = _normalize_ui_date_range(date_from=date_from, date_to=date_to)
        items = list(self._iter_all_events(camera_id=camera_id, date_range=date_range))
        items = self._attach_review_data(items)
        normalized_status = _normalize_review_filter(status_filter)
        if normalized_status != REVIEW_STATUS_ALL:
            items = [item for item in items if _text(item.get("review_status")) == normalized_status]

        items.sort(key=_event_sort_key)
        return items[: max(1, int(limit))]

    def review_queue_payload(
        self,
        *,
        limit: int = 100,
        camera_id: Optional[str] = None,
        status_filter: str = REVIEW_STATUS_PENDING,
        current_event_uid: Optional[str] = None,
        page: int = 1,
        page_size: int = DEFAULT_REVIEW_PAGE_SIZE,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        date_range = _normalize_ui_date_range(date_from=date_from, date_to=date_to)
        all_items = self._attach_review_data(list(self._iter_all_events(camera_id=camera_id, date_range=date_range)))
        normalized_status = _normalize_review_filter(status_filter)
        queue = list(all_items)
        if normalized_status != REVIEW_STATUS_ALL:
            queue = [item for item in queue if _text(item.get("review_status")) == normalized_status]
        queue.sort(key=_event_sort_key)
        cameras = self.list_cameras()
        normalized_page_size = _normalize_review_page_size(page_size)
        target_uid = _text(current_event_uid)
        total_items = len(queue)
        total_pages = max(1, (total_items + normalized_page_size - 1) // normalized_page_size)
        current_page = max(1, int(page))

        target_index = None
        for idx, item in enumerate(queue):
            item_page = (idx // normalized_page_size) + 1
            event_uid = _text(item.get("event_uid")) or ""
            item["queue_position"] = idx + 1
            item["queue_page"] = item_page
            item["queue_select_url"] = _ui_review_queue_url(
                camera_id=camera_id,
                status=normalized_status,
                event_uid=event_uid,
                page=item_page,
                page_size=normalized_page_size,
                date_from=date_range.date_from,
                date_to=date_range.date_to,
            )
            item["detail_url"] = _ui_event_detail_url(
                event_uid,
                camera_id=camera_id,
                status=normalized_status,
                page=item_page,
                page_size=normalized_page_size,
                date_from=date_range.date_from,
                date_to=date_range.date_to,
            )
            if target_uid is not None and event_uid == target_uid:
                target_index = idx
        if target_index is not None:
            current_page = (target_index // normalized_page_size) + 1

        current_page = min(current_page, total_pages)
        start_index = (current_page - 1) * normalized_page_size
        end_index = start_index + normalized_page_size
        page_items = queue[start_index:end_index]

        current_index = 0
        if target_index is not None and start_index <= target_index < end_index:
            current_index = target_index - start_index

        current = page_items[current_index] if page_items else None
        current_absolute_index: Optional[int] = None
        if current is not None:
            current_absolute_index = start_index + current_index

        previous_item = None
        next_item = None
        if current_absolute_index is not None:
            if current_absolute_index > 0:
                previous_item = queue[current_absolute_index - 1]
            if current_absolute_index + 1 < len(queue):
                next_item = queue[current_absolute_index + 1]
        review_counts = self._review_counts_for_items(all_items)
        page_start = start_index + 1 if total_items > 0 else 0
        page_end = min(end_index, total_items)
        page_window_start = max(1, current_page - 2)
        page_window_end = min(total_pages, current_page + 2)
        if page_window_end - page_window_start < 4:
            if page_window_start == 1:
                page_window_end = min(total_pages, page_window_start + 4)
            elif page_window_end == total_pages:
                page_window_start = max(1, total_pages - 4)

        pagination_pages = []
        for page_number in range(page_window_start, page_window_end + 1):
            pagination_pages.append(
                {
                    "number": page_number,
                    "url": _ui_review_queue_url(
                        camera_id=camera_id,
                        status=normalized_status,
                        page=page_number,
                        page_size=normalized_page_size,
                        date_from=date_range.date_from,
                        date_to=date_range.date_to,
                    ),
                    "current": page_number == current_page,
                }
            )

        return {
            "items": page_items,
            "current": current,
            "current_index": current_index + 1 if current is not None else 0,
            "current_absolute_index": current_absolute_index + 1 if current_absolute_index is not None else 0,
            "selected_event_uid": _text(current.get("event_uid")) if current is not None else None,
            "queue_total": total_items,
            "page_item_count": len(page_items),
            "previous_item": previous_item,
            "next_item": next_item,
            "camera_id": _text(camera_id),
            "status_filter": normalized_status,
            "cameras": cameras,
            "page_size": normalized_page_size,
            "page_size_options": list(REVIEW_PAGE_SIZE_OPTIONS),
            "date_from": date_range.date_from or "",
            "date_to": date_range.date_to or "",
            "date_filter": _ui_date_range_context(date_range),
            "pagination": {
                "current_page": current_page,
                "page_size": normalized_page_size,
                "total_pages": total_pages,
                "total_items": total_items,
                "start_item": page_start,
                "end_item": page_end,
                "has_previous": current_page > 1,
                "has_next": current_page < total_pages,
                "previous_url": _ui_review_queue_url(
                    camera_id=camera_id,
                    status=normalized_status,
                    page=current_page - 1,
                    page_size=normalized_page_size,
                    date_from=date_range.date_from,
                    date_to=date_range.date_to,
                ) if current_page > 1 else None,
                "next_url": _ui_review_queue_url(
                    camera_id=camera_id,
                    status=normalized_status,
                    page=current_page + 1,
                    page_size=normalized_page_size,
                    date_from=date_range.date_from,
                    date_to=date_range.date_to,
                ) if current_page < total_pages else None,
                "show_first": page_window_start > 1,
                "show_last": page_window_end < total_pages,
                "first_url": _ui_review_queue_url(
                    camera_id=camera_id,
                    status=normalized_status,
                    page=1,
                    page_size=normalized_page_size,
                    date_from=date_range.date_from,
                    date_to=date_range.date_to,
                ),
                "last_url": _ui_review_queue_url(
                    camera_id=camera_id,
                    status=normalized_status,
                    page=total_pages,
                    page_size=normalized_page_size,
                    date_from=date_range.date_from,
                    date_to=date_range.date_to,
                ),
                "pages": pagination_pages,
            },
            "counts": {
                "pending": review_counts["pending"],
                "qualified_yes": review_counts[DECISION_YES],
                "qualified_no": review_counts[DECISION_NO],
                "reviewed_total": review_counts["reviewed_total"],
            },
        }

    def save_review(
        self,
        event_uid: str,
        *,
        decision: str,
        reviewed_class: Optional[str] = None,
        notes: str = "",
        camera_id: Optional[str] = None,
        status_filter: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = self.get_event(event_uid)
        if event is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"event_uid not found: {event_uid}")

        queue_camera_id = _text(camera_id)
        normalized_status = _normalize_review_filter(status_filter)
        normalized_reviewed_class = _normalize_reviewed_class_name(
            reviewed_class,
            model_class_name=_text(event.get("class_name")),
        )
        normalized_page = max(1, int(page or 1))
        normalized_page_size = _normalize_review_page_size(page_size) if page_size is not None else DEFAULT_REVIEW_PAGE_SIZE
        date_range = _normalize_ui_date_range(date_from=date_from, date_to=date_to)
        queue_context = self.review_queue_payload(
            camera_id=queue_camera_id,
            status_filter=normalized_status,
            current_event_uid=event_uid,
            page=normalized_page,
            page_size=normalized_page_size,
            date_from=date_range.date_from,
            date_to=date_range.date_to,
        )
        in_queue_before_save = _text(queue_context.get("selected_event_uid")) == event_uid
        continuation_item = None
        if in_queue_before_save:
            continuation_item = queue_context.get("next_item") or queue_context.get("previous_item")
        queue_page_number = int(queue_context["pagination"]["current_page"])

        record = self.review_store.save_review(
            event_uid=str(event["event_uid"]),
            run_uid=_text(event.get("run_uid")),
            site_id=_text(event.get("site_id")),
            camera_id=_text(event.get("camera_id")),
            decision=decision,
            reviewed_class=normalized_reviewed_class,
            notes=notes,
            now_utc=_utcnow_iso(),
        )
        updated_event = self.get_event(event_uid)
        next_pending = self.list_review_queue(
            limit=1,
            camera_id=queue_camera_id,
            status_filter=REVIEW_STATUS_PENDING,
            date_from=date_range.date_from,
            date_to=date_range.date_to,
        )
        next_event_uid = _text(next_pending[0].get("event_uid")) if next_pending else None
        return {
            "ok": True,
            "event_uid": event_uid,
            "review": record.to_dict(),
            "event": updated_event,
            "next_event_uid": next_event_uid,
            "next_detail_url": _text(continuation_item.get("detail_url")) if isinstance(continuation_item, Mapping) else None,
            "queue_page_url": _ui_review_queue_url(
                camera_id=queue_camera_id,
                status=normalized_status,
                page=queue_page_number,
                page_size=normalized_page_size,
                date_from=date_range.date_from,
                date_to=date_range.date_to,
            ),
            "next_pending_detail_url": _ui_event_detail_url(
                next_event_uid,
                camera_id=queue_camera_id,
                status=REVIEW_STATUS_PENDING,
                page=1,
                page_size=normalized_page_size,
                date_from=date_range.date_from,
                date_to=date_range.date_to,
            ) if next_event_uid else None,
            "next_pending_queue_url": _ui_review_queue_url(
                camera_id=queue_camera_id,
                status=REVIEW_STATUS_PENDING,
                page=1,
                page_size=normalized_page_size,
                date_from=date_range.date_from,
                date_to=date_range.date_to,
            ),
        }

    def list_cameras(self) -> List[str]:
        cameras = {_text(item.get("camera_id")) for item in self._iter_all_events()}
        return sorted(item for item in cameras if item)

    def dashboard_payload(
        self,
        *,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        date_range = _normalize_ui_date_range(date_from=date_from, date_to=date_to)
        metrics = self.metrics_payload(date_from=date_range.date_from, date_to=date_range.date_to)
        recent_events = self.list_recent_events(limit=8, date_from=date_range.date_from, date_to=date_range.date_to)
        for item in recent_events:
            item["detail_url"] = _ui_event_detail_url(
                _text(item.get("event_uid")) or "",
                status=REVIEW_STATUS_ALL,
                date_from=date_range.date_from,
                date_to=date_range.date_to,
            )
        recent_runs = self.list_recent_runs(limit=5, date_from=date_range.date_from, date_to=date_range.date_to)
        latest_run = metrics.get("latest_run") or {}
        review_counts = metrics.get("review_counts") or {}
        pending_reviews = max(0, int(metrics.get("events_total", 0)) - int(review_counts.get("reviewed_total", 0)))
        trend = self._build_dashboard_trend(date_range=date_range)
        return {
            "metrics": metrics,
            "recent_events": recent_events,
            "recent_runs": recent_runs,
            "latest_run": latest_run,
            "sync_overview": self._build_sync_overview(
                run_summaries=list(self._iter_all_runs(date_range=date_range)),
            ),
            "pending_reviews": pending_reviews,
            "review_counts": review_counts,
            "cameras": self.list_cameras(),
            "trend": trend,
            "date_from": date_range.date_from or "",
            "date_to": date_range.date_to or "",
            "date_filter": _ui_date_range_context(date_range),
            "review_queue_url": _ui_review_queue_url(
                status=REVIEW_STATUS_PENDING,
                date_from=date_range.date_from,
                date_to=date_range.date_to,
            ),
        }

    def _build_dashboard_trend(self, *, date_range: UiDateRange) -> Dict[str, Any]:
        events = self._attach_review_data(list(self._iter_all_events(date_range=date_range)))
        points: List[tuple[datetime, Dict[str, Any]]] = []
        for event in events:
            event_dt = _event_occurrence_utc(event)
            if event_dt is None:
                continue
            points.append((event_dt, event))

        if not points:
            return _empty_dashboard_trend(date_range=date_range)

        range_start, range_end_exclusive = _trend_window_bounds(date_range=date_range, points=points)
        bucket_mode = _select_trend_bucket_mode(range_start=range_start, range_end_exclusive=range_end_exclusive)
        buckets = _build_trend_buckets(
            range_start=range_start,
            range_end_exclusive=range_end_exclusive,
            bucket_mode=bucket_mode,
        )
        bucket_index = {
            bucket["start"]: index
            for index, bucket in enumerate(buckets)
        }

        for event_dt, event in points:
            event_bucket = _align_datetime_to_bucket(event_dt, bucket_mode=bucket_mode)
            offset = bucket_index.get(event_bucket)
            if offset is None:
                continue
            bucket = buckets[offset]
            if _text(event.get("direction")) == "A_TO_B":
                bucket["a_to_b"] += 1
            elif _text(event.get("direction")) == "B_TO_A":
                bucket["b_to_a"] += 1
            if _text(event.get("review_status")) == REVIEW_STATUS_PENDING:
                bucket["pending"] += 1

        y_max = max(
            1,
            max(
                max(int(bucket["a_to_b"]), int(bucket["b_to_a"]), int(bucket["pending"]))
                for bucket in buckets
            ),
        )
        bucket_count = len(buckets)
        x_positions = _trend_x_positions(bucket_count)
        series_specs = [
            ("a_to_b", "A_TO_B", "dir-a"),
            ("b_to_a", "B_TO_A", "dir-b"),
            ("pending", "Pending", "no"),
        ]
        series = []
        for key, label, css_class in series_specs:
            series_points = []
            for index, bucket in enumerate(buckets):
                count = int(bucket[key])
                series_points.append(
                    {
                        "x": x_positions[index],
                        "y": _trend_y_position(count, y_max),
                        "count": count,
                        "label": bucket["label"],
                    }
                )
            series.append(
                {
                    "key": key,
                    "label": label,
                    "css_class": css_class,
                    "path": _trend_svg_path(series_points),
                    "points": series_points,
                    "window_total": sum(int(bucket[key]) for bucket in buckets),
                    "latest_value": int(series_points[-1]["count"]) if series_points else 0,
                }
            )

        return {
            "empty": False,
            "bucket_hours": bucket_count if bucket_mode == "hour" else None,
            "bucket_mode": bucket_mode,
            "time_basis_label": "Time (UTC)",
            "window_label": _trend_window_label(date_range=date_range, range_start=range_start, range_end_exclusive=range_end_exclusive),
            "buckets": buckets,
            "series": series,
            "grid_lines": _trend_grid_lines(y_max),
            "window_start_label": _format_datetime(range_start.isoformat().replace("+00:00", "Z")),
            "window_end_label": _format_datetime((range_end_exclusive - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")),
            "window_totals": {
                "a_to_b": sum(int(bucket["a_to_b"]) for bucket in buckets),
                "b_to_a": sum(int(bucket["b_to_a"]) for bucket in buckets),
                "pending": sum(int(bucket["pending"]) for bucket in buckets),
            },
        }

    def _iter_all_runs(self, *, date_range: Optional[UiDateRange] = None) -> Iterable[Dict[str, Any]]:
        state_filename = self._state_filename()
        for run_dir in iter_spool_runs(self.spool_dir):
            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue
            status_meta = _load_json_dict(run_dir / "status.json")
            state_meta = _load_json_dict(run_dir / state_filename)
            summary = _build_run_summary(run_dir, run_meta, status_meta, state_meta)
            if date_range is not None and not _datetime_in_date_range(_run_occurrence_utc(summary), date_range):
                continue
            yield summary

    def _iter_all_events(
        self,
        *,
        camera_id: Optional[str] = None,
        date_range: Optional[UiDateRange] = None,
    ) -> Iterable[Dict[str, Any]]:
        target_camera_id = _text(camera_id)
        for run_dir in iter_spool_runs(self.spool_dir):
            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue
            for event in _iter_jsonl_records(run_dir / "events.jsonl"):
                summary = _build_event_summary(run_dir, run_meta, event, spool_dir=self.spool_dir)
                if target_camera_id is not None and _text(summary.get("camera_id")) != target_camera_id:
                    continue
                if date_range is not None and not _datetime_in_date_range(_event_occurrence_utc(summary), date_range):
                    continue
                yield summary

    def _review_counts_for_items(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        yes_count = sum(1 for item in items if _text(item.get("review_status")) == DECISION_YES)
        no_count = sum(1 for item in items if _text(item.get("review_status")) == DECISION_NO)
        pending_count = sum(1 for item in items if _text(item.get("review_status")) == REVIEW_STATUS_PENDING)
        return {
            DECISION_YES: yes_count,
            DECISION_NO: no_count,
            REVIEW_STATUS_PENDING: pending_count,
            "reviewed_total": yes_count + no_count,
        }

    def _attach_review_data(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        review_map = self.review_store.get_reviews(
            [_text(item.get("event_uid")) or "" for item in items]
        )
        merged: List[Dict[str, Any]] = []
        for item in items:
            event_uid = _text(item.get("event_uid"))
            review = review_map.get(event_uid or "")
            row = dict(item)
            row["review"] = review.to_dict() if review is not None else None
            row["review_status"] = review.decision if review is not None else REVIEW_STATUS_PENDING
            row["review_label"] = _review_label(row["review_status"])
            row["model_class_name"] = _text(item.get("class_name"))
            row["reviewed_class_name"] = review.reviewed_class if review is not None else None
            row["effective_class_name"] = _effective_class_name(
                row["model_class_name"],
                row["reviewed_class_name"],
                row["review_status"],
            )
            row["thumb_url"] = _path_to_public_url(self.spool_dir, row.get("thumb_path"))
            row["scene_url"] = _path_to_public_url(self.spool_dir, row.get("scene_path"))
            merged.append(row)
        return merged

    def _build_event_timeline(self, event: Mapping[str, Any]) -> List[Dict[str, str]]:
        timeline = [
            {
                "time": _coalesce_text(event.get("occurred_at_local"), event.get("occurred_at_utc")) or "Unknown",
                "description": "Crossing event captured and saved to the local spool.",
            }
        ]
        review = event.get("review")
        if isinstance(review, Mapping):
            reviewed_class = _text(review.get("reviewed_class"))
            timeline.append(
                {
                    "time": _text(review.get("updated_at_utc")) or "Unknown",
                    "description": (
                        f"Reviewed as {_review_label(_text(review.get('decision')))}"
                        + (f"; corrected class: {reviewed_class}" if reviewed_class else "")
                        + (f" with notes: {_text(review.get('notes'))}" if _text(review.get("notes")) else ".")
                    ),
                }
            )
        else:
            timeline.append(
                {
                    "time": "Pending",
                    "description": "Awaiting reviewer confirmation in the MVP queue.",
                }
            )
        return timeline

    def retry_pending_runs(
        self,
        *,
        force: bool = False,
        dry_run: bool = False,
        max_runs: Optional[int] = None,
    ) -> Dict[str, Any]:
        cfg = self._require_uploader_cfg()
        summary = process_pending_runs(
            cfg,
            force=bool(force),
            dry_run=bool(dry_run),
            max_runs=max_runs,
        )
        return {
            "ok": True,
            "mode": "pending_runs",
            "force": bool(force),
            "dry_run": bool(dry_run),
            "max_runs": max_runs,
            "summary": {
                "discovered_runs": int(summary.discovered_runs),
                "completed_runs": int(summary.completed_runs),
                "skipped_runs": int(summary.skipped_runs),
                "failed_runs": int(summary.failed_runs),
            },
        }

    def retry_single_run(
        self,
        run_uid: str,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        cfg = self._require_uploader_cfg()
        target_uid = str(run_uid).strip()
        run_dir = self._find_run_dir(target_uid)
        if run_dir is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"run_uid not found: {target_uid}")

        client = DeliveryApiClient(
            base_url=cfg.api_base_url,
            api_key=cfg.api_key,
            timeout_s=cfg.timeout_s,
        )
        run_status = process_single_run(
            run_dir,
            cfg=cfg,
            client=client,
            force=bool(force),
            dry_run=bool(dry_run),
        )
        return {
            "ok": True,
            "mode": "single_run",
            "run_uid": target_uid,
            "run_dir": str(run_dir),
            "force": bool(force),
            "dry_run": bool(dry_run),
            "status": str(run_status),
        }

    def _require_uploader_cfg(self) -> UploaderConfig:
        if self.uploader_cfg is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Uploader is not configured for this service instance.",
            )
        return self.uploader_cfg

    def _find_run_dir(self, run_uid: str) -> Optional[Path]:
        for run_dir in iter_spool_runs(self.spool_dir):
            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue
            if _text(run_meta.get("run_uid")) == run_uid:
                return run_dir
        return None

    def _state_filename(self) -> str:
        if self.uploader_cfg is not None and str(self.uploader_cfg.state_filename).strip():
            return str(self.uploader_cfg.state_filename).strip()
        if str(self.retention_cfg.state_filename).strip():
            return str(self.retention_cfg.state_filename).strip()
        return DEFAULT_STATE_FILENAME

    def _retention_payload(
        self,
        *,
        dry_run: bool,
        max_age_days: Optional[int] = None,
        max_total_bytes: Optional[int] = None,
        min_free_bytes: Optional[int] = None,
        state_filename: Optional[str] = None,
        protect_incomplete_runs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        resolved_state_filename = _coalesce_text(state_filename, self.retention_cfg.state_filename, DEFAULT_STATE_FILENAME)
        if resolved_state_filename is None:
            resolved_state_filename = DEFAULT_STATE_FILENAME

        summary = apply_retention_policy(
            self.spool_dir,
            max_age_days=int(max_age_days if max_age_days is not None else self.retention_cfg.max_age_days),
            max_total_bytes=(
                int(max_total_bytes) if max_total_bytes is not None else self.retention_cfg.max_total_bytes
            ),
            min_free_bytes=(
                int(min_free_bytes) if min_free_bytes is not None else self.retention_cfg.min_free_bytes
            ),
            state_filename=resolved_state_filename,
            protect_incomplete_runs=(
                bool(protect_incomplete_runs)
                if protect_incomplete_runs is not None
                else bool(self.retention_cfg.protect_incomplete_runs)
            ),
            dry_run=bool(dry_run),
        )
        return _serialize_retention_summary(summary)

    def _collect_delivery_counts(
        self,
        *,
        run_summaries: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> Dict[str, int]:
        counts = {
            "runs_total": 0,
            "completed": 0,
            "pending": 0,
            "failed": 0,
            "in_progress": 0,
            "unknown": 0,
        }
        items = run_summaries if run_summaries is not None else self._iter_all_runs()
        for summary in items:
            counts["runs_total"] += 1
            state_name = _text(summary.get("delivery_state")) or "unknown"
            if state_name in counts:
                counts[state_name] += 1
            else:
                counts["unknown"] += 1
        return counts

    def _build_sync_overview(
        self,
        *,
        run_summaries: Iterable[Mapping[str, Any]],
        counts: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        items = [dict(item) for item in run_summaries]
        delivery_counts = dict(counts) if counts is not None else self._collect_delivery_counts(run_summaries=items)

        latest_sync: Optional[Dict[str, Any]] = None
        latest_sync_dt: Optional[datetime] = None
        latest_issue: Optional[Dict[str, Any]] = None
        latest_issue_dt: Optional[datetime] = None

        for item in items:
            sync_at = _parse_iso_datetime(item.get("last_sync_at_utc"))
            if sync_at is not None and (latest_sync_dt is None or sync_at > latest_sync_dt):
                latest_sync_dt = sync_at
                latest_sync = {
                    "run_uid": _text(item.get("run_uid")),
                    "camera_id": _text(item.get("camera_id")),
                    "at_utc": _text(item.get("last_sync_at_utc")),
                    "delivery_state": _text(item.get("delivery_state")),
                    "delivery_state_label": _delivery_state_label(item.get("delivery_state")),
                    "delivery_state_pill_class": _delivery_state_pill_class(item.get("delivery_state")),
                }

            issue_message = _text(item.get("last_error_short"))
            issue_at = _parse_iso_datetime(item.get("last_error_at_utc")) or _parse_iso_datetime(item.get("updated_at_utc"))
            if issue_message is not None and (latest_issue is None or (issue_at is not None and (latest_issue_dt is None or issue_at > latest_issue_dt))):
                latest_issue_dt = issue_at
                latest_issue = {
                    "run_uid": _text(item.get("run_uid")),
                    "camera_id": _text(item.get("camera_id")),
                    "at_utc": _coalesce_text(item.get("last_error_at_utc"), item.get("updated_at_utc")),
                    "message": issue_message,
                }

        status_cards = [
            {
                "key": "pending",
                "label": "Pending",
                "value": int(delivery_counts.get("pending", 0)),
                "card_class": "ink",
                "pill_class": "ink",
            },
            {
                "key": "in_progress",
                "label": "Sedang dikirim",
                "value": int(delivery_counts.get("in_progress", 0)),
                "card_class": "brand",
                "pill_class": "brand",
            },
            {
                "key": "failed",
                "label": "Gagal",
                "value": int(delivery_counts.get("failed", 0)),
                "card_class": "no",
                "pill_class": "no",
            },
            {
                "key": "completed",
                "label": "Selesai",
                "value": int(delivery_counts.get("completed", 0)),
                "card_class": "yes",
                "pill_class": "yes",
            },
        ]

        return {
            "uploader_enabled": self.uploader_cfg is not None,
            "runs_total": int(delivery_counts.get("runs_total", 0)),
            "status_cards": status_cards,
            "latest_sync": latest_sync,
            "latest_issue": latest_issue,
        }


def create_app(
    *,
    spool_dir: Path,
    uploader_cfg: Optional[UploaderConfig] = None,
    retention_cfg: Optional[SpoolRetentionConfig] = None,
    service_cfg: Optional[ServiceConfig] = None,
    mutation_auth_cfg: Optional[MutationAuthConfig] = None,
    review_db_path: Optional[Path] = None,
    ui_auth_cfg: Optional[UiAuthConfig] = None,
    title: str = "Pedestrian Line Edge Service",
) -> FastAPI:
    resolved_service_cfg = service_cfg if service_cfg is not None else ServiceConfig()
    docs_url = "/docs" if bool(resolved_service_cfg.enable_docs) else None
    redoc_url = "/redoc" if bool(resolved_service_cfg.enable_docs) else None
    openapi_url = "/openapi.json" if bool(resolved_service_cfg.enable_docs) else None
    app = FastAPI(
        title=title,
        version="0.1.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url
    )

    app.state.runtime = EdgeApiRuntime(
        spool_dir=Path(spool_dir),
        uploader_cfg=uploader_cfg,
        retention_cfg=retention_cfg if retention_cfg is not None else SpoolRetentionConfig(),
        service_cfg=resolved_service_cfg,
        mutation_auth_cfg=mutation_auth_cfg if mutation_auth_cfg is not None else MutationAuthConfig(),
        ui_auth_cfg=ui_auth_cfg if ui_auth_cfg is not None else UiAuthConfig(),
        review_store=ReviewStore(Path(review_db_path) if review_db_path is not None else Path(spool_dir) / DEFAULT_REVIEW_DB_FILENAME),
    )
    app.state.templates = Jinja2Templates(directory=str(UI_TEMPLATE_DIR))

    trusted_hosts = [
        str(host).strip()
        for host in resolved_service_cfg.trusted_hosts
        if str(host).strip()
    ]
    if trusted_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

    if UI_STATIC_DIR.exists():
        app.mount("/ui-static", StaticFiles(directory=str(UI_STATIC_DIR)), name="ui-static")
    if UI_ASSET_DIR.exists():
        app.mount("/ui-assets", StaticFiles(directory=str(UI_ASSET_DIR)), name="ui-assets")
    app.mount("/evidence", StaticFiles(directory=str(spool_dir), check_dir=False), name="evidence")

    router = APIRouter()

    @router.get("/favicon.ico", include_in_schema=False)
    def favicon() -> RedirectResponse:
        return RedirectResponse(url="/ui-static/favicon.svg", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    @router.get("/", include_in_schema=False)
    def ui_root(runtime: EdgeApiRuntime = Depends(_get_runtime)) -> RedirectResponse:
        target = f"{UI_BASE_PATH}/login" if runtime.ui_auth_cfg.enabled() else f"{UI_BASE_PATH}/dashboard"
        return RedirectResponse(url=target, status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    @router.get(f"{UI_BASE_PATH}", include_in_schema=False)
    def ui_index(runtime: EdgeApiRuntime = Depends(_get_runtime)) -> RedirectResponse:
        target = f"{UI_BASE_PATH}/login" if runtime.ui_auth_cfg.enabled() else f"{UI_BASE_PATH}/dashboard"
        return RedirectResponse(url=target, status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    @router.post("/api/auth/login", tags=["auth"])
    def login(
        payload: LoginRequest,
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> JSONResponse:
        cfg = runtime.ui_auth_cfg
        if not cfg.enabled():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ui_auth_disabled")
        if payload.username != cfg.username or payload.password != cfg.password:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_credentials")

        token = issue_session_token(cfg=cfg, username=cfg.username)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            key=cfg.cookie_name,
            value=token,
            max_age=int(cfg.cookie_max_age_s),
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response

    @router.post("/api/auth/logout", tags=["auth"])
    def logout(runtime: EdgeApiRuntime = Depends(_get_runtime)) -> JSONResponse:
        response = JSONResponse({"ok": True})
        response.delete_cookie(key=runtime.ui_auth_cfg.cookie_name, path="/")
        return response

    @router.post(f"{UI_BASE_PATH}/login", include_in_schema=False)
    async def login_page_submit(
        request: Request,
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> RedirectResponse:
        cfg = runtime.ui_auth_cfg
        if not cfg.enabled():
            return RedirectResponse(url=_login_success_target(None), status_code=status.HTTP_303_SEE_OTHER)

        raw_body = (await request.body()).decode("utf-8", errors="ignore")
        form_fields = parse_qs(raw_body, keep_blank_values=True)
        username = _text((form_fields.get("username") or [""])[0]) or ""
        password = _text((form_fields.get("password") or [""])[0]) or ""
        next_path = _safe_next_path(_text((form_fields.get("next") or [""])[0]))

        if username != cfg.username or password != cfg.password:
            return RedirectResponse(
                url=_ui_login_url(next_path=next_path, error="invalid_credentials", username=username),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        token = issue_session_token(cfg=cfg, username=cfg.username)
        response = RedirectResponse(
            url=_login_success_target(next_path),
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.set_cookie(
            key=cfg.cookie_name,
            value=token,
            max_age=int(cfg.cookie_max_age_s),
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response

    @router.get(f"{UI_BASE_PATH}/login", response_class=HTMLResponse, include_in_schema=False)
    def login_page(
        request: Request,
        next_path: Optional[str] = Query(default=None, alias="next"),
        error: Optional[str] = Query(default=None),
        username: Optional[str] = Query(default=None),
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> HTMLResponse:
        if _is_authenticated(request, runtime=runtime):
            return RedirectResponse(
                url=_login_success_target(next_path),
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            )
        templates = _get_templates(request)
        context = _build_ui_context(
            runtime=runtime,
            request=request,
            page_name="login",
            page_title="Operator Login",
            page_subtitle="Sign in to open the dashboard, review queue, and event detail pages.",
        )
        status_message = "Sign in diperulkan untuk mengakses dashboard, dan  melakukan review data."
        status_kind = ""
        if _text(error) == "invalid_credentials":
            status_message = "Login failed. Check your credentials and try again."
            status_kind = "error"
        context["next_path"] = _login_success_target(next_path)
        context["login_status_message"] = status_message
        context["login_status_kind"] = status_kind
        context["login_username"] = _text(username) or ""
        return templates.TemplateResponse(request, "login.html", context)

    @router.get("/healthz", tags=["health"])
    def healthz(runtime: EdgeApiRuntime = Depends(_get_runtime)) -> Dict[str, Any]:
        return runtime.health_payload()

    @router.get("/status", tags=["health"])
    def status_view(runtime: EdgeApiRuntime = Depends(_get_runtime)) -> Dict[str, Any]:
        return runtime.status_payload()

    @router.get("/metrics", tags=["health"])
    def metrics_view(runtime: EdgeApiRuntime = Depends(_get_runtime)) -> Dict[str, Any]:
        return runtime.metrics_payload()

    @router.get("/config", tags=["admin"])
    def config_view(runtime: EdgeApiRuntime = Depends(_get_runtime)) -> Dict[str, Any]:
        return runtime.config_payload()

    @router.get("/runs/recent", tags=["runs"])
    def recent_runs(
        limit: int = Query(default=20, ge=1, le=MAX_RUNS_LIMIT),
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> Dict[str, Any]:
        items = runtime.list_recent_runs(limit=limit)
        return {
            "items": items,
            "count": len(items),
            "limit": int(limit),
        }

    @router.get("/events/recent", tags=["events"])
    def recent_events(
        limit: int = Query(default=50, ge=1, le=MAX_EVENTS_LIMIT),
        run_uid: Optional[str] = Query(default=None),
        camera_id: Optional[str] = Query(default=None),
        date_from: Optional[str] = Query(default=None),
        date_to: Optional[str] = Query(default=None),
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> Dict[str, Any]:
        items = runtime.list_recent_events(
            limit=limit,
            run_uid=run_uid,
            camera_id=camera_id,
            date_from=date_from,
            date_to=date_to,
        )
        return {
            "items": items,
            "count": len(items),
            "limit": int(limit),
            "run_uid": run_uid,
            "camera_id": camera_id,
            "date_from": date_from,
            "date_to": date_to,
        }

    @router.get("/events/{event_uid}", tags=["events"])
    def event_detail_payload(
        event_uid: str,
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> Dict[str, Any]:
        item = runtime.get_event(event_uid)
        if item is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"event_uid not found: {event_uid}")
        return item

    @router.get("/review/queue", tags=["review"])
    def review_queue(
        limit: int = Query(default=100, ge=1, le=1000),
        camera_id: Optional[str] = Query(default=None),
        status_filter: str = Query(default=REVIEW_STATUS_PENDING, alias="status"),
        event_uid: Optional[str] = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=DEFAULT_REVIEW_PAGE_SIZE, ge=1, le=100),
        date_from: Optional[str] = Query(default=None),
        date_to: Optional[str] = Query(default=None),
        runtime: EdgeApiRuntime = Depends(_get_runtime),
        _auth: None = Depends(_require_ui_auth_json),
    ) -> Dict[str, Any]:
        return runtime.review_queue_payload(
            limit=limit,
            camera_id=camera_id,
            status_filter=status_filter,
            current_event_uid=event_uid,
            page=page,
            page_size=page_size,
            date_from=date_from,
            date_to=date_to,
        )

    @router.post("/events/{event_uid}/review", tags=["review"])
    def save_event_review(
        event_uid: str,
        payload: ReviewUpdateRequest,
        runtime: EdgeApiRuntime = Depends(_get_runtime),
        _auth: None = Depends(_require_ui_auth_json),
    ) -> Dict[str, Any]:
        return runtime.save_review(
            event_uid,
            decision=payload.decision,
            reviewed_class=payload.reviewed_class,
            notes=payload.notes,
            camera_id=payload.camera_id,
            status_filter=payload.status_filter,
            page=payload.page,
            page_size=payload.page_size,
            date_from=payload.date_from,
            date_to=payload.date_to,
        )

    @router.post(f"{UI_BASE_PATH}/events/{{event_uid}}/review", include_in_schema=False)
    async def save_event_review_page(
        request: Request,
        event_uid: str,
        runtime: EdgeApiRuntime = Depends(_get_runtime),
        _auth: None = Depends(_require_ui_auth_page),
    ) -> RedirectResponse:
        raw_body = (await request.body()).decode("utf-8", errors="ignore")
        form_fields = parse_qs(raw_body, keep_blank_values=True)
        decision = _text((form_fields.get("decision") or [None])[0])
        if decision == "":
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="decision is required")
        reviewed_class = _text((form_fields.get("reviewed_class") or [""])[0]) or None
        notes = _text((form_fields.get("notes") or [""])[0])
        camera_id = _text((form_fields.get("camera_id") or [""])[0]) or None
        status_filter = _text((form_fields.get("status_filter") or [REVIEW_STATUS_PENDING])[0]) or REVIEW_STATUS_PENDING
        page = int(_text((form_fields.get("page") or ["1"])[0]) or "1")
        page_size = int(_text((form_fields.get("page_size") or [str(DEFAULT_REVIEW_PAGE_SIZE)])[0]) or str(DEFAULT_REVIEW_PAGE_SIZE))
        date_from = _text((form_fields.get("date_from") or [""])[0]) or None
        date_to = _text((form_fields.get("date_to") or [""])[0]) or None
        payload = runtime.save_review(
            event_uid,
            decision=decision,
            reviewed_class=reviewed_class,
            notes=notes,
            camera_id=camera_id,
            status_filter=status_filter,
            page=page,
            page_size=page_size,
            date_from=date_from,
            date_to=date_to,
        )
        redirect_target = (
            _text(payload.get("next_detail_url"))
            or _text(payload.get("next_pending_detail_url"))
            or _text(payload.get("queue_page_url"))
            or _text(payload.get("next_pending_queue_url"))
            or _ui_review_queue_url(status=REVIEW_STATUS_PENDING)
        )
        return RedirectResponse(
            url=str(request.url_for("review_page")) if redirect_target == UI_BASE_PATH else redirect_target,
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @router.get(f"{UI_BASE_PATH}/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard_page(
        request: Request,
        date_from: Optional[str] = Query(default=None),
        date_to: Optional[str] = Query(default=None),
        runtime: EdgeApiRuntime = Depends(_get_runtime),
        _auth: None = Depends(_require_ui_auth_page),
    ) -> HTMLResponse:
        templates = _get_templates(request)
        context = _build_ui_context(
            runtime=runtime,
            request=request,
            page_name="dashboard",
            page_title="Traffic Monitoring Dashboard",
            page_subtitle="Total data dari live ditambah dengan data yang akan direview.",
        )
        context.update(runtime.dashboard_payload(date_from=date_from, date_to=date_to))
        return templates.TemplateResponse(request, "dashboard.html", context)

    @router.get(f"{UI_BASE_PATH}/review", response_class=HTMLResponse, include_in_schema=False)
    def review_page(
        request: Request,
        camera_id: Optional[str] = Query(default=None),
        status_filter: str = Query(default=REVIEW_STATUS_PENDING, alias="status"),
        event_uid: Optional[str] = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=DEFAULT_REVIEW_PAGE_SIZE, ge=1, le=100),
        date_from: Optional[str] = Query(default=None),
        date_to: Optional[str] = Query(default=None),
        runtime: EdgeApiRuntime = Depends(_get_runtime),
        _auth: None = Depends(_require_ui_auth_page),
    ) -> HTMLResponse:
        templates = _get_templates(request)
        context = _build_ui_context(
            runtime=runtime,
            request=request,
            page_name="review",
            page_title="Antrian Review",
            page_subtitle="Verifikasi APPROVED/REJECT yang menggunakan database lokal (SQLite).",
        )
        context.update(
            runtime.review_queue_payload(
                camera_id=camera_id,
                status_filter=status_filter,
                current_event_uid=event_uid,
                page=page,
                page_size=page_size,
                date_from=date_from,
                date_to=date_to,
            )
        )
        return templates.TemplateResponse(request, "review_queue.html", context)

    @router.get(f"{UI_BASE_PATH}/events/{{event_uid}}", response_class=HTMLResponse, include_in_schema=False)
    def event_detail_page(
        request: Request,
        event_uid: str,
        camera_id: Optional[str] = Query(default=None),
        status_filter: str = Query(default=REVIEW_STATUS_PENDING, alias="status"),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=DEFAULT_REVIEW_PAGE_SIZE, ge=1, le=100),
        date_from: Optional[str] = Query(default=None),
        date_to: Optional[str] = Query(default=None),
        runtime: EdgeApiRuntime = Depends(_get_runtime),
        _auth: None = Depends(_require_ui_auth_page),
    ) -> HTMLResponse:
        templates = _get_templates(request)
        item = runtime.get_event(event_uid)
        if item is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"event_uid not found: {event_uid}")
        queue_context = runtime.review_queue_payload(
            camera_id=camera_id,
            status_filter=status_filter,
            current_event_uid=event_uid,
            page=page,
            page_size=page_size,
            date_from=date_from,
            date_to=date_to,
        )
        in_queue = _text(queue_context.get("selected_event_uid")) == event_uid

        context = _build_ui_context(
            runtime=runtime,
            request=request,
            page_name="event_detail",
            page_title="Event Detail",
            page_subtitle="Bukti, metadata, dan hasil review untuk satu data.",
        )
        context["event"] = item
        context["review_counts"] = runtime.review_store.summary()
        context["back_to_queue_url"] = _ui_review_queue_url(
            camera_id=camera_id,
            status=status_filter,
            event_uid=event_uid if in_queue else None,
            page=page,
            page_size=page_size,
            date_from=queue_context.get("date_from"),
            date_to=queue_context.get("date_to"),
        )
        context["queue_page_url"] = _ui_review_queue_url(
            camera_id=camera_id,
            status=status_filter,
            page=int(queue_context["pagination"]["current_page"]),
            page_size=page_size,
            date_from=queue_context.get("date_from"),
            date_to=queue_context.get("date_to"),
        )
        context["pending_queue_url"] = _ui_review_queue_url(
            camera_id=camera_id,
            status=REVIEW_STATUS_PENDING,
            page=1,
            page_size=page_size,
            date_from=queue_context.get("date_from"),
            date_to=queue_context.get("date_to"),
        )
        context["detail_queue"] = {
            "in_queue": in_queue,
            "position": int(queue_context.get("current_absolute_index") or 0),
            "total": int(queue_context.get("queue_total") or 0),
            "previous_item": queue_context.get("previous_item"),
            "next_item": queue_context.get("next_item"),
            "status_filter": queue_context.get("status_filter"),
            "camera_id": queue_context.get("camera_id"),
            "date_from": queue_context.get("date_from"),
            "date_to": queue_context.get("date_to"),
            "date_filter": queue_context.get("date_filter"),
        }
        context["queue_page_number"] = int(queue_context["pagination"]["current_page"])
        return templates.TemplateResponse(request, "event_detail.html", context)

    @router.get("/retention/preview", tags=["admin"])
    def retention_preview(runtime: EdgeApiRuntime = Depends(_get_runtime)) -> Dict[str, Any]:
        return runtime.retention_preview_payload()

    @router.post("/retention/run", tags=["admin"])
    def retention_run(
        payload: RetentionRequest,
        runtime: EdgeApiRuntime = Depends(_get_runtime),
        _auth: None = Depends(_require_mutation_auth),
    ) -> Dict[str, Any]:
        return runtime.run_retention(
            dry_run=bool(payload.dry_run),
            max_age_days=payload.max_age_days,
            max_total_bytes=payload.max_total_bytes,
            min_free_bytes=payload.min_free_bytes,
            state_filename=payload.state_filename,
            protect_incomplete_runs=payload.protect_incomplete_runs,
        )

    @router.post("/sync/retry", tags=["sync"])
    def sync_retry(
        payload: SyncRequest,
        runtime: EdgeApiRuntime = Depends(_get_runtime),
        _auth: None = Depends(_require_mutation_auth),
    ) -> Dict[str, Any]:
        return runtime.retry_pending_runs(
            force=bool(payload.force),
            dry_run=bool(payload.dry_run),
            max_runs=payload.max_runs,
        )

    @router.post("/sync/run/{run_uid}", tags=["sync"])
    def sync_single_run(
        run_uid: str,
        payload: SingleRunSyncRequest,
        runtime: EdgeApiRuntime = Depends(_get_runtime),
        _auth: None = Depends(_require_mutation_auth),
    ) -> Dict[str, Any]:
        return runtime.retry_single_run(
            run_uid,
            force=bool(payload.force),
            dry_run=bool(payload.dry_run),
        )

    app.include_router(router)
    return app


def _normalize_reviewed_class_name(
    value: Optional[str],
    *,
    model_class_name: Optional[str],
) -> Optional[str]:
    normalized = _text(value)
    if normalized is None:
        return None
    if normalized == _text(model_class_name):
        return None
    return normalized

def _effective_class_name(
    model_class_name: Optional[str],
    reviewed_class_name: Optional[str],
    reviewed_status: Optional[str],

)-> Optional[str]:
    if _text(reviewed_status) != DECISION_YES:
        return None
    return _text(reviewed_class_name) or _text(model_class_name)

def _get_runtime(request: Request) -> EdgeApiRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, EdgeApiRuntime):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Service runtime is not initialized.")
    return runtime


def _get_templates(request: Request) -> Jinja2Templates:
    templates = getattr(request.app.state, "templates", None)
    if not isinstance(templates, Jinja2Templates):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="UI templates are not initialized.")
    return templates


def _is_authenticated(request: Request, *, runtime: EdgeApiRuntime) -> bool:
    cfg = runtime.ui_auth_cfg
    if not cfg.enabled():
        return True
    token = request.cookies.get(cfg.cookie_name, "")
    return validate_session_token(cfg=cfg, token=token)


def _safe_next_path(value: Optional[str]) -> Optional[str]:
    text = _text(value)
    if text is None:
        return None
    if not text.startswith("/"):
        return None
    if text.startswith("//"):
        return None
    return text


def _login_success_target(next_path: Optional[str]) -> str:
    return _safe_next_path(next_path) or f"{UI_BASE_PATH}/dashboard"


def _ui_login_url(
    *,
    next_path: Optional[str] = None,
    error: Optional[str] = None,
    username: Optional[str] = None,
) -> str:
    params: Dict[str, str] = {}
    safe_next = _safe_next_path(next_path)
    if safe_next:
        params["next"] = safe_next
    if _text(error):
        params["error"] = str(error)
    if _text(username):
        params["username"] = str(username)
    query = urlencode(params)
    if query:
        return f"{UI_BASE_PATH}/login?{query}"
    return f"{UI_BASE_PATH}/login"


def _require_ui_auth_page(
    request: Request,
    runtime: EdgeApiRuntime = Depends(_get_runtime),
) -> None:
    if _is_authenticated(request, runtime=runtime):
        return
    next_path = _safe_next_path(request.url.path)
    if request.url.query:
        next_path = f"{next_path}?{request.url.query}" if next_path else None
    target = f"{UI_BASE_PATH}/login"
    if next_path:
        target += f"?next={next_path}"
    raise HTTPException(
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Location": target},
    )


def _require_ui_auth_json(
    request: Request,
    runtime: EdgeApiRuntime = Depends(_get_runtime),
) -> None:
    if _is_authenticated(request, runtime=runtime):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="ui_auth_required")


def _require_mutation_auth(
    request: Request,
    runtime: EdgeApiRuntime = Depends(_get_runtime),
) -> None:
    cfg = runtime.mutation_auth_cfg
    if not cfg.enabled():
        return

    header_name = _text(cfg.header_name) or DEFAULT_MUTATION_API_KEY_HEADER
    supplied = (request.headers.get(header_name, "") or "").strip()
    expected = str(cfg.api_key or "").strip()
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_api_key",
        )


app = create_app(spool_dir=ROOT_DIR / "spool")


__all__ = [
    "DEFAULT_STATE_FILENAME",
    "DEFAULT_MUTATION_API_KEY_HEADER",
    "EdgeApiRuntime",
    "MutationAuthConfig",
    "RetentionRequest",
    "SingleRunSyncRequest",
    "SyncRequest",
    "app",
    "create_app",
]
