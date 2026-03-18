from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import secrets
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import parse_qs, urlencode

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .config import ROOT_DIR, SpoolRetentionConfig
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


DEFAULT_STATE_FILENAME = ".portal_upload_state.json"
DEFAULT_MUTATION_API_KEY_HEADER = "X-API-Key"
MAX_RUNS_LIMIT = 200
MAX_EVENTS_LIMIT = 500
DEFAULT_REVIEW_DB_FILENAME = ".edge_ui_reviews.sqlite3"
UI_BASE_PATH = "/ui"
UI_TEMPLATE_DIR = Path(__file__).resolve().parent / "ui_templates"
UI_STATIC_DIR = Path(__file__).resolve().parent / "ui_static"
UI_ASSET_DIR = ROOT_DIR / "portal" / "mockups" / "assets"
REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_ALL = "all"
DEFAULT_REVIEW_PAGE_SIZE = 25
REVIEW_PAGE_SIZE_OPTIONS = (15, 20, 25)
UI_STATIC_VERSION = max(
    int(path.stat().st_mtime)
    for path in (UI_STATIC_DIR / "app.css", UI_STATIC_DIR / "app.js")
)
TREND_BUCKET_HOURS = 12


class SyncRequest(BaseModel):
    force: bool = False
    dry_run: bool = False
    max_runs: Optional[int] = Field(default=None, ge=1)


class SingleRunSyncRequest(BaseModel):
    force: bool = False
    dry_run: bool = False


class RetentionRequest(BaseModel):
    dry_run: bool = True
    max_age_days: Optional[int] = Field(default=None, ge=0)
    state_filename: Optional[str] = None
    protect_incomplete_runs: Optional[bool] = None


class ReviewUpdateRequest(BaseModel):
    decision: str
    notes: str = ""
    camera_id: Optional[str] = None
    status_filter: Optional[str] = None
    page: Optional[int] = Field(default=None, ge=1)
    page_size: Optional[int] = Field(default=None, ge=1, le=100)


class LoginRequest(BaseModel):
    username: str
    password: str


@dataclass
class MutationAuthConfig:
    api_key: str = ""
    header_name: str = DEFAULT_MUTATION_API_KEY_HEADER

    def enabled(self) -> bool:
        return bool(str(self.api_key).strip())


@dataclass
class EdgeApiRuntime:
    spool_dir: Path
    uploader_cfg: Optional[UploaderConfig] = None
    retention_cfg: SpoolRetentionConfig = field(default_factory=SpoolRetentionConfig)
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
            "mutation_auth_enabled": self.mutation_auth_cfg.enabled(),
            "ui_auth_enabled": self.ui_auth_cfg.enabled(),
        }

    def status_payload(self) -> Dict[str, Any]:
        runs = self.list_recent_runs(limit=1)
        counts = self._collect_delivery_counts()
        review_counts = self.review_store.summary()
        return {
            "ok": True,
            "service_started_at_utc": self.service_started_at_utc,
            "spool_dir": str(self.spool_dir),
            "spool_exists": self.spool_dir.exists(),
            "uploader_enabled": self.uploader_cfg is not None,
            "retention_enabled": bool(self.retention_cfg.enabled),
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
            "review_counts": {
                "qualified_yes": review_counts[DECISION_YES],
                "qualified_no": review_counts[DECISION_NO],
                "reviewed_total": review_counts["reviewed_total"],
            },
            "latest_run": runs[0] if runs else None,
        }

    def metrics_payload(self) -> Dict[str, Any]:
        delivery_counts = self._collect_delivery_counts()
        review_counts = self.review_store.summary()
        lifecycle_status_counts: Dict[str, int] = {}
        totals = {
            "events_total": 0,
            "frames_total": 0,
            "frames_processed": 0,
            "events_emitted_total": 0,
            "count_a_to_b_total": 0,
            "count_b_to_a_total": 0,
        }
        latest_run: Optional[Dict[str, Any]] = None
        latest_key: tuple[str, str] = ("", "")

        for run_dir in iter_spool_runs(self.spool_dir):
            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue
            status_meta = _load_json_dict(run_dir / "status.json")
            state_meta = _load_json_dict(run_dir / self._state_filename())
            summary = _build_run_summary(run_dir, run_meta, status_meta, state_meta)
            totals["events_total"] += len(_iter_jsonl_records(run_dir / "events.jsonl"))
            for key in (
                "frames_total",
                "frames_processed",
                "events_emitted_total",
                "count_a_to_b",
                "count_b_to_a",
            ):
                value = _mapping_get_int(summary, key)
                if value is not None:
                    totals[f"{key}_total" if key.startswith("count_") else key] += value

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
                "protect_incomplete_runs": bool(self.retention_cfg.protect_incomplete_runs),
                "state_filename": str(self.retention_cfg.state_filename),
            },
        }

    def retention_preview_payload(self) -> Dict[str, Any]:
        return self._retention_payload(dry_run=True)

    def run_retention(
        self,
        *,
        dry_run: bool = True,
        max_age_days: Optional[int] = None,
        state_filename: Optional[str] = None,
        protect_incomplete_runs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return self._retention_payload(
            dry_run=bool(dry_run),
            max_age_days=max_age_days,
            state_filename=state_filename,
            protect_incomplete_runs=protect_incomplete_runs,
        )

    def list_recent_runs(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        state_filename = self._state_filename()
        for run_dir in iter_spool_runs(self.spool_dir):
            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue
            status_meta = _load_json_dict(run_dir / "status.json")
            state_meta = _load_json_dict(run_dir / state_filename)
            rows.append(_build_run_summary(run_dir, run_meta, status_meta, state_meta))

        rows.sort(key=_run_sort_key, reverse=True)
        return rows[: _clamp_limit(limit, MAX_RUNS_LIMIT)]

    def list_recent_events(
        self,
        *,
        limit: int = 50,
        run_uid: Optional[str] = None,
        camera_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        target_run_uid = str(run_uid).strip() if run_uid else None
        target_camera_id = str(camera_id).strip() if camera_id else None

        for run_dir in iter_spool_runs(self.spool_dir):
            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue
            current_run_uid = _text(run_meta.get("run_uid"))
            if target_run_uid is not None and current_run_uid != target_run_uid:
                continue
            for event in _iter_jsonl_records(run_dir / "events.jsonl"):
                summary = _build_event_summary(run_dir, run_meta, event, spool_dir=self.spool_dir)
                if target_camera_id is not None and _text(summary.get("camera_id")) != target_camera_id:
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
    ) -> List[Dict[str, Any]]:
        items = list(self._iter_all_events(camera_id=camera_id))
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
    ) -> Dict[str, Any]:
        all_items = self._attach_review_data(list(self._iter_all_events(camera_id=camera_id)))
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
            )
            item["detail_url"] = _ui_event_detail_url(
                event_uid,
                camera_id=camera_id,
                status=normalized_status,
                page=item_page,
                page_size=normalized_page_size,
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
        review_counts = self.review_store.summary()
        pending_count = sum(1 for item in all_items if _text(item.get("review_status")) == REVIEW_STATUS_PENDING)
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
                ) if current_page > 1 else None,
                "next_url": _ui_review_queue_url(
                    camera_id=camera_id,
                    status=normalized_status,
                    page=current_page + 1,
                    page_size=normalized_page_size,
                ) if current_page < total_pages else None,
                "show_first": page_window_start > 1,
                "show_last": page_window_end < total_pages,
                "first_url": _ui_review_queue_url(
                    camera_id=camera_id,
                    status=normalized_status,
                    page=1,
                    page_size=normalized_page_size,
                ),
                "last_url": _ui_review_queue_url(
                    camera_id=camera_id,
                    status=normalized_status,
                    page=total_pages,
                    page_size=normalized_page_size,
                ),
                "pages": pagination_pages,
            },
            "counts": {
                "pending": max(0, pending_count),
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
        notes: str = "",
        camera_id: Optional[str] = None,
        status_filter: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        event = self.get_event(event_uid)
        if event is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"event_uid not found: {event_uid}")

        queue_camera_id = _text(camera_id)
        normalized_status = _normalize_review_filter(status_filter)
        normalized_page = max(1, int(page or 1))
        normalized_page_size = _normalize_review_page_size(page_size) if page_size is not None else DEFAULT_REVIEW_PAGE_SIZE
        queue_context = self.review_queue_payload(
            camera_id=queue_camera_id,
            status_filter=normalized_status,
            current_event_uid=event_uid,
            page=normalized_page,
            page_size=normalized_page_size,
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
            notes=notes,
            now_utc=_utcnow_iso(),
        )
        updated_event = self.get_event(event_uid)
        next_pending = self.list_review_queue(
            limit=1,
            camera_id=queue_camera_id,
            status_filter=REVIEW_STATUS_PENDING,
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
            ),
            "next_pending_detail_url": _ui_event_detail_url(
                next_event_uid,
                camera_id=queue_camera_id,
                status=REVIEW_STATUS_PENDING,
                page=1,
                page_size=normalized_page_size,
            ) if next_event_uid else None,
            "next_pending_queue_url": _ui_review_queue_url(
                camera_id=queue_camera_id,
                status=REVIEW_STATUS_PENDING,
                page=1,
                page_size=normalized_page_size,
            ),
        }

    def list_cameras(self) -> List[str]:
        cameras = {_text(item.get("camera_id")) for item in self._iter_all_events()}
        return sorted(item for item in cameras if item)

    def dashboard_payload(self) -> Dict[str, Any]:
        metrics = self.metrics_payload()
        recent_events = self.list_recent_events(limit=8)
        recent_runs = self.list_recent_runs(limit=5)
        latest_run = metrics.get("latest_run") or {}
        review_counts = metrics.get("review_counts") or {}
        pending_reviews = max(0, int(metrics.get("events_total", 0)) - int(review_counts.get("reviewed_total", 0)))
        trend = self._build_dashboard_trend()
        return {
            "metrics": metrics,
            "recent_events": recent_events,
            "recent_runs": recent_runs,
            "latest_run": latest_run,
            "pending_reviews": pending_reviews,
            "review_counts": review_counts,
            "cameras": self.list_cameras(),
            "trend": trend,
        }

    def _build_dashboard_trend(self) -> Dict[str, Any]:
        events = self._attach_review_data(list(self._iter_all_events()))
        points: List[tuple[datetime, Dict[str, Any]]] = []
        for event in events:
            event_dt = _event_bucket_datetime(event)
            if event_dt is None:
                continue
            points.append((event_dt, event))

        if not points:
            return _empty_dashboard_trend()

        latest_dt = max(item[0] for item in points)
        latest_bucket = latest_dt.replace(minute=0, second=0, microsecond=0)
        bucket_count = TREND_BUCKET_HOURS
        bucket_start = latest_bucket - timedelta(hours=bucket_count - 1)
        bucket_index = {
            bucket_start + timedelta(hours=offset): offset
            for offset in range(bucket_count)
        }
        buckets = [
            {
                "start": bucket_start + timedelta(hours=offset),
                "label": (bucket_start + timedelta(hours=offset)).strftime("%H:%M"),
                "a_to_b": 0,
                "b_to_a": 0,
                "pending": 0,
            }
            for offset in range(bucket_count)
        ]

        for event_dt, event in points:
            normalized_dt = event_dt.astimezone(latest_bucket.tzinfo)
            event_bucket = normalized_dt.replace(minute=0, second=0, microsecond=0)
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
            "bucket_hours": bucket_count,
            "time_basis_label": "Time (local)" if any(_text(event.get("occurred_at_local")) for _, event in points) else "Time (UTC)",
            "window_label": f"Last {bucket_count} hours",
            "buckets": buckets,
            "series": series,
            "grid_lines": _trend_grid_lines(y_max),
            "window_start_label": buckets[0]["start"].strftime("%Y-%m-%d %H:%M"),
            "window_end_label": buckets[-1]["start"].strftime("%Y-%m-%d %H:%M"),
            "window_totals": {
                "a_to_b": sum(int(bucket["a_to_b"]) for bucket in buckets),
                "b_to_a": sum(int(bucket["b_to_a"]) for bucket in buckets),
                "pending": sum(int(bucket["pending"]) for bucket in buckets),
            },
        }

    def _iter_all_events(self, *, camera_id: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        target_camera_id = _text(camera_id)
        for run_dir in iter_spool_runs(self.spool_dir):
            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue
            for event in _iter_jsonl_records(run_dir / "events.jsonl"):
                summary = _build_event_summary(run_dir, run_meta, event, spool_dir=self.spool_dir)
                if target_camera_id is not None and _text(summary.get("camera_id")) != target_camera_id:
                    continue
                yield summary

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
            timeline.append(
                {
                    "time": _text(review.get("updated_at_utc")) or "Unknown",
                    "description": (
                        f"Reviewed as {_review_label(_text(review.get('decision')))}"
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
        state_filename: Optional[str] = None,
        protect_incomplete_runs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        resolved_state_filename = _coalesce_text(state_filename, self.retention_cfg.state_filename, DEFAULT_STATE_FILENAME)
        if resolved_state_filename is None:
            resolved_state_filename = DEFAULT_STATE_FILENAME

        summary = apply_retention_policy(
            self.spool_dir,
            max_age_days=int(max_age_days if max_age_days is not None else self.retention_cfg.max_age_days),
            state_filename=resolved_state_filename,
            protect_incomplete_runs=(
                bool(protect_incomplete_runs)
                if protect_incomplete_runs is not None
                else bool(self.retention_cfg.protect_incomplete_runs)
            ),
            dry_run=bool(dry_run),
        )
        return _serialize_retention_summary(summary)

    def _collect_delivery_counts(self) -> Dict[str, int]:
        counts = {
            "runs_total": 0,
            "completed": 0,
            "pending": 0,
            "failed": 0,
            "in_progress": 0,
            "unknown": 0,
        }
        state_filename = self._state_filename()
        for run_dir in iter_spool_runs(self.spool_dir):
            counts["runs_total"] += 1
            state_meta = _load_json_dict(run_dir / state_filename)
            state_name = _delivery_state_name(state_meta)
            if state_name in counts:
                counts[state_name] += 1
            else:
                counts["unknown"] += 1
        return counts


def create_app(
    *,
    spool_dir: Path,
    uploader_cfg: Optional[UploaderConfig] = None,
    retention_cfg: Optional[SpoolRetentionConfig] = None,
    mutation_auth_cfg: Optional[MutationAuthConfig] = None,
    review_db_path: Optional[Path] = None,
    ui_auth_cfg: Optional[UiAuthConfig] = None,
    title: str = "Pedestrian Line Edge Service",
) -> FastAPI:
    app = FastAPI(title=title, version="0.1.0")
    app.state.runtime = EdgeApiRuntime(
        spool_dir=Path(spool_dir),
        uploader_cfg=uploader_cfg,
        retention_cfg=retention_cfg if retention_cfg is not None else SpoolRetentionConfig(),
        mutation_auth_cfg=mutation_auth_cfg if mutation_auth_cfg is not None else MutationAuthConfig(),
        ui_auth_cfg=ui_auth_cfg if ui_auth_cfg is not None else UiAuthConfig(),
        review_store=ReviewStore(Path(review_db_path) if review_db_path is not None else Path(spool_dir) / DEFAULT_REVIEW_DB_FILENAME),
    )
    app.state.templates = Jinja2Templates(directory=str(UI_TEMPLATE_DIR))

    if UI_STATIC_DIR.exists():
        app.mount("/ui-static", StaticFiles(directory=str(UI_STATIC_DIR)), name="ui-static")
    if UI_ASSET_DIR.exists():
        app.mount("/ui-assets", StaticFiles(directory=str(UI_ASSET_DIR)), name="ui-assets")
    app.mount("/evidence", StaticFiles(directory=str(spool_dir), check_dir=False), name="evidence")

    router = APIRouter()

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

    @router.get(f"{UI_BASE_PATH}/login", response_class=HTMLResponse, include_in_schema=False)
    def login_page(
        request: Request,
        next_path: Optional[str] = Query(default=None, alias="next"),
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> HTMLResponse:
        if _is_authenticated(request, runtime=runtime):
            return RedirectResponse(url=f"{UI_BASE_PATH}/dashboard", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
        templates = _get_templates(request)
        context = _build_ui_context(
            runtime=runtime,
            request=request,
            page_name="login",
            page_title="Operator Login",
            page_subtitle="Sign in to open the dashboard, review queue, and event detail pages.",
        )
        context["next_path"] = _safe_next_path(next_path) or f"{UI_BASE_PATH}/dashboard"
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
        runtime: EdgeApiRuntime = Depends(_get_runtime),
    ) -> Dict[str, Any]:
        items = runtime.list_recent_events(limit=limit, run_uid=run_uid, camera_id=camera_id)
        return {
            "items": items,
            "count": len(items),
            "limit": int(limit),
            "run_uid": run_uid,
            "camera_id": camera_id,
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
            notes=payload.notes,
            camera_id=payload.camera_id,
            status_filter=payload.status_filter,
            page=payload.page,
            page_size=payload.page_size,
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
        notes = _text((form_fields.get("notes") or [""])[0])
        camera_id = _text((form_fields.get("camera_id") or [""])[0]) or None
        status_filter = _text((form_fields.get("status_filter") or [REVIEW_STATUS_PENDING])[0]) or REVIEW_STATUS_PENDING
        page = int(_text((form_fields.get("page") or ["1"])[0]) or "1")
        page_size = int(_text((form_fields.get("page_size") or [str(DEFAULT_REVIEW_PAGE_SIZE)])[0]) or str(DEFAULT_REVIEW_PAGE_SIZE))
        payload = runtime.save_review(
            event_uid,
            decision=decision,
            notes=notes,
            camera_id=camera_id,
            status_filter=status_filter,
            page=page,
            page_size=page_size,
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
        context.update(runtime.dashboard_payload())
        return templates.TemplateResponse(request, "dashboard.html", context)

    @router.get(f"{UI_BASE_PATH}/review", response_class=HTMLResponse, include_in_schema=False)
    def review_page(
        request: Request,
        camera_id: Optional[str] = Query(default=None),
        status_filter: str = Query(default=REVIEW_STATUS_PENDING, alias="status"),
        event_uid: Optional[str] = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=DEFAULT_REVIEW_PAGE_SIZE, ge=1, le=100),
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
        )
        in_queue = _text(queue_context.get("selected_event_uid")) == event_uid

        context = _build_ui_context(
            runtime=runtime,
            request=request,
            page_name="event_detail",
            page_title="Event Detail",
            page_subtitle="Evidence, metadata, and review state for a single crossing.",
        )
        context["event"] = item
        context["review_counts"] = runtime.review_store.summary()
        context["back_to_queue_url"] = _ui_review_queue_url(
            camera_id=camera_id,
            status=status_filter,
            event_uid=event_uid if in_queue else None,
            page=page,
            page_size=page_size,
        )
        context["queue_page_url"] = _ui_review_queue_url(
            camera_id=camera_id,
            status=status_filter,
            page=int(queue_context["pagination"]["current_page"]),
            page_size=page_size,
        )
        context["pending_queue_url"] = _ui_review_queue_url(
            camera_id=camera_id,
            status=REVIEW_STATUS_PENDING,
            page=1,
            page_size=page_size,
        )
        context["detail_queue"] = {
            "in_queue": in_queue,
            "position": int(queue_context.get("current_absolute_index") or 0),
            "total": int(queue_context.get("queue_total") or 0),
            "previous_item": queue_context.get("previous_item"),
            "next_item": queue_context.get("next_item"),
            "status_filter": queue_context.get("status_filter"),
            "camera_id": queue_context.get("camera_id"),
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


def _load_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    """Load a json file and return its content
        into a dictionary. if that saild path does not exist
        then it will return None    
     """
    if not path.exists():
        return None
    
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        #if the data is not a dict then it will return 
        return None
    
    #if all check passed, then return the json loads
    return data

def _iter_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            try:
                obj = json.loads(row)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    break
                continue
            if isinstance(obj, dict):
                items.append(obj)
    return items


def _build_run_summary(
    run_dir: Path,
    run_meta: Mapping[str, Any],
    status_meta: Optional[Mapping[str, Any]],
    state_meta: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    health_summary = status_meta.get("health_summary") if isinstance(status_meta, Mapping) else None
    if not isinstance(health_summary, Mapping):
        health_summary = run_meta.get("health_summary") if isinstance(run_meta.get("health_summary"), Mapping) else None

    return {
        "run_uid": _text(run_meta.get("run_uid")),
        "run_dir": str(run_dir),
        "site_id": _text(run_meta.get("site_id")),
        "camera_id": _text(run_meta.get("camera_id")),
        "started_at_utc": _text(run_meta.get("started_at_utc")),
        "updated_at_utc": _coalesce_text(
            (status_meta or {}).get("updated_at_utc") if isinstance(status_meta, Mapping) else None,
            run_meta.get("updated_at_utc"),
        ),
        "ended_at_utc": _coalesce_text(
            run_meta.get("ended_at_utc"),
            (health_summary or {}).get("ended_at_utc") if isinstance(health_summary, Mapping) else None,
        ),
        "source_type": _coalesce_text(run_meta.get("source_type"), _mapping_get_text(run_meta.get("source"), "type")),
        "source_value": _coalesce_text(run_meta.get("source_value"), _mapping_get_text(run_meta.get("source"), "value")),
        "line_mode": _text(run_meta.get("line_mode")),
        "delivery_state": _delivery_state_name(state_meta),
        "delivery_completed_at_utc": _mapping_get_text(state_meta, "completed_at_utc"),
        "delivery_last_error": _mapping_get_text(state_meta, "last_error"),
        "report_csv_relpath": _text(run_meta.get("report_csv_relpath")),
        "lifecycle_status": _mapping_get_text(health_summary, "lifecycle_status"),
        "frames_total": _mapping_get_int(health_summary, "frames_total"),
        "frames_processed": _mapping_get_int(health_summary, "frames_processed"),
        "events_emitted_total": _mapping_get_int(health_summary, "events_emitted_total"),
        "count_a_to_b": _mapping_get_int(health_summary, "count_a_to_b"),
        "count_b_to_a": _mapping_get_int(health_summary, "count_b_to_a"),
        "effective_fps": _mapping_get_float(health_summary, "effective_fps"),
        "processed_fps": _mapping_get_float(health_summary, "processed_fps"),
    }


def _build_event_summary(
    run_dir: Path,
    run_meta: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    spool_dir: Path,
) -> Dict[str, Any]:
    thumb_rel = _text(event.get("thumb_relpath"))
    scene_rel = _text(event.get("scene_relpath"))
    return {
        "event_uid": _text(event.get("event_uid")),
        "run_uid": _coalesce_text(event.get("run_uid"), run_meta.get("run_uid")),
        "site_id": _coalesce_text(event.get("site_id"), run_meta.get("site_id")),
        "camera_id": _coalesce_text(event.get("camera_id"), run_meta.get("camera_id")),
        "occurred_at_utc": _text(event.get("occurred_at_utc")),
        "occurred_at_local": _text(event.get("occurred_at_local")),
        "direction": _text(event.get("direction")),
        "track_id": _mapping_get_int(event, "track_id"),
        "class_id": _mapping_get_int(event, "class_id"),
        "class_name": _text(event.get("class_name")),
        "confidence": _mapping_get_float(event, "confidence"),
        "frame_index": _mapping_get_int(event, "frame_index"),
        "video_time_s": _mapping_get_float(event, "video_time_s"),
        "thumb_relpath": thumb_rel,
        "scene_relpath": scene_rel,
        "thumb_path": str(run_dir / thumb_rel) if thumb_rel else None,
        "scene_path": str(run_dir / scene_rel) if scene_rel else None,
        "thumb_url": _relpath_to_public_url(spool_dir, run_dir / thumb_rel) if thumb_rel else None,
        "scene_url": _relpath_to_public_url(spool_dir, run_dir / scene_rel) if scene_rel else None,
    }


def _delivery_state_name(state_meta: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(state_meta, Mapping):
        return "pending"
    if _text(state_meta.get("completed_at_utc")):
        return "completed"
    if _text(state_meta.get("last_error")):
        return "failed"
    if _mapping_get_int(state_meta, "events_uploaded_count") is not None:
        return "in_progress"
    return "pending"


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
    value = mapping.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _mapping_get_float(mapping: Any, key: str) -> Optional[float]:
    if not isinstance(mapping, Mapping):
        return None
    value = mapping.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coalesce_text(*values: Any) -> Optional[str]:
    for value in values:
        out = _text(value)
        if out is not None:
            return out
    return None


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _clamp_limit(value: int, max_value: int) -> int:
    return max(1, min(int(value), int(max_value)))


def _serialize_retention_summary(summary: Any) -> Dict[str, Any]:
    return {
        "ok": True,
        "root_dir": str(summary.root_dir),
        "now_utc": summary.now_utc,
        "max_age_days": int(summary.max_age_days),
        "state_filename": str(summary.state_filename),
        "dry_run": bool(summary.dry_run),
        "scanned_runs": int(summary.scanned_runs),
        "deleted_runs": int(summary.deleted_runs),
        "protected_runs": int(summary.protected_runs),
        "retained_recent_runs": int(summary.retained_recent_runs),
        "eligible_runs": int(summary.eligible_runs),
        "bytes_reclaimable": int(summary.bytes_reclaimable),
        "bytes_deleted": int(summary.bytes_deleted),
        "items": [_serialize_retention_run_info(info) for info in summary.runs],
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
    }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_ui_context(
    *,
    runtime: EdgeApiRuntime,
    request: Request,
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


def _ui_query_string(
    *,
    camera_id: Optional[str] = None,
    status: Optional[str] = None,
    event_uid: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
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
    return urlencode(params)


def _ui_review_queue_url(
    *,
    camera_id: Optional[str] = None,
    status: Optional[str] = None,
    event_uid: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
) -> str:
    query = _ui_query_string(
        camera_id=camera_id,
        status=status,
        event_uid=event_uid,
        page=page,
        page_size=page_size,
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
) -> str:
    query = _ui_query_string(
        camera_id=camera_id,
        status=status,
        page=page,
        page_size=page_size,
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


def _event_bucket_datetime(event: Mapping[str, Any]) -> Optional[datetime]:
    return _parse_iso_datetime(event.get("occurred_at_local")) or _parse_iso_datetime(event.get("occurred_at_utc"))


def _empty_dashboard_trend() -> Dict[str, Any]:
    bucket_count = TREND_BUCKET_HOURS
    fallback_labels = [f"{index:02d}:00" for index in range(bucket_count)]
    x_positions = _trend_x_positions(bucket_count)
    empty_points = [
        {"x": x_positions[index], "y": _trend_y_position(0, 1), "count": 0, "label": fallback_labels[index]}
        for index in range(bucket_count)
    ]
    return {
        "empty": True,
        "bucket_hours": bucket_count,
        "time_basis_label": "Time (local)",
        "window_label": f"Last {bucket_count} hours",
        "buckets": [{"label": label, "a_to_b": 0, "b_to_a": 0, "pending": 0} for label in fallback_labels],
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
        "window_start_label": "Unavailable",
        "window_end_label": "Unavailable",
        "window_totals": {"a_to_b": 0, "b_to_a": 0, "pending": 0},
    }


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
