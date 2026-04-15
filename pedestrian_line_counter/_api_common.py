from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

DEFAULT_STATE_FILENAME = ".portal_upload_state.json"
DEFAULT_MUTATION_API_KEY_HEADER = "X-API-Key"
MAX_RUNS_LIMIT = 200
MAX_EVENTS_LIMIT = 500
DEFAULT_REVIEW_DB_FILENAME = ".edge_ui_reviews.sqlite3"
UI_BASE_PATH = "/ui"
UI_TEMPLATE_DIR = Path(__file__).resolve().parent / "ui_templates"
UI_STATIC_DIR = Path(__file__).resolve().parent / "ui_static"
REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_ALL = "all"
DEFAULT_REVIEW_PAGE_SIZE = 25
REVIEW_PAGE_SIZE_OPTIONS = (15, 20, 25)
UI_STATIC_VERSION = max(
    int(path.stat().st_mtime)
    for path in UI_STATIC_DIR.iterdir()
    if path.is_file()
)
TREND_BUCKET_HOURS = 12
TREND_MAX_BUCKETS = 31


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
    max_total_bytes: Optional[int] = Field(default=None, ge=0)
    min_free_bytes: Optional[int] = Field(default=None, ge=0)
    state_filename: Optional[str] = None
    protect_incomplete_runs: Optional[bool] = None


class ReviewUpdateRequest(BaseModel):
    decision: str
    reviewed_class: Optional[str] = None
    notes: str = ""
    camera_id: Optional[str] = None
    status_filter: Optional[str] = None
    page: Optional[int] = Field(default=None, ge=1)
    page_size: Optional[int] = Field(default=None, ge=1, le=100)
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


@dataclass
class MutationAuthConfig:
    api_key: str = ""
    header_name: str = DEFAULT_MUTATION_API_KEY_HEADER

    def enabled(self) -> bool:
        return bool(str(self.api_key).strip())


@dataclass(frozen=True)
class UiDateRange:
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    start_utc: Optional[datetime] = None
    end_utc_exclusive: Optional[datetime] = None

    @property
    def active(self) -> bool:
        return self.start_utc is not None or self.end_utc_exclusive is not None
