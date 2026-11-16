from __future__ import annotations

import heapq
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from datetime import datetime
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

from ._api_helpers import (
    _build_event_summary,
    _event_occurrence_utc,
    _event_sort_key,
    _load_json_dict,
    _text,
)
from .event_uploader import iter_spool_runs


@dataclass(frozen=True)
class CatalogRefreshResult:
    loaded_events: int
    malformed_records: int
    processed_records: int
    rebuilt_runs: int
    incremental_runs: int
    removed_runs: int


@dataclass(frozen=True)
class _RunCatalogState:
    events_device: int
    events_inode: int
    events_mtime_ns: int
    observed_size: int
    confirmed_offset: int
    event_uids: FrozenSet[str]


class EventCatalog:
    """In-memory index for raw events stored in the spool directory."""

    def __init__(self, spool_dir: Path) -> None:
        self.spool_dir = Path(spool_dir)
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._events_by_uid: Dict[str, Mapping[str, Any]] = {}
        self._ordered_events: Tuple[Mapping[str, Any], ...] = ()
        self._run_states: Dict[Path, _RunCatalogState] = {}
        # Parsed once per event so request-time filters never re-parse ISO text.
        # Only refresh() writes it; readers only do single-key lookups.
        self._occurred_at: Dict[str, Optional[datetime]] = {}
        self._camera_ids: FrozenSet[str] = frozenset()
        self._class_names: FrozenSet[str] = frozenset()

    def refresh(self) -> CatalogRefreshResult:
        """Read only new complete JSONL records and atomically publish a snapshot."""
        with self._refresh_lock:
            return self._refresh_once()

    #refresh the catalog (rebuild or incremental)
    def _refresh_once(self) -> CatalogRefreshResult:
        with self._lock:
            current_events = self._events_by_uid
            current_ordered_events = self._ordered_events
            current_camera_ids = self._camera_ids
            current_class_names = self._class_names
            run_states = dict(self._run_states)

        events_by_uid = current_events
        catalog_changed = False
        requires_full_sort = False
        added_events = []
        malformed_records = 0
        processed_records = 0
        rebuilt_runs = 0
        incremental_runs = 0
        removed_runs = 0

        def writable_events() -> Dict[str, Mapping[str, Any]]:
            nonlocal events_by_uid
            if events_by_uid is current_events:
                events_by_uid = dict(current_events)
            return events_by_uid

        run_dirs = set(iter_spool_runs(self.spool_dir))
        for removed_run_dir in set(run_states) - run_dirs:
            old_state = run_states.pop(removed_run_dir)
            mutable_events = writable_events()
            for event_uid in old_state.event_uids:
                mutable_events.pop(event_uid, None)
                self._occurred_at.pop(event_uid, None)
            catalog_changed = catalog_changed or bool(old_state.event_uids)
            requires_full_sort = requires_full_sort or bool(old_state.event_uids)
            removed_runs += 1

        for run_dir in sorted(run_dirs):
            events_path = run_dir / "events.jsonl"
            events_stat = _safe_stat(events_path)
            if events_stat is None:
                continue

            previous_state = run_states.get(run_dir)
            rebuild = _run_requires_rebuild(
                previous_state,
                events_device=int(events_stat.st_dev),
                events_inode=int(events_stat.st_ino),
                events_mtime_ns=int(events_stat.st_mtime_ns),
                events_size=int(events_stat.st_size),
            )
            if not rebuild and previous_state is not None:
                if int(events_stat.st_size) == previous_state.observed_size:
                    continue
                start_offset = previous_state.confirmed_offset
            else:
                start_offset = 0

            run_meta = _load_json_dict(run_dir / "run.json")
            if run_meta is None:
                continue

            read_result = _read_complete_jsonl_records(
                events_path,
                start_offset=start_offset,
                end_offset=int(events_stat.st_size),
            )
            if read_result is None:
                continue
            records, malformed_count, confirmed_offset = read_result
            malformed_records += malformed_count
            processed_records += len(records) + malformed_count

            if rebuild:
                rebuilt_runs += 1
                old_event_uids = previous_state.event_uids if previous_state is not None else frozenset()
                if old_event_uids:
                    mutable_events = writable_events()
                    for event_uid in old_event_uids:
                        mutable_events.pop(event_uid, None)
                        self._occurred_at.pop(event_uid, None)
                    catalog_changed = True
                    requires_full_sort = True
                event_uids = set()
            else:
                incremental_runs += 1
                event_uids = set(previous_state.event_uids) if previous_state is not None else set()

            for record in records:
                summary = _build_event_summary(
                    run_dir,
                    run_meta,
                    record,
                    spool_dir=self.spool_dir,
                )
                event_uid = _required_text(summary.get("event_uid"))
                if event_uid is None:
                    malformed_records += 1
                    continue
                event_already_exists = event_uid in events_by_uid
                cached_summary = MappingProxyType(summary)
                writable_events()[event_uid] = cached_summary
                self._occurred_at[event_uid] = _event_occurrence_utc(cached_summary)
                event_uids.add(event_uid)
                if event_already_exists:
                    requires_full_sort = True
                else:
                    added_events.append(cached_summary)
                catalog_changed = True

            run_states[run_dir] = _RunCatalogState(
                events_device=int(events_stat.st_dev),
                events_inode=int(events_stat.st_ino),
                events_mtime_ns=int(events_stat.st_mtime_ns),
                observed_size=int(events_stat.st_size),
                confirmed_offset=confirmed_offset,
                event_uids=frozenset(event_uids),
            )

        ordered_events = current_ordered_events
        camera_ids = current_camera_ids
        class_names = current_class_names
        if catalog_changed:
            if requires_full_sort:
                ordered_events = tuple(sorted(events_by_uid.values(), key=_event_sort_key))
                camera_ids = _distinct_text(events_by_uid.values(), "camera_id")
                class_names = _distinct_text(events_by_uid.values(), "class_name")
            else:
                ordered_events = _merge_added_events(current_ordered_events, added_events)
                camera_ids = current_camera_ids | _distinct_text(added_events, "camera_id")
                class_names = current_class_names | _distinct_text(added_events, "class_name")
        with self._lock:
            self._events_by_uid = events_by_uid
            self._ordered_events = ordered_events
            self._camera_ids = camera_ids
            self._class_names = class_names
            self._run_states = run_states

        return CatalogRefreshResult(
            loaded_events=len(ordered_events),
            malformed_records=malformed_records,
            processed_records=processed_records,
            rebuilt_runs=rebuilt_runs,
            incremental_runs=incremental_runs,
            removed_runs=removed_runs,
        )

    def get(self, event_uid: str) -> Optional[Mapping[str, Any]]:
        """Return one event by UID without scanning the JSONL files."""
        key = str(event_uid or "").strip()
        if not key:
            return None
        with self._lock:
            return self._events_by_uid.get(key)

    def list_events(self) -> Tuple[Mapping[str, Any], ...]:
        """Return the current immutable, chronologically ordered snapshot."""
        with self._lock:
            return self._ordered_events

    def occurred_at(self, event: Mapping[str, Any]) -> Optional[datetime]:
        """Return the event's UTC occurrence time parsed at load time."""
        event_uid = _text(event.get("event_uid")) or ""
        if event_uid in self._occurred_at:
            return self._occurred_at[event_uid]
        return _event_occurrence_utc(event)

    def camera_ids(self) -> FrozenSet[str]:
        with self._lock:
            return self._camera_ids

    def class_names(self) -> FrozenSet[str]:
        with self._lock:
            return self._class_names


def _run_requires_rebuild(
    state: Optional[_RunCatalogState],
    *,
    events_device: int,
    events_inode: int,
    events_mtime_ns: int,
    events_size: int,
) -> bool:
    # run.json is ignored on purpose: the live run rewrites it every status
    # tick, and the fields events use from it (site/camera) never change.
    if state is None:
        return True
    if (state.events_device, state.events_inode) != (events_device, events_inode):
        return True
    if events_size < state.confirmed_offset:
        return True
    if events_size == state.observed_size and events_mtime_ns != state.events_mtime_ns:
        return True
    return False


def _merge_added_events(
    ordered_events: Tuple[Mapping[str, Any], ...],
    added_events: list[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], ...]:
    """
    Merge consecutive events that addeed into the catalogue
    """
    if not added_events:
        return ordered_events
    ordered_additions = tuple(sorted(added_events, key=_event_sort_key))
    if not ordered_events or _event_sort_key(ordered_additions[0]) >= _event_sort_key(ordered_events[-1]):
        return ordered_events + ordered_additions
    return tuple(
        heapq.merge(
            ordered_events,
            ordered_additions,
            key=_event_sort_key,
        )
    )


def _read_complete_jsonl_records(
    path: Path,
    *,
    start_offset: int,
    end_offset: int,
) -> Optional[Tuple[Tuple[Dict[str, Any], ...], int, int]]:
    """Read a fixed byte range and confirm only records ending in a newline."""
    try:
        with path.open("rb") as handle:
            handle.seek(start_offset)
            data = handle.read(max(end_offset - start_offset, 0))
    except OSError:
        return None

    last_newline = data.rfind(b"\n")
    if last_newline < 0:
        return (), 0, start_offset

    complete_data = data[: last_newline + 1]
    confirmed_offset = start_offset + len(complete_data)
    records = []
    malformed_records = 0
    for raw_line in complete_data.splitlines():
        text = raw_line.strip()
        if not text:
            continue
        try:
            record = json.loads(text.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed_records += 1
            continue
        if not isinstance(record, dict):
            malformed_records += 1
            continue
        records.append(record)
    return tuple(records), malformed_records, confirmed_offset


def _safe_stat(path: Path) -> Optional[Any]:
    """Get the file or folder info"""
    try:
        return path.stat()
    except OSError:
        return None

def _distinct_text(events: Iterable[Mapping[str, Any]], key: str) -> FrozenSet[str]:
    return frozenset(value for event in events if (value := _text(event.get(key))) is not None)


#another text helper function smhhh, how many of this function do we truly need bruh
def _required_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["CatalogRefreshResult", "EventCatalog"]
