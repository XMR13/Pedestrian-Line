from __future__ import annotations

import json
from pathlib import Path

import pytest

from pedestrian_line_counter.event_catalog import EventCatalog


def _write_run(root: Path) -> None:
    run_dir = root / "2026-08-11" / "run_catalog"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_uid": "run_catalog",
                "site_id": "site_a",
                "camera_id": "cam_01",
            }
        ),
        encoding="utf-8",
    )
    events = [
        {
            "event_uid": "event_newer",
            "run_uid": "run_catalog",
            "occurred_at_utc": "2026-08-11T02:00:00Z",
            "direction": "B_TO_A",
            "class_name": "tronton",
            "confidence": 0.92,
            "thumb_relpath": "thumbs/event_newer.jpg",
        },
        {
            "event_uid": "event_older",
            "run_uid": "run_catalog",
            "occurred_at_utc": "2026-08-11T01:00:00Z",
            "direction": "A_TO_B",
            "class_name": "pickup",
            "confidence": 0.88,
            "scene_relpath": "scene/event_older.jpg",
        },
    ]
    (run_dir / "events.jsonl").write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )


def test_catalog_loads_existing_events_and_gets_by_uid(tmp_path: Path) -> None:
    _write_run(tmp_path)
    catalog = EventCatalog(tmp_path)

    result = catalog.refresh()

    assert result.loaded_events == 2
    assert result.malformed_records == 0
    assert result.processed_records == 2
    assert result.rebuilt_runs == 1
    assert result.incremental_runs == 0
    assert catalog.get("event_newer")["camera_id"] == "cam_01"
    assert catalog.get("event_newer")["class_name"] == "tronton"
    assert catalog.get("missing") is None
    assert [event["event_uid"] for event in catalog.list_events()] == [
        "event_older",
        "event_newer",
    ]


def test_catalog_exposes_cached_events_as_read_only(tmp_path: Path) -> None:
    _write_run(tmp_path)
    catalog = EventCatalog(tmp_path)
    catalog.refresh()

    event = catalog.get("event_newer")

    assert event is not None
    with pytest.raises(TypeError):
        event["camera_id"] = "changed"  # type: ignore[index]


def test_catalog_waits_for_an_incomplete_tail_record(tmp_path: Path) -> None:
    _write_run(tmp_path)
    events_path = tmp_path / "2026-08-11" / "run_catalog" / "events.jsonl"
    incomplete_event = {
        "event_uid": "event_incomplete",
        "run_uid": "run_catalog",
        "occurred_at_utc": "2026-08-11T03:00:00Z",
        "class_name": "bus",
    }
    with events_path.open("ab") as handle:
        handle.write(json.dumps(incomplete_event).encode("utf-8"))

    catalog = EventCatalog(tmp_path)

    first_result = catalog.refresh()

    assert first_result.loaded_events == 2
    assert first_result.malformed_records == 0
    assert first_result.processed_records == 2
    assert catalog.get("event_incomplete") is None

    with events_path.open("ab") as handle:
        handle.write(b"\n")

    second_result = catalog.refresh()

    assert second_result.loaded_events == 3
    assert second_result.malformed_records == 0
    assert second_result.processed_records == 1
    assert second_result.rebuilt_runs == 0
    assert second_result.incremental_runs == 1
    assert catalog.get("event_incomplete") is not None


def test_catalog_skips_complete_malformed_records(tmp_path: Path) -> None:
    _write_run(tmp_path)
    catalog = EventCatalog(tmp_path)
    catalog.refresh()
    events_path = tmp_path / "2026-08-11" / "run_catalog" / "events.jsonl"
    valid_event = {
        "event_uid": "event_after_malformed",
        "run_uid": "run_catalog",
        "occurred_at_utc": "2026-08-11T04:00:00Z",
        "class_name": "bus",
    }
    with events_path.open("ab") as handle:
        handle.write(b"{not valid json}\n")
        handle.write(json.dumps(valid_event).encode("utf-8") + b"\n")

    result = catalog.refresh()

    assert result.loaded_events == 3
    assert result.malformed_records == 1
    assert result.processed_records == 2
    assert result.incremental_runs == 1
    assert catalog.get("event_after_malformed") is not None


def test_catalog_refresh_replaces_removed_disk_events(tmp_path: Path) -> None:
    _write_run(tmp_path)
    catalog = EventCatalog(tmp_path)
    catalog.refresh()

    events_path = tmp_path / "2026-08-11" / "run_catalog" / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    result = catalog.refresh()

    assert result.loaded_events == 0
    assert result.rebuilt_runs == 1
    assert catalog.get("event_newer") is None
    assert catalog.list_events() == ()


def test_catalog_unchanged_refresh_processes_no_records(tmp_path: Path) -> None:
    _write_run(tmp_path)
    catalog = EventCatalog(tmp_path)
    catalog.refresh()
    original_event = catalog.get("event_newer")

    result = catalog.refresh()

    assert result.loaded_events == 2
    assert result.processed_records == 0
    assert result.rebuilt_runs == 0
    assert result.incremental_runs == 0
    assert catalog.get("event_newer") is original_event


def test_catalog_incremental_refresh_reads_only_appended_records(tmp_path: Path) -> None:
    _write_run(tmp_path)
    catalog = EventCatalog(tmp_path)
    catalog.refresh()
    original_event = catalog.get("event_newer")
    events_path = tmp_path / "2026-08-11" / "run_catalog" / "events.jsonl"
    appended_events = [
        {
            "event_uid": f"event_appended_{index}",
            "run_uid": "run_catalog",
            "occurred_at_utc": f"2026-08-11T03:00:0{index}Z",
            "class_name": "bus",
        }
        for index in range(3)
    ]
    with events_path.open("a", encoding="utf-8", newline="\n") as handle:
        for event in appended_events:
            handle.write(f"{json.dumps(event)}\n")

    result = catalog.refresh()

    assert result.loaded_events == 5
    assert result.processed_records == 3
    assert result.rebuilt_runs == 0
    assert result.incremental_runs == 1
    assert catalog.get("event_newer") is original_event
    assert catalog.get("event_appended_2") is not None


def test_catalog_evicts_a_run_removed_by_retention(tmp_path: Path) -> None:
    _write_run(tmp_path)
    catalog = EventCatalog(tmp_path)
    catalog.refresh()
    run_dir = tmp_path / "2026-08-11" / "run_catalog"

    (run_dir / "events.jsonl").unlink()

    result = catalog.refresh()

    assert result.loaded_events == 0
    assert result.removed_runs == 1
    assert catalog.get("event_newer") is None


def test_catalog_merges_an_out_of_order_appended_event(tmp_path: Path) -> None:
    _write_run(tmp_path)
    catalog = EventCatalog(tmp_path)
    catalog.refresh()
    events_path = tmp_path / "2026-08-11" / "run_catalog" / "events.jsonl"
    out_of_order_event = {
        "event_uid": "event_earliest",
        "run_uid": "run_catalog",
        "occurred_at_utc": "2026-08-11T00:30:00Z",
        "class_name": "bus",
    }
    with events_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{json.dumps(out_of_order_event)}\n")

    result = catalog.refresh()

    assert result.processed_records == 1
    assert [event["event_uid"] for event in catalog.list_events()] == [
        "event_earliest",
        "event_older",
        "event_newer",
    ]
