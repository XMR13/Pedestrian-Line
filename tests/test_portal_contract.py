from __future__ import annotations

import pytest

from pedestrian_line_counter import portal_contract
from pedestrian_line_counter.event_contract import (
    PortalContractError,
    build_event_upsert_payload,
    build_run_upsert_payload,
    iter_event_records,
)


def test_build_run_upsert_payload_maps_flat_and_nested_fields() -> None:
    run_meta = {
        "run_uid": "run_001",
        "site_id": "site_a",
        "camera_id": "cam_01",
        "started_at_utc": "2026-02-20T00:00:00Z",
        "source": {"type": "rtsp", "value": "rtsp://camera-1"},
        "source_type": "rtsp",
        "source_value": "rtsp://camera-1",
        "model_version": "model.onnx",
        "cfg_version": "cfg-a",
        "line_mode": "single",
        "line_id": "line_01",
        "fps": 30,
        "frame_size": {"width": 1920, "height": 1080},
        "report_csv_relpath": "report.csv",
        "health_summary": {"ended_at_utc": "2026-02-20T00:10:00Z", "frames_total": 100},
    }

    payload = build_run_upsert_payload(run_meta)

    assert payload["run_uid"] == "run_001"
    assert payload["source_type"] == "rtsp"
    assert payload["source_value"] == "rtsp://camera-1"
    assert payload["frame_width"] == 1920
    assert payload["frame_height"] == 1080
    assert payload["ended_at_utc"] == "2026-02-20T00:10:00Z"
    assert payload["health_summary_json"]["frames_total"] == 100


def test_build_event_upsert_payload_supports_bbox_aliases() -> None:
    event = {
        "event_uid": "ev_001",
        "run_uid": "run_001",
        "site_id": "site_a",
        "camera_id": "cam_01",
        "frame_index": 12,
        "track_id": 77,
        "bbox": [1, 2, 30, 40],
    }

    payload = build_event_upsert_payload(event)

    assert payload["event_uid"] == "ev_001"
    assert payload["bbox_xyxy"] == [1, 2, 30, 40]
    assert payload["track_id"] == 77


def test_iter_event_records_raises_for_invalid_json(tmp_path) -> None:
    run_dir = tmp_path / "2026-02-20" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text('{"ok":1}\n{bad-json}\n', encoding="utf-8")

    with pytest.raises(PortalContractError):
        list(iter_event_records(run_dir))


def test_iter_event_records_ignores_trailing_partial_line(tmp_path) -> None:
    run_dir = tmp_path / "2026-02-20" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text('{"ok":1}\n{"partial":', encoding="utf-8")

    rows = list(iter_event_records(run_dir))

    assert len(rows) == 1
    assert rows[0]["ok"] == 1


def test_portal_contract_module_remains_compatible() -> None:
    payload = portal_contract.build_event_upsert_payload(
        {
            "event_uid": "ev_compat",
            "run_uid": "run_compat",
            "site_id": "site_a",
            "camera_id": "cam_01",
            "bbox": [1, 2, 3, 4],
        }
    )
    assert payload["event_uid"] == "ev_compat"
    assert portal_contract.PORTAL_CONTRACT_VERSION == "v1"
