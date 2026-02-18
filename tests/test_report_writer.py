import csv
import json
from pathlib import Path

import numpy as np

from pedestrian_line_counter.report_writer import ReportWriter, ReportWriterConfig
from pedestrian_line_counter.structures import CrossingEvent
from pedestrian_line_counter.traffic_spool import TrafficSpoolConfig, TrafficSpoolWriter


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_report_writer_writes_expected_columns_and_values(tmp_path) -> None:
    report_path = tmp_path / "report.csv"
    writer = ReportWriter(
        ReportWriterConfig(path=report_path, fps=20.0, include_extra_cols=True),
        class_names={2: "truck"},
    )

    events = [
        CrossingEvent(
            track_id=101,
            direction="A_TO_B",
            frame_index=40,
            class_id=2,
            confidence=0.91,
            bbox_xyxy=(10, 20, 30, 40),
            line_mode="line",
        ),
        CrossingEvent(
            track_id=102,
            direction="B_TO_A",
            frame_index=7,
            class_id=None,
            confidence=None,
            bbox_xyxy=None,
            line_mode="gate",
        ),
    ]
    records = [
        {"thumb_relpath": "thumbs/a.jpg", "scene_relpath": "scene/a.jpg", "occurred_at_utc": "2026-02-18T01:00:00Z"},
        {"thumb_relpath": None, "scene_relpath": None, "occurred_at_utc": None},
    ]

    written = writer.record_events(events, event_records=records)
    writer.close()

    rows = _read_csv_rows(report_path)
    assert written == 2
    assert len(rows) == 2
    assert rows[0]["event_no"] == "1"
    assert rows[1]["event_no"] == "2"
    assert rows[0]["timestamp_s"] == "2.000"
    assert rows[1]["timestamp_s"] == "0.350"
    assert rows[0]["vehicle_type"] == "truck"
    assert rows[1]["vehicle_type"] == "unknown"
    assert rows[0]["direction"] == "A_TO_B"
    assert rows[1]["direction"] == "B_TO_A"
    assert rows[0]["thumb_relpath"] == "thumbs/a.jpg"
    assert rows[1]["thumb_relpath"] == ""
    assert rows[0]["scene_relpath"] == "scene/a.jpg"
    assert rows[1]["scene_relpath"] == ""
    assert rows[0]["occurred_at_utc"] == "2026-02-18T01:00:00Z"
    assert rows[1]["occurred_at_utc"] == ""


def test_report_csv_and_spool_events_stay_in_sync(tmp_path) -> None:
    spool = TrafficSpoolWriter(
        TrafficSpoolConfig(
            root_dir=tmp_path / "runs",
            site_id="site_a",
            camera_id="cam_01",
            write_thumbnails=True,
            write_scene_thumbnails=True,
            thumb_pad=5,
            thumb_max_side=128,
            scene_thumb_max_side=160,
            scene_thumb_quality=85,
        ),
        source={"type": "video", "value": "media/input.mp4"},
        model_version="model.onnx",
        cfg_version="test",
        line_mode="line",
        line_id="line_1",
        lines=[((50, 0), (50, 100))],
        fps=30.0,
        frame_size=(100, 100),
        class_names={1: "pickup"},
        run_uid="run_fixed",
    )
    report = ReportWriter(
        ReportWriterConfig(path=spool.run_dir / "report.csv", fps=30.0, include_extra_cols=True),
        class_names={1: "pickup"},
    )

    frame = np.full((100, 100, 3), 180, dtype=np.uint8)
    events = [
        CrossingEvent(
            track_id=10,
            direction="A_TO_B",
            frame_index=30,
            class_id=1,
            confidence=0.88,
            bbox_xyxy=(10, 10, 40, 60),
            line_mode="line",
        ),
        CrossingEvent(
            track_id=11,
            direction="B_TO_A",
            frame_index=45,
            class_id=1,
            confidence=0.77,
            bbox_xyxy=(35, 15, 70, 70),
            line_mode="line",
        ),
    ]
    captured: list[dict[str, object]] = []

    count = spool.record_events(
        events,
        frame_bgr=frame,
        occurred_at_ts=1738791000.0,
        occurred_at_utc_source="video_start",
        capture_records=captured,
    )
    report_count = report.record_events(events, event_records=captured)

    report.close()
    spool.close()

    assert count == 2
    assert report_count == 2
    assert len(captured) == 2

    events_path = spool.run_dir / "events.jsonl"
    jsonl_lines = [json.loads(x) for x in events_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    csv_rows = _read_csv_rows(spool.run_dir / "report.csv")

    assert len(jsonl_lines) == 2
    assert len(csv_rows) == 2
    assert csv_rows[0]["frame_index"] == str(jsonl_lines[0]["frame_index"])
    assert csv_rows[0]["direction"] == str(jsonl_lines[0]["direction"])
    assert csv_rows[1]["track_id"] == str(jsonl_lines[1]["track_id"])

    thumb_rel = csv_rows[0]["thumb_relpath"]
    assert thumb_rel.startswith("thumbs/")
    assert (spool.run_dir / thumb_rel).exists()

    scene_rel = csv_rows[0]["scene_relpath"]
    assert scene_rel.startswith("scene/")
    assert (spool.run_dir / scene_rel).exists()
