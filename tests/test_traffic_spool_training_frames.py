from __future__ import annotations

import json

import cv2
import numpy as np

from pedestrian_line_counter.structures import CrossingEvent
from pedestrian_line_counter.traffic_spool import TrafficSpoolConfig, TrafficSpoolWriter


def _event(*, track_id: int, frame_index: int, bbox: tuple[int, int, int, int]) -> CrossingEvent:
    return CrossingEvent(
        track_id=track_id,
        direction="A_TO_B",
        frame_index=frame_index,
        class_id=2,
        confidence=0.9,
        bbox_xyxy=bbox,
        line_mode="line",
    )


def _make_spool(tmp_path, *, write_training_frames: bool) -> TrafficSpoolWriter:
    return TrafficSpoolWriter(
        TrafficSpoolConfig(
            root_dir=tmp_path / "runs",
            site_id="site_a",
            camera_id="cam_01",
            write_thumbnails=False,
            write_scene_thumbnails=False,
            write_training_frames=write_training_frames,
        ),
        source={"type": "video", "value": "media/input.mp4"},
        model_version="model.onnx",
        cfg_version="test",
        line_mode="line",
        line_id="line_1",
        fps=30.0,
        frame_size=(160, 90),
        class_names={2: "trailer"},
        run_uid="run_fixed",
    )


def test_training_frames_are_disabled_by_default(tmp_path) -> None:
    spool = _make_spool(tmp_path, write_training_frames=False)
    captured: list[dict[str, object]] = []

    spool.record_events(
        [_event(track_id=1, frame_index=10, bbox=(20, 10, 80, 70))],
        frame_bgr=np.full((90, 160, 3), 120, dtype=np.uint8),
        occurred_at_ts=1738791000.0,
        occurred_at_utc_source="video_start",
        capture_records=captured,
    )
    spool.close()

    assert captured[0]["training_frame_relpath"] is None
    assert captured[0]["frame_width"] == 160
    assert captured[0]["frame_height"] == 90
    assert not spool.training_frames_dir.exists()


def test_training_frame_is_clean_full_resolution_and_shared_per_frame(tmp_path) -> None:
    spool = _make_spool(tmp_path, write_training_frames=True)
    captured: list[dict[str, object]] = []
    frame = np.full((90, 160, 3), (40, 90, 140), dtype=np.uint8)

    spool.record_events(
        [
            _event(track_id=1, frame_index=10, bbox=(20, 10, 80, 70)),
            _event(track_id=2, frame_index=10, bbox=(90, 15, 150, 75)),
        ],
        frame_bgr=frame,
        occurred_at_ts=1738791000.0,
        occurred_at_utc_source="video_start",
        capture_records=captured,
    )
    spool.close()

    assert len(captured) == 2
    assert captured[0]["training_frame_relpath"] == captured[1]["training_frame_relpath"]
    assert captured[0]["frame_width"] == 160
    assert captured[0]["frame_height"] == 90

    saved_paths = list(spool.training_frames_dir.glob("*.jpg"))
    assert len(saved_paths) == 1
    saved = cv2.imread(str(saved_paths[0]))
    assert saved is not None
    assert saved.shape == frame.shape
    assert np.max(np.abs(saved.astype(np.int16) - frame.astype(np.int16))) <= 2

    records = [
        json.loads(line)
        for line in (spool.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[0]["training_frame_relpath"] == captured[0]["training_frame_relpath"]
    assert records[1]["training_frame_relpath"] == captured[1]["training_frame_relpath"]
