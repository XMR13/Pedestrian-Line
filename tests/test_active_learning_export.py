from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pedestrian_line_counter._api_common import DEFAULT_REVIEW_DB_FILENAME
from pedestrian_line_counter.active_learning_export import export_reviewed_coco
from pedestrian_line_counter.review_store import DECISION_NO, DECISION_YES, ReviewStore
from yolo_kitv2.datasets.qa import run as run_dataset_qa


TAXONOMY = {
    "0": "pickup",
    "1": "trailer",
    "2": "tronton",
}


def _event(
    event_uid: str,
    *,
    class_name: str,
    frame_index: int,
    bbox: list[int] | None = None,
    training_frame: bool = True,
    day: str = "2026-07-20",
) -> dict[str, object]:
    return {
        "event_uid": event_uid,
        "run_uid": "run_01",
        "site_id": "site_a",
        "camera_id": "cam_01",
        "occurred_at_local": f"{day}T10:00:00+07:00",
        "occurred_at_utc": f"{day}T03:00:00+00:00",
        "frame_index": frame_index,
        "class_name": class_name,
        "confidence": 0.88,
        "bbox_xyxy": bbox or [10, 10, 60, 70],
        "training_frame_relpath": (
            f"training_frames/frame_{frame_index:012d}.jpg" if training_frame else None
        ),
        "frame_width": 100,
        "frame_height": 80,
    }


def _write_run(
    spool_dir: Path,
    events: list[dict[str, object]],
    *,
    missing_frame_uids: set[str] | None = None,
) -> None:
    run_dir = spool_dir / "2026-07-20" / "run_01"
    training_dir = run_dir / "training_frames"
    training_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_uid": "run_01",
                "class_names": TAXONOMY,
                "frame_width": 100,
                "frame_height": 80,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    missing = missing_frame_uids or set()
    written: set[str] = set()
    for event in events:
        if event["event_uid"] in missing:
            continue
        relpath = event.get("training_frame_relpath")
        if not isinstance(relpath, str) or relpath in written:
            continue
        written.add(relpath)
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        assert cv2.imwrite(str(run_dir / relpath), image)


def _save_review(
    store: ReviewStore,
    event_uid: str,
    *,
    decision: str,
    reviewed_class: str | None = None,
) -> None:
    store.save_review(
        event_uid=event_uid,
        run_uid="run_01",
        site_id="site_a",
        camera_id="cam_01",
        decision=decision,
        reviewed_class=reviewed_class,
        notes="",
        now_utc="2026-07-20T04:00:00+00:00",
    )


def test_export_uses_reviewed_class_groups_shared_frames_and_excludes_unsafe_rows(
    tmp_path: Path,
) -> None:
    spool_dir = tmp_path / "runs"
    events = [
        _event(
            "corrected_pickup",
            class_name="trailer",
            frame_index=10,
            bbox=[-5, 10, 60, 70],
        ),
        _event("accepted_trailer", class_name="trailer", frame_index=10),
        _event("rejected", class_name="pickup", frame_index=20),
        _event("pending", class_name="pickup", frame_index=30),
        _event("missing_frame", class_name="pickup", frame_index=40),
    ]
    _write_run(spool_dir, events, missing_frame_uids={"missing_frame"})
    store = ReviewStore(spool_dir / DEFAULT_REVIEW_DB_FILENAME)
    _save_review(store, "corrected_pickup", decision=DECISION_YES, reviewed_class="pickup")
    _save_review(store, "accepted_trailer", decision=DECISION_YES)
    _save_review(store, "rejected", decision=DECISION_NO)
    _save_review(store, "missing_frame", decision=DECISION_YES)

    output_dir = tmp_path / "export"
    result = export_reviewed_coco(
        spool_dir=spool_dir,
        output_dir=output_dir,
        date_from="2026-07-20",
        date_to="2026-07-20",
    )

    coco = json.loads(
        (
            output_dir
            / "reviewed"
            / "annotations"
            / "instances_default.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    pending_coco = json.loads(
        (
            output_dir
            / "pending"
            / "annotations"
            / "instances_default.json"
        ).read_text(encoding="utf-8")
    )
    audit = json.loads((output_dir / "audit_manifest.json").read_text(encoding="utf-8"))

    assert result["reviewed_images"] == 1
    assert result["reviewed_annotations"] == 2
    assert result["pending_images"] == 1
    assert result["pending_annotations"] == 1
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2
    assert len(pending_coco["images"]) == 1
    assert len(pending_coco["annotations"]) == 1
    assert {category["name"] for category in coco["categories"]} == {
        "pickup",
        "trailer",
        "tronton",
    }

    category_name_by_id = {
        category["id"]: category["name"] for category in coco["categories"]
    }
    annotations_by_class = {
        category_name_by_id[annotation["category_id"]]: annotation
        for annotation in coco["annotations"]
    }
    assert annotations_by_class["pickup"]["bbox"] == [0.0, 10.0, 60.0, 60.0]
    assert audit["class_distribution"]["pickup"] == {
        "reviewed_eligible": 1,
        "reviewed_exported": 1,
        "pending_eligible": 1,
        "pending_exported": 1,
    }
    assert audit["class_distribution"]["trailer"] == {
        "reviewed_eligible": 1,
        "reviewed_exported": 1,
        "pending_eligible": 0,
        "pending_exported": 0,
    }
    assert audit["class_distribution"]["tronton"] == {
        "reviewed_eligible": 0,
        "reviewed_exported": 0,
        "pending_eligible": 0,
        "pending_exported": 0,
    }
    assert audit["classification_corrections"] == {"trailer": {"pickup": 1}}
    assert audit["counts"]["pending_reviews"] == 1
    assert audit["counts"]["rejected_reviews"] == 1
    assert audit["counts"]["accepted_reviews"] == 3
    assert audit["counts"]["excluded_candidates"] == 1
    assert audit["excluded"] == [
        {
            "event_uid": "missing_frame",
            "review_status": "reviewed",
            "reason": "missing_training_frame_file",
        }
    ]

    reviewed_qa_report = tmp_path / "reviewed_qa_report.json"
    assert run_dataset_qa(
        [
            "--dataset-dir",
            str(output_dir / "reviewed"),
            "--labels-dir",
            str(
                output_dir
                / "reviewed"
                / "annotations"
                / "instances_default.json"
            ),
            "--format",
            "coco",
            "--recursive",
            "--output-json",
            str(reviewed_qa_report),
            "--strict",
        ]
    ) == 0
    pending_qa_report = tmp_path / "pending_qa_report.json"
    assert run_dataset_qa(
        [
            "--dataset-dir",
            str(output_dir / "pending"),
            "--labels-dir",
            str(
                output_dir
                / "pending"
                / "annotations"
                / "instances_default.json"
            ),
            "--format",
            "coco",
            "--recursive",
            "--output-json",
            str(pending_qa_report),
            "--strict",
        ]
    ) == 0


def test_soft_class_cap_preserves_rare_class_and_spreads_majority_over_time(
    tmp_path: Path,
) -> None:
    spool_dir = tmp_path / "runs"
    events = [
        _event(f"pickup_{index}", class_name="pickup", frame_index=index + 1)
        for index in range(6)
    ]
    events.append(_event("rare_trailer", class_name="trailer", frame_index=100))
    _write_run(spool_dir, events)
    store = ReviewStore(spool_dir / DEFAULT_REVIEW_DB_FILENAME)
    for event in events:
        _save_review(store, str(event["event_uid"]), decision=DECISION_YES)

    output_dir = tmp_path / "balanced_export"
    export_reviewed_coco(
        spool_dir=spool_dir,
        output_dir=output_dir,
        max_per_class=2,
    )
    audit = json.loads((output_dir / "audit_manifest.json").read_text(encoding="utf-8"))

    assert audit["class_distribution"]["pickup"] == {
        "reviewed_eligible": 6,
        "reviewed_exported": 2,
        "pending_eligible": 0,
        "pending_exported": 0,
    }
    assert audit["class_distribution"]["trailer"] == {
        "reviewed_eligible": 1,
        "reviewed_exported": 1,
        "pending_eligible": 0,
        "pending_exported": 0,
    }
    assert audit["counts"]["exported_reviewed_images"] == 3
    assert audit["counts"]["exported_reviewed_annotations"] == 3
    exported_uids = {row["event_uid"] for row in audit["annotations"]}
    assert exported_uids == {"pickup_0", "pickup_5", "rare_trailer"}
    assert sum(
        row["reason"] == "excluded_by_class_cap" for row in audit["excluded"]
    ) == 4


def test_date_filter_is_inclusive(tmp_path: Path) -> None:
    spool_dir = tmp_path / "runs"
    events = [
        _event("inside", class_name="pickup", frame_index=1, day="2026-07-20"),
        _event("outside", class_name="trailer", frame_index=2, day="2026-07-19"),
    ]
    _write_run(spool_dir, events)
    store = ReviewStore(spool_dir / DEFAULT_REVIEW_DB_FILENAME)
    _save_review(store, "inside", decision=DECISION_YES)
    _save_review(store, "outside", decision=DECISION_YES)

    output_dir = tmp_path / "dated_export"
    export_reviewed_coco(
        spool_dir=spool_dir,
        output_dir=output_dir,
        date_from="2026-07-20",
        date_to="2026-07-20",
    )
    audit = json.loads((output_dir / "audit_manifest.json").read_text(encoding="utf-8"))

    assert [row["event_uid"] for row in audit["annotations"]] == ["inside"]
    assert audit["counts"]["events_outside_date_range"] == 1
