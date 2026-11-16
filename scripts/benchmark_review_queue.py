from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pedestrian_line_counter._api_common import REVIEW_STATUS_PENDING  # noqa: E402
from pedestrian_line_counter._api_helpers import (  # noqa: E402
    _event_sort_key,
    _text,
    _ui_event_detail_url,
    _ui_review_queue_url,
)
from pedestrian_line_counter.api import create_app  # noqa: E402


DEFAULT_EVENT_COUNTS = (13_000, 100_000)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy and bounded review-queue preparation.",
    )
    parser.add_argument(
        "--events",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVENT_COUNTS),
        help="Synthetic event counts to measure (default: 13000 100000).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Warm queue requests per implementation (default: 5).",
    )
    return parser.parse_args()


def _write_synthetic_spool(spool_dir: Path, event_count: int) -> None:
    run_dir = spool_dir / "2026-08-13" / "benchmark_review_queue"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_uid": "benchmark_review_queue",
                "site_id": "benchmark_site",
                "camera_id": "benchmark_camera",
            }
        ),
        encoding="utf-8",
    )
    with (run_dir / "events.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(event_count):
            handle.write(json.dumps(_synthetic_event(index), separators=(",", ":")))
            handle.write("\n")


def _synthetic_event(index: int) -> Dict[str, Any]:
    day_offset, seconds_in_day = divmod(index, 86_400)
    hour, seconds_in_hour = divmod(seconds_in_day, 3_600)
    minute, second = divmod(seconds_in_hour, 60)
    return {
        "event_uid": f"benchmark_event_{index:07d}",
        "run_uid": "benchmark_review_queue",
        "site_id": "benchmark_site",
        "camera_id": "benchmark_camera" if index % 2 == 0 else "benchmark_camera_2",
        "occurred_at_utc": (
            f"2026-08-{13 + day_offset:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"
        ),
        "frame_index": index,
        "direction": "A_TO_B" if index % 2 == 0 else "B_TO_A",
        "class_name": "pickup" if index % 2 == 0 else "tronton",
        "confidence": 0.9,
        "thumb_relpath": f"thumbs/{index:07d}.jpg",
        "scene_relpath": f"scene/{index:07d}.jpg",
    }


def _legacy_queue_work(runtime: Any, *, page: int, page_size: int) -> int:
    all_items = runtime._attach_review_data(list(runtime._iter_all_events()))
    queue = [
        item
        for item in all_items
        if _text(item.get("review_status")) == REVIEW_STATUS_PENDING
    ]
    queue.sort(key=_event_sort_key)
    runtime.list_cameras()
    for index, item in enumerate(queue):
        item_page = (index // page_size) + 1
        event_uid = _text(item.get("event_uid")) or ""
        item["queue_select_url"] = _ui_review_queue_url(
            status=REVIEW_STATUS_PENDING,
            event_uid=event_uid,
            page=item_page,
            page_size=page_size,
        )
        item["detail_url"] = _ui_event_detail_url(
            event_uid,
            status=REVIEW_STATUS_PENDING,
            page=item_page,
            page_size=page_size,
        )
    start = (page - 1) * page_size
    return len(queue[start : start + page_size])


def _bounded_queue_work(runtime: Any, *, page: int, page_size: int) -> int:
    payload = runtime.review_queue_payload(
        status_filter=REVIEW_STATUS_PENDING,
        page=page,
        page_size=page_size,
    )
    return int(payload["page_item_count"])


def _seed_reviewed_prefix(runtime: Any, reviewed_count: int) -> None:
    now_utc = "2026-08-13T12:00:00Z"
    rows = (
        (
            f"benchmark_event_{index:07d}",
            "benchmark_review_queue",
            "benchmark_site",
            "benchmark_camera" if index % 2 == 0 else "benchmark_camera_2",
            "qualified_yes",
            None,
            "",
            now_utc,
            now_utc,
        )
        for index in range(reviewed_count)
    )
    with runtime.review_store._connect() as conn:
        conn.executemany(
            """
            INSERT INTO event_reviews (
                event_uid,
                run_uid,
                site_id,
                camera_id,
                decision,
                reviewed_class,
                notes,
                created_at_utc,
                updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _legacy_next_pending_uid(runtime: Any) -> str:
    items = runtime.list_review_queue(
        limit=1,
        camera_id="benchmark_camera",
        status_filter=REVIEW_STATUS_PENDING,
    )
    return _text(items[0].get("event_uid")) if items else ""


def _bounded_next_pending_uid(runtime: Any) -> str:
    return runtime._find_first_pending_event_uid(
        camera_id="benchmark_camera",
    ) or ""


def _measure(operation, repeats: int) -> List[float]:
    measurements = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        elapsed = time.perf_counter() - started
        if result <= 0:
            raise RuntimeError("benchmark queue unexpectedly returned no rows")
        measurements.append(elapsed)
    return measurements


def _measure_uid(operation, *, repeats: int, expected_uid: str) -> List[float]:
    measurements = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        elapsed = time.perf_counter() - started
        if result != expected_uid:
            raise RuntimeError(
                f"benchmark lookup returned {result!r}, expected {expected_uid!r}"
            )
        measurements.append(elapsed)
    return measurements


def _benchmark_one(event_count: int, repeats: int) -> Mapping[str, float]:
    with tempfile.TemporaryDirectory(prefix="review-queue-benchmark-") as temp_dir:
        spool_dir = Path(temp_dir)
        _write_synthetic_spool(spool_dir, event_count)
        app = create_app(
            spool_dir=spool_dir,
            review_db_path=spool_dir / "reviews.sqlite3",
        )
        runtime = app.state.runtime
        page_size = 25
        page = max(1, (event_count // 2) // page_size)

        _bounded_queue_work(runtime, page=page, page_size=page_size)
        legacy_times = []
        legacy_failed = False
        try:
            legacy_times = _measure(
                lambda: _legacy_queue_work(runtime, page=page, page_size=page_size),
                repeats,
            )
        except sqlite3.OperationalError as exc:
            if "too many SQL variables" not in str(exc):
                raise
            legacy_failed = True
        bounded_times = _measure(
            lambda: _bounded_queue_work(runtime, page=page, page_size=page_size),
            repeats,
        )

        reviewed_count = event_count // 2
        _seed_reviewed_prefix(runtime, reviewed_count)
        first_pending_index = reviewed_count
        if first_pending_index % 2 != 0:
            first_pending_index += 1
        expected_uid = f"benchmark_event_{first_pending_index:07d}"

        legacy_next_times = []
        legacy_next_failed = False
        try:
            legacy_next_times = _measure_uid(
                lambda: _legacy_next_pending_uid(runtime),
                repeats=repeats,
                expected_uid=expected_uid,
            )
        except sqlite3.OperationalError as exc:
            if "too many SQL variables" not in str(exc):
                raise
            legacy_next_failed = True
        bounded_next_times = _measure_uid(
            lambda: _bounded_next_pending_uid(runtime),
            repeats=repeats,
            expected_uid=expected_uid,
        )

        legacy_median = (
            statistics.median(legacy_times) * 1_000
            if legacy_times
            else 0.0
        )
        bounded_median = statistics.median(bounded_times) * 1_000
        legacy_next_median = (
            statistics.median(legacy_next_times) * 1_000
            if legacy_next_times
            else 0.0
        )
        bounded_next_median = statistics.median(bounded_next_times) * 1_000
        return {
            "events": float(event_count),
            "legacy_median_ms": legacy_median,
            "bounded_median_ms": bounded_median,
            "speedup": legacy_median / bounded_median if legacy_median else 0.0,
            "legacy_failed": float(legacy_failed),
            "legacy_next_median_ms": legacy_next_median,
            "bounded_next_median_ms": bounded_next_median,
            "next_speedup": (
                legacy_next_median / bounded_next_median
                if legacy_next_median
                else 0.0
            ),
            "legacy_next_failed": float(legacy_next_failed),
        }


def _print_results(results: Sequence[Mapping[str, float]]) -> None:
    print("Review Queue preparation")
    print("events     legacy median   bounded median   speedup")
    print("---------  --------------  ---------------  -------")
    for result in results:
        legacy_text = (
            "SQL limit"
            if result["legacy_failed"]
            else f"{result['legacy_median_ms']:.2f} ms"
        )
        speedup_text = (
            "n/a"
            if result["legacy_failed"]
            else f"{result['speedup']:.2f}x"
        )
        print(
            f"{int(result['events']):>9,}  "
            f"{legacy_text:>14}  "
            f"{result['bounded_median_ms']:>12.2f} ms  "
            f"{speedup_text:>7}"
        )

    print()
    print("First Pending lookup after 50% of events are reviewed")
    print("events     legacy median   catalog median   speedup")
    print("---------  --------------  --------------   -------")
    for result in results:
        legacy_text = (
            "SQL limit"
            if result["legacy_next_failed"]
            else f"{result['legacy_next_median_ms']:.2f} ms"
        )
        speedup_text = (
            "n/a"
            if result["legacy_next_failed"]
            else f"{result['next_speedup']:.2f}x"
        )
        print(
            f"{int(result['events']):>9,}  "
            f"{legacy_text:>14}  "
            f"{result['bounded_next_median_ms']:>11.2f} ms  "
            f"{speedup_text:>7}"
        )


def main() -> int:
    args = _parse_args()
    results = [
        _benchmark_one(event_count, max(1, int(args.repeats)))
        for event_count in args.events
        if event_count > 0
    ]
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
