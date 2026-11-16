from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pedestrian_line_counter._api_helpers import (  # noqa: E402
    _build_event_summary,
    _iter_jsonl_records,
    _load_json_dict,
)
from pedestrian_line_counter.event_catalog import EventCatalog  # noqa: E402
from pedestrian_line_counter.event_uploader import iter_spool_runs  # noqa: E402


DEFAULT_EVENT_COUNTS = (13_000, 65_000, 100_000)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare repeated JSONL event reads with the in-memory event catalog.",
    )
    parser.add_argument(
        "--events",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVENT_COUNTS),
        help="Synthetic event counts to measure (default: 13000 65000 100000).",
    )
    parser.add_argument(
        "--scan-repeats",
        type=int,
        default=3,
        help="Number of repeated full JSONL UID lookups per event count (default: 3).",
    )
    parser.add_argument(
        "--cache-lookups",
        type=int,
        default=10_000,
        help="Number of warm dictionary lookups per event count (default: 10000).",
    )
    return parser.parse_args()


def _write_synthetic_spool(spool_dir: Path, event_count: int) -> str:
    run_dir = spool_dir / "2026-08-11" / "benchmark_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_uid": "benchmark_run",
                "site_id": "benchmark_site",
                "camera_id": "benchmark_camera",
            }
        ),
        encoding="utf-8",
    )

    target_uid = f"benchmark_event_{event_count - 1:07d}"
    with (run_dir / "events.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(event_count):
            handle.write(json.dumps(_synthetic_event(index), separators=(",", ":")))
            handle.write("\n")
    return target_uid


def _append_synthetic_events(spool_dir: Path, start_index: int, count: int) -> None:
    events_path = spool_dir / "2026-08-11" / "benchmark_run" / "events.jsonl"
    with events_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index in range(start_index, start_index + count):
            event = _synthetic_event(index)
            event["occurred_at_utc"] = f"2026-08-12T00:00:{index % 60:02d}Z"
            handle.write(json.dumps(event, separators=(",", ":")))
            handle.write("\n")


def _synthetic_event(index: int) -> Dict[str, Any]:
    return {
        "event_uid": f"benchmark_event_{index:07d}",
        "run_uid": "benchmark_run",
        "occurred_at_utc": f"2026-08-11T02:{(index // 60) % 60:02d}:{index % 60:02d}Z",
        "frame_index": index * 10,
        "direction": "A_TO_B" if index % 2 == 0 else "B_TO_A",
        "track_id": index,
        "class_id": index % 5,
        "class_name": ("car", "motorcycle", "bus", "truck", "person")[index % 5],
        "confidence": 0.90,
        "thumb_relpath": f"thumbs/{index:07d}.jpg",
        "scene_relpath": f"scene/{index:07d}.jpg",
    }


def _iter_events_like_current_runtime(spool_dir: Path) -> Iterable[Dict[str, Any]]:
    for run_dir in iter_spool_runs(spool_dir):
        run_meta = _load_json_dict(run_dir / "run.json")
        if run_meta is None:
            continue
        for event in _iter_jsonl_records(run_dir / "events.jsonl"):
            yield _build_event_summary(run_dir, run_meta, event, spool_dir=spool_dir)


def _scan_for_uid(spool_dir: Path, event_uid: str) -> Optional[Mapping[str, Any]]:
    for event in _iter_events_like_current_runtime(spool_dir):
        if event.get("event_uid") == event_uid:
            return event
    return None


def _measure_seconds(operation, repeats: int) -> List[float]:
    measurements = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        measurements.append(time.perf_counter() - started)
    return measurements


def _measure_catalog_build(spool_dir: Path) -> Tuple[EventCatalog, float, float, float]:
    gc.collect()
    started = time.perf_counter()
    timed_catalog = EventCatalog(spool_dir)
    result = timed_catalog.refresh()
    elapsed = time.perf_counter() - started
    if result.malformed_records:
        raise RuntimeError(f"synthetic spool produced {result.malformed_records} malformed records")

    del timed_catalog
    gc.collect()
    tracemalloc.start()
    catalog = EventCatalog(spool_dir)
    catalog.refresh()
    retained_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (
        catalog,
        elapsed,
        retained_bytes / (1024 * 1024),
        peak_bytes / (1024 * 1024),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return ordered[rank]


def _benchmark_one(event_count: int, scan_repeats: int, cache_lookups: int) -> Dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="event-catalog-benchmark-") as temp_dir:
        spool_dir = Path(temp_dir)
        target_uid = _write_synthetic_spool(spool_dir, event_count)

        scan_times = _measure_seconds(
            lambda: _scan_for_uid(spool_dir, target_uid),
            scan_repeats,
        )
        catalog, build_seconds, retained_mib, peak_mib = _measure_catalog_build(spool_dir)
        if catalog.get(target_uid) is None:
            raise RuntimeError("catalog lookup did not find the target event")

        lookup_times = _measure_seconds(
            lambda: catalog.get(target_uid),
            cache_lookups,
        )
        list_times = _measure_seconds(catalog.list_events, cache_lookups)
        unchanged_refresh_times = _measure_seconds(catalog.refresh, scan_repeats)

        _append_synthetic_events(spool_dir, event_count, 3)
        started = time.perf_counter()
        incremental_result = catalog.refresh()
        incremental_seconds = time.perf_counter() - started
        if incremental_result.processed_records != 3 or incremental_result.incremental_runs != 1:
            raise RuntimeError(
                "incremental refresh did not process exactly the three appended records"
            )

        return {
            "events": float(event_count),
            "scan_median_ms": statistics.median(scan_times) * 1_000,
            "scan_p95_ms": _percentile(scan_times, 0.95) * 1_000,
            "build_ms": build_seconds * 1_000,
            "cache_lookup_median_us": statistics.median(lookup_times) * 1_000_000,
            "cache_lookup_p95_us": _percentile(lookup_times, 0.95) * 1_000_000,
            "list_median_us": statistics.median(list_times) * 1_000_000,
            "unchanged_refresh_ms": statistics.median(unchanged_refresh_times) * 1_000,
            "append_three_ms": incremental_seconds * 1_000,
            "retained_mib": retained_mib,
            "peak_mib": peak_mib,
        }


def _print_results(results: Sequence[Mapping[str, float]]) -> None:
    headers = (
        "events",
        "JSONL UID scan median",
        "JSONL UID scan p95",
        "catalog cold build",
        "cache get median",
        "cache get p95",
        "list snapshot median",
        "unchanged refresh",
        "append 3 refresh",
        "catalog RAM*",
        "startup peak RAM*",
    )
    rows = [
        (
            f"{int(result['events']):,}",
            f"{result['scan_median_ms']:.2f} ms",
            f"{result['scan_p95_ms']:.2f} ms",
            f"{result['build_ms']:.2f} ms",
            f"{result['cache_lookup_median_us']:.2f} us",
            f"{result['cache_lookup_p95_us']:.2f} us",
            f"{result['list_median_us']:.2f} us",
            f"{result['unchanged_refresh_ms']:.2f} ms",
            f"{result['append_three_ms']:.2f} ms",
            f"{result['retained_mib']:.2f} MiB",
            f"{result['peak_mib']:.2f} MiB",
        )
        for result in results
    ]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.rjust(widths[index]) for index, value in enumerate(row)))
    print("\n* Python allocations measured with tracemalloc in a separate build; image bytes are not loaded.")


def main() -> int:
    args = _parse_args()
    if any(event_count <= 0 for event_count in args.events):
        raise SystemExit("--events values must be positive")
    if args.scan_repeats <= 0 or args.cache_lookups <= 0:
        raise SystemExit("repeat counts must be positive")

    results = []
    for event_count in args.events:
        print(f"Measuring {event_count:,} events...", flush=True)
        results.append(_benchmark_one(event_count, args.scan_repeats, args.cache_lookups))
    print()
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
