from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2

from ._api_common import DEFAULT_REVIEW_DB_FILENAME
from .event_uploader import iter_spool_runs
from .review_store import DECISION_NO, DECISION_YES, ReviewRecord, ReviewStore


COCO_FILENAME = "instances_default.json"
COCO_SUBSET = "default"
AUDIT_FILENAME = "audit_manifest.json"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REVIEW_BATCH_SIZE = 500


@dataclass(frozen=True)
class _EventSource:
    run_dir: Path
    run_meta: Mapping[str, Any]
    event: Mapping[str, Any]
    line_number: int


@dataclass(frozen=True)
class _Candidate:
    """
    Parameter - parameter sebagai penanda untuk per kandidate frame
    yang akan dijadikan data point untuk export
    """

    event_uid: str
    run_uid: str
    run_dir: Path
    source_event_relpath: str
    frame_path: str
    source_frame_relpath: str
    frame_width: int
    frame_height: int
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_xywh: Tuple[float, float, float, float]
    model_class_name: Optional[str]
    reviewed_class_name: Optional[str]
    effective_class_name: str
    confidence: Optional[float]
    occurred_at: str
    reviewed_status: str
    reviewed_updated_at_utc: Optional[str]
    #####
    
    @property
    def frame_key(self) -> str:
        return self.frame_path.as_posix()

    @property
    def sort_key(self) -> Tuple[str, str]:
        return (self.occurred_at, self.event_uid)


def export_reviewed_coco(
    *,
    spool_dir: Path,
    output_dir: Path,
    review_db_path: Optional[Path] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    max_per_class: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Export gambar yang diperlukan sebagai kandidat COCO yang ada 


    The saved boxes are detector/tracker output at the crossing frame. They are
    candidates, not final ground truth, and every exported image must still be
    checked for box geometry and unannotated visible vehicles before training.
    """

    spool_root = Path(spool_dir).resolve()
    destination = Path(output_dir).resolve()
    review_path = (
        Path(review_db_path).resolve()
        if review_db_path is not None
        else spool_root / DEFAULT_REVIEW_DB_FILENAME
    )
    start_date, end_date = _normalize_date_range(date_from, date_to)

    if not spool_root.is_dir():
        raise FileNotFoundError(f"Spool directory not found: {spool_root}")
    if not review_path.is_file():
        raise FileNotFoundError(f"Review database not found: {review_path}")
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    if max_per_class is not None and int(max_per_class) < 1:
        raise ValueError("max_per_class must be at least 1")

    sources, taxonomy, scan_counts = _load_event_sources(
        spool_root,
        start_date=start_date,
        end_date=end_date,
    )
    review_map = _load_reviews(review_path, [source.event for source in sources])

    reviewed_candidates: List[_Candidate] = []
    pending_candidates: List[_Candidate] = []
    excluded: List[Dict[str, str]] = []
    review_counts: Counter[str] = Counter()
    reviewed_distribution: Counter[str] = Counter()
    pending_distribution: Counter[str] = Counter()
    model_distribution: Counter[str] = Counter()
    correction_matrix: Dict[str, Counter[str]] = defaultdict(Counter)

    for source in sources:
        event_uid = _text(source.event.get("event_uid"))
        if event_uid is None:
            scan_counts["events_missing_uid"] += 1
            continue

        review = review_map.get(event_uid)
        if review is None:
            review_counts["pending"] += 1
            candidate, reason = _build_candidate(source, None, spool_root=spool_root)
            if candidate is None:
                excluded.append(
                    {
                        "event_uid": event_uid,
                        "review_status": "pending",
                        "reason": reason or "invalid_candidate",
                    }
                )
                continue
            pending_candidates.append(candidate)
            taxonomy.add(candidate.effective_class_name)
            pending_distribution[candidate.effective_class_name] += 1
            if candidate.model_class_name:
                model_distribution[candidate.model_class_name] += 1
            continue
        review_counts[review.decision] += 1
        if review.decision == DECISION_NO:
            continue
        if review.decision != DECISION_YES:
            excluded.append(
                {
                    "event_uid": event_uid,
                    "review_status": review.decision,
                    "reason": "unsupported_review_decision",
                }
            )
            continue

        candidate, reason = _build_candidate(source, review, spool_root=spool_root)
        if candidate is None:
            excluded.append(
                {
                    "event_uid": event_uid,
                    "review_status": "reviewed",
                    "reason": reason or "invalid_candidate",
                }
            )
            continue

        reviewed_candidates.append(candidate)
        taxonomy.add(candidate.effective_class_name)
        reviewed_distribution[candidate.effective_class_name] += 1
        if candidate.model_class_name:
            model_distribution[candidate.model_class_name] += 1
            if candidate.model_class_name != candidate.effective_class_name:
                correction_matrix[candidate.model_class_name][candidate.effective_class_name] += 1

    selected_reviewed = _select_candidates(
        reviewed_candidates,
        max_per_class=max_per_class,
    )
    selected_pending = _select_candidates(
        pending_candidates,
        max_per_class=max_per_class,
    )
    selected_reviewed_uids = {candidate.event_uid for candidate in selected_reviewed}
    selected_pending_uids = {candidate.event_uid for candidate in selected_pending}
    for candidate in reviewed_candidates:
        if candidate.event_uid not in selected_reviewed_uids:
            excluded.append(
                {
                    "event_uid": candidate.event_uid,
                    "review_status": "reviewed",
                    "reason": "excluded_by_class_cap",
                }
            )
    for candidate in pending_candidates:
        if candidate.event_uid not in selected_pending_uids:
            excluded.append(
                {
                    "event_uid": candidate.event_uid,
                    "review_status": "pending",
                    "reason": "excluded_by_class_cap",
                }
            )

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(parent)))
    try:
        result = _write_export(
            temporary,
            selected_reviewed=selected_reviewed,
            selected_pending=selected_pending,
            taxonomy=taxonomy,
            spool_root=spool_root,
            review_path=review_path,
            start_date=start_date,
            end_date=end_date,
            max_per_class=max_per_class,
            scan_counts=scan_counts,
            review_counts=review_counts,
            reviewed_distribution=reviewed_distribution,
            pending_distribution=pending_distribution,
            model_distribution=model_distribution,
            correction_matrix=correction_matrix,
            excluded=excluded,
        )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    result["output_dir"] = str(destination)
    return result


def _load_event_sources(
    spool_root: Path,
    *,
    start_date: Optional[date],
    end_date: Optional[date],
) -> Tuple[List[_EventSource], set[str], Counter[str]]:
    sources: List[_EventSource] = []
    taxonomy: set[str] = set()
    counts: Counter[str] = Counter()

    for run_dir in iter_spool_runs(spool_root):
        run_meta = _load_json_object(run_dir / "run.json")
        if run_meta is None:
            counts["runs_invalid_metadata"] += 1
            continue
        taxonomy.update(_taxonomy_names(run_meta))

        events_path = run_dir / "events.jsonl"
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            counts["runs_unreadable_events"] += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            counts["event_lines_scanned"] += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                counts["malformed_event_lines"] += 1
                continue
            if not isinstance(event, dict):
                counts["malformed_event_lines"] += 1
                continue

            event_date = _event_date(event)
            if (start_date is not None or end_date is not None) and event_date is None:
                counts["events_missing_date"] += 1
                continue
            if start_date is not None and event_date is not None and event_date < start_date:
                counts["events_outside_date_range"] += 1
                continue
            if end_date is not None and event_date is not None and event_date > end_date:
                counts["events_outside_date_range"] += 1
                continue

            counts["events_in_date_range"] += 1
            sources.append(
                _EventSource(
                    run_dir=run_dir.resolve(),
                    run_meta=run_meta,
                    event=event,
                    line_number=line_number,
                )
            )

    return sources, taxonomy, counts


def _load_reviews(
    review_path: Path,
    events: Iterable[Mapping[str, Any]],
) -> Dict[str, ReviewRecord]:
    event_uids = sorted(
        {
            event_uid
            for event in events
            if (event_uid := _text(event.get("event_uid"))) is not None
        }
    )
    store = ReviewStore(review_path)
    reviews: Dict[str, ReviewRecord] = {}
    for offset in range(0, len(event_uids), REVIEW_BATCH_SIZE):
        reviews.update(store.get_reviews(event_uids[offset : offset + REVIEW_BATCH_SIZE]))
    return reviews


def _build_candidate(
    source: _EventSource,
    review: Optional[ReviewRecord],
    *,
    spool_root: Path,
) -> Tuple[Optional[_Candidate], Optional[str]]:
    event = source.event
    event_uid = _text(event.get("event_uid"))
    if event_uid is None:
        return None, "missing_event_uid"

    model_class_name = _text(event.get("class_name"))
    reviewed_class_name = _text(review.reviewed_class) if review is not None else None
    effective_class_name = reviewed_class_name or model_class_name
    if effective_class_name is None:
        return None, "missing_effective_class"

    frame_relpath = _text(event.get("training_frame_relpath"))
    if frame_relpath is None:
        return None, "missing_training_frame_reference"
    frame_path = (source.run_dir / frame_relpath).resolve()
    try:
        source_frame_relpath = frame_path.relative_to(spool_root).as_posix()
        frame_path.relative_to(source.run_dir)
    except ValueError:
        return None, "unsafe_training_frame_path"
    if frame_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return None, "unsupported_training_frame_type"
    if not frame_path.is_file():
        return None, "missing_training_frame_file"

    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, "unreadable_training_frame"
    frame_height, frame_width = image.shape[:2]
    if frame_width < 1 or frame_height < 1:
        return None, "invalid_frame_dimensions"

    bbox_xyxy = _valid_bbox(event.get("bbox_xyxy") or event.get("bbox"))
    if bbox_xyxy is None:
        return None, "invalid_bbox"
    x1, y1, x2, y2 = bbox_xyxy
    x1 = min(max(x1, 0.0), float(frame_width))
    y1 = min(max(y1, 0.0), float(frame_height))
    x2 = min(max(x2, 0.0), float(frame_width))
    y2 = min(max(y2, 0.0), float(frame_height))
    if x2 <= x1 or y2 <= y1:
        return None, "bbox_outside_frame"

    run_uid = (
        _text(event.get("run_uid"))
        or _text(source.run_meta.get("run_uid"))
        or source.run_dir.name
    )
    confidence = _finite_float(event.get("confidence"))
    occurred_at = (
        _text(event.get("occurred_at_local"))
        or _text(event.get("occurred_at_utc"))
        or ""
    )
    source_event_relpath = (source.run_dir / "events.jsonl").relative_to(spool_root).as_posix()

    return (
        _Candidate(
            event_uid=event_uid,
            run_uid=run_uid,
            run_dir=source.run_dir,
            source_event_relpath=f"{source_event_relpath}:{source.line_number}",
            frame_path=frame_path,
            source_frame_relpath=source_frame_relpath,
            frame_width=frame_width,
            frame_height=frame_height,
            bbox_xyxy=(x1, y1, x2, y2),
            bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
            model_class_name=model_class_name,
            reviewed_class_name=reviewed_class_name,
            effective_class_name=effective_class_name,
            confidence=confidence,
            occurred_at=occurred_at,
            review_status="reviewed" if review is not None else "pending",
            review_updated_at_utc=review.updated_at_utc if review is not None else None,
        ),
        None,
    )


def _select_candidates(
    candidates: Sequence[_Candidate],
    *,
    max_per_class: Optional[int],
) -> List[_Candidate]:
    ordered = sorted(candidates, key=lambda item: item.sort_key)
    if max_per_class is None:
        return ordered

    by_class: Dict[str, List[_Candidate]] = defaultdict(list)
    for candidate in ordered:
        by_class[candidate.effective_class_name].append(candidate)

    selected_frames: set[str] = set()
    for class_name in sorted(by_class, key=str.casefold):
        class_candidates = by_class[class_name]
        for candidate in _evenly_spaced(class_candidates, int(max_per_class)):
            selected_frames.add(candidate.frame_key)

    # Keep every known, accepted box on a selected frame. The cap is therefore
    # soft when multiple reviewed vehicles share one image.
    return [candidate for candidate in ordered if candidate.frame_key in selected_frames]


def _evenly_spaced(items: Sequence[_Candidate], limit: int) -> List[_Candidate]:
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    last_index = len(items) - 1
    indexes = [round(position * last_index / (limit - 1)) for position in range(limit)]
    return [items[index] for index in indexes]


def _write_export(
    output_dir: Path,
    *,
    selected_reviewed: Sequence[_Candidate],
    selected_pending: Sequence[_Candidate],
    taxonomy: set[str],
    spool_root: Path,
    review_path: Path,
    start_date: Optional[date],
    end_date: Optional[date],
    max_per_class: Optional[int],
    scan_counts: Counter[str],
    review_counts: Counter[str],
    reviewed_distribution: Counter[str],
    pending_distribution: Counter[str],
    model_distribution: Counter[str],
    correction_matrix: Mapping[str, Counter[str]],
    excluded: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    category_names = sorted(taxonomy, key=lambda value: (value.casefold(), value))
    category_ids = {name: index for index, name in enumerate(category_names, start=1)}
    categories = [
        {"id": category_ids[name], "name": name, "supercategory": "vehicle"}
        for name in category_names
    ]
    created_at = datetime.now(timezone.utc).isoformat()
    reviewed_result = _write_coco_subset(
        output_dir / "reviewed",
        candidates=selected_reviewed,
        categories=categories,
        category_ids=category_ids,
        coco_filename=COCO_FILENAME,
        description=(
            "Operator-accepted production crossing candidates; validate boxes "
            "and all visible vehicles in CVAT before training"
        ),
        created_at=created_at,
    )
    pending_result = _write_coco_subset(
        output_dir / "pending",
        candidates=selected_pending,
        categories=categories,
        category_ids=category_ids,
        coco_filename=COCO_FILENAME,
        description=(
            "Unreviewed production crossing pre-annotations; review every label "
            "and box in CVAT before using as ground truth"
        ),
        created_at=created_at,
    )

    all_distribution_names = sorted(
        taxonomy
        | set(reviewed_distribution)
        | set(pending_distribution)
        | set(reviewed_result["distribution"])
        | set(pending_result["distribution"]),
        key=lambda value: (value.casefold(), value),
    )
    audit = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "warning": (
            "Candidate boxes come from the deployed detector/tracker. Validate every box "
            "and label every visible target vehicle in CVAT before training."
        ),
        "source": {
            "spool_dir": str(spool_root),
            "review_db": str(review_path),
        },
        "filters": {
            "date_from": start_date.isoformat() if start_date is not None else None,
            "date_to": end_date.isoformat() if end_date is not None else None,
            "reviewed_decision": DECISION_YES,
            "pending_exported_separately": True,
            "max_per_class": max_per_class,
            "class_cap_is_soft_for_shared_frames": True,
        },
        "counts": {
            **dict(sorted(scan_counts.items())),
            "pending_reviews": int(review_counts["pending"]),
            "accepted_reviews": int(review_counts[DECISION_YES]),
            "rejected_reviews": int(review_counts[DECISION_NO]),
            "eligible_reviewed_annotations": int(sum(reviewed_distribution.values())),
            "eligible_pending_annotations": int(sum(pending_distribution.values())),
            "exported_reviewed_images": reviewed_result["image_count"],
            "exported_reviewed_annotations": reviewed_result["annotation_count"],
            "exported_pending_images": pending_result["image_count"],
            "exported_pending_annotations": pending_result["annotation_count"],
            "excluded_candidates": len(excluded),
        },
        "class_distribution": {
            name: {
                "reviewed_eligible": int(reviewed_distribution[name]),
                "reviewed_exported": int(reviewed_result["distribution"][name]),
                "pending_eligible": int(pending_distribution[name]),
                "pending_exported": int(pending_result["distribution"][name]),
            }
            for name in all_distribution_names
        },
        "model_prediction_distribution": dict(
            sorted(model_distribution.items(), key=lambda item: item[0].casefold())
        ),
        "classification_corrections": {
            source_name: dict(
                sorted(targets.items(), key=lambda item: item[0].casefold())
            )
            for source_name, targets in sorted(
                correction_matrix.items(), key=lambda item: item[0].casefold()
            )
        },
        "annotations": reviewed_result["audit"] + pending_result["audit"],
        "excluded": list(excluded),
    }
    (output_dir / AUDIT_FILENAME).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.txt").write_text(
        "\n".join(
            [
                "Active-learning candidate export",
                "",
                (
                    "Reviewed COCO: "
                    f"reviewed/annotations/{COCO_FILENAME}"
                ),
                (
                    "Pending COCO: "
                    f"pending/annotations/{COCO_FILENAME}"
                ),
                f"Audit manifest: {AUDIT_FILENAME}",
                "",
                "IMPORTANT:",
                "- Reviewed and pending candidates are intentionally separate.",
                "- These boxes were saved automatically by the deployed model.",
                "- Operator review verifies the event/class, not exact box geometry.",
                "- Pending labels are unverified model predictions.",
                "- Open each candidate set in CVAT and verify every box.",
                "- Add boxes for every visible target vehicle before training.",
                "- Do not train directly from the pending candidate set.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "reviewed_images": reviewed_result["image_count"],
        "reviewed_annotations": reviewed_result["annotation_count"],
        "pending_images": pending_result["image_count"],
        "pending_annotations": pending_result["annotation_count"],
        "class_distribution": {
            name: {
                "reviewed": int(reviewed_result["distribution"][name]),
                "pending": int(pending_result["distribution"][name]),
            }
            for name in all_distribution_names
        },
        "reviewed_coco_path": str(
            Path("reviewed") / "annotations" / COCO_FILENAME
        ),
        "pending_coco_path": str(
            Path("pending") / "annotations" / COCO_FILENAME
        ),
        "audit_path": AUDIT_FILENAME,
    }


def _write_coco_subset(
    dataset_dir: Path,
    *,
    candidates: Sequence[_Candidate],
    categories: Sequence[Mapping[str, Any]],
    category_ids: Mapping[str, int],
    coco_filename: str,
    description: str,
    created_at: str,
) -> Dict[str, Any]:
    images_dir = dataset_dir / "images" / COCO_SUBSET
    annotations_dir = dataset_dir / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    grouped: Dict[str, List[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.frame_key].append(candidate)

    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    audit_annotations: List[Dict[str, Any]] = []
    distribution: Counter[str] = Counter()
    annotation_id = 1

    for image_id, frame_key in enumerate(sorted(grouped), start=1):
        frame_candidates = sorted(grouped[frame_key], key=lambda item: item.event_uid)
        primary = frame_candidates[0]
        copied_name = _export_image_name(primary)
        copied_relpath = Path("images") / COCO_SUBSET / copied_name
        shutil.copy2(primary.frame_path, images_dir / copied_name)
        coco_images.append(
            {
                "id": image_id,
                "file_name": copied_relpath.as_posix(),
                "width": primary.frame_width,
                "height": primary.frame_height,
            }
        )

        for candidate in frame_candidates:
            x, y, width, height = candidate.bbox_xywh
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[candidate.effective_class_name],
                    "bbox": [_rounded(x), _rounded(y), _rounded(width), _rounded(height)],
                    "area": _rounded(width * height),
                    "iscrowd": 0,
                }
            )
            distribution[candidate.effective_class_name] += 1
            audit_annotations.append(
                {
                    "dataset": candidate.review_status,
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "event_uid": candidate.event_uid,
                    "run_uid": candidate.run_uid,
                    "source_event": candidate.source_event_relpath,
                    "source_frame": candidate.source_frame_relpath,
                    "copied_image": (Path(candidate.review_status) / copied_relpath).as_posix(),
                    "model_class_name": candidate.model_class_name,
                    "reviewed_class_name": candidate.reviewed_class_name,
                    "effective_class_name": candidate.effective_class_name,
                    "confidence": candidate.confidence,
                    "bbox_xyxy": [_rounded(value) for value in candidate.bbox_xyxy],
                    "review_updated_at_utc": candidate.review_updated_at_utc,
                }
            )
            annotation_id += 1

    coco = {
        "info": {
            "description": description,
            "version": "1.0",
            "date_created": created_at,
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": list(categories),
    }
    (annotations_dir / coco_filename).write_text(
        json.dumps(coco, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "image_count": len(coco_images),
        "annotation_count": len(coco_annotations),
        "distribution": distribution,
        "audit": audit_annotations,
    }


def _normalize_date_range(
    date_from: Optional[str],
    date_to: Optional[str],
) -> Tuple[Optional[date], Optional[date]]:
    start = _parse_date(date_from, option="date_from")
    end = _parse_date(date_to, option="date_to")
    if start is not None and end is not None and start > end:
        raise ValueError("date_from cannot be after date_to")
    return start, end


def _parse_date(value: Optional[str], *, option: str) -> Optional[date]:
    text = _text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{option} must use YYYY-MM-DD") from exc


def _event_date(event: Mapping[str, Any]) -> Optional[date]:
    for key in ("occurred_at_local", "occurred_at_utc"):
        text = _text(event.get(key))
        if text is None:
            continue
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            continue
    return None


def _taxonomy_names(run_meta: Mapping[str, Any]) -> set[str]:
    raw = run_meta.get("class_names")
    if isinstance(raw, Mapping):
        values = raw.values()
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    return {name for value in values if (name := _text(value)) is not None}


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _valid_bbox(value: object) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = tuple(_finite_float(item) for item in value)
    if any(item is None for item in numbers):
        return None
    x1, y1, x2, y2 = numbers
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _finite_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: object) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _rounded(value: float) -> float:
    return round(float(value), 4)


def _export_image_name(candidate: _Candidate) -> str:
    run_uid = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in candidate.run_uid
    ).strip("_")
    if not run_uid:
        run_uid = "run"
    digest = hashlib.sha1(candidate.source_frame_relpath.encode("utf-8")).hexdigest()[:10]
    return f"{run_uid}_{candidate.frame_path.stem}_{digest}{candidate.frame_path.suffix.lower()}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export accepted and pending crossing events as separate "
            "CVAT-compatible COCO candidate datasets."
        )
    )
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--review-db",
        type=Path,
        default=None,
        help=f"Defaults to <spool-dir>/{DEFAULT_REVIEW_DB_FILENAME}",
    )
    parser.add_argument("--date-from", default=None, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--date-to", default=None, help="Inclusive YYYY-MM-DD")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help=(
            "Optional soft cap for majority classes. Samples are spread across time; "
            "all candidate boxes sharing a selected image stay together."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = export_reviewed_coco(
            spool_dir=args.spool_dir,
            output_dir=args.output_dir,
            review_db_path=args.review_db,
            date_from=args.date_from,
            date_to=args.date_to,
            max_per_class=args.max_per_class,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"[active-learning-export] {exc}", file=sys.stderr)
        return 2

    print(
        f"[active-learning-export] reviewed="
        f"{result['reviewed_images']} images/{result['reviewed_annotations']} boxes "
        f"pending={result['pending_images']} images/{result['pending_annotations']} boxes "
        f"output={result['output_dir']}"
    )
    for class_name, counts in result["class_distribution"].items():
        print(
            f"  {class_name}: reviewed={counts['reviewed']} "
            f"pending={counts['pending']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
