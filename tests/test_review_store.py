from __future__ import annotations

from pathlib import Path

from pedestrian_line_counter.review_store import DECISION_NO, DECISION_YES, ReviewStore


def _save(store: ReviewStore, event_uid: str, decision: str) -> None:
    store.save_review(
        event_uid=event_uid,
        run_uid="run_a",
        site_id="site_a",
        camera_id="cam_01",
        decision=decision,
        reviewed_class=None,
        notes="",
        now_utc="2026-11-16T00:00:00Z",
    )


def test_get_reviews_handles_lookups_above_sqlite_variable_limit(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    _save(store, "event_first", DECISION_YES)
    _save(store, "event_last", DECISION_NO)
    # 40k keys exceed SQLite's default 32,766-variable limit for one statement.
    event_uids = ["event_first"] + [f"event_{index:05d}" for index in range(40_000)] + ["event_last"]

    reviews = store.get_reviews(event_uids)

    assert set(reviews) == {"event_first", "event_last"}
    assert reviews["event_last"].decision == DECISION_NO


def test_get_decisions_returns_decision_per_reviewed_event(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    _save(store, "event_yes", DECISION_YES)
    _save(store, "event_no", DECISION_NO)

    assert store.get_decisions() == {"event_yes": DECISION_YES, "event_no": DECISION_NO}
