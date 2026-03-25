from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


DECISION_YES = "qualified_yes"
DECISION_NO = "qualified_no"
VALID_DECISIONS = {DECISION_YES, DECISION_NO}


@dataclass(frozen=True)
class ReviewRecord:
    event_uid: str
    run_uid: Optional[str]
    site_id: Optional[str]
    camera_id: Optional[str]
    decision: str
    reviewed_class: Optional[str]
    notes: str
    created_at_utc: str
    updated_at_utc: str

    def to_dict(self) -> Dict[str, Optional[str]]:
        """
        Get all the attributes output as a json type file
        to return it again as a dictionary
        """
        return {
            "event_uid": self.event_uid,
            "run_uid": self.run_uid,
            "site_id": self.site_id,
            "camera_id": self.camera_id,
            "decision": self.decision,
            "reciewed_class": self.reviewed_class,
            "notes": self.notes,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc
        }



class ReviewStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_reviews (
                    event_uid TEXT PRIMARY KEY,
                    run_uid TEXT,
                    site_id TEXT,
                    camera_id TEXT,
                    decision TEXT NOT NULL,
                    reviewed_class TEXT,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(event_reviews)").fetchall()
            }
            if "reviewed_class" not in columns:
                conn.execute("ALTER TABLE event_reviews ADD COLUMN reviewed_class TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_reviews_camera_id ON event_reviews(camera_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_reviews_updated_at_utc ON event_reviews(updated_at_utc)"
            )

    def save_review(
        self,
        *,
        event_uid: str,
        run_uid: Optional[str],
        site_id: Optional[str],
        camera_id: Optional[str],
        decision: str,
        reviewed_class: Optional[str],
        notes: str,
        now_utc: str,
    ) -> ReviewRecord:
        normalized_decision = str(decision or "").strip().lower()
        if normalized_decision not in VALID_DECISIONS:
            raise ValueError(f"Unsupported review decision: {decision}")

        normalized_reviewed_class = _optional_text(reviewed_class)
        normalized_notes = str(notes or "").strip()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT created_at_utc FROM event_reviews WHERE event_uid = ?",
                (event_uid,),
            ).fetchone()
            created_at_utc = str(row["created_at_utc"]) if row is not None else now_utc
            conn.execute(
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
                ON CONFLICT(event_uid) DO UPDATE SET
                    run_uid = excluded.run_uid,
                    site_id = excluded.site_id,
                    camera_id = excluded.camera_id,
                    decision = excluded.decision,
                    reviewed_class = excluded.reviewed_class,
                    notes = excluded.notes,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    event_uid,
                    run_uid,
                    site_id,
                    camera_id,
                    normalized_decision,
                    normalized_reviewed_class,
                    normalized_notes,
                    created_at_utc,
                    now_utc,
                ),
            )

        return ReviewRecord(
            event_uid=event_uid,
            run_uid=run_uid,
            site_id=site_id,
            camera_id=camera_id,
            decision=normalized_decision,
            reviewed_class=normalized_reviewed_class,
            notes=normalized_notes,
            created_at_utc=created_at_utc,
            updated_at_utc=now_utc,
        )

    def get_review(self, event_uid: str) -> Optional[ReviewRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT event_uid, run_uid, site_id, camera_id, decision, reviewed_class, notes, created_at_utc, updated_at_utc
                FROM event_reviews
                WHERE event_uid = ?
                """,
                (event_uid,),
            ).fetchone()
        return _row_to_review(row)

    def get_reviews(self, event_uids: Iterable[str]) -> Dict[str, ReviewRecord]:
        keys = [str(item).strip() for item in event_uids if str(item).strip()]
        if not keys:
            return {}

        placeholders = ",".join("?" for _ in keys)
        query = (
            "SELECT event_uid, run_uid, site_id, camera_id, decision, reviewed_class, notes, created_at_utc, updated_at_utc "
            f"FROM event_reviews WHERE event_uid IN ({placeholders})"
        )
        with self._connect() as conn:
            rows = conn.execute(query, keys).fetchall()
        return {
            str(row["event_uid"]): review
            for row in rows
            if (review := _row_to_review(row)) is not None
        }

    def summary(self) -> Dict[str, int]:
        counts = {
            DECISION_YES: 0,
            DECISION_NO: 0,
            "reviewed_total": 0,
        }
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT decision, COUNT(*) AS count FROM event_reviews GROUP BY decision"
            ).fetchall()

        for row in rows:
            decision = str(row["decision"])
            count = int(row["count"])
            if decision in counts:
                counts[decision] = count
            counts["reviewed_total"] += count
        return counts


def _row_to_review(row: Optional[sqlite3.Row]) -> Optional[ReviewRecord]:
    if row is None:
        return None
    return ReviewRecord(
        event_uid=str(row["event_uid"]),
        run_uid=_optional_text(row["run_uid"]),
        site_id=_optional_text(row["site_id"]),
        camera_id=_optional_text(row["camera_id"]),
        decision=str(row["decision"]),
        reviewed_class=_optional_text(row["reviewed_class"]),
        notes=str(row["notes"] or ""),
        created_at_utc=str(row["created_at_utc"]),
        updated_at_utc=str(row["updated_at_utc"]),
    )


def _optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "DECISION_NO",
    "DECISION_YES",
    "ReviewRecord",
    "ReviewStore",
    "VALID_DECISIONS",
]
