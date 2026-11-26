import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from app.schemas import ClassificationResult, IncidentCategory, IncidentCreate, IncidentPriority, IncidentTriageResponse

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "storeops_triage.db"


@contextmanager
def get_connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                store_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                suggested_category TEXT,
                suggested_priority TEXT,
                needs_review INTEGER NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _row_to_response(row: sqlite3.Row) -> IncidentTriageResponse:
    return IncidentTriageResponse(
        id=row["id"],
        store_id=row["store_id"],
        title=row["title"],
        description=row["description"],
        suggested_category=IncidentCategory(row["suggested_category"]) if row["suggested_category"] else None,
        suggested_priority=IncidentPriority(row["suggested_priority"]) if row["suggested_priority"] else None,
        needs_review=bool(row["needs_review"]),
        confidence=row["confidence"],
        reason=row["reason"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def save_incident(
    payload: IncidentCreate,
    classification: ClassificationResult,
    db_path: Path = DEFAULT_DB_PATH,
) -> IncidentTriageResponse:
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO incidents (
                store_id, title, description,
                suggested_category, suggested_priority,
                needs_review, confidence, reason, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.store_id,
                payload.title,
                payload.description,
                classification.suggested_category.value if classification.suggested_category else None,
                classification.suggested_priority.value if classification.suggested_priority else None,
                int(classification.needs_review),
                classification.confidence,
                classification.reason,
                created_at,
            ),
        )
        conn.commit()
        incident_id = cursor.lastrowid
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        assert row is not None
        return _row_to_response(row)


def get_incident_by_id(incident_id: int, db_path: Path = DEFAULT_DB_PATH) -> Optional[IncidentTriageResponse]:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        return _row_to_response(row) if row else None


def list_incidents(
    needs_review: Optional[bool] = None,
    limit: int = 50,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[IncidentTriageResponse]:
    with get_connection(db_path) as conn:
        if needs_review is None:
            rows = conn.execute(
                "SELECT * FROM incidents ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM incidents WHERE needs_review = ? ORDER BY id DESC LIMIT ?",
                (int(needs_review), limit),
            ).fetchall()
        return [_row_to_response(row) for row in rows]
