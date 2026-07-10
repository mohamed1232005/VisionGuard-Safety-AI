"""SQLite-backed persistence for analysis runs and safety events.

SQLite keeps Phase 1 zero-configuration and fully demoable on any machine; the
schema is deliberately simple so a later swap to PostgreSQL (Phase 2) is a
matter of changing the connection layer, not the callers.

Connections are opened per operation, which makes the store safe to use from
Streamlit's threads without shared-connection headaches.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from visionguard.safety.events import EventType, SafetyEvent, Severity

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_source TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    fps REAL,
    frames_processed INTEGER,
    duration_seconds REAL,
    compliance_rate REAL,
    stats_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    video_time REAL NOT NULL,
    wall_time TEXT NOT NULL,
    track_id INTEGER,
    track_label TEXT,
    zone_name TEXT,
    confidence REAL NOT NULL,
    description TEXT NOT NULL,
    screenshot_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
"""


@dataclass(frozen=True)
class RunRecord:
    """A finished (or in-progress) analysis run."""

    id: int
    video_source: str
    started_at: str
    finished_at: str | None
    fps: float | None
    frames_processed: int | None
    duration_seconds: float | None
    compliance_rate: float | None
    stats: dict[str, Any]


class EventStore:
    """Stores runs and their safety events in a SQLite database."""

    def __init__(self, database_path: Path | str) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Runs
    # ------------------------------------------------------------------ #
    def create_run(self, video_source: str) -> int:
        """Register a new analysis run and return its id."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO runs (video_source, started_at) VALUES (?, ?)",
                (video_source, datetime.now(timezone.utc).isoformat()),
            )
            return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        fps: float,
        frames_processed: int,
        duration_seconds: float,
        compliance_rate: float,
        stats: dict[str, Any],
    ) -> None:
        """Record final metrics once a run completes."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE runs SET finished_at = ?, fps = ?, frames_processed = ?,
                   duration_seconds = ?, compliance_rate = ?, stats_json = ?
                   WHERE id = ?""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    fps,
                    frames_processed,
                    duration_seconds,
                    compliance_rate,
                    json.dumps(stats),
                    run_id,
                ),
            )

    def list_runs(self) -> list[RunRecord]:
        """All runs, newest first."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
        return [self._row_to_run(row) for row in rows]

    def get_run(self, run_id: int) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._row_to_run(row) if row else None

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=row["id"],
            video_source=row["video_source"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            fps=row["fps"],
            frames_processed=row["frames_processed"],
            duration_seconds=row["duration_seconds"],
            compliance_rate=row["compliance_rate"],
            stats=json.loads(row["stats_json"]) if row["stats_json"] else {},
        )

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def add_event(self, run_id: int, event: SafetyEvent) -> int:
        """Persist one safety event and return its id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO events
                   (run_id, event_type, severity, frame_index, video_time,
                    wall_time, track_id, track_label, zone_name, confidence,
                    description, screenshot_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    event.event_type.value,
                    event.severity.value,
                    event.frame_index,
                    event.video_time,
                    event.wall_time.isoformat(),
                    event.track_id,
                    event.track_label,
                    event.zone_name,
                    event.confidence,
                    event.description,
                    event.screenshot_path,
                ),
            )
            return int(cursor.lastrowid)

    def events_for_run(self, run_id: int) -> list[SafetyEvent]:
        """All events of a run in chronological order."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY video_time",
                (run_id,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> SafetyEvent:
        event = SafetyEvent(
            event_type=EventType(row["event_type"]),
            frame_index=row["frame_index"],
            video_time=row["video_time"],
            track_id=row["track_id"],
            track_label=row["track_label"],
            zone_name=row["zone_name"],
            confidence=row["confidence"],
            description=row["description"],
            wall_time=datetime.fromisoformat(row["wall_time"]),
            screenshot_path=row["screenshot_path"],
        )
        event.severity = Severity(row["severity"])
        return event

    def event_type_counts(self, run_id: int) -> dict[str, int]:
        """Event counts by type — powers the 'most common violation' KPI."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT event_type, COUNT(*) AS n FROM events
                   WHERE run_id = ? GROUP BY event_type""",
                (run_id,),
            ).fetchall()
        return {row["event_type"]: row["n"] for row in rows}
