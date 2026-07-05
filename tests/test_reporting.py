"""Tests for the PDF incident report builder."""

from pathlib import Path

import pytest

from visionguard.reporting.pdf import IncidentReportBuilder
from visionguard.safety.events import EventType, SafetyEvent
from visionguard.storage.event_store import EventStore


@pytest.fixture()
def store_with_run(tmp_path: Path) -> tuple[EventStore, int, Path]:
    store = EventStore(tmp_path / "events.db")
    run_id = store.create_run("demo.mp4")
    store.add_event(
        run_id,
        SafetyEvent(
            event_type=EventType.PPE_VIOLATION,
            frame_index=120,
            video_time=4.0,
            track_id=3,
            track_label="Worker #3",
            confidence=0.9,
            description="Worker #3 detected without helmet",
        ),
    )
    store.add_event(
        run_id,
        SafetyEvent(
            event_type=EventType.FALL,
            frame_index=600,
            video_time=20.0,
            track_id=5,
            track_label="Worker #5",
            confidence=0.8,
            description="FALL DETECTED: Worker #5 down for 2.4s",
        ),
    )
    store.finish_run(
        run_id,
        fps=30.0,
        frames_processed=900,
        duration_seconds=30.0,
        compliance_rate=0.87,
        stats={
            "events_by_type": {"ppe_violation": 1, "fall": 1},
            "unique_counts": {"worker": 6, "vehicle": 1},
            "most_dangerous_zone": "Crane area",
            "falls_detected": 1,
        },
    )
    return store, run_id, tmp_path


def test_report_pdf_is_generated(store_with_run) -> None:
    store, run_id, tmp_path = store_with_run
    builder = IncidentReportBuilder(store, tmp_path / "reports")

    pdf_path = builder.build(run_id)

    assert pdf_path.exists()
    assert pdf_path.suffix == ".pdf"
    assert pdf_path.stat().st_size > 1000  # a real document, not an empty shell
    assert pdf_path.read_bytes()[:5] == b"%PDF-"


def test_missing_screenshot_does_not_crash(store_with_run) -> None:
    """Events whose screenshot file vanished must not break report generation."""
    store, run_id, tmp_path = store_with_run
    event = store.events_for_run(run_id)[0]
    event.screenshot_path = "definitely/not/here.jpg"

    builder = IncidentReportBuilder(store, tmp_path / "reports")
    assert builder.build(run_id).exists()


def test_unknown_run_raises(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    builder = IncidentReportBuilder(store, tmp_path / "reports")
    with pytest.raises(ValueError, match="999"):
        builder.build(999)
