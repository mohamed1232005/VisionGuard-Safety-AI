"""Tests for the SQLite event store."""

from pathlib import Path

from visionguard.safety.events import EventType, SafetyEvent, Severity
from visionguard.storage.event_store import EventStore


def make_event(video_time: float = 1.0) -> SafetyEvent:
    return SafetyEvent(
        event_type=EventType.PPE_VIOLATION,
        frame_index=int(video_time * 30),
        video_time=video_time,
        track_id=12,
        track_label="Worker #12",
        confidence=0.85,
        description="Worker #12 detected without helmet",
    )


def test_round_trip(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    run_id = store.create_run("video.mp4")

    event = make_event()
    event.screenshot_path = "shot.jpg"
    store.add_event(run_id, event)

    loaded = store.events_for_run(run_id)
    assert len(loaded) == 1
    restored = loaded[0]
    assert restored.event_type is EventType.PPE_VIOLATION
    assert restored.severity is Severity.WARNING
    assert restored.track_label == "Worker #12"
    assert restored.screenshot_path == "shot.jpg"
    assert restored.confidence == 0.85


def test_events_ordered_by_video_time(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    run_id = store.create_run("video.mp4")
    for t in (5.0, 1.0, 3.0):
        store.add_event(run_id, make_event(video_time=t))

    times = [e.video_time for e in store.events_for_run(run_id)]
    assert times == sorted(times)


def test_finish_run_records_metrics(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    run_id = store.create_run("video.mp4")
    store.finish_run(
        run_id,
        fps=30.0,
        frames_processed=900,
        duration_seconds=30.0,
        compliance_rate=0.93,
        stats={"falls_detected": 0},
    )

    run = store.get_run(run_id)
    assert run is not None
    assert run.finished_at is not None
    assert run.compliance_rate == 0.93
    assert run.stats["falls_detected"] == 0


def test_event_type_counts(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    run_id = store.create_run("video.mp4")
    for _ in range(3):
        store.add_event(run_id, make_event())

    assert store.event_type_counts(run_id) == {"ppe_violation": 3}


def test_runs_isolated(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    run_a = store.create_run("a.mp4")
    run_b = store.create_run("b.mp4")
    store.add_event(run_a, make_event())

    assert len(store.events_for_run(run_a)) == 1
    assert store.events_for_run(run_b) == []
