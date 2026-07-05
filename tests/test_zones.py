"""Tests for restricted-zone monitoring (synthetic square zone)."""

from collections import deque
from pathlib import Path

from visionguard.detection.types import BoundingBox, ObjectClass
from visionguard.safety.events import EventType
from visionguard.safety.zones import Zone, ZoneMonitor
from visionguard.tracking.tracker import TrackedObject
from visionguard.utils.config import ZoneSettings

FRAME_W, FRAME_H = 100, 100

# Bottom-right quadrant of the frame.
ZONE = Zone(
    name="Test Zone",
    zone_type="restricted",
    risk_level=3,
    polygon=((0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)),
)

SETTINGS = ZoneSettings(definitions_file=Path("unused.json"), dwell_alert_seconds=2.0)


def worker_at(ground_x: float, ground_y: float, track_id: int = 1) -> TrackedObject:
    """Worker whose ground point (bottom-center) is at the given pixel."""
    return TrackedObject(
        track_id=track_id,
        object_class=ObjectClass.WORKER,
        box=BoundingBox(ground_x - 5, ground_y - 30, ground_x + 5, ground_y),
        confidence=0.9,
        trajectory=deque(maxlen=10),
    )


def test_entry_raises_single_intrusion() -> None:
    monitor = ZoneMonitor([ZONE], SETTINGS)
    inside = worker_at(75, 75)

    first = monitor.update(0, 0.0, [inside], FRAME_W, FRAME_H)
    second = monitor.update(1, 0.5, [inside], FRAME_W, FRAME_H)

    assert [e.event_type for e in first] == [EventType.ZONE_INTRUSION]
    assert first[0].zone_name == "Test Zone"
    assert second == []  # still inside -> no duplicate alert


def test_outside_worker_is_ignored() -> None:
    monitor = ZoneMonitor([ZONE], SETTINGS)
    outside = worker_at(25, 25)
    assert monitor.update(0, 0.0, [outside], FRAME_W, FRAME_H) == []


def test_dwell_alert_after_threshold() -> None:
    monitor = ZoneMonitor([ZONE], SETTINGS)
    worker = worker_at(75, 75)

    monitor.update(0, 0.0, [worker], FRAME_W, FRAME_H)          # entry
    early = monitor.update(1, 1.0, [worker], FRAME_W, FRAME_H)  # 1s: too soon
    late = monitor.update(2, 2.5, [worker], FRAME_W, FRAME_H)   # 2.5s: dwell

    assert early == []
    assert [e.event_type for e in late] == [EventType.ZONE_DWELL]
    assert monitor.update(3, 3.0, [worker], FRAME_W, FRAME_H) == []  # only once


def test_reentry_counts_as_new_intrusion() -> None:
    monitor = ZoneMonitor([ZONE], SETTINGS)

    monitor.update(0, 0.0, [worker_at(75, 75)], FRAME_W, FRAME_H)   # in
    monitor.update(1, 1.0, [worker_at(25, 25)], FRAME_W, FRAME_H)   # out
    again = monitor.update(2, 2.0, [worker_at(75, 75)], FRAME_W, FRAME_H)

    assert [e.event_type for e in again] == [EventType.ZONE_INTRUSION]
    assert monitor.stats.intrusions["Test Zone"] == 2


def test_vehicle_not_restricted_by_worker_zone() -> None:
    monitor = ZoneMonitor([ZONE], SETTINGS)
    vehicle = TrackedObject(
        track_id=9,
        object_class=ObjectClass.VEHICLE,
        box=BoundingBox(70, 45, 80, 75),
        confidence=0.9,
        trajectory=deque(maxlen=10),
    )
    assert monitor.update(0, 0.0, [vehicle], FRAME_W, FRAME_H) == []


def test_most_dangerous_zone_statistic() -> None:
    quiet = Zone("Quiet", "restricted", 1, ((0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)))
    monitor = ZoneMonitor([ZONE, quiet], SETTINGS)

    monitor.update(0, 0.0, [worker_at(75, 75, track_id=1)], FRAME_W, FRAME_H)
    monitor.update(1, 1.0, [worker_at(75, 75, track_id=2)], FRAME_W, FRAME_H)
    monitor.update(2, 2.0, [worker_at(20, 20, track_id=3)], FRAME_W, FRAME_H)

    assert monitor.stats.most_dangerous_zone() == "Test Zone"
