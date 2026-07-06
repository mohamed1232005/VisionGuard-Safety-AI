"""Tests for worker-vehicle proximity monitoring (synthetic top-down plane)."""

from collections import deque
from pathlib import Path

import numpy as np

from visionguard.detection.types import BoundingBox, ObjectClass
from visionguard.safety.events import EventType, Severity
from visionguard.safety.proximity import ProximityMonitor, RiskLevel
from visionguard.spatial.homography import GroundPlane
from visionguard.tracking.tracker import TrackedObject
from visionguard.utils.config import ProximitySettings

FRAME = (1000, 1000)

# Top-down plane where 10 px = 1 m (identity * 0.1).
PLANE = GroundPlane(np.diag([0.1, 0.1, 1.0]), FRAME)

SETTINGS = ProximitySettings(
    calibration_file=Path("unused.json"),
    high_risk_distance_m=2.0,
    medium_risk_distance_m=5.0,
    cooldown_seconds=10.0,
)


def entity(cls: ObjectClass, track_id: int, ground_x_px: float) -> TrackedObject:
    """Object whose ground point sits at (ground_x_px, 500)."""
    return TrackedObject(
        track_id=track_id,
        object_class=cls,
        box=BoundingBox(ground_x_px - 10, 440, ground_x_px + 10, 500),
        confidence=0.9,
        trajectory=deque(maxlen=10),
    )


def worker(x_px: float) -> TrackedObject:
    return entity(ObjectClass.WORKER, 1, x_px)


def forklift(x_px: float) -> TrackedObject:
    return entity(ObjectClass.MACHINERY, 50, x_px)


def test_far_apart_is_silent() -> None:
    monitor = ProximityMonitor(SETTINGS, PLANE)
    # 200 px apart = 20 m
    events, pairs = monitor.update(0, 0.0, [worker(100), forklift(300)], *FRAME)
    assert events == [] and pairs == []


def test_medium_distance_warns_once() -> None:
    monitor = ProximityMonitor(SETTINGS, PLANE)
    # 40 px = 4 m -> medium risk
    first, pairs = monitor.update(0, 0.0, [worker(100), forklift(140)], *FRAME)
    second, _ = monitor.update(1, 1.0, [worker(100), forklift(140)], *FRAME)

    assert len(first) == 1
    assert first[0].event_type is EventType.PROXIMITY
    assert first[0].severity is Severity.WARNING
    assert "4.0 m" in first[0].description
    assert pairs[0].level is RiskLevel.MEDIUM
    assert second == []  # still medium: no repeat


def test_escalation_to_high_risk_alerts_again() -> None:
    monitor = ProximityMonitor(SETTINGS, PLANE)
    monitor.update(0, 0.0, [worker(100), forklift(140)], *FRAME)   # medium
    events, pairs = monitor.update(1, 1.0, [worker(100), forklift(115)], *FRAME)  # 1.5 m

    assert len(events) == 1
    assert events[0].severity is Severity.CRITICAL
    assert "High-risk" in events[0].description
    assert pairs[0].level is RiskLevel.HIGH
    assert monitor.stats.near_misses == 1


def test_min_distance_tracked() -> None:
    monitor = ProximityMonitor(SETTINGS, PLANE)
    monitor.update(0, 0.0, [worker(100), forklift(140)], *FRAME)
    monitor.update(1, 1.0, [worker(100), forklift(118)], *FRAME)
    assert monitor.stats.min_distance_m == 1.8


def test_retreat_and_reapproach_after_cooldown() -> None:
    monitor = ProximityMonitor(SETTINGS, PLANE)
    monitor.update(0, 0.0, [worker(100), forklift(140)], *FRAME)     # medium alert
    monitor.update(1, 5.0, [worker(100), forklift(400)], *FRAME)     # retreats
    events, _ = monitor.update(2, 20.0, [worker(100), forklift(140)], *FRAME)

    assert len(events) == 1  # cooldown (10 s) elapsed -> fresh alert


def test_two_workers_alert_independently() -> None:
    monitor = ProximityMonitor(SETTINGS, PLANE)
    workers = [entity(ObjectClass.WORKER, 1, 100), entity(ObjectClass.WORKER, 2, 180)]
    events, _ = monitor.update(0, 0.0, [*workers, forklift(140)], *FRAME)
    assert len(events) == 2  # forklift is 4 m from both
