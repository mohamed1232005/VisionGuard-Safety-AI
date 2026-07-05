"""Tests for the PPE compliance engine (synthetic workers and evidence)."""

from collections import deque

import pytest

from visionguard.detection.types import BoundingBox, Detection, ObjectClass
from visionguard.safety.events import EventType
from visionguard.safety.ppe import PPEComplianceEngine
from visionguard.tracking.tracker import TrackedObject
from visionguard.utils.config import PPESettings


def make_settings(**overrides: object) -> PPESettings:
    defaults = dict(
        required_equipment=("helmet",),
        window_seconds=1.0,
        violation_ratio=0.6,
        min_observations=3,
        cooldown_seconds=10.0,
    )
    defaults.update(overrides)
    return PPESettings(**defaults)


def make_worker(track_id: int = 1) -> TrackedObject:
    """Worker occupying x 100-200, y 100-400 (300 px tall)."""
    return TrackedObject(
        track_id=track_id,
        object_class=ObjectClass.WORKER,
        box=BoundingBox(100, 100, 200, 400),
        confidence=0.9,
        trajectory=deque(maxlen=10),
    )


def no_helmet_evidence() -> Detection:
    """A NO-Hardhat box at the worker's head (center ~(150, 130))."""
    return Detection(BoundingBox(130, 110, 170, 150), ObjectClass.NO_HELMET, 0.8)


def helmet_evidence() -> Detection:
    return Detection(BoundingBox(130, 110, 170, 150), ObjectClass.HELMET, 0.8)


def test_violation_raised_after_sustained_evidence() -> None:
    """No alert until min_observations frames, then exactly one alert."""
    engine = PPEComplianceEngine(make_settings(), fps=10)
    worker = make_worker()

    events = []
    for frame in range(6):
        events += engine.update(frame, frame / 10, [worker], [no_helmet_evidence()])

    violations = [e for e in events if e.event_type is EventType.PPE_VIOLATION]
    assert len(violations) == 1  # debounced: one alert, not one per frame
    assert violations[0].track_id == worker.track_id
    assert "helmet" in violations[0].description


def test_compliant_worker_never_alerts() -> None:
    engine = PPEComplianceEngine(make_settings(), fps=10)
    worker = make_worker()

    events = []
    for frame in range(20):
        events += engine.update(frame, frame / 10, [worker], [helmet_evidence()])

    assert events == []
    assert engine.stats.compliance_rate == 1.0


def test_flicker_below_ratio_does_not_alert() -> None:
    """A single missing frame among many worn frames stays silent."""
    engine = PPEComplianceEngine(make_settings(), fps=10)
    worker = make_worker()

    events = []
    for frame in range(10):
        evidence = no_helmet_evidence() if frame == 5 else helmet_evidence()
        events += engine.update(frame, frame / 10, [worker], [evidence])

    assert events == []


def test_helmet_evidence_at_feet_is_ignored() -> None:
    """Association is anatomical: a helmet box at foot level can't belong."""
    engine = PPEComplianceEngine(make_settings(), fps=10)
    worker = make_worker()
    at_feet = Detection(BoundingBox(130, 360, 170, 395), ObjectClass.NO_HELMET, 0.8)

    events = []
    for frame in range(10):
        events += engine.update(frame, frame / 10, [worker], [at_feet])

    assert events == []  # evidence never associated -> never judged


def test_overlapping_workers_evidence_goes_to_tightest_box() -> None:
    """When two workers overlap, the smaller box owns the evidence."""
    engine = PPEComplianceEngine(make_settings(), fps=10)
    big = make_worker(track_id=1)
    small = TrackedObject(
        track_id=2,
        object_class=ObjectClass.WORKER,
        box=BoundingBox(120, 105, 180, 300),
        confidence=0.9,
        trajectory=deque(maxlen=10),
    )

    events = []
    for frame in range(6):
        events += engine.update(
            frame, frame / 10, [big, small], [no_helmet_evidence()]
        )

    assert {e.track_id for e in events} == {2}


def test_unknown_equipment_in_config_fails_fast() -> None:
    with pytest.raises(ValueError, match="jetpack"):
        PPEComplianceEngine(make_settings(required_equipment=("jetpack",)), fps=10)
