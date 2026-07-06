"""Tests for the 0-100 Safety Risk Score."""

import pytest

from visionguard.safety.events import EventType, SafetyEvent, Severity
from visionguard.safety.risk import RiskScoreCalculator, risk_band
from visionguard.utils.config import RiskScoreSettings

SETTINGS = RiskScoreSettings(
    window_seconds=60.0,
    weights={
        "ppe_violation": 8,
        "zone_intrusion": 12,
        "zone_dwell": 6,
        "fall": 45,
        "proximity": 10,
        "proximity_high": 25,
    },
)


def event(
    event_type: EventType, video_time: float, severity: Severity | None = None
) -> SafetyEvent:
    return SafetyEvent(
        event_type=event_type,
        frame_index=int(video_time * 30),
        video_time=video_time,
        track_id=1,
        track_label="Worker #1",
        description="test",
        severity=severity,
    )


def test_quiet_site_scores_zero() -> None:
    calc = RiskScoreCalculator(SETTINGS)
    assert calc.score(30.0) == 0.0
    assert risk_band(0.0) == "Safe"


def test_fresh_event_contributes_full_weight() -> None:
    calc = RiskScoreCalculator(SETTINGS)
    calc.add_events([event(EventType.FALL, 10.0)])
    assert calc.score(10.0) == pytest.approx(45.0)
    assert risk_band(45.0) == "Moderate Risk"


def test_events_decay_over_the_window() -> None:
    calc = RiskScoreCalculator(SETTINGS)
    calc.add_events([event(EventType.FALL, 0.0)])
    assert calc.score(30.0) == pytest.approx(22.5)   # half the window -> half weight
    assert calc.score(60.0) == 0.0                   # fully expired


def test_score_is_clamped_at_100() -> None:
    calc = RiskScoreCalculator(SETTINGS)
    calc.add_events([event(EventType.FALL, 5.0) for _ in range(10)])
    assert calc.score(5.0) == 100.0
    assert risk_band(100.0) == "Critical Risk"


def test_high_risk_proximity_uses_escalated_weight() -> None:
    calc = RiskScoreCalculator(SETTINGS)
    calc.add_events([event(EventType.PROXIMITY, 1.0, severity=Severity.CRITICAL)])
    assert calc.score(1.0) == pytest.approx(25.0)    # proximity_high, not proximity


def test_peak_score_is_remembered() -> None:
    calc = RiskScoreCalculator(SETTINGS)
    calc.add_events([event(EventType.ZONE_INTRUSION, 2.0)])
    calc.score(2.0)
    calc.score(50.0)  # decayed almost away
    assert calc.peak_score == pytest.approx(12.0)
    assert calc.peak_time == 2.0


def test_bands() -> None:
    assert risk_band(15) == "Safe"
    assert risk_band(45) == "Moderate Risk"
    assert risk_band(70) == "High Risk"
    assert risk_band(95) == "Critical Risk"
