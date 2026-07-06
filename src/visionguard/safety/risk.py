"""Site Safety Risk Score (0-100).

Synthesizes all safety events into one interpretable number. Each event adds
its configured weight, which decays linearly to zero over the rolling window —
so the score jumps when incidents happen and cools down when the site is calm.
Critical events (falls, near misses) carry the heaviest weights.

Bands:  0-30 Safe · 31-60 Moderate · 61-80 High · 81-100 Critical.
"""

from __future__ import annotations

from dataclasses import dataclass

from visionguard.safety.events import EventType, SafetyEvent, Severity
from visionguard.utils.config import RiskScoreSettings

RISK_BANDS: tuple[tuple[float, str], ...] = (
    (30.0, "Safe"),
    (60.0, "Moderate Risk"),
    (80.0, "High Risk"),
    (100.0, "Critical Risk"),
)


def risk_band(score: float) -> str:
    """Human-readable band for a score."""
    for upper, label in RISK_BANDS:
        if score <= upper:
            return label
    return RISK_BANDS[-1][1]


@dataclass(frozen=True)
class _WeightedEvent:
    time: float
    weight: float


class RiskScoreCalculator:
    """Rolling, time-decayed 0-100 risk score over the event stream."""

    def __init__(self, settings: RiskScoreSettings) -> None:
        self._settings = settings
        self._events: list[_WeightedEvent] = []
        self.peak_score = 0.0
        self.peak_time = 0.0

    def _weight_for(self, event: SafetyEvent) -> float:
        weights = self._settings.weights
        if (
            event.event_type is EventType.PROXIMITY
            and event.severity is Severity.CRITICAL
        ):
            return weights.get("proximity_high", weights.get("proximity", 10.0))
        return weights.get(event.event_type.value, 10.0)

    def add_events(self, events: list[SafetyEvent]) -> None:
        """Feed the new events of the current frame."""
        for event in events:
            self._events.append(
                _WeightedEvent(time=event.video_time, weight=self._weight_for(event))
            )

    def score(self, video_time: float) -> float:
        """Current 0-100 score at ``video_time``.

        Each event contributes weight * (1 - age / window); expired events are
        dropped so long runs stay O(recent events).
        """
        window = self._settings.window_seconds
        self._events = [e for e in self._events if video_time - e.time < window]

        total = sum(
            e.weight * (1.0 - (video_time - e.time) / window) for e in self._events
        )
        score = min(round(total, 1), 100.0)
        if score > self.peak_score:
            self.peak_score = score
            self.peak_time = video_time
        return score
