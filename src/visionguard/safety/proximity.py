"""Worker-vehicle proximity risk (calibrated, real-world meters).

For every (worker, vehicle/machinery) pair the monitor measures ground-plane
distance in meters via the camera homography and classifies it:

    distance <= high_risk_distance_m    -> HIGH risk   (critical event)
    distance <= medium_risk_distance_m  -> MEDIUM risk (warning event)

Alerts are debounced per pair and per level with a cooldown, and each level
only fires on *entering* it — a vehicle idling next to a worker produces one
event, not one per frame. A high-risk encounter is also counted as a
"near miss" for the run statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from visionguard.detection.types import ObjectClass
from visionguard.safety.events import EventType, SafetyEvent, Severity
from visionguard.spatial.homography import GroundPlane
from visionguard.tracking.tracker import TrackedObject
from visionguard.utils.config import ProximitySettings

logger = logging.getLogger(__name__)

_VEHICLE_CLASSES = (ObjectClass.VEHICLE, ObjectClass.MACHINERY)


class RiskLevel(int, Enum):
    """Ordered proximity risk levels (higher = closer = worse)."""

    NONE = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class _PairState:
    """Alert state for one (worker, vehicle) pair."""

    level: RiskLevel = RiskLevel.NONE
    last_alert_time: dict[RiskLevel, float] = field(default_factory=dict)


@dataclass
class ProximityStats:
    """Aggregates for dashboards and reports."""

    near_misses: int = 0             # entries into HIGH risk
    medium_alerts: int = 0
    min_distance_m: float | None = None

    def observe_distance(self, distance: float) -> None:
        if self.min_distance_m is None or distance < self.min_distance_m:
            self.min_distance_m = round(distance, 2)


@dataclass(frozen=True)
class ProximityPair:
    """A worker-vehicle pair with its current distance (for drawing)."""

    worker: TrackedObject
    vehicle: TrackedObject
    distance_m: float
    level: RiskLevel


class ProximityMonitor:
    """Raises debounced proximity events for worker-vehicle encounters."""

    def __init__(self, settings: ProximitySettings, plane: GroundPlane) -> None:
        self._settings = settings
        self._plane = plane
        self._pairs: dict[tuple[int, int], _PairState] = {}
        self.stats = ProximityStats()

    def _classify(self, distance: float) -> RiskLevel:
        if distance <= self._settings.high_risk_distance_m:
            return RiskLevel.HIGH
        if distance <= self._settings.medium_risk_distance_m:
            return RiskLevel.MEDIUM
        return RiskLevel.NONE

    def update(
        self,
        frame_index: int,
        video_time: float,
        tracked: list[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> tuple[list[SafetyEvent], list[ProximityPair]]:
        """Measure all worker-vehicle pairs for one frame.

        Returns:
            (new events, currently-close pairs for overlay drawing).
        """
        plane = self._plane.scaled_to(frame_width, frame_height)
        workers = [t for t in tracked if t.object_class is ObjectClass.WORKER]
        vehicles = [t for t in tracked if t.object_class in _VEHICLE_CLASSES]

        events: list[SafetyEvent] = []
        close_pairs: list[ProximityPair] = []

        for worker in workers:
            for vehicle in vehicles:
                distance = plane.distance_m(
                    worker.box.ground_point, vehicle.box.ground_point
                )
                self.stats.observe_distance(distance)
                level = self._classify(distance)
                key = (worker.track_id, vehicle.track_id)
                state = self._pairs.setdefault(key, _PairState())
                previous = state.level
                state.level = level

                if level is not RiskLevel.NONE:
                    close_pairs.append(
                        ProximityPair(worker, vehicle, distance, level)
                    )

                # Alert only when the pair got *closer* than before, with a
                # per-level cooldown so hovering at a boundary stays quiet.
                if level.value <= previous.value or level is RiskLevel.NONE:
                    continue
                last = state.last_alert_time.get(level, float("-inf"))
                if video_time - last < self._settings.cooldown_seconds:
                    continue
                state.last_alert_time[level] = video_time

                if level is RiskLevel.HIGH:
                    self.stats.near_misses += 1
                    severity = Severity.CRITICAL
                    label = "High-risk proximity"
                else:
                    self.stats.medium_alerts += 1
                    severity = Severity.WARNING
                    label = "Proximity warning"

                events.append(
                    SafetyEvent(
                        event_type=EventType.PROXIMITY,
                        frame_index=frame_index,
                        video_time=video_time,
                        track_id=worker.track_id,
                        track_label=worker.label,
                        confidence=min(worker.confidence, vehicle.confidence),
                        severity=severity,
                        description=(
                            f"{label}: {vehicle.label} within "
                            f"{distance:.1f} m of {worker.label}"
                        ),
                    )
                )
                logger.warning("Proximity: %s", events[-1].description)

        return events, close_pairs
