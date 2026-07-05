"""Safety event types — the common language of all rule engines.

Every safety module (PPE, zones, falls) emits :class:`SafetyEvent` objects; the
pipeline enriches them with screenshots and persists them to the event store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class EventType(str, Enum):
    """Kinds of safety events VisionGuard can raise."""

    PPE_VIOLATION = "ppe_violation"
    ZONE_INTRUSION = "zone_intrusion"
    ZONE_DWELL = "zone_dwell"
    FALL = "fall"


class Severity(str, Enum):
    """How urgently a human should react to the event."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Default severity per event type (falls are always emergencies).
DEFAULT_SEVERITY: dict[EventType, Severity] = {
    EventType.PPE_VIOLATION: Severity.WARNING,
    EventType.ZONE_INTRUSION: Severity.WARNING,
    EventType.ZONE_DWELL: Severity.WARNING,
    EventType.FALL: Severity.CRITICAL,
}

# Recommended action shown in incident reports, per event type.
RECOMMENDED_ACTIONS: dict[EventType, str] = {
    EventType.PPE_VIOLATION: (
        "Notify the site supervisor to stop the worker and enforce PPE policy "
        "before work continues. Log the violation against the worker's record."
    ),
    EventType.ZONE_INTRUSION: (
        "Dispatch a supervisor to remove the worker from the restricted area and "
        "verify physical barriers and signage are in place."
    ),
    EventType.ZONE_DWELL: (
        "Investigate why the worker remained in a hazardous area; review whether "
        "the task requires a permit-to-work or additional controls."
    ),
    EventType.FALL: (
        "EMERGENCY: send first-aid responders to the location immediately and "
        "verify the worker's condition. Preserve the scene for investigation."
    ),
}


@dataclass
class SafetyEvent:
    """A single safety incident detected in the video stream.

    Attributes:
        event_type: What kind of incident occurred.
        severity: Urgency level (defaults per event type).
        frame_index: Frame number in the source video where it was confirmed.
        video_time: Seconds from the start of the video.
        wall_time: Real-world UTC timestamp when the event was raised.
        track_id: Persistent ID of the involved worker/vehicle (if any).
        track_label: Human-readable label, e.g. "Worker #12".
        zone_name: Involved zone, for zone events.
        confidence: Detector/heuristic confidence in [0, 1].
        description: One-line human-readable summary used in alerts and reports.
        screenshot_path: Evidence image saved by the pipeline (filled in later).
    """

    event_type: EventType
    frame_index: int
    video_time: float
    track_id: int | None
    track_label: str | None
    description: str
    confidence: float = 1.0
    zone_name: str | None = None
    severity: Severity = field(default=Severity.WARNING)
    wall_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    screenshot_path: str | None = None

    def __post_init__(self) -> None:
        # Apply the canonical severity unless a caller overrode it explicitly.
        self.severity = DEFAULT_SEVERITY.get(self.event_type, self.severity)

    @property
    def recommended_action(self) -> str:
        return RECOMMENDED_ACTIONS.get(self.event_type, "Review the incident.")

    def timestamp_str(self) -> str:
        """Video timestamp formatted as MM:SS.d for logs and overlays."""
        minutes, seconds = divmod(self.video_time, 60)
        return f"{int(minutes):02d}:{seconds:04.1f}"
