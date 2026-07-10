"""Restricted-zone intrusion detection.

Zones are polygons drawn on the camera view (``scripts/define_zones.py``) and
stored in JSON with *normalized* coordinates (0-1 of frame size), so the same
zone file works regardless of processing resolution.

An object is inside a zone when its ground point (bottom-center of its box —
i.e. its feet/wheels) is inside the polygon. Entry raises an intrusion event;
staying longer than the configured dwell time raises a second, escalated event.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from visionguard.detection.types import ObjectClass
from visionguard.safety.events import EventType, SafetyEvent
from visionguard.tracking.tracker import TrackedObject
from visionguard.utils.config import ZoneSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Zone:
    """A user-defined danger area on the camera view.

    Attributes:
        name: Display name, e.g. "Crane swing radius".
        zone_type: Category label (restricted / vehicle_only / high_risk ...).
        risk_level: 1 (low) - 3 (high); scales the zone's weight in analytics.
        polygon: Normalized (x, y) vertices in [0, 1].
        applies_to: Object classes that are NOT allowed inside.
    """

    name: str
    zone_type: str
    risk_level: int
    polygon: tuple[tuple[float, float], ...]
    applies_to: tuple[ObjectClass, ...] = (ObjectClass.WORKER,)

    def pixel_polygon(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Polygon scaled to pixel coordinates for a given frame size."""
        return np.array(
            [(x * frame_width, y * frame_height) for x, y in self.polygon],
            dtype=np.float32,
        )

    def contains(
        self, point: tuple[float, float], frame_width: int, frame_height: int
    ) -> bool:
        """Whether a pixel-space point lies inside the zone."""
        polygon = self.pixel_polygon(frame_width, frame_height)
        result = cv2.pointPolygonTest(polygon, point, measureDist=False)
        return result >= 0


def load_zones(definitions_file: Path) -> list[Zone]:
    """Load zone definitions from JSON; a missing file simply means no zones."""
    if not Path(definitions_file).exists():
        logger.info("No zone definitions at %s — zone monitoring idle", definitions_file)
        return []

    with Path(definitions_file).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    zones = [
        Zone(
            name=z["name"],
            zone_type=z.get("zone_type", "restricted"),
            risk_level=int(z.get("risk_level", 2)),
            polygon=tuple((float(x), float(y)) for x, y in z["polygon"]),
            applies_to=tuple(
                ObjectClass(c) for c in z.get("applies_to", ["worker"])
            ),
        )
        for z in raw.get("zones", [])
    ]
    logger.info("Loaded %d zone(s) from %s", len(zones), definitions_file)
    return zones


@dataclass
class _Presence:
    """State for one object currently inside one zone."""

    entered_at: float
    dwell_alerted: bool = False


@dataclass
class ZoneStats:
    """Per-zone counters for dashboards and reports."""

    intrusions: dict[str, int] = field(default_factory=dict)
    dwell_alerts: dict[str, int] = field(default_factory=dict)

    def most_dangerous_zone(self) -> str | None:
        """Zone with the most combined violations (None if no violations)."""
        totals: dict[str, int] = dict(self.intrusions)
        for name, count in self.dwell_alerts.items():
            totals[name] = totals.get(name, 0) + count
        if not totals:
            return None
        return max(totals, key=lambda name: totals[name])


class ZoneMonitor:
    """Tracks zone entry/exit/dwell for all tracked objects."""

    def __init__(self, zones: list[Zone], settings: ZoneSettings) -> None:
        self._zones = zones
        self._settings = settings
        self._presence: dict[tuple[str, int], _Presence] = {}
        # Survives exit: quick re-entries (boundary flicker, track blips)
        # within the cooldown don't re-alert.
        self._last_alert: dict[tuple[str, int], float] = {}
        self.stats = ZoneStats()

    @property
    def zones(self) -> list[Zone]:
        return self._zones

    def update(
        self,
        frame_index: int,
        video_time: float,
        tracked: list[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> list[SafetyEvent]:
        """Check every tracked object against every zone for one frame."""
        events: list[SafetyEvent] = []
        seen_inside: set[tuple[str, int]] = set()

        for zone in self._zones:
            for obj in tracked:
                if obj.object_class not in zone.applies_to:
                    continue
                ground = obj.box.ground_point
                if not zone.contains(ground, frame_width, frame_height):
                    continue

                key = (zone.name, obj.track_id)
                seen_inside.add(key)
                presence = self._presence.get(key)

                if presence is None:  # new entry
                    self._presence[key] = _Presence(entered_at=video_time)
                    last = self._last_alert.get(key, float("-inf"))
                    if video_time - last < self._settings.reentry_cooldown_seconds:
                        continue  # boundary flicker — already alerted recently
                    self._last_alert[key] = video_time
                    self.stats.intrusions[zone.name] = (
                        self.stats.intrusions.get(zone.name, 0) + 1
                    )
                    events.append(
                        SafetyEvent(
                            event_type=EventType.ZONE_INTRUSION,
                            frame_index=frame_index,
                            video_time=video_time,
                            track_id=obj.track_id,
                            track_label=obj.label,
                            zone_name=zone.name,
                            confidence=obj.confidence,
                            description=(
                                f"{obj.label} entered restricted zone "
                                f"'{zone.name}'"
                            ),
                        )
                    )
                    logger.warning("Zone intrusion: %s", events[-1].description)
                elif (
                    not presence.dwell_alerted
                    and video_time - presence.entered_at
                    >= self._settings.dwell_alert_seconds
                ):
                    presence.dwell_alerted = True
                    self.stats.dwell_alerts[zone.name] = (
                        self.stats.dwell_alerts.get(zone.name, 0) + 1
                    )
                    dwell = video_time - presence.entered_at
                    events.append(
                        SafetyEvent(
                            event_type=EventType.ZONE_DWELL,
                            frame_index=frame_index,
                            video_time=video_time,
                            track_id=obj.track_id,
                            track_label=obj.label,
                            zone_name=zone.name,
                            confidence=obj.confidence,
                            description=(
                                f"{obj.label} loitering in zone '{zone.name}' "
                                f"for {dwell:.0f}s"
                            ),
                        )
                    )
                    logger.warning("Zone dwell: %s", events[-1].description)

        # Objects no longer inside a zone lose their presence state, so a
        # re-entry later counts as a fresh intrusion.
        for key in list(self._presence):
            if key not in seen_inside:
                del self._presence[key]

        return events
