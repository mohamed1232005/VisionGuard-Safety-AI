"""PPE compliance engine.

Associates per-frame PPE evidence (helmet / no-helmet / vest / no-vest boxes)
with tracked workers, smooths it over a rolling time window to kill detector
flicker, and raises debounced violation events.

Association rule: a PPE box belongs to the worker whose box contains its
center — helmets must sit in the upper part of the worker box, vests in the
torso band. When several workers overlap, the tightest (smallest) box wins.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from visionguard.detection.types import BoundingBox, Detection, ObjectClass
from visionguard.safety.events import EventType, SafetyEvent
from visionguard.tracking.tracker import TrackedObject
from visionguard.utils.config import PPESettings

logger = logging.getLogger(__name__)

# Evidence classes per equipment item: (worn, missing)
_EVIDENCE: dict[str, tuple[ObjectClass, ObjectClass]] = {
    "helmet": (ObjectClass.HELMET, ObjectClass.NO_HELMET),
    "vest": (ObjectClass.VEST, ObjectClass.NO_VEST),
}

# Vertical band of the worker box where each item's evidence must appear
# (as fractions of box height from the top). Helmets belong near the head;
# vests on the torso. Head gear may poke slightly above the box, hence -0.15.
_VERTICAL_BAND: dict[str, tuple[float, float]] = {
    "helmet": (-0.15, 0.45),
    "vest": (0.10, 0.85),
}


@dataclass
class _ItemState:
    """Rolling compliance state for one (worker, equipment item) pair."""

    observations: deque[bool]  # True = worn, False = missing (evidence frames only)
    violation_active: bool = False
    last_alert_time: float = float("-inf")


@dataclass
class ComplianceStats:
    """Aggregated compliance numbers for dashboards and reports."""

    judged_frames: int = 0     # (worker, item, frame) triples with enough data
    compliant_frames: int = 0
    violations_by_item: dict[str, int] = field(default_factory=dict)

    @property
    def compliance_rate(self) -> float:
        """Fraction of judged worker-frames that were compliant (1.0 = perfect)."""
        if self.judged_frames == 0:
            return 1.0
        return self.compliant_frames / self.judged_frames


class PPEComplianceEngine:
    """Raises debounced PPE violation events for tracked workers.

    Args:
        settings: PPE section of the app config.
        fps: Effective processing frame rate — converts the config's
            ``window_seconds`` into a frame-count window.
    """

    def __init__(self, settings: PPESettings, fps: float) -> None:
        unknown = set(settings.required_equipment) - set(_EVIDENCE)
        if unknown:
            raise ValueError(f"Unknown PPE items in config: {sorted(unknown)}")
        self._settings = settings
        self._window_frames = max(int(settings.window_seconds * fps), 1)
        self._states: dict[tuple[int, str], _ItemState] = {}
        self.stats = ComplianceStats()

    # ------------------------------------------------------------------ #
    # Association
    # ------------------------------------------------------------------ #
    @staticmethod
    def _owner_of(
        ppe_box: BoundingBox, item: str, workers: list[TrackedObject]
    ) -> TrackedObject | None:
        """Find the worker a PPE evidence box belongs to (None if nobody)."""
        cx, cy = ppe_box.center
        top_frac, bottom_frac = _VERTICAL_BAND[item]
        candidates = []
        for worker in workers:
            wb = worker.box
            band_top = wb.y1 + top_frac * wb.height
            band_bottom = wb.y1 + bottom_frac * wb.height
            if wb.x1 <= cx <= wb.x2 and band_top <= cy <= band_bottom:
                candidates.append(worker)
        if not candidates:
            return None
        return min(candidates, key=lambda w: w.box.area)  # tightest box wins

    # ------------------------------------------------------------------ #
    # Per-frame update
    # ------------------------------------------------------------------ #
    def update(
        self,
        frame_index: int,
        video_time: float,
        workers: list[TrackedObject],
        detections: list[Detection],
    ) -> list[SafetyEvent]:
        """Ingest one frame of evidence and return any new violation events."""
        events: list[SafetyEvent] = []

        for item in self._settings.required_equipment:
            worn_class, missing_class = _EVIDENCE[item]

            # Collect evidence per worker for this frame.
            evidence: dict[int, bool] = {}
            for det in detections:
                if det.object_class not in (worn_class, missing_class):
                    continue
                owner = self._owner_of(det.box, item, workers)
                if owner is None:
                    continue
                worn = det.object_class is worn_class
                # A "missing" observation overrides a "worn" one in the same
                # frame (conflicts usually mean the worn box matched wrongly).
                evidence[owner.track_id] = evidence.get(owner.track_id, True) and worn

            for worker in workers:
                if worker.track_id not in evidence:
                    continue  # no evidence this frame (occlusion) — don't guess
                state = self._states.setdefault(
                    (worker.track_id, item),
                    _ItemState(observations=deque(maxlen=self._window_frames)),
                )
                state.observations.append(evidence[worker.track_id])

                if len(state.observations) < self._settings.min_observations:
                    continue  # not enough history to judge yet

                missing_ratio = 1.0 - (
                    sum(state.observations) / len(state.observations)
                )
                violating = missing_ratio >= self._settings.violation_ratio

                # Aggregate stats for the compliance-rate KPI.
                self.stats.judged_frames += 1
                if not violating:
                    self.stats.compliant_frames += 1

                if violating and not state.violation_active:
                    state.violation_active = True
                    if (
                        video_time - state.last_alert_time
                        >= self._settings.cooldown_seconds
                    ):
                        state.last_alert_time = video_time
                        self.stats.violations_by_item[item] = (
                            self.stats.violations_by_item.get(item, 0) + 1
                        )
                        events.append(
                            SafetyEvent(
                                event_type=EventType.PPE_VIOLATION,
                                frame_index=frame_index,
                                video_time=video_time,
                                track_id=worker.track_id,
                                track_label=worker.label,
                                confidence=round(missing_ratio, 3),
                                description=(
                                    f"{worker.label} detected without {item}"
                                ),
                            )
                        )
                        logger.warning("PPE violation: %s", events[-1].description)
                elif not violating and missing_ratio < self._settings.violation_ratio / 2:
                    # Hysteresis: require clearly-compliant before re-arming, so a
                    # worker hovering at the threshold doesn't re-alert endlessly.
                    state.violation_active = False

        return events
