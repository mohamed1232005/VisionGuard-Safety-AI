"""Multi-object tracking with ByteTrack.

Assigns persistent IDs to *trackable* detections (workers, vehicles, machinery)
and maintains a short trajectory of ground points per track. PPE detections are
not tracked — they are per-frame evidence that the compliance engine associates
with tracked workers.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from visionguard.detection.types import (
    TRACKABLE_CLASSES,
    BoundingBox,
    Detection,
    ObjectClass,
)
from visionguard.utils.config import TrackingSettings

logger = logging.getLogger(__name__)

# Stable numeric ids for supervision's class_id array (order must not change).
_CLASS_TO_ID = {cls: i for i, cls in enumerate(ObjectClass)}
_ID_TO_CLASS = {i: cls for cls, i in _CLASS_TO_ID.items()}


@dataclass
class TrackedObject:
    """A detection enriched with a persistent identity and its recent path."""

    track_id: int
    object_class: ObjectClass
    box: BoundingBox
    confidence: float
    trajectory: deque[tuple[float, float]] = field(default_factory=deque)
    frames_seen: int = 1

    @property
    def label(self) -> str:
        """Human-readable label like 'Worker #12' used in overlays and alerts."""
        name = self.object_class.value.replace("_", " ").title()
        return f"{name} #{self.track_id}"


class Tracker:
    """ByteTrack wrapper producing :class:`TrackedObject` instances.

    Args:
        settings: Tracking section of the app config.
        frame_rate: Video FPS — ByteTrack uses it to scale its motion model.
    """

    def __init__(self, settings: TrackingSettings, frame_rate: float = 30.0) -> None:
        self._settings = settings
        self._tracker = sv.ByteTrack(
            track_activation_threshold=settings.track_activation_threshold,
            lost_track_buffer=settings.lost_track_buffer,
            minimum_matching_threshold=settings.minimum_matching_threshold,
            frame_rate=int(round(frame_rate)),
        )
        # Track state that must survive between frames (trajectories, counters).
        self._tracks: dict[int, TrackedObject] = {}

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Advance the tracker by one frame.

        Args:
            detections: All detections for the frame; non-trackable classes are
                filtered out here so callers can pass the full list.

        Returns:
            Currently visible tracked objects with up-to-date boxes and paths.
        """
        trackable = [d for d in detections if d.object_class in TRACKABLE_CLASSES]

        sv_detections = sv.Detections(
            xyxy=np.array(
                [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in trackable],
                dtype=np.float32,
            ).reshape(-1, 4),
            confidence=np.array([d.confidence for d in trackable], dtype=np.float32),
            class_id=np.array(
                [_CLASS_TO_ID[d.object_class] for d in trackable], dtype=int
            ),
        )

        tracked = self._tracker.update_with_detections(sv_detections)

        visible: list[TrackedObject] = []
        for (x1, y1, x2, y2), conf, class_id, track_id in zip(
            tracked.xyxy,
            tracked.confidence,
            tracked.class_id,
            tracked.tracker_id,
        ):
            box = BoundingBox(float(x1), float(y1), float(x2), float(y2))
            obj = self._tracks.get(int(track_id))
            if obj is None:
                obj = TrackedObject(
                    track_id=int(track_id),
                    object_class=_ID_TO_CLASS[int(class_id)],
                    box=box,
                    confidence=float(conf),
                    trajectory=deque(maxlen=self._settings.trajectory_length),
                )
                self._tracks[int(track_id)] = obj
            else:
                obj.box = box
                obj.confidence = float(conf)
                obj.frames_seen += 1
            obj.trajectory.append(box.ground_point)
            visible.append(obj)

        return visible

    @property
    def total_tracks(self) -> int:
        """Number of unique identities seen so far (for run statistics)."""
        return len(self._tracks)

    def unique_counts(self) -> dict[ObjectClass, int]:
        """Unique track count per class over the whole run."""
        counts: dict[ObjectClass, int] = {}
        for obj in self._tracks.values():
            counts[obj.object_class] = counts.get(obj.object_class, 0) + 1
        return counts
