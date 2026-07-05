"""Pose-based fall detection.

A fall is confirmed when a tracked worker's torso goes from upright to
horizontal AND stays down for a configurable number of seconds. The stay-down
confirmation is what separates a real fall from bending over, kneeling, or a
single-frame pose glitch.

Posture signal, per frame:
  1. Torso angle — the vector from mid-hip to mid-shoulder, measured against
     vertical. Beyond ``torso_angle_threshold`` degrees counts as "down".
  2. Fallback: if the keypoints are unreliable, a bounding box wider than tall
     (aspect ratio above threshold) counts as "down".

This module is pure logic (no model inference) so it is fully unit-testable
with synthetic keypoints.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from visionguard.detection.types import (
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    PoseObservation,
)
from visionguard.safety.events import EventType, SafetyEvent
from visionguard.tracking.tracker import TrackedObject
from visionguard.utils.config import FallSettings

logger = logging.getLogger(__name__)

_MATCH_IOU = 0.3  # min IoU to associate a pose observation with a worker track


def torso_angle_degrees(
    pose: PoseObservation, min_confidence: float
) -> float | None:
    """Torso tilt from vertical in degrees (0 = standing, 90 = horizontal).

    Uses the midpoint of the shoulders and the midpoint of the hips. Returns
    None when either end of the torso has no keypoint above ``min_confidence``.
    """
    xy, conf = pose.keypoints_xy, pose.keypoints_conf

    def midpoint(idx_a: int, idx_b: int) -> np.ndarray | None:
        points = [xy[i] for i in (idx_a, idx_b) if conf[i] >= min_confidence]
        if not points:
            return None
        return np.mean(points, axis=0)

    shoulders = midpoint(KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
    hips = midpoint(KP_LEFT_HIP, KP_RIGHT_HIP)
    if shoulders is None or hips is None:
        return None

    dx = float(shoulders[0] - hips[0])
    dy = float(shoulders[1] - hips[1])  # image y grows downward
    if dx == 0.0 and dy == 0.0:
        return None
    # Angle between the torso vector and the vertical axis.
    return math.degrees(math.atan2(abs(dx), abs(dy)))


@dataclass
class _FallState:
    """Fall state machine for one tracked worker."""

    down_since: float | None = None      # video time when they first went down
    fall_reported: bool = False
    last_alert_time: float = float("-inf")
    angle_sum: float = 0.0               # accumulated while down (for confidence)
    angle_frames: int = 0


class FallDetector:
    """Confirms falls from per-frame posture and raises emergency events."""

    def __init__(self, settings: FallSettings) -> None:
        self._settings = settings
        self._states: dict[int, _FallState] = {}
        self.falls_detected = 0

    def _is_down(
        self, pose: PoseObservation | None, worker: TrackedObject
    ) -> tuple[bool, float | None]:
        """Classify one frame's posture. Returns (down, torso_angle or None)."""
        if pose is not None:
            angle = torso_angle_degrees(pose, self._settings.keypoint_confidence)
            if angle is not None:
                return angle >= self._settings.torso_angle_threshold, angle
        # No reliable keypoints: fall back to box shape.
        wide = worker.box.aspect_ratio >= self._settings.aspect_ratio_threshold
        return wide, None

    @staticmethod
    def _match_pose(
        worker: TrackedObject, poses: list[PoseObservation]
    ) -> PoseObservation | None:
        """Best-overlapping pose observation for a worker (None if no overlap)."""
        best, best_iou = None, _MATCH_IOU
        for pose in poses:
            iou = worker.box.iou(pose.box)
            if iou > best_iou:
                best, best_iou = pose, iou
        return best

    def update(
        self,
        frame_index: int,
        video_time: float,
        workers: list[TrackedObject],
        poses: list[PoseObservation],
    ) -> list[SafetyEvent]:
        """Advance every worker's fall state machine by one frame."""
        events: list[SafetyEvent] = []

        for worker in workers:
            state = self._states.setdefault(worker.track_id, _FallState())
            pose = self._match_pose(worker, poses)
            down, angle = self._is_down(pose, worker)

            if not down:
                # Upright again — reset (a confirmed fall stays reported until
                # the worker gets up, then the machine re-arms).
                state.down_since = None
                state.fall_reported = False
                state.angle_sum = 0.0
                state.angle_frames = 0
                continue

            if state.down_since is None:
                state.down_since = video_time
            if angle is not None:
                state.angle_sum += angle
                state.angle_frames += 1

            down_duration = video_time - state.down_since
            if (
                not state.fall_reported
                and down_duration >= self._settings.confirm_seconds
                and video_time - state.last_alert_time
                >= self._settings.cooldown_seconds
            ):
                state.fall_reported = True
                state.last_alert_time = video_time
                self.falls_detected += 1

                # Confidence: how decisively horizontal the torso was while
                # down (90 degrees = flat on the ground = 1.0). Box-shape-only
                # confirmations get a conservative fixed confidence.
                if state.angle_frames > 0:
                    mean_angle = state.angle_sum / state.angle_frames
                    confidence = min(mean_angle / 90.0, 1.0)
                else:
                    confidence = 0.5

                events.append(
                    SafetyEvent(
                        event_type=EventType.FALL,
                        frame_index=frame_index,
                        video_time=video_time,
                        track_id=worker.track_id,
                        track_label=worker.label,
                        confidence=round(confidence, 3),
                        description=(
                            f"FALL DETECTED: {worker.label} down for "
                            f"{down_duration:.1f}s"
                        ),
                    )
                )
                logger.critical("Fall detected: %s", events[-1].description)

        return events
