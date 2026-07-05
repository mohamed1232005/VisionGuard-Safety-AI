"""Tests for fall detection (synthetic poses; no model inference)."""

from collections import deque
from pathlib import Path

import numpy as np

from visionguard.detection.types import (
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    BoundingBox,
    ObjectClass,
    PoseObservation,
)
from visionguard.safety.events import EventType, Severity
from visionguard.safety.falls import FallDetector, torso_angle_degrees
from visionguard.tracking.tracker import TrackedObject
from visionguard.utils.config import FallSettings

SETTINGS = FallSettings(
    model_path=Path("unused.pt"),
    torso_angle_threshold=55.0,
    aspect_ratio_threshold=1.1,
    confirm_seconds=1.0,
    cooldown_seconds=10.0,
    keypoint_confidence=0.3,
)


def make_pose(
    shoulders: tuple[float, float], hips: tuple[float, float], box: BoundingBox
) -> PoseObservation:
    """Pose with only shoulders/hips confident; all other keypoints are junk."""
    xy = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros(17, dtype=np.float32)
    for idx in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER):
        xy[idx] = shoulders
        conf[idx] = 0.9
    for idx in (KP_LEFT_HIP, KP_RIGHT_HIP):
        xy[idx] = hips
        conf[idx] = 0.9
    return PoseObservation(box=box, keypoints_xy=xy, keypoints_conf=conf)


def make_worker(box: BoundingBox) -> TrackedObject:
    return TrackedObject(
        track_id=7,
        object_class=ObjectClass.WORKER,
        box=box,
        confidence=0.9,
        trajectory=deque(maxlen=10),
    )


STANDING_BOX = BoundingBox(100, 100, 150, 250)
LYING_BOX = BoundingBox(100, 200, 250, 250)


def standing_pose() -> PoseObservation:
    return make_pose(shoulders=(125, 120), hips=(125, 200), box=STANDING_BOX)


def lying_pose() -> PoseObservation:
    return make_pose(shoulders=(230, 230), hips=(120, 225), box=LYING_BOX)


def test_torso_angle_vertical_vs_horizontal() -> None:
    assert torso_angle_degrees(standing_pose(), 0.3) < 10
    assert torso_angle_degrees(lying_pose(), 0.3) > 80


def test_torso_angle_none_without_confident_keypoints() -> None:
    pose = standing_pose()
    blind = PoseObservation(
        box=pose.box,
        keypoints_xy=pose.keypoints_xy,
        keypoints_conf=np.zeros(17, dtype=np.float32),
    )
    assert torso_angle_degrees(blind, 0.3) is None


def test_fall_confirmed_only_after_stay_down() -> None:
    detector = FallDetector(SETTINGS)
    worker = make_worker(LYING_BOX)

    early = detector.update(0, 0.0, [worker], [lying_pose()])   # goes down
    mid = detector.update(1, 0.5, [worker], [lying_pose()])     # 0.5s down
    late = detector.update(2, 1.2, [worker], [lying_pose()])    # 1.2s down

    assert early == [] and mid == []
    assert [e.event_type for e in late] == [EventType.FALL]
    assert late[0].severity is Severity.CRITICAL
    assert late[0].confidence > 0.8  # torso was ~horizontal the whole time


def test_brief_stumble_is_not_a_fall() -> None:
    detector = FallDetector(SETTINGS)

    detector.update(0, 0.0, [make_worker(LYING_BOX)], [lying_pose()])
    # Back upright before confirm_seconds elapses:
    detector.update(1, 0.5, [make_worker(STANDING_BOX)], [standing_pose()])
    events = detector.update(2, 1.5, [make_worker(STANDING_BOX)], [standing_pose()])

    assert events == []
    assert detector.falls_detected == 0


def test_standing_worker_never_falls() -> None:
    detector = FallDetector(SETTINGS)
    events = []
    for frame in range(30):
        events += detector.update(
            frame, frame / 10, [make_worker(STANDING_BOX)], [standing_pose()]
        )
    assert events == []


def test_no_duplicate_alert_while_still_down() -> None:
    detector = FallDetector(SETTINGS)
    worker = make_worker(LYING_BOX)

    events = []
    for frame in range(40):  # 4 seconds down
        events += detector.update(frame, frame / 10, [worker], [lying_pose()])

    assert len(events) == 1


def test_box_shape_fallback_without_pose() -> None:
    """No usable keypoints: a wide box sustained on the ground still confirms."""
    detector = FallDetector(SETTINGS)
    worker = make_worker(LYING_BOX)  # aspect ratio 3.0

    events = []
    for frame in range(15):
        events += detector.update(frame, frame / 10, [worker], [])

    assert [e.event_type for e in events] == [EventType.FALL]
    assert events[0].confidence == 0.5  # conservative without pose data
