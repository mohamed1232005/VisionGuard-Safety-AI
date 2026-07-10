"""Core detection types shared across the whole pipeline.

This module has NO heavy dependencies (no torch/ultralytics), so safety rules,
tests, and the dashboard can import these types without loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ObjectClass(str, Enum):
    """Canonical VisionGuard object taxonomy.

    Detector implementations translate raw model class names into these values,
    so downstream logic (tracking, safety rules, reports) never depends on any
    specific model's label spelling.
    """

    WORKER = "worker"
    HELMET = "helmet"
    NO_HELMET = "no_helmet"
    VEST = "vest"
    NO_VEST = "no_vest"
    MACHINERY = "machinery"
    VEHICLE = "vehicle"
    SAFETY_CONE = "safety_cone"


# Classes that get persistent track IDs (PPE items are attributes of a worker,
# not independently tracked entities).
TRACKABLE_CLASSES = frozenset(
    {ObjectClass.WORKER, ObjectClass.MACHINERY, ObjectClass.VEHICLE}
)

# PPE evidence classes used by the compliance engine.
PPE_CLASSES = frozenset(
    {ObjectClass.HELMET, ObjectClass.NO_HELMET, ObjectClass.VEST, ObjectClass.NO_VEST}
)

# Mapping from the Construction-Hazard-Detection model's labels to our taxonomy.
# Labels absent from this dict (Mask, Utility Pole, ...) are dropped on purpose.
# NOTE: the repo documents capitalized names ("Machinery") but the shipped
# weights embed some labels lowercase ("machinery"), so both spellings are
# mapped — verified against the actual model.names at load time.
CONSTRUCTION_MODEL_CLASS_MAP: dict[str, ObjectClass] = {
    "Person": ObjectClass.WORKER,
    "Hardhat": ObjectClass.HELMET,
    "NO-Hardhat": ObjectClass.NO_HELMET,
    "Safety Vest": ObjectClass.VEST,
    "NO-Safety Vest": ObjectClass.NO_VEST,
    "Machinery": ObjectClass.MACHINERY,
    "machinery": ObjectClass.MACHINERY,
    "Vehicle": ObjectClass.VEHICLE,
    "vehicle": ObjectClass.VEHICLE,
    "Safety Cone": ObjectClass.SAFETY_CONE,
    "safety cone": ObjectClass.SAFETY_CONE,
}


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates (x1, y1 = top-left corner)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def ground_point(self) -> tuple[float, float]:
        """Bottom-center of the box — where the object touches the ground.

        Used for zone tests and heatmaps: a worker is "inside" a floor zone when
        their feet are, not when their head overlaps it in 2D.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def aspect_ratio(self) -> float:
        """Width / height. > 1 means wider than tall (e.g. a lying person)."""
        return self.width / self.height if self.height > 0 else 0.0

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Whether (x, y) lies inside the box, optionally expanded by ``margin`` px."""
        return (
            self.x1 - margin <= x <= self.x2 + margin
            and self.y1 - margin <= y <= self.y2 + margin
        )

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-union with another box (0 = disjoint, 1 = identical)."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame."""

    box: BoundingBox
    object_class: ObjectClass
    confidence: float


# COCO-17 keypoint indices used by the fall detector.
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12


@dataclass(frozen=True)
class PoseObservation:
    """A person's pose in one frame (COCO-17 keypoint layout).

    Attributes:
        box: Person bounding box from the pose model.
        keypoints_xy: (17, 2) array of pixel coordinates.
        keypoints_conf: (17,) per-keypoint confidence in [0, 1].
    """

    box: BoundingBox
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
