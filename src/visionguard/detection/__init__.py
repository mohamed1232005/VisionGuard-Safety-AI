"""Object detection: workers, PPE (helmet, vest), and vehicles. (Phase 1, Feature 1)"""

from visionguard.detection.types import (
    PPE_CLASSES,
    TRACKABLE_CLASSES,
    BoundingBox,
    Detection,
    ObjectClass,
)

__all__ = [
    "BoundingBox",
    "Detection",
    "ObjectClass",
    "PPE_CLASSES",
    "TRACKABLE_CLASSES",
]
