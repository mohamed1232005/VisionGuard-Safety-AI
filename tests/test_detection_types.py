"""Tests for the core geometry/taxonomy types."""

from visionguard.detection.types import (
    CONSTRUCTION_MODEL_CLASS_MAP,
    PPE_CLASSES,
    TRACKABLE_CLASSES,
    BoundingBox,
    ObjectClass,
)


class TestBoundingBox:
    def test_geometry_properties(self) -> None:
        box = BoundingBox(10, 20, 50, 100)
        assert box.width == 40
        assert box.height == 80
        assert box.area == 3200
        assert box.center == (30, 60)
        assert box.ground_point == (30, 100)  # feet = bottom-center

    def test_aspect_ratio_flags_lying_person(self) -> None:
        standing = BoundingBox(0, 0, 40, 120)
        lying = BoundingBox(0, 0, 120, 40)
        assert standing.aspect_ratio < 1 < lying.aspect_ratio

    def test_iou_identical_and_disjoint(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        assert a.iou(a) == 1.0
        assert a.iou(BoundingBox(20, 20, 30, 30)) == 0.0

    def test_iou_partial_overlap(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 0, 15, 10)  # half overlap
        assert abs(a.iou(b) - (50 / 150)) < 1e-9

    def test_contains_point_with_margin(self) -> None:
        box = BoundingBox(10, 10, 20, 20)
        assert box.contains_point(15, 15)
        assert not box.contains_point(22, 15)
        assert box.contains_point(22, 15, margin=3)


class TestTaxonomy:
    def test_model_map_covers_required_classes(self) -> None:
        """The PPE model must supply workers, PPE evidence, and vehicles."""
        mapped = set(CONSTRUCTION_MODEL_CLASS_MAP.values())
        assert ObjectClass.WORKER in mapped
        assert PPE_CLASSES <= mapped
        assert ObjectClass.VEHICLE in mapped

    def test_ppe_classes_are_not_trackable(self) -> None:
        """PPE items are worker attributes, never independent tracks."""
        assert not (PPE_CLASSES & TRACKABLE_CLASSES)
