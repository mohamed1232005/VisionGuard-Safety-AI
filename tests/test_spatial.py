"""Tests for ground-plane homography (synthetic calibrations)."""

import json
from pathlib import Path

import numpy as np
import pytest

from visionguard.spatial.homography import GroundPlane, load_ground_plane

# Identity-scale calibration: a 100x100 px square maps to a 10x10 m square,
# so 10 px = 1 m everywhere (an orthographic top-down camera).
TOP_DOWN = {
    "reference_size": [1000, 1000],
    "image_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
    "world_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
}


def write_calibration(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(data), "utf-8")
    return path


def test_top_down_distances_scale_linearly(tmp_path: Path) -> None:
    plane = load_ground_plane(write_calibration(tmp_path, TOP_DOWN))
    assert plane is not None
    # 30 px apart horizontally -> 3 m
    assert plane.distance_m((0, 0), (30, 0)) == pytest.approx(3.0, abs=1e-6)
    # 3-4-5 triangle in pixels -> 0.5 m
    assert plane.distance_m((0, 0), (3, 4)) == pytest.approx(0.5, abs=1e-6)


def test_perspective_camera_unwarps_correctly(tmp_path: Path) -> None:
    """A trapezoid in the image (perspective) maps back to a 10x10 m square."""
    perspective = {
        "reference_size": [1280, 720],
        # Far edge appears smaller/higher in the image than the near edge.
        "image_points": [[500, 300], [780, 300], [1180, 700], [100, 700]],
        "world_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
    }
    plane = load_ground_plane(write_calibration(tmp_path, perspective))
    # The near edge spans 1080 px but must still measure 10 m.
    assert plane.distance_m((100, 700), (1180, 700)) == pytest.approx(10.0, abs=1e-6)
    # The far edge spans only 280 px yet ALSO measures 10 m — the whole point.
    assert plane.distance_m((500, 300), (780, 300)) == pytest.approx(10.0, abs=1e-6)


def test_scaled_to_other_resolution(tmp_path: Path) -> None:
    """Processing at half resolution must not change measured meters."""
    plane = load_ground_plane(write_calibration(tmp_path, TOP_DOWN))
    half = plane.scaled_to(500, 500)
    # 15 px at half resolution = 30 px at reference = 3 m.
    assert half.distance_m((0, 0), (15, 0)) == pytest.approx(3.0, abs=1e-6)


def test_missing_calibration_returns_none(tmp_path: Path) -> None:
    assert load_ground_plane(tmp_path / "nope.json") is None


def test_too_few_points_rejected(tmp_path: Path) -> None:
    bad = dict(TOP_DOWN, image_points=[[0, 0], [1, 0], [1, 1]],
               world_points=[[0, 0], [1, 0], [1, 1]])
    with pytest.raises(ValueError, match=">= 4"):
        load_ground_plane(write_calibration(tmp_path, bad))


def test_bad_matrix_shape_rejected() -> None:
    with pytest.raises(ValueError, match="3x3"):
        GroundPlane(np.eye(2), (100, 100))
