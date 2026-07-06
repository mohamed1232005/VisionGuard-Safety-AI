"""Ground-plane homography: pixel coordinates -> real-world meters.

Why this matters: pixel distance is meaningless for safety. Two objects 50 px
apart can be 1 m or 15 m from each other depending on where they stand relative
to the camera. A homography H maps points on the *image's ground plane* to a
metric top-down coordinate system, so distances between workers and vehicles
come out in actual meters.

Calibration (once per fixed camera) is done with
``scripts/calibrate_camera.py``: click at least 4 ground points whose
real-world positions are known (e.g. corners of a slab whose dimensions you
know) and their pixel/world pairs are stored in a JSON file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GroundPlane:
    """Metric view of the ground as seen by one camera.

    Args:
        homography: 3x3 matrix mapping image pixels (at ``reference_size``)
            to world coordinates in meters.
        reference_size: (width, height) of the frames the calibration points
            were clicked on. Use :meth:`scaled_to` when processing at another
            resolution.
    """

    def __init__(self, homography: np.ndarray, reference_size: tuple[int, int]) -> None:
        self._h = np.asarray(homography, dtype=np.float64)
        if self._h.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3, got {self._h.shape}")
        self._reference_size = reference_size

    @property
    def reference_size(self) -> tuple[int, int]:
        return self._reference_size

    def scaled_to(self, width: int, height: int) -> "GroundPlane":
        """Adapt the calibration to a different processing resolution.

        A point at (x, y) in the new resolution corresponds to
        (x * ref_w / width, y * ref_h / height) in the calibrated resolution,
        expressed as a scale matrix composed with the original homography.
        """
        ref_w, ref_h = self._reference_size
        if (width, height) == (ref_w, ref_h):
            return self
        scale = np.array(
            [[ref_w / width, 0.0, 0.0], [0.0, ref_h / height, 0.0], [0.0, 0.0, 1.0]]
        )
        return GroundPlane(self._h @ scale, (width, height))

    def image_to_world(self, point: tuple[float, float]) -> tuple[float, float]:
        """Project one image point (on the ground) to world meters."""
        src = np.array([[[point[0], point[1]]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self._h)
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])

    def distance_m(
        self, point_a: tuple[float, float], point_b: tuple[float, float]
    ) -> float:
        """Real-world distance in meters between two image ground points."""
        ax, ay = self.image_to_world(point_a)
        bx, by = self.image_to_world(point_b)
        return float(np.hypot(ax - bx, ay - by))


def load_ground_plane(calibration_file: Path | str) -> GroundPlane | None:
    """Load a calibration JSON; None (with a log line) when not calibrated.

    Expected JSON:
        {
          "reference_size": [1280, 720],
          "image_points": [[x, y], ...],   # >= 4 pixel points on the ground
          "world_points": [[X, Y], ...]    # matching positions in meters
        }
    """
    path = Path(calibration_file)
    if not path.exists():
        logger.info(
            "No camera calibration at %s — proximity monitoring idle "
            "(run scripts/calibrate_camera.py to enable)",
            path,
        )
        return None

    data = json.loads(path.read_text("utf-8"))
    image_points = np.array(data["image_points"], dtype=np.float64)
    world_points = np.array(data["world_points"], dtype=np.float64)
    if len(image_points) < 4 or len(image_points) != len(world_points):
        raise ValueError(
            f"Calibration needs >= 4 matched point pairs, got "
            f"{len(image_points)} image / {len(world_points)} world"
        )

    homography, _ = cv2.findHomography(image_points, world_points)
    if homography is None:
        raise ValueError("Could not compute homography from calibration points")

    plane = GroundPlane(homography, tuple(data["reference_size"]))
    logger.info("Loaded ground-plane calibration from %s", path)
    return plane
