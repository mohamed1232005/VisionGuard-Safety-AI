"""Human pose estimation (YOLO11-pose) for fall detection.

Thin wrapper that turns model output into plain :class:`PoseObservation`
objects, keeping torch/ultralytics out of the fall-detection logic itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from visionguard.detection.detector import resolve_device
from visionguard.detection.types import BoundingBox, PoseObservation
from visionguard.utils.config import FallSettings

logger = logging.getLogger(__name__)


class PoseEstimator:
    """Estimates COCO-17 keypoints for every person in a frame."""

    def __init__(self, settings: FallSettings, device: str = "auto") -> None:
        from ultralytics import YOLO  # local import: heavy

        model_path = Path(settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}. "
                "Run scripts/download_assets.py first."
            )
        self._device = resolve_device(device)
        self._model = YOLO(str(model_path))
        logger.info("Loaded pose model %s on %s", model_path.name, self._device)

    def estimate(self, frame: np.ndarray) -> list[PoseObservation]:
        """Return one :class:`PoseObservation` per detected person."""
        results = self._model.predict(frame, device=self._device, verbose=False)[0]

        observations: list[PoseObservation] = []
        if results.keypoints is None or results.boxes is None:
            return observations

        xyxy = results.boxes.xyxy.cpu().numpy()
        kp_xy = results.keypoints.xy.cpu().numpy()          # (n, 17, 2)
        kp_conf = (
            results.keypoints.conf.cpu().numpy()             # (n, 17)
            if results.keypoints.conf is not None
            else np.ones(kp_xy.shape[:2], dtype=np.float32)
        )

        for (x1, y1, x2, y2), xy, conf in zip(xyxy, kp_xy, kp_conf):
            observations.append(
                PoseObservation(
                    box=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                    keypoints_xy=xy,
                    keypoints_conf=conf,
                )
            )
        return observations
