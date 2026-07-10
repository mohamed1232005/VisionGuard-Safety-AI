"""YOLO-based multi-class detector for workers, PPE, and vehicles.

Wraps an Ultralytics YOLO model and returns plain :class:`Detection` objects in
VisionGuard's canonical taxonomy. Ultralytics/torch are imported lazily so that
modules which only need the *types* never pay the import cost.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from visionguard.detection.types import (
    CONSTRUCTION_MODEL_CLASS_MAP,
    BoundingBox,
    Detection,
    ObjectClass,
)
from visionguard.utils.config import DetectionSettings

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """Turn the config's device string into a concrete torch device.

    "auto" picks CUDA when available so the same config runs on any machine.
    """
    if requested != "auto":
        return requested
    import torch  # local import: keep module importable without torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Detector:
    """Detects workers, PPE evidence, and vehicles in a frame.

    Args:
        settings: Detection section of the app config.
        class_map: Raw model label -> canonical :class:`ObjectClass`. Labels not
            in the map are silently discarded (e.g. Mask, Utility Pole).
    """

    def __init__(
        self,
        settings: DetectionSettings,
        class_map: dict[str, ObjectClass] | None = None,
    ) -> None:
        from ultralytics import YOLO  # local import: heavy

        model_path = Path(settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Detection model not found: {model_path}. "
                "Run scripts/download_assets.py first."
            )

        self._settings = settings
        self._device = resolve_device(settings.device)
        self._model = YOLO(str(model_path))
        self._class_map = class_map or CONSTRUCTION_MODEL_CLASS_MAP

        # Map the model's numeric class ids -> canonical classes once, so the
        # per-frame hot path is a dict lookup instead of string matching.
        self._id_map: dict[int, ObjectClass] = {
            class_id: self._class_map[name]
            for class_id, name in self._model.names.items()
            if name in self._class_map
        }
        skipped = [n for n in self._model.names.values() if n not in self._class_map]
        logger.info(
            "Loaded detector %s on %s (%d classes mapped, ignored: %s)",
            model_path.name,
            self._device,
            len(self._id_map),
            skipped or "none",
        )

    @property
    def device(self) -> str:
        return self._device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame.

        Returns:
            Detections above the confidence threshold whose class is part of
            the VisionGuard taxonomy.
        """
        results = self._model.predict(
            frame,
            conf=self._settings.confidence_threshold,
            iou=self._settings.iou_threshold,
            device=self._device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids):
            object_class = self._id_map.get(class_id)
            if object_class is None:  # class we deliberately don't monitor
                continue
            detections.append(
                Detection(
                    box=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                    object_class=object_class,
                    confidence=float(conf),
                )
            )
        return detections
