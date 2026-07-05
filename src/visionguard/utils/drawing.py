"""Frame annotation: boxes, labels, trajectories, zones, alert banner, HUD.

All drawing is plain OpenCV so the overlay style is fully ours (and one less
dependency in the hot path). Colors are BGR.
"""

from __future__ import annotations

import cv2
import numpy as np

from visionguard.detection.types import Detection, ObjectClass
from visionguard.safety.events import SafetyEvent, Severity
from visionguard.safety.zones import Zone
from visionguard.tracking.tracker import TrackedObject

CLASS_COLORS: dict[ObjectClass, tuple[int, int, int]] = {
    ObjectClass.WORKER: (80, 200, 120),      # green
    ObjectClass.HELMET: (255, 180, 40),      # light blue
    ObjectClass.NO_HELMET: (0, 0, 230),      # red
    ObjectClass.VEST: (200, 160, 0),         # teal-ish
    ObjectClass.NO_VEST: (0, 80, 255),       # orange-red
    ObjectClass.MACHINERY: (0, 200, 255),    # yellow
    ObjectClass.VEHICLE: (255, 0, 200),      # magenta
    ObjectClass.SAFETY_CONE: (100, 100, 255),
}

_SEVERITY_COLORS: dict[Severity, tuple[int, int, int]] = {
    Severity.INFO: (200, 200, 200),
    Severity.WARNING: (0, 165, 255),
    Severity.CRITICAL: (0, 0, 255),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
) -> None:
    """Text with a filled background so it stays readable over any scene."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = origin
    cv2.rectangle(frame, (x, y - th - baseline), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - baseline // 2), _FONT, scale, (0, 0, 0), 1)


def draw_tracked_object(frame: np.ndarray, obj: TrackedObject) -> None:
    """Box + persistent ID label + recent trajectory trail."""
    color = CLASS_COLORS.get(obj.object_class, (255, 255, 255))
    b = obj.box
    cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
    _label(frame, obj.label, (int(b.x1), int(b.y1)), color)

    if len(obj.trajectory) > 1:
        points = np.array(obj.trajectory, dtype=np.int32)
        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)


def draw_ppe_evidence(frame: np.ndarray, detection: Detection) -> None:
    """Thin box for PPE evidence; 'missing' classes drawn in warning colors."""
    color = CLASS_COLORS.get(detection.object_class, (255, 255, 255))
    b = detection.box
    cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 1)
    name = detection.object_class.value.replace("_", "-")
    _label(frame, name, (int(b.x1), int(b.y2) + 14), color, scale=0.4)


def draw_zones(frame: np.ndarray, zones: list[Zone]) -> None:
    """Translucent red fill + outline + name for every zone."""
    if not zones:
        return
    height, width = frame.shape[:2]
    overlay = frame.copy()
    for zone in zones:
        polygon = zone.pixel_polygon(width, height).astype(np.int32)
        cv2.fillPoly(overlay, [polygon], (0, 0, 200))
        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 220), thickness=2)
        anchor = polygon.min(axis=0)
        _label(frame, f"ZONE: {zone.name}", (int(anchor[0]), int(anchor[1]) + 18),
               (0, 0, 220), scale=0.5)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)


def draw_alert_banner(
    frame: np.ndarray, alerts: list[SafetyEvent], max_lines: int = 3
) -> None:
    """Red banner across the top listing the most recent active alerts."""
    if not alerts:
        return
    lines = alerts[-max_lines:]
    banner_height = 26 * len(lines) + 8
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (30, 30, 30), -1)
    for i, event in enumerate(reversed(lines)):
        color = _SEVERITY_COLORS[event.severity]
        text = f"[{event.timestamp_str()}] {event.description}"
        cv2.putText(frame, text, (10, 22 + i * 26), _FONT, 0.6, color, 2)


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    """Small status panel (compliance %, counts, FPS) in the bottom-left."""
    if not lines:
        return
    height = frame.shape[0]
    panel_height = 20 * len(lines) + 10
    top = height - panel_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (250, height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (8, top + 20 + i * 20), _FONT, 0.5, (255, 255, 255), 1
        )
