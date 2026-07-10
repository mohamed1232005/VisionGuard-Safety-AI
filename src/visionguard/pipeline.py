"""End-to-end safety analysis pipeline.

Per processed frame:
    detect -> track -> PPE compliance -> zone checks -> fall detection
    -> persist events (with screenshot evidence) -> annotated output frame

A run produces: rows in the event database, evidence screenshots, an annotated
video, a danger heatmap image, and summary statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from visionguard.detection.detector import Detector
from visionguard.detection.pose import PoseEstimator
from visionguard.detection.types import PPE_CLASSES, ObjectClass
from visionguard.safety.events import SafetyEvent
from visionguard.safety.falls import FallDetector
from visionguard.safety.ppe import PPEComplianceEngine
from visionguard.safety.proximity import ProximityMonitor
from visionguard.safety.risk import RiskScoreCalculator, risk_band
from visionguard.safety.zones import ZoneMonitor, load_zones
from visionguard.spatial.homography import load_ground_plane
from visionguard.storage.event_store import EventStore
from visionguard.tracking.tracker import Tracker
from visionguard.utils.config import AppConfig, portable_path
from visionguard.utils.drawing import (
    draw_alert_banner,
    draw_hud,
    draw_ppe_evidence,
    draw_proximity_line,
    draw_tracked_object,
    draw_zones,
)
from visionguard.utils.video import FPSMeter, VideoReader, VideoWriter, reencode_h264

logger = logging.getLogger(__name__)

_ALERT_BANNER_SECONDS = 4.0  # how long an alert stays on the video banner


@dataclass(frozen=True)
class RunResult:
    """Summary of one completed analysis run."""

    run_id: int
    frames_processed: int
    events_total: int
    compliance_rate: float
    processing_fps: float
    annotated_video: Path
    annotated_video_h264: Path | None
    heatmap_image: Path | None
    stats: dict[str, Any]


class SafetyPipeline:
    """Runs the full Phase 1 analysis over a video source."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._detector = Detector(config.detection)
        self._pose = PoseEstimator(config.fall, device=config.detection.device)
        self._zones = load_zones(config.zones.definitions_file)
        self._ground_plane = load_ground_plane(config.proximity.calibration_file)
        self._store = EventStore(config.events.database_path)

    @property
    def store(self) -> EventStore:
        return self._store

    def run(
        self,
        source: str | int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> RunResult:
        """Analyze a video and return the run summary.

        Args:
            source: Video path or camera index; defaults to the config value.
            progress: Optional callback (frames_done, total_frames) for UIs.
        """
        config = self._config
        source = source if source is not None else config.video.source
        source_name = str(source)

        run_id = self._store.create_run(source_name)
        screenshots_dir = config.events.screenshots_dir / f"run_{run_id}"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = config.output.annotated_dir / f"run_{run_id}.mp4"

        logger.info("Run %d started on %s", run_id, source_name)

        with VideoReader(source) as reader:
            source_fps = reader.fps
            effective_fps = source_fps / config.video.frame_skip
            total_frames = reader.frame_count

            tracker = Tracker(config.tracking, frame_rate=effective_fps)
            ppe_engine = PPEComplianceEngine(config.ppe, fps=effective_fps)
            zone_monitor = ZoneMonitor(self._zones, config.zones)
            fall_detector = FallDetector(config.fall)
            proximity_monitor = (
                ProximityMonitor(config.proximity, self._ground_plane)
                if self._ground_plane is not None
                else None
            )
            risk_calculator = RiskScoreCalculator(config.risk_score)
            risk_timeline: list[tuple[float, float]] = []
            last_risk_sample = -1.0

            heat_cols, heat_rows = config.output.heatmap_grid
            heatmap = np.zeros((heat_rows, heat_cols), dtype=np.float64)

            meter = FPSMeter()
            banner_alerts: list[SafetyEvent] = []
            events_total = 0
            frames_processed = 0
            writer: VideoWriter | None = None
            last_frame: np.ndarray | None = None

            try:
                for frame_index, frame in reader.frames(
                    config.video.frame_skip, config.video.resize_width
                ):
                    video_time = frame_index / source_fps
                    height, width = frame.shape[:2]

                    # ---- Perception ------------------------------------- #
                    detections = self._detector.detect(frame)
                    tracked = tracker.update(detections)
                    workers = [
                        t for t in tracked if t.object_class is ObjectClass.WORKER
                    ]
                    poses = self._pose.estimate(frame) if workers else []

                    # ---- Safety rules ----------------------------------- #
                    new_events = ppe_engine.update(
                        frame_index, video_time, workers, detections
                    )
                    new_events += zone_monitor.update(
                        frame_index, video_time, tracked, width, height
                    )
                    new_events += fall_detector.update(
                        frame_index, video_time, workers, poses
                    )
                    close_pairs = []
                    if proximity_monitor is not None:
                        proximity_events, close_pairs = proximity_monitor.update(
                            frame_index, video_time, tracked, width, height
                        )
                        new_events += proximity_events

                    # ---- Risk score --------------------------------------- #
                    risk_calculator.add_events(new_events)
                    risk_now = risk_calculator.score(video_time)
                    if video_time - last_risk_sample >= 1.0:
                        risk_timeline.append((round(video_time, 1), risk_now))
                        last_risk_sample = video_time

                    # ---- Heatmap accumulation --------------------------- #
                    for worker in workers:
                        gx, gy = worker.box.ground_point
                        col = min(int(gx / width * heat_cols), heat_cols - 1)
                        row = min(int(gy / height * heat_rows), heat_rows - 1)
                        heatmap[row, col] += 1.0

                    # ---- Annotation ------------------------------------- #
                    draw_zones(frame, self._zones)
                    for obj in tracked:
                        draw_tracked_object(frame, obj)
                    for det in detections:
                        if det.object_class in PPE_CLASSES:
                            draw_ppe_evidence(frame, det)
                    for pair in close_pairs:
                        draw_proximity_line(frame, pair)

                    # ---- Event persistence (screenshot AFTER annotation,
                    #      so evidence images show boxes and IDs) ---------- #
                    for event in new_events:
                        event.screenshot_path = portable_path(
                            self._save_screenshot(frame, event, screenshots_dir)
                        )
                        self._store.add_event(run_id, event)
                        banner_alerts.append(event)
                        events_total += 1

                    banner_alerts = [
                        e
                        for e in banner_alerts
                        if video_time - e.video_time <= _ALERT_BANNER_SECONDS
                    ]
                    draw_alert_banner(frame, banner_alerts)

                    meter.tick()
                    draw_hud(
                        frame,
                        [
                            f"Risk score: {risk_now:.0f} ({risk_band(risk_now)})",
                            f"PPE compliance: {ppe_engine.stats.compliance_rate:.0%}",
                            f"Workers: {len(workers)}  Events: {events_total}",
                            f"Processing: {meter.fps:.1f} FPS",
                        ],
                    )

                    # ---- Output video ----------------------------------- #
                    if writer is None:
                        writer = VideoWriter(
                            annotated_path, effective_fps, (width, height)
                        )
                    writer.write(frame)
                    last_frame = frame
                    frames_processed += 1

                    if progress is not None and total_frames:
                        progress(frames_processed, total_frames)
                    if frames_processed % 100 == 0:
                        logger.info(
                            "Run %d: %d frames, %d events, %.1f FPS",
                            run_id, frames_processed, events_total, meter.fps,
                        )
            finally:
                if writer is not None:
                    writer.__exit__()

        # ---- Post-run artifacts ------------------------------------------ #
        heatmap_path = self._save_heatmap(heatmap, last_frame, run_id)
        h264_path = reencode_h264(annotated_path) if frames_processed else None

        compliance = ppe_engine.stats.compliance_rate
        stats: dict[str, Any] = {
            "source": source_name,
            "unique_counts": {
                cls.value: n for cls, n in tracker.unique_counts().items()
            },
            "events_by_type": self._store.event_type_counts(run_id),
            "ppe_violations_by_item": ppe_engine.stats.violations_by_item,
            "zone_intrusions": zone_monitor.stats.intrusions,
            "zone_dwell_alerts": zone_monitor.stats.dwell_alerts,
            "most_dangerous_zone": zone_monitor.stats.most_dangerous_zone(),
            "falls_detected": fall_detector.falls_detected,
            "proximity": (
                {
                    "near_misses": proximity_monitor.stats.near_misses,
                    "medium_alerts": proximity_monitor.stats.medium_alerts,
                    "min_distance_m": proximity_monitor.stats.min_distance_m,
                }
                if proximity_monitor is not None
                else None  # camera not calibrated
            ),
            "risk_score": {
                "final": risk_calculator.score(
                    frames_processed / effective_fps if effective_fps else 0.0
                ),
                "peak": risk_calculator.peak_score,
                "peak_time": round(risk_calculator.peak_time, 1),
                "timeline": risk_timeline,
            },
            "processing_fps": round(meter.fps, 2),
            "annotated_video": portable_path(annotated_path),
            "annotated_video_h264": portable_path(h264_path) if h264_path else None,
            "heatmap_image": portable_path(heatmap_path) if heatmap_path else None,
        }
        self._store.finish_run(
            run_id,
            fps=source_fps,
            frames_processed=frames_processed,
            duration_seconds=frames_processed / effective_fps if effective_fps else 0,
            compliance_rate=compliance,
            stats=stats,
        )
        logger.info(
            "Run %d finished: %d frames, %d events, compliance %.0f%%",
            run_id, frames_processed, events_total, compliance * 100,
        )
        return RunResult(
            run_id=run_id,
            frames_processed=frames_processed,
            events_total=events_total,
            compliance_rate=compliance,
            processing_fps=meter.fps,
            annotated_video=annotated_path,
            annotated_video_h264=h264_path,
            heatmap_image=heatmap_path,
            stats=stats,
        )

    # ------------------------------------------------------------------ #
    # Artifacts
    # ------------------------------------------------------------------ #
    @staticmethod
    def _save_screenshot(
        frame: np.ndarray, event: SafetyEvent, directory: Path
    ) -> Path:
        """Save annotated evidence for an event, highlighting the description."""
        evidence = frame.copy()
        cv2.putText(
            evidence,
            event.description,
            (10, evidence.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        name = (
            f"f{event.frame_index:06d}_{event.event_type.value}"
            f"_t{event.track_id or 0}.jpg"
        )
        path = directory / name
        cv2.imwrite(str(path), evidence)
        return path

    def _save_heatmap(
        self, heatmap: np.ndarray, last_frame: np.ndarray | None, run_id: int
    ) -> Path | None:
        """Render worker-position density as a colored overlay on the scene."""
        if last_frame is None or heatmap.max() <= 0:
            return None
        height, width = last_frame.shape[:2]

        # Log scale keeps rarely-visited areas visible next to hotspots.
        normalized = np.log1p(heatmap) / np.log1p(heatmap.max())
        heat_image = cv2.resize(
            (normalized * 255).astype(np.uint8), (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        heat_color = cv2.applyColorMap(heat_image, cv2.COLORMAP_JET)
        mask = heat_image > 8  # leave un-visited areas as the raw scene
        blended = last_frame.copy()
        blended[mask] = cv2.addWeighted(last_frame, 0.4, heat_color, 0.6, 0)[mask]

        path = self._config.output.heatmap_dir / f"run_{run_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), blended)
        return path
