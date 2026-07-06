"""Typed configuration loading for VisionGuard.

The single source of truth for settings is ``configs/config.yaml``. This module
parses that file into frozen dataclasses so the rest of the codebase gets
autocomplete, type checking, and a loud error at startup if the config is
malformed — instead of a silent KeyError deep inside the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Project root = three levels above this file (src/visionguard/utils -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class AppSettings:
    """General application settings."""

    name: str
    log_level: str
    log_dir: Path


@dataclass(frozen=True)
class PathSettings:
    """Locations of data, model weights, and generated outputs."""

    data_dir: Path
    models_dir: Path
    outputs_dir: Path


@dataclass(frozen=True)
class VideoSettings:
    """How input video is read and pre-processed."""

    source: str | int
    frame_skip: int
    resize_width: int | None


@dataclass(frozen=True)
class DetectionSettings:
    """Object detection model and thresholds."""

    model_path: Path
    confidence_threshold: float
    iou_threshold: float
    device: str


@dataclass(frozen=True)
class TrackingSettings:
    """ByteTrack multi-object tracking parameters."""

    track_activation_threshold: float
    lost_track_buffer: int
    minimum_matching_threshold: float
    trajectory_length: int


@dataclass(frozen=True)
class PPESettings:
    """PPE compliance engine thresholds."""

    required_equipment: tuple[str, ...]
    window_seconds: float
    violation_ratio: float
    min_observations: int
    cooldown_seconds: float


@dataclass(frozen=True)
class ZoneSettings:
    """Restricted-zone monitoring settings."""

    definitions_file: Path
    dwell_alert_seconds: float


@dataclass(frozen=True)
class FallSettings:
    """Pose-based fall detection thresholds."""

    model_path: Path
    torso_angle_threshold: float
    aspect_ratio_threshold: float
    confirm_seconds: float
    cooldown_seconds: float
    keypoint_confidence: float


@dataclass(frozen=True)
class ProximitySettings:
    """Worker-vehicle proximity thresholds (real-world meters)."""

    calibration_file: Path
    high_risk_distance_m: float
    medium_risk_distance_m: float
    cooldown_seconds: float


@dataclass(frozen=True)
class RiskScoreSettings:
    """Safety Risk Score weighting."""

    window_seconds: float
    weights: dict[str, float]


@dataclass(frozen=True)
class AssistantSettings:
    """RAG safety assistant configuration."""

    model: str
    max_tokens: int
    embedding_model: str
    top_k: int


@dataclass(frozen=True)
class EventSettings:
    """Event persistence locations."""

    database_path: Path
    screenshots_dir: Path


@dataclass(frozen=True)
class OutputSettings:
    """Where generated artifacts (videos, heatmaps, reports) are written."""

    annotated_dir: Path
    heatmap_dir: Path
    reports_dir: Path
    heatmap_grid: tuple[int, int]


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object handed to every module."""

    app: AppSettings
    paths: PathSettings
    video: VideoSettings
    detection: DetectionSettings
    tracking: TrackingSettings
    ppe: PPESettings
    zones: ZoneSettings
    fall: FallSettings
    proximity: ProximitySettings
    risk_score: RiskScoreSettings
    assistant: AssistantSettings
    events: EventSettings
    output: OutputSettings


def _resolve(path_value: str) -> Path:
    """Resolve a config path relative to the project root.

    Absolute paths are kept as-is, so users can point at data anywhere on disk.
    """
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate the YAML config into an :class:`AppConfig`.

    Args:
        config_path: Path to the YAML file. Defaults to ``configs/config.yaml``.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required section or key is missing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    return AppConfig(
        app=AppSettings(
            name=raw["app"]["name"],
            log_level=raw["app"]["log_level"],
            log_dir=_resolve(raw["app"]["log_dir"]),
        ),
        paths=PathSettings(
            data_dir=_resolve(raw["paths"]["data_dir"]),
            models_dir=_resolve(raw["paths"]["models_dir"]),
            outputs_dir=_resolve(raw["paths"]["outputs_dir"]),
        ),
        video=VideoSettings(
            source=raw["video"]["source"],
            frame_skip=int(raw["video"]["frame_skip"]),
            resize_width=(
                int(raw["video"]["resize_width"])
                if raw["video"]["resize_width"] is not None
                else None
            ),
        ),
        detection=DetectionSettings(
            model_path=_resolve(raw["detection"]["model_path"]),
            confidence_threshold=float(raw["detection"]["confidence_threshold"]),
            iou_threshold=float(raw["detection"]["iou_threshold"]),
            device=raw["detection"]["device"],
        ),
        tracking=TrackingSettings(
            track_activation_threshold=float(
                raw["tracking"]["track_activation_threshold"]
            ),
            lost_track_buffer=int(raw["tracking"]["lost_track_buffer"]),
            minimum_matching_threshold=float(
                raw["tracking"]["minimum_matching_threshold"]
            ),
            trajectory_length=int(raw["tracking"]["trajectory_length"]),
        ),
        ppe=PPESettings(
            required_equipment=tuple(raw["ppe"]["required_equipment"]),
            window_seconds=float(raw["ppe"]["window_seconds"]),
            violation_ratio=float(raw["ppe"]["violation_ratio"]),
            min_observations=int(raw["ppe"]["min_observations"]),
            cooldown_seconds=float(raw["ppe"]["cooldown_seconds"]),
        ),
        zones=ZoneSettings(
            definitions_file=_resolve(raw["zones"]["definitions_file"]),
            dwell_alert_seconds=float(raw["zones"]["dwell_alert_seconds"]),
        ),
        fall=FallSettings(
            model_path=_resolve(raw["fall"]["model_path"]),
            torso_angle_threshold=float(raw["fall"]["torso_angle_threshold"]),
            aspect_ratio_threshold=float(raw["fall"]["aspect_ratio_threshold"]),
            confirm_seconds=float(raw["fall"]["confirm_seconds"]),
            cooldown_seconds=float(raw["fall"]["cooldown_seconds"]),
            keypoint_confidence=float(raw["fall"]["keypoint_confidence"]),
        ),
        proximity=ProximitySettings(
            calibration_file=_resolve(raw["proximity"]["calibration_file"]),
            high_risk_distance_m=float(raw["proximity"]["high_risk_distance_m"]),
            medium_risk_distance_m=float(raw["proximity"]["medium_risk_distance_m"]),
            cooldown_seconds=float(raw["proximity"]["cooldown_seconds"]),
        ),
        risk_score=RiskScoreSettings(
            window_seconds=float(raw["risk_score"]["window_seconds"]),
            weights={
                name: float(value)
                for name, value in raw["risk_score"]["weights"].items()
            },
        ),
        assistant=AssistantSettings(
            model=raw["assistant"]["model"],
            max_tokens=int(raw["assistant"]["max_tokens"]),
            embedding_model=raw["assistant"]["embedding_model"],
            top_k=int(raw["assistant"]["top_k"]),
        ),
        events=EventSettings(
            database_path=_resolve(raw["events"]["database_path"]),
            screenshots_dir=_resolve(raw["events"]["screenshots_dir"]),
        ),
        output=OutputSettings(
            annotated_dir=_resolve(raw["output"]["annotated_dir"]),
            heatmap_dir=_resolve(raw["output"]["heatmap_dir"]),
            reports_dir=_resolve(raw["output"]["reports_dir"]),
            heatmap_grid=tuple(int(v) for v in raw["output"]["heatmap_grid"]),
        ),
    )
