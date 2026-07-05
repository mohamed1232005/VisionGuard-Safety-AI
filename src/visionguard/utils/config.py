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

# Project root = two levels above this file's package (src/visionguard/utils -> root)
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

    model: str
    confidence_threshold: float
    iou_threshold: float
    device: str


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object handed to every module."""

    app: AppSettings
    paths: PathSettings
    video: VideoSettings
    detection: DetectionSettings


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
            model=raw["detection"]["model"],
            confidence_threshold=float(raw["detection"]["confidence_threshold"]),
            iou_threshold=float(raw["detection"]["iou_threshold"]),
            device=raw["detection"]["device"],
        ),
    )
