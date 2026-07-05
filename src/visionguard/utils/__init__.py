"""Shared utilities: configuration loading and logging setup."""

from visionguard.utils.config import AppConfig, load_config
from visionguard.utils.logger import setup_logging

__all__ = ["AppConfig", "load_config", "setup_logging"]
