"""Central logging setup for VisionGuard.

Call :func:`setup_logging` once at application startup; afterwards every module
simply does ``logger = logging.getLogger(__name__)``. Output goes to the
console and to a rotating file in the configured log directory, so long video
runs never fill the disk.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
MAX_LOG_BYTES = 5 * 1024 * 1024  # rotate after 5 MB
BACKUP_COUNT = 3  # keep at most 3 old log files


def setup_logging(level: str = "INFO", log_dir: Path | str = "outputs/logs") -> None:
    """Configure root logging with console + rotating file handlers.

    Args:
        level: Minimum level to record ("DEBUG", "INFO", "WARNING", "ERROR").
        log_dir: Directory for the log file; created if missing.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        RotatingFileHandler(
            log_dir / "visionguard.log",
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        ),
    ]

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,  # replace any handlers configured by imported libraries
    )
