"""Video reading/writing helpers used by the pipeline and scripts."""

from __future__ import annotations

import logging
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Context-manager wrapper around ``cv2.VideoCapture``.

    Handles file paths and camera indices, exposes stream properties, and
    yields frames with optional skipping and resizing.
    """

    def __init__(self, source: str | int) -> None:
        self._source = source
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoReader":
        self._capture = cv2.VideoCapture(self._source)
        if not self._capture.isOpened():
            raise IOError(f"Cannot open video source: {self._source!r}")
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._capture is not None:
            self._capture.release()

    @property
    def fps(self) -> float:
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0  # webcams often report 0

    @property
    def frame_count(self) -> int:
        """Total frames (0 for live sources)."""
        return max(int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)

    def frames(
        self, frame_skip: int = 1, resize_width: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, BGR frame), honoring skip and resize settings.

        ``frame_index`` is the index in the *source* video (so timestamps stay
        correct even when frames are skipped).
        """
        index = -1
        while True:
            ok, frame = self._capture.read()
            if not ok:
                return
            index += 1
            if index % frame_skip != 0:
                continue
            if resize_width is not None and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(
                    frame, (resize_width, int(frame.shape[0] * scale))
                )
            yield index, frame


class VideoWriter:
    """Context-manager wrapper around ``cv2.VideoWriter`` (mp4v codec).

    The mp4v output plays everywhere locally; use :func:`reencode_h264` on the
    result when browser playback (Streamlit) is needed.
    """

    def __init__(self, path: Path | str, fps: float, size: tuple[int, int]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path), fourcc, fps, size)
        if not self._writer.isOpened():
            raise IOError(f"Cannot open video writer for {self._path}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._writer.release()

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)


def reencode_h264(source: Path | str, destination: Path | str | None = None) -> Path:
    """Re-encode a video to H.264 so browsers (and Streamlit) can play it.

    Uses the ffmpeg binary bundled with ``imageio-ffmpeg`` — no system ffmpeg
    install required. Returns the destination path (falls back to the original
    file if ffmpeg is unavailable).
    """
    source = Path(source)
    destination = (
        Path(destination)
        if destination is not None
        else source.with_name(source.stem + "_h264.mp4")
    )
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - depends on optional install
        logger.warning("imageio-ffmpeg unavailable; skipping H.264 re-encode")
        return source

    command = [
        ffmpeg, "-y", "-i", str(source),
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-an", str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("H.264 re-encode failed: %s", result.stderr[-400:])
        return source
    return destination


class FPSMeter:
    """Rolling measurement of end-to-end processing speed."""

    def __init__(self, window: int = 60) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
