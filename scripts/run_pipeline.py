"""Run the full VisionGuard safety analysis on a video.

Usage:
    python scripts/run_pipeline.py                     # uses config video source
    python scripts/run_pipeline.py --source path.mp4   # analyze a specific file
    python scripts/run_pipeline.py --source 0          # live webcam
"""

from __future__ import annotations

import argparse
import logging

from visionguard.pipeline import SafetyPipeline
from visionguard.utils.config import DEFAULT_CONFIG_PATH, load_config
from visionguard.utils.logger import setup_logging

logger = logging.getLogger("visionguard.run")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionGuard safety analysis")
    parser.add_argument(
        "--source",
        default=None,
        help="Video file path or camera index (default: config video.source)",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="Path to config YAML"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.app.log_level, config.app.log_dir)

    source: str | int | None = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)  # "0" -> webcam index 0

    pipeline = SafetyPipeline(config)
    result = pipeline.run(source=source)

    print("\n=== VisionGuard run summary ===")
    print(f"Run ID:            {result.run_id}")
    print(f"Frames processed:  {result.frames_processed}")
    print(f"Safety events:     {result.events_total}")
    print(f"PPE compliance:    {result.compliance_rate:.1%}")
    print(f"Processing speed:  {result.processing_fps:.1f} FPS")
    print(f"Annotated video:   {result.annotated_video}")
    if result.heatmap_image:
        print(f"Danger heatmap:    {result.heatmap_image}")


if __name__ == "__main__":
    main()
