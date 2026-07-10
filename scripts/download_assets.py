"""Download the model weights and sample video the project needs.

Run once after cloning:
    python scripts/download_assets.py

Everything is skipped if already present, so the script is safe to re-run.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSETS: dict[str, str] = {
    # PPE detection model — yihong1120/Construction-Hazard-Detection (AGPL-3.0).
    # Classes: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
    #          Safety Cone, Safety Vest, Machinery, Utility Pole, Vehicle.
    "models/ppe_yolo11s.pt": (
        "https://huggingface.co/yihong1120/Construction-Hazard-Detection"
        "/resolve/main/models/yolo11/pt/yolo11s.pt"
    ),
    # Pose model for fall detection (Ultralytics, AGPL-3.0).
    "models/yolo11n-pose.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0"
        "/yolo11n-pose.pt"
    ),
    # Sample construction video (Pexels free license, by manas patra).
    "data/videos/construction_steelwork.mp4": (
        "https://videos.pexels.com/video-files/11798561"
        "/11798561-hd_1920_1080_50fps.mp4"
    ),
    # Feature-test videos (Pexels free license):
    # PPE violations + fall ("person down" emergency), by Ron Lach.
    "data/videos/test_person_down.mp4": (
        "https://www.pexels.com/download/video/8526604/"
    ),
    # Workers digging (zone-intrusion test), by K.
    "data/videos/test_ppe_digging.mp4": (
        "https://videos.pexels.com/video-files/3967264"
        "/3967264-uhd_2732_1440_24fps.mp4"
    ),
    # Forklift in warehouse (vehicle detection + proximity test), by kelly.
    "data/videos/test_forklift.mp4": (
        "https://www.pexels.com/download/video/6079419/"
    ),
}


def main() -> None:
    for relative_path, url in ASSETS.items():
        destination = PROJECT_ROOT / relative_path
        if destination.exists():
            print(f"[skip] {relative_path} already exists")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[down] {relative_path} <- {url}")
        urllib.request.urlretrieve(url, destination)  # noqa: S310 - fixed URLs
        print(f"       done ({destination.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
