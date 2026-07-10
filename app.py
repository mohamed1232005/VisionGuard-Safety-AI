"""Hugging Face Space entry point for the VisionGuard Safety Command Center.

On first boot it downloads the model weights and two demo videos, then hands
off to the Streamlit dashboard. Two pre-computed analysis runs ship with the
Space so the dashboard is instantly explorable; visitors can also trigger a
fresh (CPU, slower) analysis from the sidebar.
"""

from __future__ import annotations

import runpy
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

STARTUP_ASSETS: dict[str, str] = {
    "models/ppe_yolo11s.pt": (
        "https://huggingface.co/yihong1120/Construction-Hazard-Detection"
        "/resolve/main/models/yolo11/pt/yolo11s.pt"
    ),
    "models/yolo11n-pose.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0"
        "/yolo11n-pose.pt"
    ),
    "data/videos/construction_steelwork.mp4": (
        "https://videos.pexels.com/video-files/11798561"
        "/11798561-hd_1920_1080_50fps.mp4"
    ),
    "data/videos/test_person_down.mp4": (
        "https://www.pexels.com/download/video/8526604/"
    ),
}

for relative, url in STARTUP_ASSETS.items():
    destination = ROOT / relative
    if destination.exists():
        continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[startup] downloading {relative} …", flush=True)
    urllib.request.urlretrieve(url, destination)

runpy.run_path(str(ROOT / "src" / "visionguard" / "dashboard" / "app.py"),
               run_name="__main__")
