"""Cross-camera Re-Identification demo.

Runs detection + tracking on two video sources ("cameras"), embeds each
worker's appearance, and matches identities across cameras: the same worker
receives the same global ID everywhere. Saves a montage image per matched
identity so you can eyeball the matches.

Usage:
    python scripts/reid_demo.py --videos camA.mp4 camB.mp4
    python scripts/reid_demo.py            # single config video split in two
                                           # halves simulating two cameras
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from visionguard.detection.detector import Detector
from visionguard.detection.types import ObjectClass
from visionguard.reid.embedder import AppearanceEmbedder
from visionguard.reid.gallery import ReIDGallery
from visionguard.tracking.tracker import Tracker
from visionguard.utils.config import DEFAULT_CONFIG_PATH, load_config
from visionguard.utils.logger import setup_logging
from visionguard.utils.video import VideoReader

logger = logging.getLogger("visionguard.reid_demo")

FRAME_STRIDE = 10          # analyze every Nth frame (appearance changes slowly)
MIN_CROP_HEIGHT = 60       # skip tiny/far-away crops (too little appearance signal)


def collect_track_crops(
    config, source: str | int, camera: str,
    frame_range: tuple[float, float] = (0.0, 1.0),
) -> dict[int, list[np.ndarray]]:
    """Run detect+track over a video segment, returning crops per track id."""
    detector = Detector(config.detection)
    crops: dict[int, list[np.ndarray]] = defaultdict(list)

    with VideoReader(source) as reader:
        total = reader.frame_count or 0
        start = int(total * frame_range[0])
        end = int(total * frame_range[1]) if total else None
        tracker = Tracker(config.tracking, frame_rate=reader.fps / FRAME_STRIDE)

        for frame_index, frame in reader.frames(
            FRAME_STRIDE, config.video.resize_width
        ):
            if frame_index < start:
                continue
            if end is not None and frame_index >= end:
                break
            tracked = tracker.update(detector.detect(frame))
            for obj in tracked:
                if obj.object_class is not ObjectClass.WORKER:
                    continue
                if len(crops[obj.track_id]) >= config.reid.crops_per_track:
                    continue
                b = obj.box
                if b.height < MIN_CROP_HEIGHT:
                    continue
                crop = frame[
                    max(int(b.y1), 0):int(b.y2), max(int(b.x1), 0):int(b.x2)
                ]
                if crop.size:
                    crops[obj.track_id].append(crop.copy())

    logger.info("%s: %d worker tracks with usable crops", camera, len(crops))
    return crops


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-camera Re-ID demo")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Two video files (one file = split-in-half demo)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.app.log_level, config.app.log_dir)

    if args.videos and len(args.videos) >= 2:
        cameras = [(f"camera-{c}", video, (0.0, 1.0))
                   for c, video in zip("ABCD", args.videos)]
    else:
        video = args.videos[0] if args.videos else str(config.video.source)
        print("One video provided - splitting into two halves as two 'cameras'.")
        cameras = [("camera-A", video, (0.0, 0.5)), ("camera-B", video, (0.5, 1.0))]

    embedder = AppearanceEmbedder(config.reid.embedding_model)
    gallery = ReIDGallery(config.reid.similarity_threshold)
    crops_by_gid: dict[int, list[np.ndarray]] = defaultdict(list)

    for camera, video, frame_range in cameras:
        track_crops = collect_track_crops(config, video, camera, frame_range)
        for track_id, crops in track_crops.items():
            if not crops:
                continue
            # Mean of several crops = a steadier appearance signature.
            embeddings = embedder.embed(crops)
            mean = embeddings.mean(axis=0)
            mean /= np.linalg.norm(mean)
            gid = gallery.assign(camera, track_id, mean)
            crops_by_gid[gid].extend(crops[:3])
            print(f"  {camera} track {track_id:>3d}  ->  global identity #{gid}")

    print("\n=== Identity gallery ===")
    cross_camera = 0
    for entry in gallery.summary():
        marker = ""
        if len(entry["cameras"]) > 1:
            cross_camera += 1
            marker = "  <-- matched across cameras"
        print(f"Global #{entry['global_id']}: {entry['observations']} tracks, "
              f"cameras {entry['cameras']}{marker}")
    print(f"\n{gallery.identity_count} unique people; "
          f"{cross_camera} matched across cameras.")

    # Montage sheets: visual proof of who was matched with whom.
    out_dir = config.paths.outputs_dir / "reid"
    out_dir.mkdir(parents=True, exist_ok=True)
    for gid, crops in crops_by_gid.items():
        if len(crops) < 2:
            continue
        tiles = [cv2.resize(c, (96, 192)) for c in crops[:8]]
        montage = cv2.hconcat(tiles)
        cv2.imwrite(str(out_dir / f"identity_{gid:03d}.jpg"), montage)
    print(f"Montage sheets saved to {out_dir}")


if __name__ == "__main__":
    main()
