"""Interactive restricted-zone editor.

Opens the first frame of a video; draw polygon zones with the mouse and save
them to the zones JSON used by the pipeline. Coordinates are stored normalized
(0-1), so zones keep working at any processing resolution.

Controls:
    Left-click   add a vertex to the current polygon
    U            undo last vertex
    N            finish current polygon (asks for name/type/risk in terminal)
    S            save all zones to the JSON file and quit
    Q / Esc      quit without saving
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from visionguard.utils.config import DEFAULT_CONFIG_PATH, load_config

WINDOW = "VisionGuard - zone editor"


def grab_first_frame(source: str | int, resize_width: int | None) -> np.ndarray:
    capture = cv2.VideoCapture(source)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise IOError(f"Cannot read a frame from {source!r}")
    if resize_width and frame.shape[1] > resize_width:
        scale = resize_width / frame.shape[1]
        frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw restricted zones on a video")
    parser.add_argument("--video", default=None, help="Video path (default: config)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    source = args.video if args.video is not None else config.video.source
    frame = grab_first_frame(source, config.video.resize_width)
    height, width = frame.shape[:2]

    zones: list[dict] = []
    if config.zones.definitions_file.exists():
        zones = json.loads(config.zones.definitions_file.read_text("utf-8")).get(
            "zones", []
        )
        print(f"Loaded {len(zones)} existing zone(s); new zones will be appended.")

    current: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)

    while True:
        canvas = frame.copy()
        for zone in zones:  # already-defined zones, in red
            pts = np.array(
                [(px * width, py * height) for px, py in zone["polygon"]],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts], True, (0, 0, 220), 2)
            cv2.putText(canvas, zone["name"], tuple(pts.min(axis=0) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
        if current:  # polygon in progress, in yellow
            pts = np.array(current, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (0, 220, 220), 2)
            for point in current:
                cv2.circle(canvas, point, 4, (0, 220, 220), -1)

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            print("Quit without saving.")
            break
        if key == ord("u") and current:
            current.pop()
        if key == ord("n") and len(current) >= 3:
            name = input("Zone name: ").strip() or f"Zone {len(zones) + 1}"
            zone_type = (
                input("Zone type [restricted/vehicle_only/high_risk] (restricted): ")
                .strip()
                or "restricted"
            )
            risk = input("Risk level 1-3 (2): ").strip() or "2"
            zones.append(
                {
                    "name": name,
                    "zone_type": zone_type,
                    "risk_level": int(risk),
                    "polygon": [
                        [round(x / width, 4), round(y / height, 4)]
                        for x, y in current
                    ],
                    "applies_to": ["worker"],
                }
            )
            current.clear()
            print(f"Zone '{name}' added ({len(zones)} total). Press S to save.")
        if key == ord("s"):
            config.zones.definitions_file.parent.mkdir(parents=True, exist_ok=True)
            config.zones.definitions_file.write_text(
                json.dumps({"zones": zones}, indent=2), "utf-8"
            )
            print(f"Saved {len(zones)} zone(s) to {config.zones.definitions_file}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
