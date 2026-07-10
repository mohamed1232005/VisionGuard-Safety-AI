"""Interactive ground-plane camera calibration.

Teaches VisionGuard how to convert pixels to real-world meters. You click at
least 4 points ON THE GROUND whose real-world positions you know (for example
the four corners of a slab or marked rectangle whose dimensions you measured),
and enter each point's (X, Y) position in meters in the terminal.

Tip: pick points that are spread out — a large rectangle beats a small one.
The world origin (0, 0) can be any point you choose; only relative distances
matter.

Controls:
    Left-click   mark a ground point (then enter its X Y meters in terminal)
    U            undo the last point
    S            save calibration (needs >= 4 points) and quit
    Q / Esc      quit without saving

Usage:
    python scripts/calibrate_camera.py                 # config video source
    python scripts/calibrate_camera.py --video x.mp4
"""

from __future__ import annotations

import argparse
import json

import cv2

from visionguard.utils.config import DEFAULT_CONFIG_PATH, load_config

WINDOW = "VisionGuard - camera calibration"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the ground plane")
    parser.add_argument("--video", default=None, help="Video path (default: config)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    source = args.video if args.video is not None else config.video.source

    capture = cv2.VideoCapture(source)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise IOError(f"Cannot read a frame from {source!r}")
    if config.video.resize_width and frame.shape[1] > config.video.resize_width:
        scale = config.video.resize_width / frame.shape[1]
        frame = cv2.resize(frame, (config.video.resize_width, int(frame.shape[0] * scale)))
    height, width = frame.shape[:2]

    image_points: list[list[float]] = []
    world_points: list[list[float]] = []
    clicked: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)

    while True:
        # Consume pending clicks: ask for the world position of each.
        while clicked:
            x, y = clicked.pop(0)
            raw = input(f"Point at pixel ({x}, {y}) — real-world 'X Y' in meters: ")
            try:
                wx, wy = (float(v) for v in raw.replace(",", " ").split())
            except ValueError:
                print("  Could not parse — expected two numbers like: 3.5 0")
                continue
            image_points.append([float(x), float(y)])
            world_points.append([wx, wy])
            print(f"  #{len(image_points)}: pixel ({x},{y}) = world ({wx} m, {wy} m)")

        canvas = frame.copy()
        for i, (px, py) in enumerate(image_points):
            cv2.circle(canvas, (int(px), int(py)), 6, (0, 220, 220), -1)
            cv2.putText(
                canvas,
                f"{i + 1}: ({world_points[i][0]}, {world_points[i][1]})m",
                (int(px) + 8, int(py) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 2,
            )
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            print("Quit without saving.")
            break
        if key == ord("u") and image_points:
            image_points.pop()
            world_points.pop()
        if key == ord("s"):
            if len(image_points) < 4:
                print(f"Need >= 4 points, have {len(image_points)}.")
                continue
            output = config.proximity.calibration_file
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "reference_size": [width, height],
                        "image_points": image_points,
                        "world_points": world_points,
                    },
                    indent=2,
                ),
                "utf-8",
            )
            print(f"Saved {len(image_points)} calibration points to {output}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
