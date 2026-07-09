"""Edge-deployment benchmark: PyTorch vs ONNX vs quantized INT8.

Exports the PPE detector to ONNX (FP32 and dynamically-quantized INT8) and
measures single-image inference latency/FPS for every available configuration
on real frames from the sample video. Results print as a markdown table ready
for the README.

Usage:
    python scripts/benchmark.py                # 100 frames per config
    python scripts/benchmark.py --frames 300
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from visionguard.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = 640  # YOLO's native inference size


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def load_frames(source: str, count: int, resize_width: int | None) -> list[np.ndarray]:
    """Sample `count` frames evenly from the video."""
    capture = cv2.VideoCapture(source)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), count).astype(int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        if resize_width and frame.shape[1] > resize_width:
            scale = resize_width / frame.shape[1]
            frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
        frames.append(frame)
    capture.release()
    return frames


def preprocess_onnx(frame: np.ndarray) -> np.ndarray:
    """Letterbox-free simple resize preprocessing for ONNX latency tests."""
    resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return blob.transpose(2, 0, 1)[None]


def timed(fn, frames: list, warmup: int = 10) -> dict:
    """Run fn over frames (after warmup) and report latency statistics."""
    for frame in frames[:warmup]:
        fn(frame)
    latencies = []
    for frame in frames:
        start = time.perf_counter()
        fn(frame)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95) - 1],
        "fps": 1000.0 / statistics.mean(latencies),
    }


def file_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


# --------------------------------------------------------------------- #
# Benchmark configurations
# --------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="VisionGuard model benchmarks")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    config = load_config()
    weights = Path(config.detection.model_path)
    frames = load_frames(
        str(config.video.source), args.frames, config.video.resize_width
    )
    print(f"Benchmarking on {len(frames)} frames from {config.video.source}\n")

    results: list[tuple[str, float, dict]] = []

    # ---- PyTorch (Ultralytics) ---------------------------------------- #
    import logging as _logging

    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER as _ULTRA_LOGGER

    # Ultralytics warns once per frame about the deprecated (but still
    # functional) `half` argument — silence warnings during timing loops.
    _ULTRA_LOGGER.setLevel(_logging.ERROR)

    model = YOLO(str(weights))
    if torch.cuda.is_available():
        for label, half in [("PyTorch FP32 (GPU)", False), ("PyTorch FP16 (GPU)", True)]:
            fn = lambda f, h=half: model.predict(  # noqa: E731
                f, device="cuda", half=h, verbose=False, imgsz=IMAGE_SIZE
            )
            results.append((label, file_mb(weights), timed(fn, frames)))
            print(f"done: {label}")

    cpu_fn = lambda f: model.predict(f, device="cpu", verbose=False, imgsz=IMAGE_SIZE)  # noqa: E731
    results.append(("PyTorch FP32 (CPU)", file_mb(weights), timed(cpu_fn, frames)))
    print("done: PyTorch FP32 (CPU)")

    # ---- ONNX export --------------------------------------------------- #
    onnx_path = weights.with_suffix(".onnx")
    if not onnx_path.exists():
        print("exporting ONNX…")
        model.export(format="onnx", imgsz=IMAGE_SIZE, simplify=True)

    import onnxruntime as ort

    def ort_benchmark(label: str, path: Path, providers: list[str]) -> None:
        try:
            session = ort.InferenceSession(str(path), providers=providers)
        except Exception as error:  # provider not available on this machine
            print(f"skip: {label} ({type(error).__name__})")
            return
        input_name = session.get_inputs()[0].name
        fn = lambda f: session.run(None, {input_name: preprocess_onnx(f)})  # noqa: E731
        results.append((label, file_mb(path), timed(fn, frames)))
        print(f"done: {label} [{session.get_providers()[0]}]")

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        ort_benchmark("ONNX Runtime FP32 (GPU)", onnx_path, ["CUDAExecutionProvider"])
    ort_benchmark("ONNX Runtime FP32 (CPU)", onnx_path, ["CPUExecutionProvider"])

    # ---- INT8 dynamic quantization ------------------------------------- #
    int8_path = weights.with_name(weights.stem + "_int8.onnx")
    if not int8_path.exists():
        print("quantizing to INT8…")
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(onnx_path), str(int8_path),
                         weight_type=QuantType.QUInt8)
    ort_benchmark("ONNX Runtime INT8 (CPU)", int8_path, ["CPUExecutionProvider"])

    # ---- Report --------------------------------------------------------- #
    print("\n| Configuration | Size (MB) | Mean latency | p95 | Throughput |")
    print("|---|---|---|---|---|")
    for label, size, stats in results:
        print(
            f"| {label} | {size:.1f} | {stats['mean_ms']:.1f} ms "
            f"| {stats['p95_ms']:.1f} ms | **{stats['fps']:.1f} FPS** |"
        )
    print("\n(Single-image inference at 640x640, includes pre/post-processing; "
          "measured on this machine.)")


if __name__ == "__main__":
    main()
