"""Procedural pose-sequence generator for training the behavior model.

We have no labeled fall videos (real falls are rare and dangerous to stage),
so training data is generated procedurally: skeletons that walk, bend over,
or fall, with randomized body size, position, speed, and keypoint noise.
This is a transparent, reproducible stand-in — the model architecture and
training loop transfer unchanged to real labeled clips when available.

Behaviors (class labels match visionguard.temporal.model.CLASS_NAMES):
    walk: upright torso, slight sway and bob, horizontal drift
    bend: torso pitches to ~60-85 degrees mid-sequence, then recovers
    fall: torso pitches past ~80 degrees fast, hips drop, stays down
"""

from __future__ import annotations

import numpy as np

SEQUENCE_LENGTH = 30  # frames per sample (~3 s at 10 FPS effective)


def _skeleton(mid_hip: np.ndarray, angle: float, torso: float,
              rng: np.random.Generator) -> np.ndarray:
    """Build a plausible 17-keypoint COCO skeleton.

    Args:
        mid_hip: (x, y) pixel position of the hip center.
        angle: torso tilt from vertical, radians (0 = upright).
        torso: hip-to-shoulder distance in pixels.
    """
    axis = np.array([np.sin(angle), -np.cos(angle)])   # hip -> shoulder
    perp = np.array([np.cos(angle), np.sin(angle)])    # across the body

    mid_shoulder = mid_hip + axis * torso
    head = mid_shoulder + axis * 0.35 * torso
    shoulder_w, hip_w = 0.40 * torso, 0.30 * torso

    kp = np.zeros((17, 2), dtype=np.float32)
    kp[0] = head                                        # nose
    kp[1] = head + perp * 0.06 * torso                  # eyes / ears cluster
    kp[2] = head - perp * 0.06 * torso
    kp[3] = head + perp * 0.12 * torso
    kp[4] = head - perp * 0.12 * torso
    kp[5] = mid_shoulder + perp * shoulder_w            # shoulders
    kp[6] = mid_shoulder - perp * shoulder_w
    kp[7] = kp[5] - axis * 0.45 * torso                 # elbows
    kp[8] = kp[6] - axis * 0.45 * torso
    kp[9] = kp[7] - axis * 0.40 * torso                 # wrists
    kp[10] = kp[8] - axis * 0.40 * torso
    kp[11] = mid_hip + perp * hip_w                     # hips
    kp[12] = mid_hip - perp * hip_w
    kp[13] = kp[11] - axis * 0.9 * torso                # knees
    kp[14] = kp[12] - axis * 0.9 * torso
    kp[15] = kp[13] - axis * 0.8 * torso                # ankles
    kp[16] = kp[14] - axis * 0.8 * torso

    kp += rng.normal(0, 0.03 * torso, kp.shape)         # keypoint jitter
    return kp


def _angle_profile(behavior: str, rng: np.random.Generator) -> np.ndarray:
    """Torso tilt (radians) over the sequence, per behavior."""
    t = np.linspace(0, 1, SEQUENCE_LENGTH)
    if behavior == "walk":
        base = np.deg2rad(rng.uniform(0, 10))
        return base + np.deg2rad(3) * np.sin(t * rng.uniform(4, 9) * np.pi)
    if behavior == "bend":
        peak = np.deg2rad(rng.uniform(55, 85))
        # smooth down-and-back-up bump centered mid-sequence
        return peak * np.sin(np.clip(t, 0, 1) * np.pi) ** 2
    if behavior == "fall":
        peak = np.deg2rad(rng.uniform(78, 95))
        onset = rng.uniform(0.15, 0.4)       # when the fall starts
        duration = rng.uniform(0.1, 0.25)    # how fast they go down
        profile = np.clip((t - onset) / duration, 0, 1)
        return peak * profile                # goes down and STAYS down
    raise ValueError(behavior)


def generate_sequence(behavior: str, rng: np.random.Generator) -> np.ndarray:
    """One (SEQUENCE_LENGTH, 17, 2) keypoint sequence for a behavior."""
    torso = rng.uniform(40, 120)              # apparent body size in pixels
    x = rng.uniform(200, 1000)
    standing_y = rng.uniform(300, 600)
    drift = rng.uniform(-2, 2) if behavior == "walk" else rng.uniform(-0.5, 0.5)
    angles = _angle_profile(behavior, rng)

    frames = []
    for i, angle in enumerate(angles):
        # Hips sink as the torso goes horizontal (falls sink fully).
        sink = np.sin(angle) * torso * (1.1 if behavior == "fall" else 0.4)
        bob = 0.03 * torso * np.sin(i * 0.9) if behavior == "walk" else 0.0
        mid_hip = np.array([x + drift * i, standing_y + sink + bob])
        frames.append(_skeleton(mid_hip, float(angle), torso, rng))
    return np.stack(frames)


def generate_dataset(
    samples_per_class: int, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced dataset -> (X: (n, T, 17, 2), y: (n,)) with labels 0/1/2."""
    from visionguard.temporal.model import CLASS_NAMES

    rng = np.random.default_rng(seed)
    sequences, labels = [], []
    for label, behavior in enumerate(CLASS_NAMES):
        for _ in range(samples_per_class):
            sequences.append(generate_sequence(behavior, rng))
            labels.append(label)
    return np.stack(sequences), np.array(labels, dtype=np.int64)
