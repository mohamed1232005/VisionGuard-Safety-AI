"""Pose-sequence featurization.

Raw keypoints depend on where the person stands and how large they appear;
the *behavior* doesn't. Each frame's 17 COCO keypoints are therefore
normalized to a body-centric frame — centered on the mid-hip, scaled by torso
length — plus the torso angle encoded as sin/cos. Translation- and
scale-invariant by construction (verified by unit tests).
"""

from __future__ import annotations

import numpy as np

from visionguard.detection.types import (
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
)

FEATURE_DIM = 36  # 17 keypoints x 2 (normalized) + torso-angle sin + cos


def featurize_frame(keypoints_xy: np.ndarray) -> np.ndarray:
    """(17, 2) pixel keypoints -> (FEATURE_DIM,) body-centric features."""
    kp = np.asarray(keypoints_xy, dtype=np.float32)
    mid_hip = (kp[KP_LEFT_HIP] + kp[KP_RIGHT_HIP]) / 2.0
    mid_shoulder = (kp[KP_LEFT_SHOULDER] + kp[KP_RIGHT_SHOULDER]) / 2.0

    torso = mid_shoulder - mid_hip
    torso_length = float(np.linalg.norm(torso))
    scale = torso_length if torso_length > 1e-6 else 1.0

    centered = (kp - mid_hip) / scale
    # Angle of the torso vs vertical (image y grows downward).
    angle = np.arctan2(torso[0], -torso[1]) if torso_length > 1e-6 else 0.0
    return np.concatenate(
        [centered.reshape(-1), [np.sin(angle), np.cos(angle)]]
    ).astype(np.float32)


def featurize_sequence(keypoint_sequence: np.ndarray) -> np.ndarray:
    """(T, 17, 2) keypoint sequence -> (T, FEATURE_DIM) feature sequence."""
    return np.stack([featurize_frame(frame) for frame in keypoint_sequence])
