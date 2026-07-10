"""GRU classifier over pose-feature sequences.

A deliberately small model (~50K parameters): behaviors like falling are
low-dimensional signals once poses are body-normalized, and a small recurrent
network trains in seconds and runs in microseconds — appropriate for an edge
safety system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from visionguard.temporal.features import FEATURE_DIM, featurize_sequence

CLASS_NAMES = ("walk", "bend", "fall")


class PoseSequenceClassifier(nn.Module):
    """GRU over (T, FEATURE_DIM) sequences -> behavior class logits."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(FEATURE_DIM, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, len(CLASS_NAMES))

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """(batch, T, FEATURE_DIM) -> (batch, num_classes) logits."""
        _, final_hidden = self.gru(sequences)
        return self.head(final_hidden[-1])

    # ------------------------------------------------------------------ #
    # Inference conveniences
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_keypoints(self, keypoint_sequence: np.ndarray) -> dict[str, float]:
        """(T, 17, 2) raw keypoints -> {class name: probability}."""
        self.eval()
        features = torch.from_numpy(
            featurize_sequence(keypoint_sequence)
        ).unsqueeze(0)
        probabilities = torch.softmax(self(features), dim=-1)[0]
        return {
            name: float(p) for name, p in zip(CLASS_NAMES, probabilities)
        }

    def save(self, path: Path | str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path | str) -> "PoseSequenceClassifier":
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu",
                                         weights_only=True))
        model.eval()
        return model
