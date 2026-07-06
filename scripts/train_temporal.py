"""Train the pose-sequence behavior classifier on procedural data.

Usage:
    python scripts/train_temporal.py                # ~30 s, saves the model
    python scripts/train_temporal.py --samples 500  # more data per class

Outputs models/temporal_behavior_gru.pt plus a per-class accuracy report on a
held-out set. Swap generate_dataset() for real labeled clips to retrain on
actual footage — the featurization and model are unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from visionguard.temporal.features import featurize_sequence
from visionguard.temporal.model import CLASS_NAMES, PoseSequenceClassifier
from visionguard.temporal.synthetic import generate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the behavior model")
    parser.add_argument("--samples", type=int, default=400,
                        help="training samples per class")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--output", default=PROJECT_ROOT / "models" /
                        "temporal_behavior_gru.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # ---- Data ---------------------------------------------------------- #
    train_x, train_y = generate_dataset(args.samples, seed=7)
    test_x, test_y = generate_dataset(max(args.samples // 5, 40), seed=99)

    def to_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            np.stack([featurize_sequence(seq) for seq in x])
        ).to(device)

    train_features, test_features = to_tensor(train_x), to_tensor(test_x)
    train_labels = torch.from_numpy(train_y).to(device)
    test_labels = torch.from_numpy(test_y).to(device)

    # ---- Train ----------------------------------------------------------#
    model = PoseSequenceClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(len(train_features))
        total_loss = 0.0
        for start in range(0, len(permutation), 64):
            batch = permutation[start:start + 64]
            optimizer.zero_grad()
            loss = loss_fn(model(train_features[batch]), train_labels[batch])
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(batch)
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1:>3}: loss {total_loss / len(train_features):.4f}")

    # ---- Evaluate ------------------------------------------------------ #
    model.eval()
    with torch.no_grad():
        predictions = model(test_features).argmax(dim=-1)
    accuracy = float((predictions == test_labels).float().mean())
    print(f"\nHeld-out accuracy: {accuracy:.1%}")
    for label, name in enumerate(CLASS_NAMES):
        mask = test_labels == label
        class_accuracy = float((predictions[mask] == label).float().mean())
        print(f"  {name:>5}: {class_accuracy:.1%} "
              f"({int(mask.sum())} samples)")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.cpu().save(output)
    print(f"\nModel saved to {output}")


if __name__ == "__main__":
    main()
