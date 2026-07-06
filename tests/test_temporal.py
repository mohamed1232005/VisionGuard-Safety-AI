"""Tests for the temporal behavior model stack (features, synthetic data, model)."""

import numpy as np
import pytest
import torch

from visionguard.temporal.features import FEATURE_DIM, featurize_frame, featurize_sequence
from visionguard.temporal.model import CLASS_NAMES, PoseSequenceClassifier
from visionguard.temporal.synthetic import SEQUENCE_LENGTH, generate_dataset, generate_sequence


RNG = np.random.default_rng(3)


class TestFeatures:
    def test_translation_invariance(self) -> None:
        """The same pose anywhere in the image gives identical features."""
        pose = generate_sequence("walk", np.random.default_rng(1))[0]
        shifted = pose + np.array([250.0, -40.0])
        np.testing.assert_allclose(
            featurize_frame(pose), featurize_frame(shifted), atol=1e-4
        )

    def test_scale_invariance(self) -> None:
        """A person twice as close (2x pixels) gives identical features."""
        pose = generate_sequence("walk", np.random.default_rng(1))[0]
        np.testing.assert_allclose(
            featurize_frame(pose), featurize_frame(pose * 2.0), atol=1e-4
        )

    def test_sequence_shape(self) -> None:
        seq = generate_sequence("fall", RNG)
        assert featurize_sequence(seq).shape == (SEQUENCE_LENGTH, FEATURE_DIM)


class TestSyntheticData:
    def test_dataset_is_balanced_and_shaped(self) -> None:
        x, y = generate_dataset(samples_per_class=5, seed=1)
        assert x.shape == (15, SEQUENCE_LENGTH, 17, 2)
        assert [int((y == label).sum()) for label in range(3)] == [5, 5, 5]

    def test_fall_ends_down_walk_stays_up(self) -> None:
        """Sanity: hips end much lower (larger y) in falls than in walks."""
        fall = generate_sequence("fall", np.random.default_rng(5))
        walk = generate_sequence("walk", np.random.default_rng(5))
        hip_drop = lambda seq: seq[-1, 11, 1] - seq[0, 11, 1]  # noqa: E731
        assert hip_drop(fall) > hip_drop(walk) + 10

    def test_unknown_behavior_rejected(self) -> None:
        with pytest.raises(ValueError):
            generate_sequence("moonwalk", RNG)


class TestModel:
    def test_forward_shape(self) -> None:
        model = PoseSequenceClassifier()
        logits = model(torch.randn(4, SEQUENCE_LENGTH, FEATURE_DIM))
        assert logits.shape == (4, len(CLASS_NAMES))

    def test_predict_keypoints_returns_probabilities(self) -> None:
        model = PoseSequenceClassifier()
        probs = model.predict_keypoints(generate_sequence("bend", RNG))
        assert set(probs) == set(CLASS_NAMES)
        assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_save_and_load_round_trip(self, tmp_path) -> None:
        model = PoseSequenceClassifier()
        model.save(tmp_path / "m.pt")
        restored = PoseSequenceClassifier.load(tmp_path / "m.pt")
        x = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM)
        with torch.no_grad():
            torch.testing.assert_close(model(x), restored(x))

    def test_learns_to_separate_behaviors_quickly(self) -> None:
        """A tiny training run must already beat random (33%) by a wide margin —
        proves the features carry the signal, not just the model capacity."""
        from visionguard.temporal.features import featurize_sequence

        x, y = generate_dataset(samples_per_class=30, seed=11)
        features = torch.from_numpy(
            np.stack([featurize_sequence(s) for s in x])
        )
        labels = torch.from_numpy(y)

        model = PoseSequenceClassifier()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(30):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(features), labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            accuracy = float((model(features).argmax(-1) == labels).float().mean())
        assert accuracy > 0.8
