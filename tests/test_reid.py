"""Tests for the Re-ID gallery (synthetic embeddings; no model)."""

import numpy as np

from visionguard.reid.gallery import ReIDGallery


def vec(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_same_appearance_across_cameras_shares_identity() -> None:
    gallery = ReIDGallery(similarity_threshold=0.8)
    person = vec(1, 0, 0)

    gid_a = gallery.assign("camera-A", 1, person)
    gid_b = gallery.assign("camera-B", 7, person)   # different camera + track id

    assert gid_a == gid_b
    assert gallery.identity_count == 1
    assert gallery.summary()[0]["cameras"] == ["camera-A", "camera-B"]


def test_different_appearances_get_different_identities() -> None:
    gallery = ReIDGallery(similarity_threshold=0.8)
    assert gallery.assign("camera-A", 1, vec(1, 0, 0)) != gallery.assign(
        "camera-A", 2, vec(0, 1, 0)
    )
    assert gallery.identity_count == 2


def test_binding_is_sticky_even_if_appearance_drifts() -> None:
    """Once a track is bound, later noisy embeddings don't reassign it."""
    gallery = ReIDGallery(similarity_threshold=0.8)
    gid = gallery.assign("camera-A", 1, vec(1, 0, 0))
    drifted = vec(0.6, 0.8, 0)  # similarity 0.6 — below threshold

    assert gallery.assign("camera-A", 1, drifted) == gid
    assert gallery.identity_count == 1


def test_below_threshold_creates_new_identity() -> None:
    gallery = ReIDGallery(similarity_threshold=0.9)
    gallery.assign("camera-A", 1, vec(1, 0, 0))
    gid2 = gallery.assign("camera-B", 1, vec(0.85, np.sqrt(1 - 0.85**2), 0))

    assert gid2 == 2  # similarity 0.85 < 0.9 threshold


def test_centroid_absorbs_observations() -> None:
    """The centroid moves toward the mean of everything it has seen."""
    gallery = ReIDGallery(similarity_threshold=0.8)
    gallery.assign("camera-A", 1, vec(1, 0.1, 0))
    gallery.assign("camera-A", 1, vec(1, -0.1, 0))

    # A third camera's clean view still matches the refined centroid.
    assert gallery.assign("camera-C", 4, vec(1, 0, 0)) == 1
    assert gallery.summary()[0]["observations"] == 3
