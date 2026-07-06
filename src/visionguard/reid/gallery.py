"""Identity gallery: appearance embeddings -> persistent global IDs.

Each global identity keeps a running-mean appearance centroid. A new track
(from any camera) is matched against all centroids by cosine similarity:
above the threshold it *joins* the best-matching identity (same worker seen
again), below it a new identity is created. Once a (camera, track) pair is
bound to an identity it stays bound — trackers already guarantee within-camera
consistency, so Re-ID only decides the cross-camera / re-appearance question.

Pure numpy: fully unit-testable with synthetic vectors, no model needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _Identity:
    global_id: int
    centroid: np.ndarray          # L2-normalized running mean
    observations: int = 1
    cameras: set[str] = field(default_factory=set)


class ReIDGallery:
    """Assigns persistent global IDs from appearance embeddings."""

    def __init__(self, similarity_threshold: float) -> None:
        self._threshold = similarity_threshold
        self._identities: list[_Identity] = []
        self._bindings: dict[tuple[str, int], int] = {}  # (camera, track) -> gid
        self._next_id = 1

    @property
    def identity_count(self) -> int:
        return len(self._identities)

    def _best_match(self, embedding: np.ndarray) -> tuple[_Identity | None, float]:
        best, best_score = None, self._threshold
        for identity in self._identities:
            score = float(np.dot(identity.centroid, embedding))
            if score >= best_score:
                best, best_score = identity, score
        return best, best_score

    def assign(self, camera: str, track_id: int, embedding: np.ndarray) -> int:
        """Global ID for a track given one appearance embedding.

        Args:
            camera: Camera/stream name (any stable string).
            track_id: The tracker's per-camera track ID.
            embedding: L2-normalized appearance vector.
        """
        embedding = np.asarray(embedding, dtype=np.float32)
        key = (camera, track_id)

        # Sticky: a bound track keeps its identity and refines the centroid.
        if key in self._bindings:
            identity = self._identities[self._bindings[key] - 1]
            self._update(identity, embedding)
            return identity.global_id

        identity, score = self._best_match(embedding)
        if identity is None:
            identity = _Identity(
                global_id=self._next_id, centroid=embedding.copy()
            )
            self._identities.append(identity)
            self._next_id += 1
            logger.debug("New identity #%d from %s/track %d", identity.global_id,
                         camera, track_id)
        else:
            self._update(identity, embedding)
            logger.info(
                "Re-identified: %s/track %d matches global #%d (similarity %.2f)",
                camera, track_id, identity.global_id, score,
            )

        identity.cameras.add(camera)
        self._bindings[key] = identity.global_id
        return identity.global_id

    @staticmethod
    def _update(identity: _Identity, embedding: np.ndarray) -> None:
        """Fold a new observation into the identity's centroid."""
        n = identity.observations
        merged = (identity.centroid * n + embedding) / (n + 1)
        norm = np.linalg.norm(merged)
        identity.centroid = merged / norm if norm > 0 else merged
        identity.observations = n + 1

    def summary(self) -> list[dict]:
        """Per-identity overview (for reports and the demo script)."""
        return [
            {
                "global_id": identity.global_id,
                "observations": identity.observations,
                "cameras": sorted(identity.cameras),
            }
            for identity in self._identities
        ]
