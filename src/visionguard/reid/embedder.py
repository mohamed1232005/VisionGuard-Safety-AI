"""Appearance embeddings for person crops (CLIP image encoder).

A person's crop is embedded into a normalized vector; the same person produces
nearby vectors even from a different camera angle, while different people land
farther apart. Cosine similarity between embeddings is the matching signal.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AppearanceEmbedder:
    """Embeds BGR person crops into L2-normalized appearance vectors."""

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        from PIL import Image  # noqa: F401 - ensure dependency present early
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        logger.info("Loaded appearance embedding model %s", model_name)

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of BGR crops -> (n, d) normalized float32 vectors."""
        from PIL import Image

        images = [
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            for crop in crops_bgr
        ]
        embeddings = self._model.encode(
            images, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(embeddings, dtype=np.float32)
