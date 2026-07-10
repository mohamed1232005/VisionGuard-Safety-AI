"""Semantic index over safety events (sentence-transformers + FAISS).

Each event's description (plus its metadata rendered as text) is embedded once;
questions are embedded at query time and matched by cosine similarity. This is
the "R" in RAG — it finds the incidents relevant to a question even when the
wording differs ("worker not wearing head protection" matches "without helmet").
"""

from __future__ import annotations

import logging

import numpy as np

from visionguard.safety.events import SafetyEvent

logger = logging.getLogger(__name__)


def event_to_text(event: SafetyEvent) -> str:
    """Render an event as the text that gets embedded and retrieved."""
    parts = [
        event.description,
        f"type: {event.event_type.value.replace('_', ' ')}",
        f"severity: {event.severity.value}",
        f"at video time {event.timestamp_str()}",
    ]
    if event.zone_name:
        parts.append(f"zone: {event.zone_name}")
    return " | ".join(parts)


class EventIndexer:
    """Builds and queries a FAISS index over one run's events."""

    def __init__(self, embedding_model: str) -> None:
        # Heavy imports kept local so the rest of the app never pays for them.
        from sentence_transformers import SentenceTransformer

        self._encoder = SentenceTransformer(embedding_model)
        self._index = None
        self._events: list[SafetyEvent] = []
        logger.info("Loaded embedding model %s", embedding_model)

    def build(self, events: list[SafetyEvent]) -> None:
        """(Re)build the index from a run's events."""
        import faiss

        self._events = list(events)
        if not self._events:
            self._index = None
            return

        texts = [event_to_text(e) for e in self._events]
        embeddings = self._encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        # Inner product on normalized vectors == cosine similarity.
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        logger.info("Indexed %d events for retrieval", len(self._events))

    def search(self, question: str, top_k: int) -> list[tuple[SafetyEvent, float]]:
        """Most relevant events for a question, with similarity scores."""
        if self._index is None or not self._events:
            return []
        query = self._encoder.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        scores, indices = self._index.search(query, min(top_k, len(self._events)))
        return [
            (self._events[i], float(score))
            for i, score in zip(indices[0], scores[0])
            if i >= 0
        ]
