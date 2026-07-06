"""The AI Safety Assistant: natural-language Q&A over the incident history.

Architecture (classic RAG with a structured-facts twist):

    question ──▶ 1. structured facts (SQL aggregates: counts, compliance, zones)
             ──▶ 2. semantic retrieval (FAISS over event descriptions)
             ──▶ 3. answer synthesis:
                    - Claude (Anthropic API) when credentials are available
                    - deterministic template otherwise (retrieval-only mode)

The structured facts matter: questions like "how many helmet violations?"
deserve an exact number from SQL, not an LLM guess from retrieved snippets.
The LLM's job is synthesis and phrasing, grounded in both.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from visionguard.safety.events import SafetyEvent
from visionguard.storage.event_store import EventStore, RunRecord
from visionguard.utils.config import AssistantSettings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are VisionGuard's safety assistant. You answer questions from safety "
    "managers about incidents detected by a computer-vision monitoring system "
    "on their worksite video.\n"
    "Ground every answer in the FACTS and INCIDENTS provided — never invent "
    "numbers or incidents. Quote exact counts from FACTS when the question is "
    "quantitative. Mention video timestamps (MM:SS) when referencing specific "
    "incidents. If the data cannot answer the question, say so plainly. "
    "Be concise and operational: a safety manager reads this on site."
)


@dataclass
class AssistantAnswer:
    """An answer plus the evidence behind it."""

    text: str
    sources: list[SafetyEvent] = field(default_factory=list)
    used_llm: bool = False


def _facts_block(run: RunRecord, counts: dict[str, int]) -> str:
    """Render the structured facts handed to the LLM (and to the fallback)."""
    stats = run.stats
    lines = [
        f"video: {run.video_source}",
        f"analyzed_at_utc: {run.started_at[:19]}",
        f"video_duration_seconds: {run.duration_seconds or 0:.0f}",
        f"ppe_compliance_rate: {(run.compliance_rate or 0) * 100:.1f}%",
        f"total_events: {sum(counts.values())}",
    ]
    for event_type, count in sorted(counts.items()):
        lines.append(f"events.{event_type}: {count}")
    lines.append(
        f"ppe_violations_by_item: {stats.get('ppe_violations_by_item') or {}}"
    )
    lines.append(f"zone_intrusions: {stats.get('zone_intrusions') or {}}")
    lines.append(
        f"most_dangerous_zone: {stats.get('most_dangerous_zone') or 'none'}"
    )
    lines.append(f"falls_detected: {stats.get('falls_detected', 0)}")
    proximity = stats.get("proximity")
    if proximity:
        lines.append(f"proximity_near_misses: {proximity.get('near_misses', 0)}")
        lines.append(f"min_worker_vehicle_distance_m: {proximity.get('min_distance_m')}")
    risk = stats.get("risk_score") or {}
    lines.append(f"risk_score_peak: {risk.get('peak', 0)} at t={risk.get('peak_time', 0)}s")
    unique = stats.get("unique_counts") or {}
    lines.append(f"workers_observed: {unique.get('worker', 0)}")
    lines.append(
        f"vehicles_machinery_observed: "
        f"{unique.get('vehicle', 0) + unique.get('machinery', 0)}"
    )
    return "\n".join(lines)


class SafetyAssistant:
    """Answers natural-language questions about a run's incident history."""

    def __init__(self, store: EventStore, settings: AssistantSettings) -> None:
        self._store = store
        self._settings = settings
        self._indexer = None          # built lazily (downloads a model once)
        self._indexed_run: int | None = None

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def _retrieve(self, question: str, run_id: int) -> list[tuple[SafetyEvent, float]]:
        """Semantic search over the run's events; empty on indexing failure."""
        try:
            if self._indexer is None:
                from visionguard.assistant.indexer import EventIndexer

                self._indexer = EventIndexer(self._settings.embedding_model)
            if self._indexed_run != run_id:
                self._indexer.build(self._store.events_for_run(run_id))
                self._indexed_run = run_id
            return self._indexer.search(question, self._settings.top_k)
        except Exception:  # retrieval is an enhancement, never a hard failure
            logger.exception("Event retrieval failed; answering from facts only")
            return []

    # ------------------------------------------------------------------ #
    # Answering
    # ------------------------------------------------------------------ #
    def answer(self, question: str, run_id: int) -> AssistantAnswer:
        """Answer a question about one analysis run."""
        run = self._store.get_run(run_id)
        if run is None:
            return AssistantAnswer(text=f"Run {run_id} does not exist.")

        counts = self._store.event_type_counts(run_id)
        facts = _facts_block(run, counts)
        matches = self._retrieve(question, run_id)
        sources = [event for event, _ in matches]

        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return self._answer_with_claude(question, facts, matches, sources)
            except Exception:
                logger.exception("Claude call failed; falling back to composed answer")

        return self._composed_answer(question, run, counts, facts, matches)

    def _answer_with_claude(
        self,
        question: str,
        facts: str,
        matches: list[tuple[SafetyEvent, float]],
        sources: list[SafetyEvent],
    ) -> AssistantAnswer:
        """Synthesize a grounded answer with the Anthropic API."""
        import anthropic  # local import: only needed when a key is configured

        incidents = "\n".join(
            f"- [{event.timestamp_str()}] ({event.severity.value}) "
            f"{event.description}"
            for event, _ in matches
        ) or "(no matching incidents)"

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self._settings.model,
            max_tokens=self._settings.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"FACTS:\n{facts}\n\n"
                        f"INCIDENTS (most relevant to the question):\n{incidents}\n\n"
                        f"QUESTION: {question}"
                    ),
                }
            ],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return AssistantAnswer(text=text, sources=sources, used_llm=True)

    @staticmethod
    def _composed_answer(
        question: str,
        run: RunRecord,
        counts: dict[str, int],
        facts: str,
        matches: list[tuple[SafetyEvent, float]],
    ) -> AssistantAnswer:
        """Deterministic answer when no LLM is available (retrieval-only mode)."""
        lines: list[str] = []
        total = sum(counts.values())
        lines.append(
            f"Run {run.id} summary: {total} safety event(s), PPE compliance "
            f"{(run.compliance_rate or 0) * 100:.1f}%."
        )
        if counts:
            readable = ", ".join(
                f"{count}x {event_type.replace('_', ' ')}"
                for event_type, count in sorted(counts.items())
            )
            lines.append(f"By type: {readable}.")
        if matches:
            lines.append("Incidents most relevant to your question:")
            for event, _ in matches:
                lines.append(
                    f"  - [{event.timestamp_str()}] ({event.severity.value}) "
                    f"{event.description}"
                )
        else:
            lines.append("No recorded incidents match your question.")
        lines.append(
            "(Set ANTHROPIC_API_KEY to enable full natural-language answers.)"
        )
        return AssistantAnswer(
            text="\n".join(lines),
            sources=[event for event, _ in matches],
            used_llm=False,
        )
