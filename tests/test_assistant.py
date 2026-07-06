"""Tests for the safety assistant (facts + fallback path; no network/LLM)."""

from pathlib import Path

import pytest

from visionguard.assistant.assistant import SafetyAssistant, _facts_block
from visionguard.safety.events import EventType, SafetyEvent
from visionguard.storage.event_store import EventStore
from visionguard.utils.config import AssistantSettings

SETTINGS = AssistantSettings(
    model="claude-opus-4-8",
    max_tokens=512,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    top_k=3,
)


@pytest.fixture()
def store_with_run(tmp_path: Path, monkeypatch) -> tuple[EventStore, int]:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = EventStore(tmp_path / "events.db")
    run_id = store.create_run("site.mp4")
    for t, description in [
        (4.0, "Worker #3 detected without helmet"),
        (9.0, "Worker #5 entered restricted zone 'Crane area'"),
    ]:
        store.add_event(
            run_id,
            SafetyEvent(
                event_type=(
                    EventType.PPE_VIOLATION if "helmet" in description
                    else EventType.ZONE_INTRUSION
                ),
                frame_index=int(t * 30),
                video_time=t,
                track_id=3,
                track_label="Worker #3",
                confidence=0.9,
                description=description,
                zone_name="Crane area" if "zone" in description else None,
            ),
        )
    store.finish_run(
        run_id, fps=30.0, frames_processed=900, duration_seconds=30.0,
        compliance_rate=0.9,
        stats={
            "unique_counts": {"worker": 5},
            "most_dangerous_zone": "Crane area",
            "falls_detected": 0,
            "risk_score": {"peak": 20.0, "peak_time": 9.0},
        },
    )
    return store, run_id


class _NoRetrievalAssistant(SafetyAssistant):
    """Assistant with retrieval stubbed out (no embedding-model download)."""

    def _retrieve(self, question: str, run_id: int):
        return []


def test_facts_block_contains_exact_counts(store_with_run) -> None:
    store, run_id = store_with_run
    facts = _facts_block(store.get_run(run_id), store.event_type_counts(run_id))
    assert "events.ppe_violation: 1" in facts
    assert "events.zone_intrusion: 1" in facts
    assert "ppe_compliance_rate: 90.0%" in facts
    assert "most_dangerous_zone: Crane area" in facts


def test_fallback_answer_without_api_key(store_with_run) -> None:
    store, run_id = store_with_run
    assistant = _NoRetrievalAssistant(store, SETTINGS)

    answer = assistant.answer("How many violations happened?", run_id)

    assert not answer.used_llm
    assert "2 safety event(s)" in answer.text
    assert "90.0%" in answer.text


def test_unknown_run_answers_gracefully(store_with_run) -> None:
    store, _ = store_with_run
    assistant = _NoRetrievalAssistant(store, SETTINGS)
    answer = assistant.answer("anything", run_id=999)
    assert "does not exist" in answer.text


def test_retrieval_failure_never_crashes(store_with_run, monkeypatch) -> None:
    """A broken embedding stack degrades to facts-only, not an exception."""

    class _BrokenIndexer:
        def __init__(self, *args: object) -> None:
            raise RuntimeError("embedding model unavailable")

    import visionguard.assistant.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "EventIndexer", _BrokenIndexer)

    store, run_id = store_with_run
    assistant = SafetyAssistant(store, SETTINGS)  # real _retrieve, broken index
    answer = assistant.answer("summarize the risks", run_id)

    assert answer.text                      # still answered from facts
    assert answer.sources == []             # retrieval contributed nothing
    assert not answer.used_llm
