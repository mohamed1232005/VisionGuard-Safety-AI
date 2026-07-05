"""Event persistence (SQLite). Feeds the dashboard, reports, and Phase 2 RAG."""

from visionguard.storage.event_store import EventStore, RunRecord

__all__ = ["EventStore", "RunRecord"]
