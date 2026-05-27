"""Short-term and long-term memory handlers for StudyMate.

Short-term : in-graph ``messages`` list (capped to the last 10 messages).
Long-term  : SQLite-backed weak topics, quiz scores, and student name.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from db.sqlite_store import (
    create_session,
    get_session,
    get_weak_topics,
    save_weak_topics,
    get_quiz_scores,
    update_session_name,
)

MAX_SHORT_TERM_MESSAGES = 10


# ── short-term memory ───────────────────────────────────────────────────────

def trim_messages(messages: list[BaseMessage], max_messages: int = MAX_SHORT_TERM_MESSAGES) -> list[BaseMessage]:
    """Keep only the last *max_messages* messages to bound context size."""
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


# ── long-term memory ────────────────────────────────────────────────────────

def load_session_memory(session_id: str) -> dict[str, Any]:
    """Load persisted state for *session_id* from SQLite.

    Returns a dict with keys that can be merged directly into ``AgentState``.
    """
    session = get_session(session_id)
    if session is None:
        return {}

    weak = get_weak_topics(session_id)
    scores = get_quiz_scores(session_id)
    latest_score = scores[0]["score"] if scores else 0.0

    return {
        "student_name": session["student_name"],
        "weak_topics": weak,
        "quiz_score": latest_score,
        "session_id": session_id,
    }


def persist_weak_topics(session_id: str, topics: list[str]) -> None:
    """Write weak topics to SQLite for long-term recall."""
    save_weak_topics(session_id, topics)


def ensure_session(student_name: str, session_id: str | None = None) -> str:
    """Return an existing session_id or create a new one."""
    if session_id:
        session = get_session(session_id)
        if session:
            # Update name if changed
            if session["student_name"] != student_name:
                update_session_name(session_id, student_name)
            return session_id
    return create_session(student_name)
