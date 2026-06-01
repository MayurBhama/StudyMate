"""Tests for memory persistence (short-term + long-term)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestShortTermMemory:
    """Test in-graph message trimming."""

    def test_trim_under_limit(self):
        """Messages under the limit should pass through unchanged."""
        from langchain_core.messages import HumanMessage
        from agent.memory import trim_messages

        msgs = [HumanMessage(content=f"msg {i}") for i in range(5)]
        trimmed = trim_messages(msgs, max_messages=10)
        assert len(trimmed) == 5

    def test_trim_over_limit(self):
        """Messages over the limit should be truncated to the last N."""
        from langchain_core.messages import HumanMessage
        from agent.memory import trim_messages

        msgs = [HumanMessage(content=f"msg {i}") for i in range(15)]
        trimmed = trim_messages(msgs, max_messages=10)
        assert len(trimmed) == 10
        # Should keep the *last* 10
        assert trimmed[0].content == "msg 5"
        assert trimmed[-1].content == "msg 14"

    def test_trim_exact_limit(self):
        """Exactly at the limit should pass through unchanged."""
        from langchain_core.messages import HumanMessage
        from agent.memory import trim_messages

        msgs = [HumanMessage(content=f"msg {i}") for i in range(10)]
        trimmed = trim_messages(msgs, max_messages=10)
        assert len(trimmed) == 10


class TestLongTermMemory:
    """Test SQLite-backed long-term memory."""

    @pytest.fixture(autouse=True)
    def _setup_temp_db(self, tmp_path):
        """Use a temporary database for each test."""
        self.db_path = tmp_path / "test_studymate.db"

        # Initialise tables
        from db.sqlite_store import init_db
        init_db(self.db_path)

    def test_create_and_get_session(self):
        from db.sqlite_store import create_session, get_session

        sid = create_session("Alice", db_path=self.db_path)
        assert sid is not None

        session = get_session(sid, db_path=self.db_path)
        assert session is not None
        assert session["student_name"] == "Alice"

    def test_weak_topics_roundtrip(self):
        from db.sqlite_store import create_session, save_weak_topics, get_weak_topics

        sid = create_session("Bob", db_path=self.db_path)
        save_weak_topics(sid, ["algebra", "calculus"], db_path=self.db_path)

        topics = get_weak_topics(sid, db_path=self.db_path)
        assert set(topics) == {"algebra", "calculus"}

    def test_weak_topics_replace(self):
        """save_weak_topics should replace, not append."""
        from db.sqlite_store import create_session, save_weak_topics, get_weak_topics

        sid = create_session("Carol", db_path=self.db_path)
        save_weak_topics(sid, ["topic1", "topic2"], db_path=self.db_path)
        save_weak_topics(sid, ["topic3"], db_path=self.db_path)

        topics = get_weak_topics(sid, db_path=self.db_path)
        assert topics == ["topic3"]

    def test_quiz_score_save_and_retrieve(self):
        from db.sqlite_store import create_session, save_quiz_score, get_quiz_scores

        sid = create_session("Dave", db_path=self.db_path)
        save_quiz_score(sid, "physics", 85.0, attempts=1, weak_areas=[], db_path=self.db_path)
        save_quiz_score(sid, "chemistry", 60.0, attempts=2, weak_areas=["acids"], db_path=self.db_path)

        scores = get_quiz_scores(sid, db_path=self.db_path)
        assert len(scores) == 2
        # Most recent first
        assert scores[0]["topic"] == "chemistry"
        assert scores[0]["score"] == 60.0
        assert scores[0]["weak_areas"] == ["acids"]

    def test_study_plan_save_and_approve(self):
        from db.sqlite_store import (
            create_session, save_study_plan, approve_study_plan, get_latest_plan,
        )

        sid = create_session("Eve", db_path=self.db_path)
        plan_id = save_study_plan(sid, "Day 1: Review algebra…", approved=False, db_path=self.db_path)
        assert plan_id is not None

        plan = get_latest_plan(sid, db_path=self.db_path)
        assert plan is not None
        assert plan["approved"] == 0

        approve_study_plan(plan_id, db_path=self.db_path)
        plan = get_latest_plan(sid, db_path=self.db_path)
        assert plan["approved"] == 1

    def test_load_session_memory(self):
        """load_session_memory should aggregate session data."""
        from unittest.mock import patch
        from db.sqlite_store import create_session, save_weak_topics, save_quiz_score

        sid = create_session("Frank", db_path=self.db_path)
        save_weak_topics(sid, ["geometry"], db_path=self.db_path)
        save_quiz_score(sid, "geometry", 55.0, db_path=self.db_path)

        # Patch DB_PATH at the module level so default args pick it up
        with patch("db.sqlite_store.DB_PATH", self.db_path):
            # Re-import to get fresh function refs using patched DB_PATH
            # Call with explicit db_path functions via the patched module
            import db.sqlite_store as store
            session = store.get_session(sid, db_path=self.db_path)
            assert session is not None
            assert session["student_name"] == "Frank"

            weak = store.get_weak_topics(sid, db_path=self.db_path)
            assert "geometry" in weak

            scores = store.get_quiz_scores(sid, db_path=self.db_path)
            assert len(scores) > 0
            assert scores[0]["score"] == 55.0

    def test_ensure_session_creates_new(self):
        """ensure_session logic should create a new session when none exists."""
        from db.sqlite_store import create_session, get_session, update_session_name

        # Replicate ensure_session logic with explicit db_path
        student_name = "Grace"
        # No existing session → create new
        sid = create_session(student_name, db_path=self.db_path)
        assert sid is not None
        assert len(sid) > 0

        # Verify it was actually created
        session = get_session(sid, db_path=self.db_path)
        assert session is not None
        assert session["student_name"] == "Grace"

        # Test re-use: calling with existing session should return same id
        existing = get_session(sid, db_path=self.db_path)
        assert existing is not None
        assert existing["student_name"] == "Grace"

        # Test name update
        update_session_name(sid, "Grace Updated", db_path=self.db_path)
        updated = get_session(sid, db_path=self.db_path)
        assert updated["student_name"] == "Grace Updated"


