"""SQLite persistence layer for StudyMate.

Stores quiz scores, weak topics, study plans, and session metadata so that
student progress survives across Streamlit reruns and server restarts.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "studymate.db"


# ── helpers ──────────────────────────────────────────────────────────────────

@contextmanager
def _get_conn(db_path: Path = DB_PATH):
    """Yield a SQLite connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't exist yet."""
    with _get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                student_name TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quiz_scores (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                topic       TEXT NOT NULL,
                score       REAL NOT NULL,
                attempts    INTEGER NOT NULL DEFAULT 1,
                weak_areas  TEXT,  -- JSON list
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS weak_topics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                topic       TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS study_plans (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                plan_text   TEXT NOT NULL,
                approved    INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            """
        )


# ── session management ───────────────────────────────────────────────────────

def create_session(student_name: str, db_path: Path = DB_PATH) -> str:
    """Create a new session and return its UUID."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    name_to_save = student_name.strip() if student_name else "Student"
    with _get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, student_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, name_to_save, now, now),
        )
    return session_id


def get_session(session_id: str, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    """Return session row as dict, or ``None`` if not found."""
    with _get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def update_session_name(session_id: str, student_name: str, db_path: Path = DB_PATH) -> None:
    """Update the student name for an existing session."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET student_name = ?, updated_at = ? WHERE session_id = ?",
            (student_name, now, session_id),
        )


def list_sessions(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Return all sessions ordered by most recent."""
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ── quiz scores ──────────────────────────────────────────────────────────────

def save_quiz_score(
    session_id: str,
    topic: str,
    score: float,
    attempts: int = 1,
    weak_areas: list[str] | None = None,
    db_path: Path = DB_PATH,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO quiz_scores (session_id, topic, score, attempts, weak_areas, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, topic, score, attempts, json.dumps(weak_areas or []), now),
        )


def get_quiz_scores(session_id: str, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM quiz_scores WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
    results = []
    seen_topics = set()
    for r in rows:
        d = dict(r)
        topic = d["topic"]
        if topic not in seen_topics:
            seen_topics.add(topic)
            d["weak_areas"] = json.loads(d["weak_areas"]) if d["weak_areas"] else []
            results.append(d)
    return results


# ── weak topics ──────────────────────────────────────────────────────────────

def save_weak_topics(session_id: str, topics: list[str], db_path: Path = DB_PATH) -> None:
    """Replace weak topics for a session."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn(db_path) as conn:
        conn.execute("DELETE FROM weak_topics WHERE session_id = ?", (session_id,))
        conn.executemany(
            "INSERT INTO weak_topics (session_id, topic, created_at) VALUES (?, ?, ?)",
            [(session_id, t, now) for t in topics],
        )


def get_weak_topics(session_id: str, db_path: Path = DB_PATH) -> list[str]:
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT topic FROM weak_topics WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
    return [r["topic"] for r in rows]


# ── study plans ──────────────────────────────────────────────────────────────

def save_study_plan(session_id: str, plan_text: str, approved: bool = False, db_path: Path = DB_PATH) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO study_plans (session_id, plan_text, approved, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, plan_text, int(approved), now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def approve_study_plan(plan_id: int, db_path: Path = DB_PATH) -> None:
    with _get_conn(db_path) as conn:
        conn.execute(
            "UPDATE study_plans SET approved = 1 WHERE id = ?", (plan_id,)
        )


def get_latest_plan(session_id: str, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    with _get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM study_plans WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


# ── initialise on import ─────────────────────────────────────────────────────
init_db()
