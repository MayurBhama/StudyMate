"""AgentState TypedDict — central state schema for the LangGraph workflow."""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Shared state that flows through every node in the StudyMate graph.

    Keys
    ----
    messages : list
        Conversation history (managed by LangGraph's ``add_messages`` reducer).
    student_name : str
        Name of the current student (persisted across sessions).
    topic : str
        The topic the student is currently studying.
    weak_topics : list[str]
        Topics the student scored poorly on (loaded from / saved to SQLite).
    quiz_score : float
        Latest quiz score (0‑100).
    quiz_questions : list[dict]
        Current batch of MCQ questions.
    quiz_attempts : int
        Number of quiz attempts in the current cycle.
    quiz_answers : list[str]
        Student's answers to the current quiz.
    pending_plan : str
        Study plan awaiting Human‑in‑the‑Loop approval.
    plan_approved : bool | None
        ``True`` / ``False`` after HITL decision; ``None`` while waiting.
    session_id : str
        Unique session identifier (UUID stored in Streamlit session_state).
    route : str
        Routing decision made by the router node.
    rag_context : str
        Retrieved context from the RAG pipeline.
    response : str
        Final response to display in the UI.
    error : str
        Error message (if any) for graceful degradation.
    """

    messages: Annotated[list, add_messages]
    student_name: str  # default is "Student"
    topic: str
    weak_topics: list[str]
    quiz_score: float
    quiz_questions: list[dict[str, Any]]
    quiz_attempts: int
    quiz_answers: list[str]
    pending_plan: str
    plan_approved: bool | None
    session_id: str
    route: str
    rag_context: str
    response: str
    error: str
