"""Main LangGraph graph definition for StudyMate.

Builds a ``StateGraph`` with conditional routing from the router node
to explain / quiz / study_plan / rag_query sub-workflows.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    router_node,
    explain_node,
    quiz_generate_node,
    quiz_evaluate_node,
    study_plan_node,
    study_plan_save_node,
    rag_query_node,
)


def _route_decision(state: AgentState) -> str:
    """Return the next node name based on the router's classification."""
    return state.get("route", "explain")


def _quiz_should_retry(state: AgentState) -> str:
    """After quiz evaluation, decide whether to retry or finish."""
    score = state.get("quiz_score", 0)
    attempts = state.get("quiz_attempts", 0)
    if score < 70 and attempts < 3:
        return "quiz_generate"
    return END


def build_graph() -> StateGraph:
    """Construct and return the compiled StudyMate agent graph."""
    graph = StateGraph(AgentState)

    # ── Add nodes ────────────────────────────────────────────────────────
    graph.add_node("router", router_node)
    graph.add_node("explain", explain_node)
    graph.add_node("quiz_generate", quiz_generate_node)
    graph.add_node("quiz_evaluate", quiz_evaluate_node)
    graph.add_node("study_plan", study_plan_node)
    graph.add_node("study_plan_save", study_plan_save_node)
    graph.add_node("rag_query", rag_query_node)

    # ── Entry point ──────────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Conditional routing from router ──────────────────────────────────
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {
            "explain": "explain",
            "quiz": "quiz_generate",
            "study_plan": "study_plan",
            "rag_query": "rag_query",
        },
    )

    # ── Terminal edges ───────────────────────────────────────────────────
    graph.add_edge("explain", END)
    graph.add_edge("rag_query", END)

    # ── Quiz iterative loop ──────────────────────────────────────────────
    # quiz_generate → END (waits for student answers via Streamlit)
    # quiz_evaluate → retry or END
    graph.add_edge("quiz_generate", END)
    graph.add_conditional_edges(
        "quiz_evaluate",
        _quiz_should_retry,
        {
            "quiz_generate": "quiz_generate",
            END: END,
        },
    )

    # ── Study plan HITL ──────────────────────────────────────────────────
    # study_plan → study_plan_save
    graph.add_edge("study_plan", "study_plan_save")
    graph.add_edge("study_plan_save", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory, interrupt_before=["study_plan_save"])


# Pre-built graph instance for import convenience
study_graph = build_graph()
