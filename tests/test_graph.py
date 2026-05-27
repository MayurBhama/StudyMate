"""Tests for the LangGraph workflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.state import AgentState


class TestAgentState:
    """Verify the AgentState TypedDict schema."""

    def test_state_keys_exist(self):
        """AgentState should define all required keys."""
        required = {
            "messages", "student_name", "topic", "weak_topics",
            "quiz_score", "quiz_questions", "quiz_attempts", "quiz_answers",
            "pending_plan", "plan_approved", "session_id", "route",
            "rag_context", "response", "error",
        }
        annotations = AgentState.__annotations__
        assert required.issubset(set(annotations.keys())), (
            f"Missing keys: {required - set(annotations.keys())}"
        )

    def test_state_is_total_false(self):
        """AgentState(total=False) means all keys are optional at construction."""
        # Should not raise
        state: AgentState = {}  # type: ignore[typeddict-item]
        assert isinstance(state, dict)


class TestRouterNode:
    """Test the router_node classification logic."""

    @patch("agent.nodes._get_llm")
    def test_router_returns_valid_route(self, mock_llm_factory):
        """Router should return one of the valid route strings."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "explain"
        mock_llm.invoke.return_value = mock_response
        mock_llm_factory.return_value = mock_llm

        from langchain_core.messages import HumanMessage
        from agent.nodes import router_node

        state = {"messages": [HumanMessage(content="Explain photosynthesis")]}
        result = router_node(state)

        assert result["route"] in {"explain", "quiz", "study_plan", "rag_query"}

    @patch("agent.nodes._get_llm")
    def test_router_empty_messages(self, mock_llm_factory):
        """Router should default to 'explain' with no messages."""
        from agent.nodes import router_node

        result = router_node({"messages": []})
        assert result["route"] == "explain"


class TestExplainNode:
    """Test the explanation node."""

    @patch("agent.nodes.retrieve_as_text", return_value="")
    @patch("agent.nodes._get_llm")
    def test_explain_returns_response(self, mock_llm_factory, mock_rag):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Photosynthesis is the process…"
        mock_llm.invoke.return_value = mock_response
        mock_llm_factory.return_value = mock_llm

        from langchain_core.messages import HumanMessage
        from agent.nodes import explain_node

        state = {
            "messages": [HumanMessage(content="Explain photosynthesis")],
            "topic": "photosynthesis",
            "student_name": "Alice",
        }
        result = explain_node(state)

        assert "response" in result
        assert len(result["response"]) > 0
        assert "messages" in result


class TestQuizNode:
    """Test quiz generation and evaluation."""

    @patch("agent.nodes._get_llm")
    def test_quiz_generate_returns_questions(self, mock_llm_factory):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """{"questions": [
            {"question": "What is 2+2?", "options": ["A) 3", "B) 4", "C) 5", "D) 6"], "correct": "B", "explanation": "Basic math"},
            {"question": "What is 3+3?", "options": ["A) 5", "B) 6", "C) 7", "D) 8"], "correct": "B", "explanation": "Basic math"},
            {"question": "What is 4+4?", "options": ["A) 7", "B) 8", "C) 9", "D) 10"], "correct": "B", "explanation": "Basic math"},
            {"question": "What is 5+5?", "options": ["A) 9", "B) 10", "C) 11", "D) 12"], "correct": "B", "explanation": "Basic math"},
            {"question": "What is 6+6?", "options": ["A) 11", "B) 12", "C) 13", "D) 14"], "correct": "B", "explanation": "Basic math"}
        ]}"""
        mock_llm.invoke.return_value = mock_response
        # bind() returns a new object whose invoke also needs to return our response
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = mock_response
        mock_llm.bind.return_value = mock_bound
        mock_llm_factory.return_value = mock_llm

        from agent.nodes import quiz_generate_node

        state = {"topic": "math", "weak_topics": [], "quiz_attempts": 0}
        result = quiz_generate_node(state)

        assert "quiz_questions" in result
        assert len(result["quiz_questions"]) == 5
        assert result["quiz_attempts"] == 1


class TestGraphStructure:
    """Test that the graph compiles and has correct structure."""

    def test_graph_compiles(self):
        """The study_graph should compile without errors."""
        from agent.graph import build_graph

        graph = build_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        """The compiled graph should contain all expected nodes."""
        from agent.graph import build_graph

        graph = build_graph()
        # LangGraph compiled graph has a .get_graph() method
        graph_def = graph.get_graph()
        # Nodes may be a dict (node_id -> schema) or list of objects
        nodes = graph_def.nodes
        if isinstance(nodes, dict):
            node_ids = set(nodes.keys())
        else:
            # Try .id attribute, fallback to str
            node_ids = set()
            for n in nodes:
                node_ids.add(n.id if hasattr(n, 'id') else str(n))
        expected = {"router", "explain", "quiz_generate", "quiz_evaluate",
                    "study_plan", "study_plan_save", "rag_query"}
        # __start__ and __end__ are implicit
        assert expected.issubset(node_ids), f"Missing nodes: {expected - node_ids}"
