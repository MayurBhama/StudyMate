"""StudyMate — Streamlit UI entry point.

A personalized AI study agent with chat, quizzes, study plans, and RAG-powered
note search.  Powered by LangGraph + Groq + ChromaDB.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

# LangSmith observability
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "studymate")

# ── Local imports (after dotenv) ─────────────────────────────────────────────
from agent.graph import study_graph  # noqa: E402
from agent.memory import ensure_session, load_session_memory, trim_messages  # noqa: E402
from db.sqlite_store import (  # noqa: E402
    get_weak_topics,
    get_quiz_scores,
    get_latest_plan,
    list_sessions,
    save_study_plan,
    approve_study_plan,
    init_db,
)
from rag.retriever import ingest_pdf, collection_count  # noqa: E402

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyMate — AI Study Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa;
}
section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}

/* ── Cards / containers ─────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    border: 1px solid rgba(167, 139, 250, .25);
    border-radius: 12px;
    background: rgba(15, 12, 41, .35);
    backdrop-filter: blur(6px);
}

/* ── Quiz radio buttons ─────────────────────────────────────────────── */
.stRadio > div {
    gap: 0.25rem;
}

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(167, 139, 250, .3);
}

/* ── Chat messages ──────────────────────────────────────────────────── */
div[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 0.5rem;
}

/* ── Metric cards ───────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(167,139,250,.1), rgba(99,102,241,.1));
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid rgba(167, 139, 250, .2);
}

/* ── Progress bars ──────────────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #a78bfa, #6366f1);
    border-radius: 10px;
}

/* ── Smooth animations ──────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.element-container {
    animation: fadeInUp 0.3s ease-out;
}

/* ── Scrollbar ──────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(167, 139, 250, .4);
    border-radius: 3px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state initialisation ─────────────────────────────────────────────

def _init_session_state() -> None:
    """Ensure all session-state keys exist with sensible defaults."""
    defaults: dict[str, Any] = {
        "session_id": "",
        "student_name": "",
        "messages": [],
        "weak_topics": [],
        "quiz_score": 0.0,
        "quiz_questions": [],
        "quiz_attempts": 0,
        "quiz_answers": [],
        "pending_plan": "",
        "plan_approved": None,
        "topic": "",
        "awaiting_quiz_answers": False,
        "pdf_uploaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🎓 StudyMate")
    st.markdown("*Your personalised AI study companion*")
    st.divider()

    # ── Student name ─────────────────────────────────────────────────────
    st.markdown("### 👤 Student Profile")
    student_name = st.text_input(
        "Your name",
        value=st.session_state.get("student_name", ""),
        placeholder="Enter your name…",
        key="name_input",
    )
    if student_name and student_name != st.session_state.get("student_name"):
        st.session_state.student_name = student_name
        st.session_state.session_id = ensure_session(
            student_name, st.session_state.get("session_id") or None
        )
        # Load long-term memory
        mem = load_session_memory(st.session_state.session_id)
        if mem:
            st.session_state.weak_topics = mem.get("weak_topics", [])
            st.session_state.quiz_score = mem.get("quiz_score", 0.0)

    # ── PDF uploader ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📄 Upload Study Notes")
    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="pdf_uploader",
        help="Your notes will be indexed for context-aware answers.",
    )
    if uploaded is not None and not st.session_state.get("pdf_uploaded"):
        with st.spinner("📚 Indexing your notes…"):
            count = ingest_pdf(uploaded)
        st.session_state.pdf_uploaded = True
        st.success(f"✅ Indexed {count} chunks from **{uploaded.name}**")

    doc_count = 0
    try:
        doc_count = collection_count()
    except Exception:
        pass
    if doc_count > 0:
        st.info(f"📑 {doc_count} chunks in knowledge base")

    # ── Weak topics tracker ──────────────────────────────────────────────
    st.divider()
    st.markdown("### 🎯 Weak Topics Tracker")
    weak = st.session_state.get("weak_topics", [])
    if weak:
        for t in weak:
            st.markdown(f"- 🔸 {t}")
    else:
        st.markdown("_No weak topics recorded yet._")

    # ── Quiz scores ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Recent Scores")
    if st.session_state.get("session_id"):
        scores = get_quiz_scores(st.session_state.session_id)
        if scores:
            for s in scores[:5]:
                emoji = "🟢" if s["score"] >= 70 else "🟡" if s["score"] >= 50 else "🔴"
                st.markdown(f"{emoji} **{s['topic']}** — {s['score']}%")
        else:
            st.markdown("_No quiz scores yet._")
    else:
        st.markdown("_Enter your name to load history._")

    # ── HITL approval ────────────────────────────────────────────────────
    if st.session_state.get("pending_plan") and st.session_state.get("plan_approved") is None:
        st.divider()
        st.markdown("### ✅ Study Plan Approval")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve", key="approve_plan", use_container_width=True):
                st.session_state.plan_approved = True
                # Save plan
                _handle_plan_approval(True)
                st.rerun()
        with col2:
            if st.button("❌ Reject", key="reject_plan", use_container_width=True):
                st.session_state.plan_approved = False
                _handle_plan_approval(False)
                st.rerun()

    # ── Session history ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🕐 Session History")
    sessions = list_sessions()
    if sessions:
        for sess in sessions[:5]:
            st.markdown(
                f"- **{sess['student_name']}** — {sess['created_at'][:10]}"
            )
    else:
        st.markdown("_No previous sessions._")


def _handle_plan_approval(approved: bool) -> None:
    """Process HITL plan approval/rejection."""
    session_id = st.session_state.get("session_id", "")
    plan = st.session_state.get("pending_plan", "")

    if approved and plan and session_id:
        try:
            from agent.nodes import study_plan_save_node

            result = study_plan_save_node(
                {
                    "pending_plan": plan,
                    "plan_approved": True,
                    "session_id": session_id,
                }
            )
            st.session_state.messages.append(
                AIMessage(content="✅ Study plan approved and saved!")
            )
        except Exception as exc:
            st.session_state.messages.append(
                AIMessage(content=f"Error saving plan: {exc}")
            )
    else:
        st.session_state.messages.append(
            AIMessage(content="❌ Study plan rejected. Ask me for a new one anytime!")
        )

    st.session_state.pending_plan = ""


# ── Main area ────────────────────────────────────────────────────────────────

# Header
st.markdown(
    """
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <h1 style="
            background: linear-gradient(135deg, #a78bfa, #6366f1, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        ">🎓 StudyMate</h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">
            Your AI-powered study companion — explain, quiz, plan, and search your notes.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick-action pills
st.markdown(
    """
    <div style="display:flex; gap:0.5rem; justify-content:center; margin-bottom:1.5rem; flex-wrap:wrap;">
        <span style="background:rgba(167,139,250,.15); color:#a78bfa; padding:0.35rem 1rem;
              border-radius:20px; font-size:0.85rem; border:1px solid rgba(167,139,250,.3);">
            💡 Explain a topic</span>
        <span style="background:rgba(99,102,241,.15); color:#818cf8; padding:0.35rem 1rem;
              border-radius:20px; font-size:0.85rem; border:1px solid rgba(99,102,241,.3);">
            📝 Take a quiz</span>
        <span style="background:rgba(129,140,248,.15); color:#a5b4fc; padding:0.35rem 1rem;
              border-radius:20px; font-size:0.85rem; border:1px solid rgba(129,140,248,.3);">
            📋 Get a study plan</span>
        <span style="background:rgba(139,92,246,.15); color:#c4b5fd; padding:0.35rem 1rem;
              border-radius:20px; font-size:0.85rem; border:1px solid rgba(139,92,246,.3);">
            🔍 Search my notes</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Display chat history ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    avatar = "🎓" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# ── Quiz answer handling ─────────────────────────────────────────────────────
if st.session_state.get("awaiting_quiz_answers") and st.session_state.get("quiz_questions"):
    questions = st.session_state.quiz_questions
    st.markdown("---")
    st.markdown("### 📝 Submit Your Answers")

    with st.form("quiz_form"):
        answers = []
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}.** {q['question']}")
            options = q.get("options", ["A", "B", "C", "D"])
            choice = st.radio(
                f"Your answer for Q{i+1}",
                options=options,
                key=f"quiz_q_{i}",
                label_visibility="collapsed",
            )
            # Extract the letter from the chosen option
            ans_letter = choice[0] if choice else "A"
            answers.append(ans_letter)

        submitted = st.form_submit_button("📨 Submit Answers", use_container_width=True)
        if submitted:
            st.session_state.quiz_answers = answers
            st.session_state.awaiting_quiz_answers = False

            # Evaluate answers
            eval_state = {
                "messages": st.session_state.messages,
                "student_name": st.session_state.student_name,
                "topic": st.session_state.topic,
                "weak_topics": st.session_state.weak_topics,
                "quiz_score": st.session_state.quiz_score,
                "quiz_questions": st.session_state.quiz_questions,
                "quiz_attempts": st.session_state.quiz_attempts,
                "quiz_answers": answers,
                "session_id": st.session_state.session_id,
            }

            from agent.nodes import quiz_evaluate_node

            with st.spinner("📊 Evaluating your answers…"):
                result = quiz_evaluate_node(eval_state)

            # Update session state
            st.session_state.quiz_score = result.get("quiz_score", 0)
            st.session_state.weak_topics = result.get("weak_topics", st.session_state.weak_topics)
            response_text = result.get("response", "")

            if result.get("quiz_questions") is not None:
                st.session_state.quiz_questions = result["quiz_questions"]
            if result.get("quiz_attempts") is not None:
                st.session_state.quiz_attempts = result["quiz_attempts"]

            st.session_state.messages.append(AIMessage(content=response_text))

            # If score < 70% and attempts < 3, generate new quiz
            score = result.get("quiz_score", 0)
            attempts = result.get("quiz_attempts", 0)
            # quiz_attempts is reset to 0 when done, so check quiz_questions
            if st.session_state.quiz_questions:
                # Retry: generate new quiz
                st.session_state.awaiting_quiz_answers = True

            st.rerun()


# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything — explain, quiz me, make a plan, or search your notes…"):
    # Ensure session exists
    if not st.session_state.get("student_name"):
        st.session_state.student_name = "Student"
    if not st.session_state.get("session_id"):
        st.session_state.session_id = ensure_session(st.session_state.student_name)

    # Add user message
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Build agent state
    agent_state: dict[str, Any] = {
        "messages": st.session_state.messages,
        "student_name": st.session_state.student_name,
        "topic": st.session_state.get("topic", ""),
        "weak_topics": st.session_state.get("weak_topics", []),
        "quiz_score": st.session_state.get("quiz_score", 0.0),
        "quiz_questions": st.session_state.get("quiz_questions", []),
        "quiz_attempts": st.session_state.get("quiz_attempts", 0),
        "quiz_answers": [],
        "pending_plan": st.session_state.get("pending_plan", ""),
        "plan_approved": st.session_state.get("plan_approved"),
        "session_id": st.session_state.session_id,
        "route": "",
        "rag_context": "",
        "response": "",
        "error": "",
    }

    # Run the graph
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("🧠 Thinking…"):
            try:
                result = study_graph.invoke(agent_state)

                response = result.get("response", "I'm not sure how to respond to that.")
                st.markdown(response)

                # Update session state from result
                st.session_state.topic = result.get("topic", st.session_state.topic)
                st.session_state.weak_topics = result.get("weak_topics", st.session_state.weak_topics)
                st.session_state.quiz_score = result.get("quiz_score", st.session_state.quiz_score)

                if result.get("quiz_questions"):
                    st.session_state.quiz_questions = result["quiz_questions"]
                    st.session_state.quiz_attempts = result.get("quiz_attempts", 0)
                    st.session_state.awaiting_quiz_answers = True

                if result.get("pending_plan"):
                    st.session_state.pending_plan = result["pending_plan"]
                    st.session_state.plan_approved = None

                # Save AI response to messages
                st.session_state.messages.append(AIMessage(content=response))

                # Trim messages for short-term memory
                st.session_state.messages = trim_messages(
                    st.session_state.messages, max_messages=10
                )

            except Exception as exc:
                error_msg = f"⚠️ Something went wrong: {exc}\n\nPlease try again!"
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))

    # Rerun to show quiz form if needed
    if st.session_state.get("awaiting_quiz_answers"):
        st.rerun()
