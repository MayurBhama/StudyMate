"""StudyMate -- Streamlit UI entry point.

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

# -- Environment ---------------------------------------------------------------
load_dotenv()

# Verify LangSmith observability is enabled via .env
assert os.environ.get("LANGCHAIN_TRACING_V2") == "true", "LANGCHAIN_TRACING_V2 not set in .env"

# -- Local imports (after dotenv) ----------------------------------------------
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

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="StudyMate",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS ----------------------------------------------------------------
st.markdown(
    """
<style>
/* -- Global ----------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* -- Sidebar ---------------------------------------------------------- */
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

/* -- Cards / containers ----------------------------------------------- */
div[data-testid="stExpander"] {
    border: 1px solid rgba(167, 139, 250, .25);
    border-radius: 12px;
    background: rgba(15, 12, 41, .35);
    backdrop-filter: blur(6px);
}

/* -- Quiz radio buttons ----------------------------------------------- */
.stRadio > div {
    gap: 0.25rem;
}

/* -- Buttons ---------------------------------------------------------- */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(167, 139, 250, .3);
}

/* -- Chat messages ---------------------------------------------------- */
div[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 0.5rem;
}

/* -- Metric cards ----------------------------------------------------- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(167,139,250,.1), rgba(99,102,241,.1));
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid rgba(167, 139, 250, .2);
}

/* -- Progress bars ---------------------------------------------------- */
.stProgress > div > div {
    background: linear-gradient(90deg, #a78bfa, #6366f1);
    border-radius: 10px;
}

/* -- Smooth animations ------------------------------------------------ */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.element-container {
    animation: fadeInUp 0.3s ease-out;
}

/* -- Scrollbar -------------------------------------------------------- */
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


# -- Session state initialisation ----------------------------------------------

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


def _handle_plan_approval(approved: bool) -> None:
    """Process HITL plan approval/rejection."""
    session_id = st.session_state.get("session_id", "")
    plan = st.session_state.get("pending_plan", "")

    if approved and plan and session_id:
        try:
            from agent.graph import study_graph
            config = {"configurable": {"thread_id": session_id}}
            study_graph.update_state(config, {"plan_approved": True})
            result = study_graph.invoke(None, config=config)

            if "response" in result:
                st.session_state.messages.append(AIMessage(content=result["response"]))

        except Exception as exc:
            st.session_state.messages.append(
                AIMessage(content=f"Error saving plan: {exc}")
            )
        st.session_state.pending_plan = ""
        st.session_state.plan_approved = True

    elif not approved:
        st.session_state.pending_plan = None
        st.session_state.plan_approved = False


# -- Sidebar -------------------------------------------------------------------

with st.sidebar:
    st.markdown("# StudyMate")
    st.markdown("*Your personalised AI study companion*")
    st.divider()

    # -- Student name ----------------------------------------------------------
    st.markdown("### Student Profile")
    student_name = st.text_input(
        "Your name",
        value=st.session_state.get("student_name", ""),
        placeholder="Enter your name",
        key="name_input",
    ).strip()
    if not student_name:
        student_name = ""
    if student_name and student_name != st.session_state.get("student_name"):
        st.session_state.student_name = student_name
        st.session_state.session_id = ensure_session(
            student_name, st.session_state.get("session_id") or None
        )
        mem = load_session_memory(st.session_state.session_id)
        if mem:
            st.session_state.weak_topics = mem.get("weak_topics", [])
            st.session_state.quiz_score = mem.get("quiz_score", 0.0)

    # -- PDF uploader ----------------------------------------------------------
    st.divider()
    st.markdown("### Study Notes")
    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="pdf_uploader",
        help="Your notes will be indexed for context-aware answers.",
    )
    if uploaded is not None and st.session_state.get("last_uploaded_filename") != uploaded.name:
        with st.spinner("Indexing your notes..."):
            count = ingest_pdf(uploaded)
        st.session_state.last_uploaded_filename = uploaded.name
        st.success(f"Indexed {count} chunks from {uploaded.name}")

    doc_count = 0
    try:
        doc_count = collection_count()
    except Exception:
        pass
    if doc_count > 0:
        st.info(f"{doc_count} chunks in knowledge base")

    # -- Weak topics tracker ---------------------------------------------------
    st.divider()
    st.markdown("### Weak Topics")
    weak = st.session_state.get("weak_topics", [])
    if weak:
        for t in weak:
            st.markdown(f"- {t}")
    else:
        st.markdown("_No weak topics recorded yet._")

# -- Main area -----------------------------------------------------------------

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
        ">StudyMate</h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 1rem;">
            Your personalised AI study companion
        </p>
        <div style="color: #cbd5e1; font-size: 0.95rem; display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap; max-width: 600px; margin: 0 auto;">
            <span>• Explain topics</span>
            <span>• Take a quiz</span>
            <span>• Get a study plan</span>
            <span>• Search your notes</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -- Display chat history ------------------------------------------------------
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# -- HITL approval -------------------------------------------------------------
if st.session_state.get("pending_plan") and st.session_state.get("plan_approved") is None:
    st.info("Please review the proposed study plan above.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve", key="approve_plan", use_container_width=True, type="primary"):
            st.session_state.plan_approved = True
            _handle_plan_approval(True)
            st.session_state.trigger_prompt = "I approve of the schedule you proposed. Let's move on!"
            st.rerun()
    with col2:
        if st.button("Reject", key="reject_plan", use_container_width=True):
            st.session_state.plan_approved = False
            _handle_plan_approval(False)
            st.session_state.trigger_prompt = "I reject the previous study plan. Please generate a new, completely different one for me."
            st.rerun()

# -- Quiz interactions ---------------------------------------------------------
if st.session_state.get("awaiting_quiz_answers") and st.session_state.get("quiz_questions"):
    questions = st.session_state.quiz_questions
    st.markdown("---")
    attempts = st.session_state.get("quiz_attempts", 1)
    st.markdown(f"### Quiz -- Attempt {attempts} of 3")

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
            ans_letter = choice[0] if choice else "A"
            answers.append(ans_letter)

        submitted = st.form_submit_button("Submit Answers", use_container_width=True)
        if submitted:
            st.session_state.quiz_answers = answers
            st.session_state.awaiting_quiz_answers = False

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

            with st.spinner("Evaluating your answers..."):
                result = quiz_evaluate_node(eval_state)

            st.session_state.quiz_score = result.get("quiz_score", 0)
            st.session_state.weak_topics = result.get("weak_topics", st.session_state.weak_topics)
            response_text = result.get("response", "")

            st.session_state.messages.append(AIMessage(content=response_text))

            score = result.get("quiz_score", 0)
            attempts = st.session_state.quiz_attempts

            if score < 70 and attempts < 3:
                with st.spinner("Generating new quiz questions..."):
                    gen_state = {
                        "topic": st.session_state.topic,
                        "weak_topics": st.session_state.weak_topics,
                        "quiz_attempts": attempts,
                        "quiz_questions": st.session_state.quiz_questions,
                    }
                    from agent.nodes import quiz_generate_node
                    gen_result = quiz_generate_node(gen_state)

                    st.session_state.quiz_questions = gen_result.get("quiz_questions", [])
                    st.session_state.quiz_attempts = gen_result.get("quiz_attempts", attempts + 1)
                    st.session_state.awaiting_quiz_answers = True

                    new_quiz_text = gen_result.get("response", "")
                    st.session_state.messages.append(AIMessage(content=new_quiz_text))
            else:
                st.session_state.quiz_questions = []
                st.session_state.quiz_attempts = 0
                st.session_state.awaiting_quiz_answers = False

            st.rerun()


# -- Chat input ----------------------------------------------------------------
user_prompt = st.chat_input("Ask me anything...")

if st.session_state.get("trigger_prompt"):
    user_prompt = st.session_state.trigger_prompt
    st.session_state.trigger_prompt = None

if user_prompt:
    if not st.session_state.get("student_name"):
        st.session_state.student_name = "Student"
    if not st.session_state.get("session_id"):
        st.session_state.session_id = ensure_session(st.session_state.student_name)

    user_msg = HumanMessage(content=user_prompt)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(user_prompt)

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

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from agent.nodes import router_node
                predicted_route = router_node(agent_state).get("route", "explain")
                config = {"configurable": {"thread_id": st.session_state.session_id}}

                if predicted_route in ["explain", "rag_query"]:
                    response_placeholder = st.empty()
                    full_response = ""
                    result = agent_state.copy()

                    for chunk in study_graph.stream(agent_state, config=config):
                        for node_name, node_output in chunk.items():
                            result.update(node_output)
                            if "messages" in node_output:
                                last_msg = node_output["messages"][-1]
                                if hasattr(last_msg, "content"):
                                    full_response += last_msg.content
                                    response_placeholder.markdown(full_response)

                    response_placeholder.markdown(full_response)
                    response = result.get("response", full_response)

                    # Confidence caption
                    notes_uploaded = collection_count() > 0
                    if predicted_route == "rag_query":
                        st.caption("Answer retrieved from your uploaded notes.")
                    elif notes_uploaded:
                        st.caption("Response based on your uploaded notes.")
                    else:
                        st.caption("Response from general knowledge -- verify from your textbook.")
                else:
                    result = study_graph.invoke(agent_state, config=config)
                    response = result.get("response", "I'm not sure how to respond to that.")
                    st.markdown(response)

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

                st.session_state.messages.append(AIMessage(content=response))

                st.session_state.messages = trim_messages(
                    st.session_state.messages, max_messages=10
                )

            except Exception as exc:
                error_msg = f"Something went wrong: {exc}\n\nPlease try again."
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))

    if st.session_state.get("awaiting_quiz_answers") or (st.session_state.get("pending_plan") and st.session_state.get("plan_approved") is None):
        st.rerun()
