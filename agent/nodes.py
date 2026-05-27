"""All node functions for the StudyMate LangGraph workflow.

Every function is a *pure* node: it receives ``AgentState``, does its work,
and returns a **partial** state update dict.  No global state is mutated.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agent.state import AgentState
from agent.memory import trim_messages, persist_weak_topics
from db.sqlite_store import save_quiz_score, save_study_plan, get_weak_topics
from rag.retriever import retrieve_as_text, collection_count

load_dotenv()

# ── LLM helper ──────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.3) -> ChatGroq:
    """Build a ChatGroq instance (reads GROQ_API_KEY from env)."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _safe_invoke(llm: ChatGroq, messages: list, fallback: str = "I'm sorry, something went wrong. Please try again.") -> str:
    """Invoke the LLM and return content, with graceful error handling."""
    try:
        resp = llm.invoke(messages)
        return resp.content
    except Exception as exc:
        return f"{fallback}\n\n(Error details: {exc})"


# ── 1. ROUTER NODE ──────────────────────────────────────────────────────────

ROUTE_PROMPT_WITHOUT_NOTES = """\
You are a routing classifier for a study assistant.
The student has NOT uploaded any notes/PDF yet.
Classify the student's message into EXACTLY one of these categories:
- explain    → the student wants a concept explained or wants general tutoring/chit-chat.
- quiz       → the student wants to be quizzed or tested.
- study_plan → the student wants a study plan or schedule.
- rag_query  → the student is asking a question explicitly about their notes, documents, or uploaded files.

Reply with ONLY the category name, nothing else.
"""

ROUTE_PROMPT_WITH_NOTES = """\
You are a routing classifier for a study assistant.
The student HAS uploaded study notes/PDF to their profile.
Classify the student's message into EXACTLY one of these categories:
- explain    → the student wants general non-academic conversation or chit-chat.
- quiz       → the student wants to be quizzed, tested, or asked questions.
- study_plan → the student wants a study plan, schedule, or curriculum.
- rag_query  → the student is asking academic questions, explaining concepts, asking for facts, or requesting information that is likely covered in their study notes. If in doubt, route to rag_query.

Reply with ONLY the category name, nothing else.
"""


def router_node(state: AgentState) -> dict[str, Any]:
    """Classify the latest user message and set ``state["route"]``."""
    messages = state.get("messages", [])
    if not messages:
        return {"route": "explain", "error": ""}

    last_msg = messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # Check if a PDF is uploaded in ChromaDB
    has_notes = collection_count() > 0
    route_prompt = ROUTE_PROMPT_WITH_NOTES if has_notes else ROUTE_PROMPT_WITHOUT_NOTES

    llm = _get_llm(temperature=0.0)
    result = _safe_invoke(
        llm,
        [SystemMessage(content=route_prompt), HumanMessage(content=user_text)],
        fallback="explain",
    )

    route = result.strip().lower()
    valid_routes = {"explain", "quiz", "study_plan", "rag_query"}
    if route not in valid_routes:
        # Fuzzy fallback
        for r in valid_routes:
            if r in route:
                route = r
                break
        else:
            route = "explain"

    # Extract topic from the message for downstream nodes
    topic = _extract_topic(user_text, state.get("topic", ""))

    return {"route": route, "topic": topic, "error": ""}


def _extract_topic(user_text: str, current_topic: str) -> str:
    """Simple topic extraction — uses the LLM for a one-shot extraction."""
    try:
        llm = _get_llm(temperature=0.0)
        result = _safe_invoke(
            llm,
            [
                SystemMessage(content="Extract the main study topic from the student's message. Reply with ONLY the topic name (2-5 words). If no clear topic, reply with 'general'."),
                HumanMessage(content=user_text),
            ],
            fallback=current_topic or "general",
        )
        topic = result.strip().strip('"').strip("'")
        return topic if topic else (current_topic or "general")
    except Exception:
        return current_topic or "general"


# ── 2. EXPLANATION NODE ─────────────────────────────────────────────────────

def explain_node(state: AgentState) -> dict[str, Any]:
    """Generate a clear, student-level explanation of the topic.

    If RAG context is available (notes uploaded), it is included as
    supporting material.
    """
    topic = state.get("topic", "the topic")
    student = state.get("student_name", "Student")
    messages = state.get("messages", [])

    # Try to get RAG context
    rag_ctx = ""
    try:
        rag_ctx = retrieve_as_text(topic, k=3)
    except Exception:
        pass

    system_content = f"""\
You are StudyMate, a friendly and knowledgeable AI tutor.
The student's name is {student}.

Explain the topic clearly and thoroughly at an undergraduate level.
Use examples, analogies, and structured formatting (headers, bullet points).
Be encouraging and supportive.
"""
    if rag_ctx:
        system_content += f"""
The student has uploaded notes. Use the following retrieved context to enrich your explanation:

{rag_ctx}
"""

    # Build conversation context (trimmed)
    conv = [SystemMessage(content=system_content)]
    for m in trim_messages(messages):
        if hasattr(m, "type"):
            conv.append(m)
        else:
            conv.append(HumanMessage(content=str(m)))

    llm = _get_llm(temperature=0.5)
    response = _safe_invoke(llm, conv)

    return {
        "response": response,
        "messages": [AIMessage(content=response)],
        "rag_context": rag_ctx,
        "error": "",
    }


# ── 3. QUIZ NODE ────────────────────────────────────────────────────────────

QUIZ_GENERATE_PROMPT = """\
You are StudyMate's quiz generator.
Generate exactly 3 multiple-choice questions on the topic: "{topic}".
{weak_focus}
{previous_questions_focus}

Return your response as a JSON object containing a "questions" array. Each element must have:
- "question": the question text
- "options": a list of 4 options labelled A, B, C, D
- "correct": the letter of the correct answer (A/B/C/D)
- "explanation": brief explanation of the correct answer

Example format:
{{
  "questions": [
    {{
      "question": "What is ...?",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct": "A",
      "explanation": "Because ..."
    }}
  ]
}}

Reply with ONLY the JSON object, no other text.
"""

QUIZ_EVALUATE_PROMPT = """\
You are evaluating a student's quiz answers.
Here are the questions, correct answers, and the student's answers:

{details}

For each question, state whether the student was correct or incorrect and why.
Then give an overall score as a percentage.
Finally, list any weak areas the student should review.

Format your response clearly with headers.
End with a line: SCORE: <number>%
And a line: WEAK_AREAS: <comma-separated topics or "none">
"""


def quiz_generate_node(state: AgentState) -> dict[str, Any]:
    """Generate 3 MCQ questions on the current topic."""
    topic = state.get("topic", "general")
    weak_topics = state.get("weak_topics", [])
    attempts = state.get("quiz_attempts", 0)

    # Check if a PDF is uploaded to generate custom quiz questions from it
    has_notes = collection_count() > 0
    rag_ctx = ""
    if has_notes:
        try:
            rag_ctx = retrieve_as_text(topic, k=3)
        except Exception:
            pass

    weak_focus = ""
    if attempts > 0 and weak_topics:
        weak_focus = f"Focus especially on these weak areas: {', '.join(weak_topics)}"

    previous_questions = state.get("quiz_questions", [])
    previous_questions_focus = ""
    if previous_questions and attempts > 0:
        prev_texts = [q.get("question", "") for q in previous_questions]
        previous_questions_focus = f"\\nIMPORTANT: Do NOT generate the following questions again:\\n- " + "\\n- ".join(prev_texts)

    llm = _get_llm(temperature=0.7)
    
    prompt = QUIZ_GENERATE_PROMPT.format(topic=topic, weak_focus=weak_focus, previous_questions_focus=previous_questions_focus)
    if rag_ctx:
        prompt += f"\n\nIMPORTANT: Use the following retrieved context from the student's uploaded notes to generate questions that test their knowledge of this material specifically:\n{rag_ctx}"

    llm_json = llm.bind(response_format={"type": "json_object"})
    result = _safe_invoke(llm_json, [SystemMessage(content=prompt)])

    # Parse JSON from response
    questions = _parse_quiz_json(result)

    if not questions:
        return {
            "response": "I had trouble generating quiz questions. Let me try explaining the topic instead.",
            "messages": [AIMessage(content="I had trouble generating quiz questions. Please try again.")],
            "route": "explain",
            "error": "quiz_generation_failed",
        }

    # Build a readable quiz message
    quiz_text = f"📝 **Quiz on {topic}** (Attempt {attempts + 1}/3)\n\n"
    for i, q in enumerate(questions, 1):
        quiz_text += f"**Q{i}.** {q['question']}\n"
        for opt in q["options"]:
            quiz_text += f"  {opt}\n"
        quiz_text += "\n"
    quiz_text += "\n_Reply with your answers like: A, B, C_"

    return {
        "quiz_questions": questions,
        "quiz_attempts": attempts + 1,
        "response": quiz_text,
        "messages": [AIMessage(content=quiz_text)],
        "error": "",
    }


def quiz_evaluate_node(state: AgentState) -> dict[str, Any]:
    """Evaluate the student's quiz answers and compute score."""
    questions = state.get("quiz_questions", [])
    answers_raw = state.get("quiz_answers", [])
    session_id = state.get("session_id", "")
    topic = state.get("topic", "general")
    attempts = state.get("quiz_attempts", 0)

    if not questions or not answers_raw:
        return {
            "response": "I don't have your answers yet. Please answer the quiz first.",
            "messages": [AIMessage(content="Please answer the quiz questions first.")],
            "error": "",
        }

    # Build evaluation details
    details = ""
    correct_count = 0
    weak_areas: list[str] = []

    for i, q in enumerate(questions):
        student_ans = answers_raw[i].strip().upper() if i < len(answers_raw) else "?"
        is_correct = student_ans == q.get("correct", "").upper()
        if is_correct:
            correct_count += 1
        else:
            weak_areas.append(q.get("question", f"Question {i+1}")[:50])

        details += f"Q{i+1}: {q['question']}\n"
        details += f"Correct: {q['correct']}\n"
        details += f"Student answered: {student_ans}\n"
        details += f"Explanation: {q.get('explanation', 'N/A')}\n\n"

    score = round((correct_count / len(questions)) * 100, 1) if questions else 0

    # Generate detailed feedback via LLM
    llm = _get_llm(temperature=0.3)
    eval_prompt = QUIZ_EVALUATE_PROMPT.format(details=details)
    feedback = _safe_invoke(llm, [SystemMessage(content=eval_prompt)])

    # Save to SQLite
    if session_id:
        try:
            save_quiz_score(session_id, topic, score, attempts, weak_areas)
            if weak_areas:
                existing_weak = state.get("weak_topics", [])
                all_weak = list(set(existing_weak + [topic] + [a[:30] for a in weak_areas]))
                persist_weak_topics(session_id, all_weak)
        except Exception:
            pass

    result_text = f"📊 **Quiz Results — {topic}**\n\n"
    result_text += f"**Score: {score}%** ({correct_count}/{len(questions)} correct)\n\n"
    result_text += feedback

    if score < 70 and attempts < 3:
        result_text += f"\n\n🔄 Score below 70%. Let's try again! (Attempt {attempts}/{3})"
        return {
            "quiz_score": score,
            "weak_topics": list(set(state.get("weak_topics", []) + [topic])),
            "response": result_text,
            "messages": [AIMessage(content=result_text)],
            "error": "",
        }

    if score >= 70:
        result_text += "\n\n🎉 Great job! You passed the quiz!"
    else:
        result_text += "\n\n📚 You've used all 3 attempts. Consider reviewing the topic and trying again later."

    return {
        "quiz_score": score,
        "quiz_questions": [],
        "quiz_attempts": 0,
        "weak_topics": list(set(state.get("weak_topics", []) + ([topic] if score < 70 else []))),
        "response": result_text,
        "messages": [AIMessage(content=result_text)],
        "error": "",
    }


def _parse_quiz_json(text: str) -> list[dict[str, Any]]:
    """Robustly extract a JSON array of quiz questions from LLM output."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "questions" in data:
            return data["questions"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON dictionary in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "questions" in data:
                return data["questions"]
        except json.JSONDecodeError:
            pass
            
    # Try array as fallback
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


# ── 4. STUDY PLAN NODE ──────────────────────────────────────────────────────

STUDY_PLAN_PROMPT = """\
You are StudyMate, a study planning assistant.
The student's name is {student}.

Create a detailed 7-day study plan based on these weak topics: {weak_topics}.
Also consider the student's current topic of interest: {topic}.

Structure the plan day by day with:
- Specific topics to cover each day
- Recommended study time
- Practice exercises or activities
- Review sessions

Make it realistic and encouraging.
"""


def study_plan_node(state: AgentState) -> dict[str, Any]:
    """Generate a 7-day study plan and pause for HITL approval."""
    student = state.get("student_name", "Student")
    topic = state.get("topic", "general")
    session_id = state.get("session_id", "")
    weak_topics = state.get("weak_topics", [])
    messages = state.get("messages", [])

    if not weak_topics:
        weak_topics = [topic]

    has_notes = collection_count() > 0
    rag_ctx = ""
    if has_notes:
        try:
            rag_ctx = retrieve_as_text(topic, k=3)
        except Exception:
            pass

    llm = _get_llm(temperature=0.7)
    prompt = STUDY_PLAN_PROMPT.format(
        student=student,
        weak_topics=", ".join(weak_topics),
        topic=topic,
    )
    if rag_ctx:
        prompt += f"\n\nIMPORTANT: The student has uploaded study notes. Please design the study plan explicitly around the following extracted content from their notes:\n{rag_ctx}"

    conv = [SystemMessage(content=prompt)]
    for m in trim_messages(messages, max_messages=4):
        if hasattr(m, "type"):
            conv.append(m)
        else:
            conv.append(HumanMessage(content=str(m)))

    plan = _safe_invoke(llm, conv)

    response = "📋 **Your Personalized 7-Day Study Plan**\n\n"
    response += plan
    response += "\n\n---\n"
    response += "👆 **Please review the plan above.**\n"
    response += "Use the **Approve** or **Reject** buttons in the sidebar to confirm."

    return {
        "pending_plan": plan,
        "plan_approved": None,  # Waiting for HITL
        "response": response,
        "messages": [AIMessage(content=response)],
        "error": "",
    }


def study_plan_save_node(state: AgentState) -> dict[str, Any]:
    """Save an approved study plan to SQLite."""
    plan = state.get("pending_plan", "")
    approved = state.get("plan_approved", False)
    session_id = state.get("session_id", "")

    if approved and plan and session_id:
        try:
            save_study_plan(session_id, plan, approved=True)
            return {
                "response": "✅ Study plan approved and saved! You can view it anytime in your session history.",
                "messages": [AIMessage(content="✅ Study plan approved and saved!")],
                "pending_plan": "",
                "plan_approved": True,
                "error": "",
            }
        except Exception as exc:
            return {
                "response": f"Failed to save the study plan: {exc}",
                "error": str(exc),
            }
    else:
        return {
            "response": "❌ Study plan was not approved. Feel free to request a new one anytime!",
            "messages": [AIMessage(content="❌ Study plan not approved.")],
            "pending_plan": "",
            "plan_approved": False,
            "error": "",
        }


# ── 5. RAG QUERY NODE ───────────────────────────────────────────────────────

def rag_query_node(state: AgentState) -> dict[str, Any]:
    """Answer a question using retrieved context from uploaded notes."""
    messages = state.get("messages", [])
    student = state.get("student_name", "Student")

    last_msg = messages[-1] if messages else None
    user_text = last_msg.content if last_msg and hasattr(last_msg, "content") else "What are the key concepts?"

    # Check if there are any documents in the vector store first
    if collection_count() == 0:
        no_ctx = ("I don't have any uploaded notes to search through yet. "
                  "Please upload a PDF in the sidebar first, then ask your question again!")
        return {
            "response": no_ctx,
            "messages": [AIMessage(content=no_ctx)],
            "rag_context": "",
            "error": "",
        }

    # Retrieve context
    rag_ctx = ""
    try:
        rag_ctx = retrieve_as_text(user_text, k=3)
    except Exception:
        pass

    system_content = f"""\
You are StudyMate, a friendly and knowledgeable AI tutor helping {student}.
We have searched the student's uploaded notes, and here is the retrieved context:

---
{rag_ctx}
---

Your task:
1. Try to answer the student's question using the retrieved context from their uploaded notes first. If the context contains the answer, ground your response in it and be specific.
2. If the retrieved context does NOT contain the answer, or if the context is empty, answer the question clearly and thoroughly using your own general knowledge. However, you MUST preface your response with a brief note indicating that you did not find a direct answer in their uploaded notes, e.g., "I couldn't find a direct mention of this in your uploaded notes, but here is a general explanation:" or similar, to be fully transparent.
3. Keep the tone encouraging, structured, and helpful.
"""

    llm = _get_llm(temperature=0.3)
    conv = [SystemMessage(content=system_content), HumanMessage(content=user_text)]
    response = _safe_invoke(llm, conv)

    return {
        "response": response,
        "messages": [AIMessage(content=response)],
        "rag_context": rag_ctx,
        "error": "",
    }
