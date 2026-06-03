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
from db.sqlite_store import save_quiz_score, save_study_plan
from rag.retriever import retrieve_as_text, collection_count, retrieve_context, get_random_chunks

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
Classify the user message into exactly one of these 4 categories.
Return ONLY the category word. Nothing else. No punctuation.

explain    -- user wants a topic explained or described
quiz       -- user wants to be tested, quizzed, or asked questions
study_plan -- user wants a plan, schedule, or roadmap
rag_query  -- user asks about their uploaded notes or document

The student has NOT uploaded any notes, so rag_query should only be used
if they explicitly mention notes, documents, or uploaded files.

Examples:
'explain photosynthesis' -> explain
'quiz me on python' -> quiz
'test my knowledge' -> quiz
'what does my pdf say about loops' -> rag_query
'make me a study plan' -> study_plan
'i want to practice' -> quiz
'tell me about recursion' -> explain

User message: {message}
"""

ROUTE_PROMPT_WITH_NOTES = """\
Classify the user message into exactly one of these 4 categories.
Return ONLY the category word. Nothing else. No punctuation.

explain    -- user wants a topic explained or described
quiz       -- user wants to be tested, quizzed, or asked questions
study_plan -- user wants a plan, schedule, or roadmap
rag_query  -- user asks about their uploaded notes or document, OR asks
              an academic question that is likely covered in their notes.
              If in doubt between explain and rag_query, choose rag_query.

The student HAS uploaded study notes/PDF.

Examples:
'explain photosynthesis' -> rag_query
'quiz me on python' -> quiz
'test my knowledge' -> quiz
'what does my pdf say about loops' -> rag_query
'make me a study plan' -> study_plan
'what is in my notes about arrays' -> rag_query
'i want to practice' -> quiz
'what is logistic regression' -> rag_query

User message: {message}
"""


def router_node(state: AgentState) -> dict[str, Any]:
    """Classify the latest user message and set ``state["route"]``."""
    messages = state.get("messages", [])
    if not messages:
        return {"route": "explain", "error": ""}

    last_msg = messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    session_id = state.get("session_id", "")
    coll_name = f"user_{session_id.replace('-', '')}" if session_id else "studymate_notes"
    
    # Check if a PDF is uploaded in ChromaDB
    has_notes = collection_count(collection_name=coll_name) > 0
    route_prompt = ROUTE_PROMPT_WITH_NOTES if has_notes else ROUTE_PROMPT_WITHOUT_NOTES

    llm = _get_llm(temperature=0.0)
    formatted_prompt = route_prompt.format(message=user_text)
    result = _safe_invoke(
        llm,
        [SystemMessage(content=formatted_prompt)],
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
        topic = result.strip().strip('"').strip("'").lower()
        if topic == "general" and current_topic and current_topic != "general":
            return current_topic
        return topic if topic else (current_topic or "general")
    except Exception:
        return current_topic or "general"


# ── 2. EXPLANATION NODE ─────────────────────────────────────────────────────

def explain_node(state: AgentState) -> dict[str, Any]:
    """Generate a clear, student-level explanation of the topic.

    If RAG context is available (notes uploaded), it is included as
    supporting material.
    """
    topic = state.get("topic", "")
    messages = state.get("messages", [])
    user_message = messages[-1].content if messages and hasattr(messages[-1], "content") else str(messages[-1]) if messages else ""
    topic = topic or user_message

    safe_name = state.get("student_name", "").strip()
    if not safe_name or safe_name.lower() in ["", "none", "null"]:
        safe_name = "Student"

    session_id = state.get("session_id", "")
    coll_name = f"user_{session_id.replace('-', '')}" if session_id else "studymate_notes"
    
    # Try to get RAG context
    chunks = retrieve_context(topic, n_results=3, collection_name=coll_name)
    
    if chunks:
        context_block = "\n\n".join(chunks)
        context_instruction = f"""
Use the following content from the student's uploaded notes 
as your primary source. Supplement with general knowledge only 
if the notes are insufficient.

NOTES CONTENT:
{context_block}"""
    else:
        context_instruction = ""

    if chunks:
        system_content = f"""\
You are StudyMate, a study tutor helping {safe_name}.
{context_instruction}
If the context does not cover the topic fully, you may use your general knowledge to explain it, but you MUST add this disclaimer at the beginning:
'Your notes do not cover this completely. Here is a general explanation -- please verify from your textbook.'
Never present uncertain information as fact. If you are not sure, say so explicitly.
Address {safe_name} by name at the start of your response.

Explain the topic clearly at an undergraduate level.
Use examples, analogies, and structured formatting (headers, bullet points).
"""
    else:
        system_content = f"""\
You are StudyMate, a study tutor helping {safe_name}.
The student has NOT uploaded any notes, so you are answering from general knowledge.
Start your response by addressing {safe_name} by name.
Then add this disclaimer at the beginning:
'I don\'t have your notes on this topic. Here is a general explanation -- please verify from your textbook.'
Explain the topic clearly at an undergraduate level.
Use examples, analogies, and structured formatting (headers, bullet points).
Never present uncertain information as fact. If you are not sure, say so explicitly.
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
        "rag_context": "\n\n".join(chunks) if chunks else "",
        "error": "",
    }


# ── 3. QUIZ NODE ────────────────────────────────────────────────────────────

QUIZ_GENERATE_PROMPT = """\
You are StudyMate's quiz generator.
Generate exactly 5 multiple-choice questions on the topic: "{topic}".
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
You are a strict quiz evaluator. Follow these rules exactly:
1. Compare student answer ONLY against the correct answer provided.
2. Do not give partial credit. An answer is correct or incorrect.
3. Format each question's evaluation as a bullet point with double newlines between them, like:
   - **Q[number]:** [Correct/Incorrect] — [brief explanation]
   
4. Do NOT output a final score or weak areas list. The system calculates those.
5. Be concise and factual.

Quiz details:
{details}
"""


def quiz_generate_node(state: AgentState) -> dict[str, Any]:
    """Generate 5 MCQ questions on the current topic."""
    topic = state.get("topic", "general")
    weak_topics = state.get("weak_topics", [])
    attempts = state.get("quiz_attempts", 0)
    
    safe_name = state.get("student_name", "").strip()
    if not safe_name or safe_name.lower() in ["", "none", "null"]:
        safe_name = "Student"

    # ── Fix 1: Detect "quiz from notes" intent ───────────────────────────
    NOTES_INTENT_KEYWORDS = [
        "note", "notes", "uploaded", "pdf", "document",
        "my file", "what i uploaded", "from notes",
        "from pdf", "general", "",
    ]

    topic_lower = topic.lower().strip()
    wants_notes_quiz = any(
        keyword in topic_lower
        for keyword in NOTES_INTENT_KEYWORDS
    ) or topic_lower in ["general", "", "none"]

    # Check if a PDF is uploaded to generate custom quiz questions from it
    session_id = state.get("session_id", "")
    coll_name = f"user_{session_id.replace('-', '')}" if session_id else "studymate_notes"

    # ── Fix 3: Intent-aware retrieval ────────────────────────────────────
    has_notes = collection_count(collection_name=coll_name) > 0
    rag_ctx = ""

    if has_notes:
        try:
            if wants_notes_quiz:
                # User wants quiz from notes broadly —
                # sample random chunks to discover topics
                rag_ctx = get_random_chunks(
                    k=15,
                    collection_name=coll_name,
                )
            else:
                # User specified a topic — search for it
                rag_ctx = retrieve_as_text(
                    topic,
                    k=10,
                    collection_name=coll_name,
                )
        except Exception:
            pass

    weak_focus = ""
    if attempts > 0 and weak_topics:
        weak_focus = f"Focus especially on these weak areas: {', '.join(weak_topics)}"

    previous_questions = state.get("quiz_questions", [])
    previous_questions_focus = ""
    if previous_questions and attempts > 0:
        prev_texts = [q.get("question", "") for q in previous_questions]
        previous_questions_focus = "\\nIMPORTANT: Do NOT generate the following questions again:\\n- " + "\\n- ".join(prev_texts)

    llm = _get_llm(temperature=0.7)
    
    prompt = QUIZ_GENERATE_PROMPT.format(topic=topic, weak_focus=weak_focus, previous_questions_focus=previous_questions_focus)

    # ── Fix 4: Intent-aware prompt augmentation ──────────────────────────
    if rag_ctx:
        if wants_notes_quiz:
            prompt += f"""

CRITICAL INSTRUCTION:
The student wants to be quizzed on their uploaded notes.
You have been given random samples from their notes below.

Your job:
1. Read the content carefully
2. Identify 3 different topics or concepts present in it
3. Generate one question per topic — all from this content only
4. Do NOT ask questions about anything outside this content
5. Do NOT ask general knowledge questions
6. Every question must be directly answerable from the text below

CONTENT FROM UPLOADED NOTES:
{rag_ctx}

Generate questions ONLY from the above content."""
        else:
            prompt += f"\n\nUse the following retrieved context from the student's uploaded notes to generate questions that test their knowledge of this material specifically:\n{rag_ctx}"
    else:
        # No notes uploaded or retrieval failed
        if wants_notes_quiz:
            return {
                "response": (
                    f"{safe_name}, I don't see any uploaded notes "
                    f"to quiz you from. Please upload a PDF first "
                    f"using the sidebar, then ask me to quiz you."
                ),
                "messages": [AIMessage(content=(
                    f"{safe_name}, please upload your notes PDF "
                    f"first, then I can quiz you from it."
                ))],
                "error": "no_notes_uploaded",
            }

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

    # ── Fix 5: Display topic update ──────────────────────────────────────
    display_topic = "your uploaded notes" if wants_notes_quiz else topic
    quiz_text = f"**Quiz on {display_topic}** (Attempt {attempts + 1}/3)\n\n"
    for i, q in enumerate(questions, 1):
        quiz_text += f"**Q{i}.** {q['question']}\n"
        for opt in q["options"]:
            quiz_text += f"  {opt}\n"
        quiz_text += "\n"
    quiz_text += f"\nGood luck, {safe_name}! Select your answers below."

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

    safe_name = state.get("student_name", "").strip()
    if not safe_name or safe_name.lower() in ["", "none", "null"]:
        safe_name = "Student"

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
        # Safely extract just the letter from the correct answer
        correct_ans = q.get("correct", "A").strip().upper()
        if correct_ans:
            correct_ans = correct_ans[0]
            
        is_correct = student_ans == correct_ans
        if is_correct:
            correct_count += 1
        else:
            weak_areas.append(q.get("question", f"Question {i+1}")[:50])

        details += f"Q{i+1}: {q['question']}\n"
        details += f"Correct Answer: {correct_ans}\n"
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

    result_text = f"**Quiz Results -- {topic}**\n\n"
    result_text += f"**Score: {score}%** ({correct_count}/{len(questions)} correct)\n\n"
    result_text += feedback

    if score < 70 and attempts < 3:
        result_text += f"\n\nKeep going, {safe_name}! Let's try again on your weak areas. (Attempt {attempts}/3)"
        return {
            "quiz_score": score,
            "weak_topics": list(set(state.get("weak_topics", []) + [topic])),
            "response": result_text,
            "messages": [AIMessage(content=result_text)],
            "error": "",
        }

    if score >= 70:
        result_text += f"\n\nWell done, {safe_name}! You passed the quiz!"
    else:
        result_text += f"\n\nGood effort, {safe_name}. Let's add these to your study plan and review them."

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
Create a personalised 7-day study plan for {student}.
Address {student} by name in the introduction of the plan.

The student has struggled with: {weak_topics}.
Create a 7-day study plan that focuses SPECIFICALLY on these topics.
Day 1 and Day 2 must cover the weakest topic first.
Each day must have: topic, what to study, one practice task.
Do not include topics not in the weak areas list.

Structure the plan day by day with:
- Specific topics to cover each day
- Recommended study time
- Practice exercises or activities
- Review sessions

Make it realistic and encouraging.
"""


def study_plan_node(state: AgentState) -> dict[str, Any]:
    """Generate a 7-day study plan and pause for HITL approval."""
    safe_name = state.get("student_name", "").strip()
    if not safe_name or safe_name.lower() in ["", "none", "null"]:
        safe_name = "Student"
        
    topic = state.get("topic", "general")
    session_id = state.get("session_id", "")
    weak_topics = state.get("weak_topics", [])
    messages = state.get("messages", [])

    if not weak_topics:
        no_data_msg = (f"{safe_name}, please complete at least one quiz first "
                       "so I can personalise your study plan based on your weak areas.")
        return {
            "response": no_data_msg,
            "messages": [AIMessage(content=no_data_msg)],
            "error": "",
        }

    coll_name = f"user_{session_id.replace('-', '')}" if session_id else "studymate_notes"
    has_notes = collection_count(collection_name=coll_name) > 0
    rag_ctx = ""
    if has_notes:
        try:
            rag_ctx = retrieve_as_text(topic, k=10, collection_name=coll_name)
        except Exception:
            pass

    llm = _get_llm(temperature=0.7)
    prompt = STUDY_PLAN_PROMPT.format(
        student=safe_name,
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

    response = "**Your Personalized 7-Day Study Plan**\n\n"
    response += plan
    response += "\n\n---\n"
    response += "**Please review the plan above.**\n"
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
                "response": "Study plan approved and saved! You can view it anytime in your session history.",
                "messages": [AIMessage(content="Study plan approved and saved!")],
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
            "response": "Study plan was not approved. Feel free to request a new one anytime!",
            "messages": [AIMessage(content="Study plan not approved.")],
            "pending_plan": "",
            "plan_approved": False,
            "error": "",
        }


# ── 5. RAG QUERY NODE ───────────────────────────────────────────────────────

def rag_query_node(state: AgentState) -> dict[str, Any]:
    """Answer a question using retrieved context from uploaded notes."""
    messages = state.get("messages", [])
    safe_name = state.get("student_name", "").strip()
    if not safe_name or safe_name.lower() in ["", "none", "null"]:
        safe_name = "Student"

    last_msg = messages[-1] if messages else None
    user_text = last_msg.content if last_msg and hasattr(last_msg, "content") else "What are the key concepts?"

    session_id = state.get("session_id", "")
    coll_name = f"user_{session_id.replace('-', '')}" if session_id else "studymate_notes"
    
    # Check if there are any documents in the vector store first
    if collection_count(collection_name=coll_name) == 0:
        no_ctx = ("I don't have any uploaded notes to search through yet. "
                  "Please upload a PDF in the sidebar first, then ask your question again!")
        return {
            "response": no_ctx,
            "messages": [AIMessage(content=no_ctx)],
            "rag_context": "",
            "error": "",
        }

    chunks = retrieve_context(user_text, n_results=3, collection_name=coll_name)
    
    if not chunks or all(c.strip() == "" for c in chunks):
        # Truly no content found
        response = (f"I searched your uploaded notes but could not "
                    f"find relevant content on this topic, {safe_name}. "
                    f"Here is a general explanation instead:\n\n")
        # Then call LLM with general knowledge fallback
        use_rag = False
        context_text = ""
    else:
        # Content found — inject into prompt
        context_text = "\n\n---\n\n".join(chunks)
        use_rag = True

    if use_rag:
        system_content = f"""You are a study tutor helping {safe_name}.
Answer using the context below. Stay grounded to this content.
If the context partially covers the topic, use what is there 
and note what is not covered.
Do not say the notes don't cover something if any relevant 
content exists in the context.

CONTEXT FROM UPLOADED NOTES:
{context_text}

Answer the student's question using this context."""
    else:
        system_content = f"""You are a study tutor helping {safe_name}.
The student's uploaded notes do not contain this topic.
Provide a clear general explanation.
Start with: '{safe_name}, your uploaded notes do not cover this 
topic directly. Here is a general explanation:'"""

    llm = _get_llm(temperature=0.3)
    conv = [SystemMessage(content=system_content), HumanMessage(content=user_text)]
    response = _safe_invoke(llm, conv)

    return {
        "response": response,
        "messages": [AIMessage(content=response)],
        "rag_context": context_text,
        "error": "",
    }
