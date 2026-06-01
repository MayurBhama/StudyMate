# StudyMate -- Complete Learning Guide
# Personal reference only. Do not push to GitHub.
# Last updated: 2026-05-28

---

## 0. How to read this guide

This guide follows the exact order we built and fixed the project. Each section explains: what we built, why we built it that way, what problems we fixed, and how to explain it in an interview.

Start from Section 1 and read sequentially. Every concept builds on the previous one. When you see a code snippet, open the actual file side by side and trace through it. The code snippets in this guide are copied verbatim from the project files -- nothing is paraphrased or simplified. If you understand every snippet in this guide, you understand the entire project.

The sections are ordered as follows:

- Section 1: What the project does and why each tool was chosen.
- Section 2: Every file in the project, what it does, what every function does.
- Section 3: LangGraph concepts as they appear in our code (the core of this guide).
- Section 4: The RAG pipeline end to end.
- Section 5: Hallucination problems we found and how we fixed them.
- Section 6: How Streamlit connects to the LangGraph agent.
- Section 7: The student name personalisation fix.
- Section 8: LangSmith observability.
- Section 9: Every test explained.
- Section 10: 15 interview questions with detailed answers.
- Section 11: ASCII architecture diagram.

---

## 1. Project overview

### What StudyMate does

StudyMate is a personal AI tutor that does four things:

1. **Explain**: You type "explain binary search" and it gives you a clear, structured explanation with examples and analogies. If you have uploaded notes, it uses your notes as context for the explanation.

2. **Quiz**: You type "quiz me on sorting algorithms" and it generates 5 multiple-choice questions. You answer them, it grades you, and if you score below 70% it generates a new quiz focused on your weak areas. This retries up to 3 times.

3. **Study Plan**: You type "make me a study plan" and it generates a 7-day study plan based on your weak topics. Before saving it, it pauses and asks you to approve or reject the plan. This is Human-in-the-Loop.

4. **RAG Query**: You upload a PDF of your class notes. Then you type "what are the key points from chapter 3" and it searches your notes, finds the relevant chunks, and answers from your own material.

Think of it this way: a normal chatbot is just one function that takes input and returns output. StudyMate is a workflow -- it routes your message to the right handler, manages state across the conversation, retries quizzes in a loop, pauses for human approval, and persists your weak topics across sessions.

### Why LangGraph and not a simple chatbot

A simple chatbot is a single function: take user input, call the LLM, return the response. That works for basic Q&A but breaks down the moment you need:

- **Branching logic**: "If the user asks for a quiz, go to the quiz handler. If they ask for an explanation, go to the explain handler." In a simple chatbot, this is a chain of if-else statements that grows ugly fast.

- **Loops**: "If the student scores below 70%, generate a new quiz and try again, up to 3 times." Without LangGraph, you would write something like:

```python
# Without LangGraph -- messy imperative code
for attempt in range(3):
    questions = generate_quiz(topic)
    answers = get_student_answers(questions)
    score = evaluate(answers, questions)
    if score >= 70:
        break
    weak_topics = find_weak_areas(answers, questions)
    # Now somehow pass weak_topics back to generate_quiz...
```

This looks simple in pseudocode but gets tangled when you add error handling, state management, and UI integration.

- **Human checkpoints**: "Generate a study plan, then PAUSE execution until the student approves it." There is no clean way to do this in a regular Python script. You would need to save state to a file, exit the program, and reload when the student clicks approve.

- **State management**: Every node needs to read and write shared state -- the current topic, weak topics, quiz score, quiz attempts, messages. Without a framework, you are passing dictionaries around and hoping nothing gets lost.

LangGraph solves all of this. It gives you a graph where:
- **Nodes** are Python functions (explain, quiz, study plan, etc.)
- **Edges** are connections between nodes (router -> explain, quiz_evaluate -> quiz_generate for retry)
- **State** is a typed dictionary that flows through every node
- **Conditional edges** let you branch based on state values
- **Interrupts** let you pause the graph for human input
- **Checkpointers** save state so you can resume later

With LangGraph, the retry loop is just a conditional edge that points back to the quiz_generate node. The human checkpoint is just `interrupt_before=["study_plan_save"]`. The routing is just `add_conditional_edges` with a mapping dictionary. The framework handles all the plumbing.

### The complete tech stack -- why each tool was chosen

| Tool | What it does in this project | Why this over alternatives | Cost |
|------|------------------------------|---------------------------|------|
| **LangGraph** | Defines the agent as a state graph with nodes, edges, conditional routing, loops, and human-in-the-loop interrupts. | LangChain alone gives you chains (linear pipelines). LangGraph gives you graphs (branching, looping, interrupts). We needed all three. CrewAI and AutoGen are for multi-agent systems; we have a single agent with multiple workflows. | Free and open source. |
| **LangChain** | Provides base classes (ChatGroq, HumanMessage, AIMessage, Document), text splitters, and the message system that LangGraph builds on. | It is the standard library for LLM applications. Every other tool in this stack integrates with it. | Free and open source. |
| **Groq API (llama-3.1-8b-instant)** | Runs the LLM inference. Every node that calls the LLM goes through Groq. The model used is llama-3.1-8b-instant. | Groq is free for development with generous rate limits. OpenAI costs money per token. For a fresher building a portfolio project, free matters. Groq also has extremely fast inference because it runs on custom LPU hardware. | Free tier available. No credit card required for development. |
| **ChromaDB** | Stores vector embeddings of uploaded PDF chunks. When the student asks a question, ChromaDB finds the most similar chunks using cosine similarity. | It is the simplest vector database to set up -- no server, no configuration, just a Python package. Pinecone requires an account and API key. FAISS has no persistence built in. ChromaDB persists to disk automatically. | Free and open source. Runs locally. |
| **sentence-transformers (all-MiniLM-L6-v2)** | Converts text into 384-dimensional embedding vectors. Used for both indexing PDF chunks and encoding search queries. | It runs locally on CPU -- no API calls, no cost, no latency. OpenAI embeddings cost money per token. all-MiniLM-L6-v2 is small (80MB), fast, and produces good quality embeddings for English text. | Free. Runs locally on CPU. |
| **SQLite** | Stores quiz scores, weak topics, study plans, and session metadata on disk so that student progress survives server restarts. | It requires zero setup -- no database server, no configuration file. The database is a single file (studymate.db). PostgreSQL would be overkill for a single-user study app. | Free. Built into Python. |
| **Streamlit** | The web UI. Renders the chat interface, quiz forms, PDF uploader, sidebar with weak topics, and approve/reject buttons. | It is the fastest way to build a Python web app. Flask or FastAPI would require writing HTML templates and JavaScript. Streamlit lets you write pure Python and get a reactive UI. | Free and open source. |
| **LangSmith** | Traces every LLM call, showing input prompts, output responses, token counts, latency, and costs. Used for debugging and observability. | It is built by the LangChain team and integrates with zero configuration -- just set three environment variables. | Free tier available with 5000 traces per month. |
| **python-dotenv** | Loads environment variables from a `.env` file so API keys are not hardcoded in source code. | Standard practice for managing secrets in Python projects. | Free. |
| **pypdf** | Reads PDF files and extracts text page by page. | Lightweight and reliable. PyMuPDF (fitz) is faster but has complex licensing. pdfplumber is heavier than needed for simple text extraction. | Free and open source. |

---

## 2. Project structure -- every file explained

### agent/state.py

**What this file does**: Defines the `AgentState` TypedDict -- the single data structure that flows through every node in the LangGraph workflow.

**Full path**: `agent/state.py`

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    student_name: str
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
```

**Classes and keys**:

- `AgentState` (TypedDict, total=False): The central state schema. `total=False` means all keys are optional at construction time -- you can create an `AgentState` with only the keys you need. Every node reads from this dict and returns a partial update dict.

**How it connects to other files**: Every node in `agent/nodes.py` takes `AgentState` as its argument. The graph in `agent/graph.py` is typed with `StateGraph(AgentState)`. The Streamlit app in `app.py` constructs an `AgentState` dict before invoking the graph.

---

### agent/memory.py

**What this file does**: Provides short-term memory (message trimming) and long-term memory (SQLite session load/save) helpers.

**Full path**: `agent/memory.py`

```python
MAX_SHORT_TERM_MESSAGES = 10
```

**Functions**:

- `trim_messages(messages, max_messages=10) -> list[BaseMessage]`: Takes the full message list and returns only the last `max_messages` messages. If there are 15 messages, it returns messages 5 through 14 (0-indexed). Called by `explain_node` and `study_plan_node` in `nodes.py` to keep the context window small.

- `load_session_memory(session_id) -> dict`: Loads persisted state from SQLite for a given session. Returns a dict with `student_name`, `weak_topics`, `quiz_score`, and `session_id` that can be merged directly into `AgentState`. Called by `app.py` when the student enters their name.

- `persist_weak_topics(session_id, topics) -> None`: Writes weak topics to SQLite. Called by `quiz_evaluate_node` in `nodes.py` after grading a quiz.

- `ensure_session(student_name, session_id=None) -> str`: If a `session_id` is provided and exists in the database, returns it (updating the name if changed). If not, creates a new session and returns its UUID. Called by `app.py` when the student enters their name or when a chat starts without a session.

**How it connects to other files**: Imports from `db/sqlite_store.py` for all database operations. Imported by `agent/nodes.py` for `trim_messages` and `persist_weak_topics`. Imported by `app.py` for `ensure_session`, `load_session_memory`, and `trim_messages`.

---

### agent/nodes.py

**What this file does**: Contains every node function for the LangGraph workflow. Each function takes `AgentState` and returns a partial state update dict. This is the largest file in the project (560 lines).

**Full path**: `agent/nodes.py`

**Helper functions**:

- `_get_llm(temperature=0.3) -> ChatGroq`: Creates a ChatGroq instance using the GROQ_API_KEY from the environment. The model is `llama-3.1-8b-instant`. Temperature defaults to 0.3 (low randomness) but is overridden to 0.0 for routing (deterministic), 0.5 for explanations (some creativity), and 0.7 for quiz generation and study plans (more variety).

- `_safe_invoke(llm, messages, fallback) -> str`: Wraps `llm.invoke(messages)` in a try/except. If the LLM call fails, returns the fallback string instead of crashing the entire workflow.

- `_extract_topic(user_text, current_topic) -> str`: Uses the LLM to extract the main study topic from the user's message (2-5 words). If no clear topic is found, keeps the current topic. Returns "general" as a last resort.

- `_parse_quiz_json(text) -> list[dict]`: Robustly extracts a JSON array of quiz questions from LLM output. Tries three strategies in order: (1) parse the entire text as JSON, (2) find a JSON object `{...}` in the text, (3) find a JSON array `[...]` in the text. Returns an empty list if all three fail.

**Node functions**:

- `router_node(state) -> dict`: Reads the last message from `state["messages"]`. Checks if a PDF has been uploaded (via `collection_count()`). Sends the message to the LLM with either `ROUTE_PROMPT_WITH_NOTES` or `ROUTE_PROMPT_WITHOUT_NOTES`. Parses the LLM response to one of `{explain, quiz, study_plan, rag_query}`. If the LLM returns something unexpected, falls back to "explain". Also extracts the topic. Returns `{"route": ..., "topic": ..., "error": ""}`.

- `explain_node(state) -> dict`: Reads `topic`, `student_name`, and `messages` from state. Tries to retrieve RAG context. Builds a system prompt that includes the student's name and (optionally) RAG context. Calls `trim_messages` on the conversation. Invokes the LLM. Returns `{"response": ..., "messages": [AIMessage(...)], "rag_context": ..., "error": ""}`.

- `quiz_generate_node(state) -> dict`: Reads `topic`, `weak_topics`, and `quiz_attempts`. If notes are uploaded, retrieves RAG context. If this is a retry (attempts > 0), adds a "focus on weak areas" instruction and tells the LLM not to repeat previous questions. Binds JSON output format to the LLM. Parses the response into a list of question dicts. Returns `{"quiz_questions": [...], "quiz_attempts": attempts + 1, "response": ..., "messages": [...], "error": ""}`.

- `quiz_evaluate_node(state) -> dict`: Reads `quiz_questions`, `quiz_answers`, `session_id`, `topic`, and `quiz_attempts`. Compares each student answer to the correct answer. Computes a score (0-100). Identifies weak areas (questions answered incorrectly). Generates detailed feedback via the LLM. Saves the score and weak topics to SQLite. If score < 70 and attempts < 3, returns state that triggers a retry. Otherwise, resets quiz state and returns final results.

- `study_plan_node(state) -> dict`: Reads `student_name`, `topic`, `weak_topics`, and `messages`. If no weak topics exist, uses the current topic. Retrieves RAG context if notes are uploaded. Generates a 7-day study plan via the LLM. Sets `pending_plan` and `plan_approved = None` (waiting for HITL). Returns the plan as `response`.

- `study_plan_save_node(state) -> dict`: Reads `pending_plan`, `plan_approved`, and `session_id`. If approved and plan exists, saves to SQLite and returns a success message. If not approved, returns a rejection message. In both cases, clears `pending_plan`.

- `rag_query_node(state) -> dict`: Reads the last message from `messages` and `student_name`. Checks if documents exist in ChromaDB. If not, returns a message asking the student to upload a PDF. If yes, retrieves the top 3 chunks, builds a system prompt that grounds the LLM in the retrieved context, and returns the answer.

**How it connects to other files**: Imports `AgentState` from `agent/state.py`. Imports `trim_messages` and `persist_weak_topics` from `agent/memory.py`. Imports `save_quiz_score`, `save_study_plan`, `get_weak_topics` from `db/sqlite_store.py`. Imports `retrieve_as_text` and `collection_count` from `rag/retriever.py`. Every function is registered as a node in `agent/graph.py`.

---

### agent/graph.py

**What this file does**: Builds the LangGraph `StateGraph`, registers all nodes, defines edges (sequential, conditional, and iterative), and compiles the graph with a `MemorySaver` checkpointer and HITL interrupt.

**Full path**: `agent/graph.py`

**Functions**:

- `_route_decision(state) -> str`: Reads `state["route"]` and returns it. If the key is missing, returns "explain" as a default. This is the function passed to `add_conditional_edges` for the router.

- `_quiz_should_retry(state) -> str`: Reads `quiz_score` and `quiz_attempts`. If score < 70 and attempts < 3, returns `"quiz_generate"` (loop back). Otherwise returns `END`. This is the function passed to `add_conditional_edges` for the quiz retry loop.

- `build_graph() -> StateGraph`: The main function. Creates a `StateGraph(AgentState)`, adds all 7 nodes, sets the entry point to "router", adds conditional edges for routing and quiz retry, adds sequential edges for explain/rag_query to END and study_plan to study_plan_save. Compiles with `MemorySaver()` checkpointer and `interrupt_before=["study_plan_save"]`.

- `study_graph`: A pre-built graph instance created by calling `build_graph()` at module level. Imported by `app.py` for convenience.

**How it connects to other files**: Imports `AgentState` from `agent/state.py`. Imports all node functions from `agent/nodes.py`. Imported by `app.py` as `study_graph`.

---

### rag/loader.py

**What this file does**: Reads PDF files and splits them into small text chunks suitable for embedding.

**Full path**: `rag/loader.py`

**Functions**:

- `load_pdf(file) -> list[Document]`: Accepts a file path (string) or an in-memory file object (from Streamlit's `st.file_uploader`). Uses `pypdf.PdfReader` to extract text from each page. Returns one `Document` per page with metadata containing the source filename and page number. Skips blank pages.

- `chunk_documents(docs, chunk_size=500, chunk_overlap=50) -> list[Document]`: Takes a list of Documents and splits them using `RecursiveCharacterTextSplitter`. The splitter tries to split on paragraph breaks first (`\n\n`), then line breaks (`\n`), then sentences (`. `), then words (` `), then characters (`""`). Each chunk is at most 500 characters with 50 characters of overlap between adjacent chunks to preserve context at boundaries.

- `load_and_chunk_pdf(file, chunk_size=500, chunk_overlap=50) -> list[Document]`: Convenience wrapper that calls `load_pdf` then `chunk_documents` in one step.

**How it connects to other files**: Imported by `rag/retriever.py` for `load_and_chunk_pdf`. The `Document` class comes from `langchain_core.documents`.

---

### rag/retriever.py

**What this file does**: Manages the ChromaDB vector store -- embedding, storing, and retrieving document chunks.

**Full path**: `rag/retriever.py`

**Constants**:

- `CHROMA_DIR`: Path to the `chroma_db` directory at the project root. ChromaDB persists its data here.
- `COLLECTION_NAME`: `"studymate_notes"` -- the name of the ChromaDB collection.
- `EMBEDDING_MODEL`: `"all-MiniLM-L6-v2"` -- the sentence-transformers model used for embeddings.

**Functions**:

- `get_embeddings() -> HuggingFaceEmbeddings`: Lazily creates and caches a `HuggingFaceEmbeddings` instance. Runs on CPU with normalized embeddings. Singleton pattern -- created once, reused for all calls.

- `get_vectorstore(persist_directory) -> Chroma`: Lazily creates and caches a `Chroma` instance. Uses the shared embeddings function and the `studymate_notes` collection. Singleton pattern.

- `reset_vectorstore() -> None`: Sets the cached vectorstore to `None`. Used by tests to get a fresh store for each test.

- `add_documents(docs, persist_directory) -> int`: Adds pre-chunked documents to ChromaDB in batches of 150. Returns the number of chunks added.

- `ingest_pdf(file, persist_directory) -> int`: The main ingestion function. Deletes the existing collection (so each PDF replaces the previous one), resets the singleton, loads and chunks the PDF, then adds the chunks. Returns the number of chunks indexed.

- `retrieve(query, k=3, persist_directory) -> list[Document]`: Runs a similarity search on ChromaDB and returns the top-k most relevant document chunks. Returns an empty list if the collection is empty or an error occurs.

- `retrieve_as_text(query, k=3, persist_directory) -> str`: Calls `retrieve` and formats the results into a single string with chunk numbers and source metadata. Each chunk is labeled like `[Chunk 1 -- bio.pdf p.3]`.

- `collection_count(persist_directory) -> int`: Returns the number of documents currently in the ChromaDB collection. Used by `router_node` to decide which routing prompt to use (with or without notes).

**How it connects to other files**: Imports `load_and_chunk_pdf` from `rag/loader.py`. Imported by `agent/nodes.py` for `retrieve_as_text` and `collection_count`. Imported by `app.py` for `ingest_pdf` and `collection_count`.

---

### db/sqlite_store.py

**What this file does**: The SQLite persistence layer. Creates and manages four tables: sessions, quiz_scores, weak_topics, and study_plans.

**Full path**: `db/sqlite_store.py`

**Helper**:

- `_get_conn(db_path) -> contextmanager`: A context manager that yields a SQLite connection with WAL mode and foreign keys enabled. Auto-commits on success, auto-rolls-back on exception.

**Functions**:

- `init_db(db_path)`: Creates the four tables if they do not exist. Called automatically at module import time (`init_db()` at the bottom of the file).

- `create_session(student_name, db_path) -> str`: Generates a UUID, inserts a new row into the sessions table with the student name and current timestamp. Returns the UUID.

- `get_session(session_id, db_path) -> dict | None`: Looks up a session by ID. Returns the row as a dict or None if not found.

- `update_session_name(session_id, student_name, db_path)`: Updates the student name for an existing session.

- `list_sessions(db_path) -> list[dict]`: Returns all sessions ordered by most recent first.

- `save_quiz_score(session_id, topic, score, attempts, weak_areas, db_path)`: Inserts a quiz score row with the topic, score (0-100), attempt count, and weak areas (stored as a JSON string).

- `get_quiz_scores(session_id, db_path) -> list[dict]`: Returns quiz scores for a session, most recent first. Deduplicates by topic (only the latest score per topic is returned). Parses the `weak_areas` JSON string back into a list.

- `save_weak_topics(session_id, topics, db_path)`: Replaces all weak topics for a session. First deletes existing rows, then inserts the new list. This is a full replacement, not an append.

- `get_weak_topics(session_id, db_path) -> list[str]`: Returns distinct weak topics for a session, ordered by most recent.

- `save_study_plan(session_id, plan_text, approved, db_path) -> int`: Inserts a study plan row. Returns the auto-incremented row ID.

- `approve_study_plan(plan_id, db_path)`: Sets `approved = 1` for a specific plan row.

- `get_latest_plan(session_id, db_path) -> dict | None`: Returns the most recent study plan for a session.

**How it connects to other files**: Imported by `agent/memory.py` for session and weak topic operations. Imported by `agent/nodes.py` for `save_quiz_score`, `save_study_plan`, `get_weak_topics`. Imported by `app.py` for all session and data retrieval operations.

---

### app.py

**What this file does**: The Streamlit entry point. Renders the UI, manages session state, handles user input, invokes the LangGraph agent, and displays responses.

**Full path**: `app.py`

**Key functions**:

- `_init_session_state()`: Initializes all `st.session_state` keys with default values on first load. Keys include `session_id`, `student_name`, `messages`, `weak_topics`, `quiz_score`, `quiz_questions`, `quiz_attempts`, `quiz_answers`, `pending_plan`, `plan_approved`, `topic`, `awaiting_quiz_answers`, and `pdf_uploaded`.

- `_handle_plan_approval(approved)`: Processes HITL plan approval or rejection. If approved, calls `study_graph.update_state` to set `plan_approved = True`, then calls `study_graph.invoke(None, config)` to resume the graph from the interrupt point. If rejected, clears the pending plan.

**UI sections** (in order of rendering):

1. **Sidebar**: Student name input, PDF uploader, weak topics display.
2. **Main header**: Title and feature list.
3. **Chat history**: Renders all messages from `st.session_state.messages`.
4. **HITL approval**: Shows Approve/Reject buttons when a plan is pending.
5. **Quiz form**: Shows radio buttons for quiz answers when `awaiting_quiz_answers` is True.
6. **Chat input**: The main text input at the bottom. Triggers graph invocation.

**How it connects to other files**: Imports `study_graph` from `agent/graph.py`. Imports memory functions from `agent/memory.py`. Imports database functions from `db/sqlite_store.py`. Imports RAG functions from `rag/retriever.py`. Imports node functions directly from `agent/nodes.py` for quiz evaluation (bypassing the graph for interactive quiz flow).

---

## 3. LangGraph concepts -- exactly as used in this project

### 3A. StateGraph and AgentState

**What is a TypedDict?**

A `TypedDict` is a Python type hint that describes a dictionary with specific keys and value types. It does not enforce types at runtime -- it is only used by type checkers (like mypy) and by LangGraph to understand the state schema.

```python
from typing_extensions import TypedDict

class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    student_name: str
    topic: str
    ...
```

The `total=False` parameter means every key is optional. You can create `AgentState` with just `{"messages": [...]}` and it will not complain about missing keys. This is important because nodes return partial updates -- `explain_node` returns `{"response": ..., "messages": [...]}` without touching `quiz_score` or `weak_topics`.

**Every key in AgentState explained**:

| Key | Type | What it stores | When updated | Read by | Written by |
|-----|------|----------------|-------------|---------|------------|
| `messages` | `Annotated[list, add_messages]` | Full conversation history. The `add_messages` reducer means new messages are appended, not replaced. | Every node that adds a response | Every node (for context) | Every node (appends AIMessage) |
| `student_name` | `str` | The student's name for personalisation | On startup from UI or SQLite | `explain_node`, `study_plan_node`, `rag_query_node` | `app.py` (from sidebar input) |
| `topic` | `str` | Current study topic (e.g. "binary search") | By `router_node` via `_extract_topic` | `explain_node`, `quiz_generate_node`, `study_plan_node` | `router_node` |
| `weak_topics` | `list[str]` | Topics the student scored poorly on | By `quiz_evaluate_node` after grading | `quiz_generate_node`, `study_plan_node` | `quiz_evaluate_node` |
| `quiz_score` | `float` | Latest quiz score (0-100) | By `quiz_evaluate_node` | `_quiz_should_retry` (in graph.py) | `quiz_evaluate_node` |
| `quiz_questions` | `list[dict]` | Current batch of MCQ questions | By `quiz_generate_node` | `quiz_evaluate_node` | `quiz_generate_node` |
| `quiz_attempts` | `int` | Number of quiz attempts in current cycle | By `quiz_generate_node` (incremented) | `_quiz_should_retry`, `quiz_generate_node`, `quiz_evaluate_node` | `quiz_generate_node` |
| `quiz_answers` | `list[str]` | Student's answers (e.g. ["A", "B", "C"]) | By `app.py` from the quiz form | `quiz_evaluate_node` | `app.py` (from Streamlit form) |
| `pending_plan` | `str` | Study plan text awaiting approval | By `study_plan_node` | `study_plan_save_node` | `study_plan_node`, `study_plan_save_node` |
| `plan_approved` | `bool or None` | HITL decision. `None` = waiting, `True` = approved, `False` = rejected | By `app.py` when student clicks Approve/Reject | `study_plan_save_node` | `app.py` via `graph.update_state` |
| `session_id` | `str` | UUID that identifies the current session in SQLite | On startup by `ensure_session` | `quiz_evaluate_node`, `study_plan_save_node` | `app.py` |
| `route` | `str` | Routing decision (explain/quiz/study_plan/rag_query) | By `router_node` | `_route_decision` (in graph.py) | `router_node` |
| `rag_context` | `str` | Retrieved text chunks from ChromaDB | By nodes that do RAG retrieval | Informational (logged) | `explain_node`, `rag_query_node` |
| `response` | `str` | Final response text for the UI | By every content-producing node | `app.py` for display | Every node |
| `error` | `str` | Error message for graceful degradation | When something goes wrong | `app.py` for error display | Any node on error |

**Why quiz_attempts exists**: Without `quiz_attempts`, the conditional edge `_quiz_should_retry` would check `score < 70` and loop back forever. The student might never score 70% on a topic, and the graph would generate quizzes indefinitely. `quiz_attempts` caps the loop at 3 iterations. After 3 attempts, the graph moves to END regardless of score.

**The add_messages reducer**: The `Annotated[list, add_messages]` annotation on the `messages` key tells LangGraph to use the `add_messages` reducer when merging state updates. Instead of replacing the entire list, new messages are appended to the existing list. This is critical because every node adds an `AIMessage` to the conversation without losing previous messages.

---

### 3B. Nodes -- what they are and how we wrote them

A node in LangGraph is just a Python function with this signature:

```python
def some_node(state: AgentState) -> dict[str, Any]:
    # 1. Read what you need from state
    topic = state.get("topic", "general")
    
    # 2. Do your work (call LLM, query database, etc.)
    response = llm.invoke(...)
    
    # 3. Return a PARTIAL state update
    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)],
        "error": "",
    }
```

The pattern every node follows:
1. Extract inputs from the state dict using `.get()` with defaults.
2. Do the actual work (LLM call, database query, RAG retrieval).
3. Return a dict containing ONLY the keys that changed. LangGraph merges this into the full state automatically.

The return dict is NOT a full `AgentState` -- it is a partial update. If `explain_node` returns `{"response": "...", "messages": [...]}`, LangGraph updates only those two keys and leaves everything else (like `quiz_score`, `weak_topics`, etc.) untouched.

**List of all nodes**:

| Node | Reads from state | Writes to state |
|------|-----------------|-----------------|
| `router_node` | `messages` | `route`, `topic`, `error` |
| `explain_node` | `topic`, `student_name`, `messages` | `response`, `messages`, `rag_context`, `error` |
| `quiz_generate_node` | `topic`, `weak_topics`, `quiz_attempts`, `quiz_questions` | `quiz_questions`, `quiz_attempts`, `response`, `messages`, `error` |
| `quiz_evaluate_node` | `quiz_questions`, `quiz_answers`, `session_id`, `topic`, `quiz_attempts`, `weak_topics` | `quiz_score`, `weak_topics`, `response`, `messages`, `error`, `quiz_questions`, `quiz_attempts` |
| `study_plan_node` | `student_name`, `topic`, `session_id`, `weak_topics`, `messages` | `pending_plan`, `plan_approved`, `response`, `messages`, `error` |
| `study_plan_save_node` | `pending_plan`, `plan_approved`, `session_id` | `response`, `messages`, `pending_plan`, `plan_approved`, `error` |
| `rag_query_node` | `messages`, `student_name` | `response`, `messages`, `rag_context`, `error` |

---

### 3C. Conditional workflow -- Router Node

The router is the entry point of the graph. Every user message goes to the router first, and the router decides which workflow to execute.

**How add_conditional_edges works**:

```python
graph.add_conditional_edges(
    "router",           # Source node
    _route_decision,    # Function that returns a string
    {                   # Mapping: string -> next node name
        "explain": "explain",
        "quiz": "quiz_generate",
        "study_plan": "study_plan",
        "rag_query": "rag_query",
    },
)
```

After `router_node` runs and sets `state["route"]`, LangGraph calls `_route_decision(state)` which returns the route string. It then looks up that string in the mapping dictionary to find the next node to execute.

**The _route_decision function**:

```python
def _route_decision(state: AgentState) -> str:
    return state.get("route", "explain")
```

This is intentionally simple. The actual classification logic lives in `router_node` in `nodes.py`. The `_route_decision` function just reads the result. The fallback `"explain"` protects against the case where `router_node` somehow fails to set the `route` key -- instead of crashing, the graph defaults to the explain workflow.

**How the router classifies intent**: The `router_node` sends the user's message to the LLM with a system prompt that says "classify into exactly one of: explain, quiz, study_plan, rag_query." There are two variants of the prompt:

- `ROUTE_PROMPT_WITHOUT_NOTES`: Used when no PDF has been uploaded. General questions go to "explain", academic questions also go to "explain" since there are no notes to search.
- `ROUTE_PROMPT_WITH_NOTES`: Used when a PDF has been uploaded. Academic questions are routed to "rag_query" so they can be answered from the student's notes. The prompt says "if in doubt, route to rag_query."

**Fuzzy fallback**: If the LLM returns something unexpected (like "I think this is an explanation request"), the router does a fuzzy match:

```python
route = result.strip().lower()
valid_routes = {"explain", "quiz", "study_plan", "rag_query"}
if route not in valid_routes:
    for r in valid_routes:
        if r in route:
            route = r
            break
    else:
        route = "explain"
```

This catches responses like "explain (the student wants an explanation)" by finding "explain" as a substring. If nothing matches, it defaults to "explain" -- the safest fallback because explaining something is never wrong.

---

### 3D. Sequential workflow -- Explain and Study Plan flow

A sequential edge is the simplest type: after node A finishes, always go to node B.

```python
graph.add_edge("explain", END)
graph.add_edge("rag_query", END)
```

These two lines mean: after `explain_node` runs, the graph ends. After `rag_query_node` runs, the graph ends. No conditions, no branching.

**The explain flow**:

```
User types "explain binary search"
  -> router_node: classifies as "explain", extracts topic "binary search"
  -> explain_node: generates explanation, returns response
  -> END
```

This is a two-node pipeline: router -> explain -> END.

**The study plan flow**:

```
User types "make me a study plan"
  -> router_node: classifies as "study_plan", extracts topic
  -> study_plan_node: generates 7-day plan, sets pending_plan, sets plan_approved=None
  -> INTERRUPT (graph pauses here, waiting for human approval)
  -> [student clicks Approve]
  -> study_plan_save_node: saves plan to SQLite, returns success message
  -> END
```

This is a three-node pipeline with an interrupt: router -> study_plan -> [INTERRUPT] -> study_plan_save -> END.

The edges that define this:

```python
graph.add_edge("study_plan", "study_plan_save")
graph.add_edge("study_plan_save", END)
```

The interrupt is defined at compile time:

```python
graph.compile(checkpointer=memory, interrupt_before=["study_plan_save"])
```

---

### 3E. Iterative workflow -- Quiz retry loop

This is the most important workflow in the project. It demonstrates something that would be very difficult without LangGraph: a conditional loop that retries based on state.

**What the loop does step by step**:

1. User types "quiz me on sorting algorithms".
2. `router_node` classifies as "quiz", extracts topic "sorting algorithms".
3. `quiz_generate_node` generates 5 MCQ questions. Sets `quiz_attempts = 1`. Graph ends (returns questions to UI).
4. Student answers the questions in the Streamlit form.
5. `quiz_evaluate_node` grades the answers. Computes score. Identifies weak areas.
6. If score >= 70%: quiz is done. Graph ends.
7. If score < 70% AND attempts < 3: generate a new quiz focused on weak areas (loop back to step 3 with `quiz_attempts = 2`).
8. If score < 70% AND attempts >= 3: quiz is done. Student has used all 3 attempts.

**The exact conditional edge code**:

```python
def _quiz_should_retry(state: AgentState) -> str:
    score = state.get("quiz_score", 0)
    attempts = state.get("quiz_attempts", 0)
    if score < 70 and attempts < 3:
        return "quiz_generate"
    return END
```

```python
graph.add_conditional_edges(
    "quiz_evaluate",
    _quiz_should_retry,
    {
        "quiz_generate": "quiz_generate",
        END: END,
    },
)
```

After `quiz_evaluate_node` finishes, LangGraph calls `_quiz_should_retry`. If the function returns `"quiz_generate"`, execution jumps back to `quiz_generate_node` -- creating a loop. If it returns `END`, the graph terminates.

**The score < 70% condition**: 70% is the passing threshold. If the student gets 4 out of 5 questions right (80%), they pass. If they get 3 out of 5 (60%), they fail and get a retry. This threshold is hardcoded in `_quiz_should_retry`.

**The quiz_attempts >= 3 condition**: This is the safety valve. Without it, a student who consistently scores below 70% would be stuck in an infinite loop. After 3 attempts, the graph moves to END regardless of score.

**What weak_topics collects during the loop**: Every time `quiz_evaluate_node` runs, it identifies questions the student got wrong and extracts the question text (truncated to 30 characters) as a "weak area". These are accumulated in `state["weak_topics"]`:

```python
all_weak = list(set(existing_weak + [topic] + [a[:30] for a in weak_areas]))
persist_weak_topics(session_id, all_weak)
```

On the next iteration, `quiz_generate_node` reads `weak_topics` and adds them to the LLM prompt:

```python
if attempts > 0 and weak_topics:
    weak_focus = f"Focus especially on these weak areas: {', '.join(weak_topics)}"
```

This means each retry quiz is harder and more targeted -- it focuses on exactly what the student got wrong.

**What happens after 3 attempts if score never hits 70%**: The `quiz_evaluate_node` detects that attempts >= 3 and returns this message:

```python
result_text += "\n\nYou've used all 3 attempts. Consider reviewing the topic and trying again later."
```

It also resets the quiz state:

```python
return {
    "quiz_score": score,
    "quiz_questions": [],
    "quiz_attempts": 0,
    "weak_topics": list(set(state.get("weak_topics", []) + ([topic] if score < 70 else []))),
    ...
}
```

The topic is still added to `weak_topics` so the study plan node can use it later.

**Why this could not be done without LangGraph easily**: In a simple chatbot, you would need to:
- Manually track `quiz_attempts` in a global variable or database
- Write a while loop that calls generate_quiz -> get_answers -> evaluate in sequence
- Handle the case where the loop needs to be interrupted (user closes the browser)
- Maintain state between iterations (weak_topics accumulating across retries)
- Integrate this with the rest of the chatbot's routing logic

With LangGraph, the loop is just one conditional edge. State is managed automatically. If you add a checkpointer, the loop can even survive server restarts.

---

### 3F. Human-in-the-Loop (HITL)

Human-in-the-Loop means the graph pauses execution and waits for a human decision before continuing. In StudyMate, this is used for study plan approval.

**What interrupt_before does**: When you compile the graph with `interrupt_before=["study_plan_save"]`, LangGraph will literally stop execution right before `study_plan_save_node` runs. The graph state is saved to the checkpointer, and `graph.invoke()` returns with the state as it was after `study_plan_node` finished.

```python
memory = MemorySaver()
return graph.compile(checkpointer=memory, interrupt_before=["study_plan_save"])
```

**Why we need MemorySaver checkpointer**: The interrupt only works if the graph can save its state and resume later. `MemorySaver` stores the state in memory (a Python dict). When the student clicks "Approve" and we call `graph.invoke(None, config)`, LangGraph loads the saved state from the checkpointer and resumes execution from where it stopped -- at the `study_plan_save_node`.

Without a checkpointer, `interrupt_before` would have no effect because there would be nowhere to save the paused state.

**What thread_id is and why every session needs its own**: The `thread_id` is a string that identifies a specific execution thread in the checkpointer. When you invoke the graph with `config = {"configurable": {"thread_id": session_id}}`, LangGraph uses this ID to look up the saved state. Each student session needs its own `thread_id` so that one student's paused graph does not interfere with another's.

```python
config = {"configurable": {"thread_id": st.session_state.session_id}}
```

We use the SQLite session UUID as the `thread_id`, which guarantees uniqueness.

**The compile() call**:

```python
memory = MemorySaver()
return graph.compile(checkpointer=memory, interrupt_before=["study_plan_save"])
```

This does three things:
1. Creates a `MemorySaver` instance (in-memory state storage).
2. Compiles the graph into an executable form.
3. Registers `study_plan_save` as an interrupt point.

**The config dict**:

```python
config = {"configurable": {"thread_id": session_id}}
```

This dict is passed to every `graph.invoke()` and `graph.stream()` call. The `thread_id` is the only required configurable for checkpointing.

**How graph.invoke(None, config) resumes**: When the student clicks "Approve" in the UI, `app.py` does this:

```python
study_graph.update_state(config, {"plan_approved": True})
result = study_graph.invoke(None, config=config)
```

The first line updates the saved state to set `plan_approved = True`. The second line invokes the graph with `None` as input (meaning "resume from where you left off") and the same `config` (same `thread_id`). LangGraph loads the saved state, sees that execution was paused before `study_plan_save_node`, and runs that node with the updated state.

**What happens if the user clicks Reject**: The `_handle_plan_approval` function sets `plan_approved = False` and clears `pending_plan`:

```python
elif not approved:
    st.session_state.pending_plan = None
    st.session_state.plan_approved = False
```

When rejected, the app also sets a trigger prompt that tells the agent to generate a new plan:

```python
st.session_state.trigger_prompt = "I reject the previous study plan. Please generate a new, completely different one for me."
```

This trigger prompt is picked up on the next rerun and sent through the graph as a new user message, which gets routed to `study_plan_node` again.

**Why this is fundamentally different from just using Streamlit buttons**: A naive approach would be to use Streamlit buttons to show/hide the plan without involving the graph at all. But that misses the point:

- With Streamlit buttons alone, the plan is just displayed text. There is no graph state, no checkpoint, no connection to the agent's workflow.
- With HITL, the graph is literally paused mid-execution. The `study_plan_save_node` has not run yet. The approval decision is injected into the graph state, and the save node runs with that decision.
- This means the graph controls the logic, not the UI. If you wanted to add a "revise plan" option that loops back to `study_plan_node`, you would just add another conditional edge -- no UI changes needed.

---

### 3G. Short-term memory

Short-term memory in StudyMate is the `messages` list in `AgentState`. It stores the conversation history as a list of `HumanMessage` and `AIMessage` objects.

**The trim_messages function**:

```python
def trim_messages(messages: list[BaseMessage], max_messages: int = MAX_SHORT_TERM_MESSAGES) -> list[BaseMessage]:
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]
```

This keeps only the last 10 messages. If the conversation has 15 messages, it returns messages at indices 5 through 14 -- the 10 most recent.

**What happens to trimmed messages**: They are gone from the context window. The LLM will not see them. But they are not deleted from `st.session_state.messages` in the UI -- the student can still scroll up and see old messages. The trimming only affects what gets sent to the LLM as context.

In `app.py`, trimming happens after every graph invocation:

```python
st.session_state.messages = trim_messages(
    st.session_state.messages, max_messages=10
)
```

**Why 10 is a reasonable limit**: The LLM has a context window limit (measured in tokens). Each message consumes tokens. With a small model like llama-3.1-8b-instant, the context window is 131072 tokens, but sending too many messages:
- Increases latency (more tokens to process)
- Increases cost (more tokens billed, if applicable)
- Can confuse the LLM (older messages may contradict newer context)

10 messages is about 5 back-and-forth exchanges, which is enough for the LLM to understand the current conversation thread without being overwhelmed by ancient history.

---

### 3H. Long-term memory with SQLite

Long-term memory stores facts that survive across sessions: the student's name, their weak topics, their quiz scores, and their approved study plans.

**What gets saved**:

| Data | Table | When saved | Why it matters |
|------|-------|------------|----------------|
| Student name | `sessions` | When student types their name | Personalisation in prompts |
| Weak topics | `weak_topics` | After every quiz evaluation | Study plan generation, quiz focus |
| Quiz score | `quiz_scores` | After every quiz evaluation | Tracking progress |
| Study plan | `study_plans` | When student approves a plan | Reference for future sessions |

**The sqlite_store.py functions**:

- `create_session`: Generates a UUID, inserts a row with student_name and timestamps. This is the "birth" of a session.
- `get_session`: Looks up a session by ID. Returns None if not found.
- `save_weak_topics`: Deletes all existing weak topics for the session, then inserts the new list. This is a full replacement -- if the student improves on a topic, it can be removed.
- `get_weak_topics`: Returns all weak topics for a session.
- `save_quiz_score`: Inserts a new quiz score row. Does not replace -- all historical scores are kept.
- `get_quiz_scores`: Returns scores most recent first, deduplicated by topic.
- `save_study_plan`: Inserts a study plan row with an `approved` flag.

**Session ID**: Every session has a UUID (e.g. "a1b2c3d4-..."). This is generated by `create_session` and stored in `st.session_state.session_id`. All database operations use this ID as a foreign key.

**The startup load**: When the student enters their name in the sidebar, `app.py` calls:

```python
st.session_state.session_id = ensure_session(
    student_name, st.session_state.get("session_id") or None
)
mem = load_session_memory(st.session_state.session_id)
if mem:
    st.session_state.weak_topics = mem.get("weak_topics", [])
    st.session_state.quiz_score = mem.get("quiz_score", 0.0)
```

This loads any previously saved weak topics and quiz scores into the session state, so the student sees their progress from previous sessions.

**The difference between short-term and long-term memory**:

- **Short-term** = the `messages` list in `AgentState`. Lives in memory (Python objects). Trimmed to 10 messages. Lost when the server restarts. Purpose: give the LLM conversation context so it can follow the thread.

- **Long-term** = SQLite database on disk (`studymate.db`). Persists across restarts. Stores structured data (names, topics, scores, plans). Purpose: remember facts about the student that matter across sessions.

Think of it like human memory: short-term is "what we were just talking about" and long-term is "I know this student is weak at calculus."

---

## 4. RAG pipeline -- complete walkthrough

### 4A. What RAG is and why we need it

RAG stands for Retrieval-Augmented Generation. It is a technique where you:
1. Store documents in a searchable database.
2. When the user asks a question, search the database for relevant passages.
3. Include those passages in the LLM prompt as context.
4. The LLM answers based on the retrieved context, not just its training data.

Why we need it: the LLM was trained on public internet data. It does not know what is in the student's class notes, textbook, or professor's slides. Without RAG, the LLM would either guess (hallucinate) or say "I don't know." With RAG, we give the LLM the actual content from the student's notes and say "answer based on this."

### 4B. PDF loading -- loader.py

The first step is getting text out of a PDF. The `load_pdf` function handles two cases:

```python
def load_pdf(file: str | BinaryIO) -> list[Document]:
    if isinstance(file, str):
        reader = PdfReader(file)
        source = file
    else:
        reader = PdfReader(io.BytesIO(file.read()) if hasattr(file, "read") else file)
        source = getattr(file, "name", "uploaded_pdf")

    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(page_content=text, metadata={"source": source, "page": i + 1})
            )
    return docs
```

- If `file` is a string (file path), it reads directly from disk.
- If `file` is a file object (from Streamlit's `st.file_uploader`), it wraps it in `BytesIO` first.
- It extracts text from each page and creates a `Document` object with the text and metadata (source filename and page number).
- Blank pages are skipped.

Then `chunk_documents` splits long pages into smaller pieces:

```python
def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
```

Why 500 characters? LLMs work best with focused, specific context. A full page of text (2000+ characters) contains too many topics and dilutes the search results. 500 characters is roughly one paragraph -- focused enough to be relevant, long enough to be coherent.

Why 50 character overlap? Without overlap, a sentence that falls exactly at a chunk boundary would be split in half. The 50-character overlap ensures that boundary sentences appear in both chunks, so they can be found by search regardless of which chunk they land in.

### 4C. Embeddings -- why local and what they produce

An embedding is a list of numbers (a vector) that represents the meaning of a piece of text. Texts with similar meanings have similar vectors.

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings
```

The `all-MiniLM-L6-v2` model converts any text into a 384-dimensional vector. For example:
- "photosynthesis in plants" might become `[0.12, -0.34, 0.56, ..., 0.78]` (384 numbers)
- "how do plants make food" might become `[0.11, -0.33, 0.55, ..., 0.79]` (very similar vector)
- "Newton's second law" might become `[0.89, 0.12, -0.45, ..., -0.23]` (very different vector)

When we search for "how do plants make food", ChromaDB compares the query vector to all stored chunk vectors and returns the ones with the highest cosine similarity.

Why local (not API)? The `all-MiniLM-L6-v2` model runs entirely on the student's machine. No API calls, no latency, no cost. OpenAI's embedding API charges per token. For a fresher's project, free is important.

The `normalize_embeddings=True` parameter ensures all vectors have unit length (magnitude = 1), which is required for cosine similarity to work correctly.

### 4D. ChromaDB -- storing and searching

ChromaDB is a vector database. It stores embedding vectors alongside the original text and metadata, and provides fast similarity search.

```python
CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "studymate_notes"
```

The vector store is persisted to the `chroma_db` directory. A collection is like a table in SQL -- all of the student's note chunks go into the `studymate_notes` collection.

**Adding documents**:

```python
def add_documents(docs, persist_directory=CHROMA_DIR):
    vs = get_vectorstore(persist_directory)
    batch_size = 150
    for i in range(0, len(docs), batch_size):
        vs.add_documents(docs[i : i + batch_size])
    return len(docs)
```

Documents are added in batches of 150 to avoid memory issues with large PDFs. Each document's text is automatically embedded by the `HuggingFaceEmbeddings` function before storage.

**Ingesting a PDF**:

```python
def ingest_pdf(file, persist_directory=CHROMA_DIR):
    vs = get_vectorstore(persist_directory)
    try:
        vs.delete_collection()
    except Exception:
        pass
    reset_vectorstore()
    chunks = load_and_chunk_pdf(file)
    if not chunks:
        return 0
    return add_documents(chunks, persist_directory)
```

Notice that `ingest_pdf` deletes the existing collection before adding new chunks. This means each PDF upload replaces the previous one. The design decision was simplicity -- a single student typically studies one subject at a time.

**Searching**:

```python
def retrieve(query, k=3, persist_directory=CHROMA_DIR):
    vs = get_vectorstore(persist_directory)
    try:
        results = vs.similarity_search(query, k=k)
    except Exception:
        results = []
    return results
```

`similarity_search` takes the query text, embeds it using the same model, and finds the `k` most similar chunks by cosine similarity.

### 4E. RAG in the graph -- end to end

Here is the complete flow when a student asks a question about their uploaded notes:

**Step 1: Student types "What is the difference between TCP and UDP?"**

The message is added to `st.session_state.messages` as a `HumanMessage`.

**Step 2: Router classifies as rag_query**

`router_node` checks `collection_count() > 0` (yes, a PDF is uploaded). It uses `ROUTE_PROMPT_WITH_NOTES` which says "if in doubt, route to rag_query." The LLM returns "rag_query". The router sets `state["route"] = "rag_query"`.

**Step 3: Conditional edge routes to rag_query_node**

`_route_decision(state)` returns "rag_query". The mapping in `add_conditional_edges` sends execution to the `rag_query` node.

**Step 4: rag_query_node retrieves context**

```python
def rag_query_node(state: AgentState) -> dict[str, Any]:
    messages = state.get("messages", [])
    student = state.get("student_name", "Student")
    last_msg = messages[-1] if messages else None
    user_text = last_msg.content if last_msg and hasattr(last_msg, "content") else "What are the key concepts?"

    if collection_count() == 0:
        # No documents uploaded
        ...

    rag_ctx = ""
    try:
        rag_ctx = retrieve_as_text(user_text, k=3)
    except Exception:
        pass
```

It takes the user's question, passes it to `retrieve_as_text`, and gets back the top 3 most relevant chunks from ChromaDB. Each chunk is formatted like:

```
[Chunk 1 -- networking.pdf p.12]
TCP is a connection-oriented protocol that ensures reliable data delivery...

[Chunk 2 -- networking.pdf p.15]
UDP is a connectionless protocol used for real-time applications...

[Chunk 3 -- networking.pdf p.13]
The key differences between TCP and UDP include...
```

**Step 5: Chunks injected into LLM prompt**

```python
system_content = f"""\
You are StudyMate, a friendly and knowledgeable AI tutor helping {student}.
We have searched the student's uploaded notes, and here is the retrieved context:

---
{rag_ctx}
---

Your task:
1. Try to answer the student's question using the retrieved context from their uploaded notes first.
2. If the retrieved context does NOT contain the answer, answer using your own general knowledge.
   However, you MUST preface your response with a brief note indicating that you did not find
   a direct answer in their uploaded notes.
3. Keep the tone encouraging, structured, and helpful.
"""
```

The retrieved chunks become part of the system message, sandwiched between `---` delimiters. The LLM is instructed to answer from the context first and be transparent if it cannot.

**Step 6: LLM generates answer**

The LLM reads the system prompt (with chunks) and the user question, then generates an answer grounded in the student's actual notes.

**Step 7: Response returned to UI**

The node returns `{"response": response, "messages": [AIMessage(content=response)], "rag_context": rag_ctx}`. The graph ends (sequential edge to END). The response is displayed in Streamlit.

---

## 5. Hallucination fixes -- what we fixed and why

### Problem 1: Explain node making things up

**What was happening**: When explaining a topic, the LLM would sometimes invent facts or cite non-existent papers. For example, "photosynthesis was discovered by Dr. James Watson in 1845" -- completely fabricated.

**What we changed**: Added RAG context to the explain node. If the student has uploaded notes, the explanation is grounded in their actual material:

```python
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
```

**Why this reduces hallucination**: By giving the LLM actual source material, it has real facts to work with instead of relying entirely on its training data. The instruction "use the following retrieved context to enrich your explanation" steers the LLM toward the provided content.

**What can still go wrong**: If the student's notes contain errors, the LLM will dutifully repeat those errors. Also, the retrieved chunks might not be relevant to the topic if the PDF does not cover it.

### Problem 2: RAG query not staying grounded

**What was happening**: Even with retrieved context, the LLM would sometimes ignore the chunks and answer from its own knowledge, contradicting the student's notes.

**What we changed**: Added an explicit instruction in the `rag_query_node` system prompt:

```python
"1. Try to answer the student's question using the retrieved context from their uploaded notes first. If the context contains the answer, ground your response in it and be specific."
"2. If the retrieved context does NOT contain the answer, or if the context is empty, answer the question clearly and thoroughly using your own general knowledge. However, you MUST preface your response with a brief note indicating that you did not find a direct answer in their uploaded notes"
```

**Why this reduces hallucination**: The LLM is now explicitly told to prioritize the retrieved context and to be transparent when it falls back to general knowledge. The student can then decide whether to trust the answer.

**What can still go wrong**: The LLM might still mix retrieved context with its own knowledge without clearly labeling which is which.

### Problem 3: Quiz evaluate giving inconsistent scores

**What was happening**: The LLM was asked to score quizzes in its feedback text, and sometimes the LLM-generated score contradicted the actual score calculated from correct answers.

**What we changed**: Moved score calculation to Python code (deterministic) and told the LLM NOT to output a score:

```python
QUIZ_EVALUATE_PROMPT = """\
...
DO NOT output a final score or a list of weak areas, as the system will calculate and display those automatically.
"""
```

The actual score is calculated deterministically:

```python
score = round((correct_count / len(questions)) * 100, 1) if questions else 0
```

**Why this reduces hallucination**: Scores are now computed by Python, not by the LLM. The LLM only provides qualitative feedback ("you got Q2 wrong because..."), which it is good at. Quantitative accuracy ("3 out of 5 = 60%") is handled by code.

**What can still go wrong**: The LLM might still mention that the student did "well" or "poorly" in its feedback, which could contradict a borderline score.

### Problem 4: Router misclassifying intent

**What was happening**: Without notes uploaded, the router would sometimes classify academic questions as "rag_query" (which would fail because there are no documents to search).

**What we changed**: Created two separate routing prompts:

```python
ROUTE_PROMPT_WITHOUT_NOTES = """\
You are a routing classifier for a study assistant.
The student has NOT uploaded any notes/PDF yet.
...
"""

ROUTE_PROMPT_WITH_NOTES = """\
You are a routing classifier for a study assistant.
The student HAS uploaded study notes/PDF to their profile.
...
"""
```

The router checks `collection_count() > 0` and selects the appropriate prompt:

```python
has_notes = collection_count() > 0
route_prompt = ROUTE_PROMPT_WITH_NOTES if has_notes else ROUTE_PROMPT_WITHOUT_NOTES
```

**Why this reduces misclassification**: The LLM now knows whether notes exist. Without notes, academic questions go to "explain" (general knowledge). With notes, they go to "rag_query" (searched from documents).

**What can still go wrong**: Edge cases where the student's intent is ambiguous, like "tell me about sorting" (could be explain or rag_query).

### Problem 5: Study plan generating generic plans without weak topics

**What was happening**: The study plan node was generating generic plans ("Day 1: Introduction, Day 2: Basics...") without incorporating the student's actual weak areas.

**What we changed**: Added weak topics and RAG context to the study plan prompt:

```python
STUDY_PLAN_PROMPT = """\
You are StudyMate, a study planning assistant.
The student's name is {student}.

Create a detailed 7-day study plan based on these weak topics: {weak_topics}.
Also consider the student's current topic of interest: {topic}.
...
"""
```

And if notes are uploaded:

```python
if rag_ctx:
    prompt += f"\n\nIMPORTANT: The student has uploaded study notes. Please design the study plan explicitly around the following extracted content from their notes:\n{rag_ctx}"
```

**Why this reduces generic plans**: The LLM now has the student's specific weak areas and their actual course material. It cannot generate a generic plan when it is explicitly told "focus on these weak topics: calculus, integration, limits."

**What can still go wrong**: If `weak_topics` is empty (new student, no quizzes taken), the plan falls back to the current topic, which might still be generic.

### Problem 6: Confidence indicators in Streamlit UI

**What was happening**: The student had no way to know whether an answer came from their notes or from the LLM's general knowledge.

**What we changed**: The RAG query node's prompt now requires the LLM to be transparent:

```python
"If the retrieved context does NOT contain the answer, or if the context is empty, answer the question clearly and thoroughly using your own general knowledge. However, you MUST preface your response with a brief note indicating that you did not find a direct answer in their uploaded notes"
```

**Why this helps**: The student can make an informed decision about whether to trust the answer. An answer from their notes is more likely to match what their professor expects on an exam.

**What can still go wrong**: The LLM might not always follow the instruction perfectly, especially if the context partially covers the answer.

---

## 6. Streamlit integration -- how UI connects to agent

### How app.py initialises the graph on startup

At the top of `app.py`:

```python
from agent.graph import study_graph
```

This imports the pre-built graph instance. Since `study_graph = build_graph()` is executed at module level in `graph.py`, the graph is compiled once when the app starts and reused for every request.

### How st.session_state holds session_id

Streamlit reruns the entire script on every interaction (button click, text input, etc.). `st.session_state` persists values across reruns. The session state is initialised once:

```python
def _init_session_state() -> None:
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
```

The `session_id` is set when the student enters their name:

```python
st.session_state.session_id = ensure_session(
    student_name, st.session_state.get("session_id") or None
)
```

### How the chat input triggers graph invocation

When the student types a message and presses Enter:

```python
user_prompt = st.chat_input("Ask me anything...")

if user_prompt:
    # Build the agent state dict from session state
    agent_state = {
        "messages": st.session_state.messages,
        "student_name": st.session_state.student_name,
        "topic": st.session_state.get("topic", ""),
        ...
    }

    # Invoke the graph
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    result = study_graph.invoke(agent_state, config=config)
```

The `agent_state` dict is constructed from `st.session_state` values. This is how the Streamlit UI state maps to the LangGraph `AgentState`.

### How streaming works

For explain and rag_query routes, the app uses `graph.stream()` instead of `graph.invoke()` to show progressive output:

```python
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
```

The `st.empty()` creates a placeholder that can be updated in-place. As each chunk arrives from the graph stream, the placeholder is updated with the accumulated text, creating a "typing" effect.

### How HITL Approve/Reject buttons connect to graph.invoke

When a study plan is pending, the UI shows two buttons:

```python
if st.session_state.get("pending_plan") and st.session_state.get("plan_approved") is None:
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
```

When "Approve" is clicked, `_handle_plan_approval(True)` is called:

```python
def _handle_plan_approval(approved: bool) -> None:
    session_id = st.session_state.get("session_id", "")
    plan = st.session_state.get("pending_plan", "")

    if approved and plan and session_id:
        config = {"configurable": {"thread_id": session_id}}
        study_graph.update_state(config, {"plan_approved": True})
        result = study_graph.invoke(None, config=config)

        if "response" in result:
            st.session_state.messages.append(AIMessage(content=result["response"]))
```

This updates the graph state to set `plan_approved = True`, then resumes the graph from the interrupt point. The `study_plan_save_node` runs, saves the plan to SQLite, and returns a success message.

### How the sidebar weak topics update after every quiz

After quiz evaluation, the app updates `st.session_state.weak_topics`:

```python
st.session_state.weak_topics = result.get("weak_topics", st.session_state.weak_topics)
```

The sidebar reads this value on every rerun:

```python
weak = st.session_state.get("weak_topics", [])
if weak:
    for t in weak:
        st.markdown(f"- {t}")
else:
    st.markdown("_No weak topics recorded yet._")
```

Since Streamlit reruns the entire script on every interaction, the sidebar automatically reflects the latest weak topics.

---

## 7. Student name -- why it matters and how it is used

**Before the fix**: The student's name was collected via the sidebar text input and stored in `st.session_state.student_name`. But none of the LLM prompts used it. The LLM would say "Here is an explanation for you" instead of "Here is an explanation for you, Alice."

**After the fix**: The student's name is injected into every node's system prompt:

In `explain_node`:
```python
student = state.get("student_name", "Student")
system_content = f"""\
You are StudyMate, a friendly and knowledgeable AI tutor.
The student's name is {student}.
...
"""
```

In `study_plan_node`:
```python
student = state.get("student_name", "Student")
prompt = STUDY_PLAN_PROMPT.format(
    student=student,
    weak_topics=", ".join(weak_topics),
    topic=topic,
)
```

In `rag_query_node`:
```python
student = state.get("student_name", "Student")
system_content = f"""\
You are StudyMate, a friendly and knowledgeable AI tutor helping {student}.
...
"""
```

**Why personalisation matters**: A personalised response ("Great question, Alice! Binary search works by...") feels like a real tutor. A generic response ("Binary search works by...") feels like a search engine. In an interview, this shows that you thought about user experience, not just functionality.

The name also persists via SQLite. If Alice comes back tomorrow, her name is loaded from the database and injected into prompts without her needing to re-enter it.

---

## 8. LangSmith observability

### What the 3 environment variables do

In the `.env` file:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=studymate
```

- `LANGCHAIN_TRACING_V2=true`: Enables automatic tracing of all LangChain/LangGraph operations. Every LLM call, every chain step, every graph node execution is logged.
- `LANGCHAIN_API_KEY`: Your LangSmith API key. Traces are sent to the LangSmith cloud service for storage and visualization.
- `LANGCHAIN_PROJECT`: The project name in LangSmith. All traces from this app appear under the "studymate" project.

The app verifies tracing is enabled at startup:

```python
assert os.environ.get("LANGCHAIN_TRACING_V2") == "true", "LANGCHAIN_TRACING_V2 not set in .env"
```

### What a trace looks like

A trace in LangSmith shows the complete execution of one graph invocation. It looks like a tree:

```
graph.invoke() [2.3s]
  |-- router_node [0.4s]
  |     |-- ChatGroq.invoke (routing) [0.3s]
  |     |-- ChatGroq.invoke (topic extraction) [0.2s]
  |-- explain_node [1.5s]
  |     |-- ChatGroq.invoke (explanation) [1.4s]
```

### What each metric means

- **Latency**: Total time for the graph invocation. Broken down per node and per LLM call. Useful for finding bottlenecks.
- **Token count**: Input tokens (prompt) and output tokens (response) for each LLM call. Shows how much context you are sending.
- **Cost**: Estimated cost per call (if using a paid API).
- **Success/Error**: Whether each step completed successfully or threw an exception.
- **Input/Output**: The exact prompts sent to the LLM and the exact responses received. Critical for debugging hallucination.

### How to use LangSmith to debug

1. Open smith.langchain.com and select the "studymate" project.
2. Find the trace for the problematic interaction.
3. Click on the router node -- check if the routing was correct.
4. Click on the content node -- read the exact system prompt and user message that were sent to the LLM.
5. Read the LLM response -- is the hallucination in the response or was the prompt bad?
6. If the prompt is bad, fix it in `nodes.py`. If the LLM response is bad despite a good prompt, adjust the temperature or add more constraints to the prompt.

### Why this is a production skill

Every production LLM application needs observability. When a user reports "the bot said something wrong," you need to trace the exact sequence of LLM calls that produced that output. LangSmith is the standard tool for this in the LangChain ecosystem. Knowing how to use it sets you apart from developers who only test manually.

---

## 9. Testing -- what the 23 tests cover

### test_graph.py (8 tests)

**Class: TestAgentState**

`test_state_keys_exist`: Verifies that `AgentState.__annotations__` contains all 15 required keys. This catches typos and missing keys that would cause runtime errors. If someone removes a key from `AgentState`, this test fails immediately.

```python
def test_state_keys_exist(self):
    required = {
        "messages", "student_name", "topic", "weak_topics",
        "quiz_score", "quiz_questions", "quiz_attempts", "quiz_answers",
        "pending_plan", "plan_approved", "session_id", "route",
        "rag_context", "response", "error",
    }
    annotations = AgentState.__annotations__
    assert required.issubset(set(annotations.keys()))
```

`test_state_is_total_false`: Verifies that `AgentState` can be constructed with an empty dict (all keys optional). This is a consequence of `total=False` in the TypedDict definition. Without it, nodes would be forced to return every single key in every return dict.

```python
def test_state_is_total_false(self):
    state: AgentState = {}
    assert isinstance(state, dict)
```

**Class: TestRouterNode**

`test_router_returns_valid_route`: Mocks the LLM to return "explain". Calls `router_node` with a real `HumanMessage`. Asserts that the returned route is one of the 4 valid routes. This test verifies the router's parsing and validation logic, not the LLM's classification (which is mocked).

```python
@patch("agent.nodes._get_llm")
def test_router_returns_valid_route(self, mock_llm_factory):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "explain"
    mock_llm.invoke.return_value = mock_response
    mock_llm_factory.return_value = mock_llm

    state = {"messages": [HumanMessage(content="Explain photosynthesis")]}
    result = router_node(state)
    assert result["route"] in {"explain", "quiz", "study_plan", "rag_query"}
```

`test_router_empty_messages`: Calls `router_node` with an empty messages list. Asserts that it defaults to "explain". This tests the edge case guard at the top of `router_node`:

```python
if not messages:
    return {"route": "explain", "error": ""}
```

**Class: TestExplainNode**

`test_explain_returns_response`: Mocks the LLM and RAG retrieval. Calls `explain_node` with a state containing a topic, student name, and message. Asserts that the result contains a non-empty "response" key and a "messages" key. This verifies the explain node's structure without depending on the actual LLM.

**Class: TestQuizNode**

`test_quiz_generate_returns_questions`: Mocks the LLM to return a valid JSON string with 5 quiz questions. Also mocks `llm.bind()` because `quiz_generate_node` uses `llm.bind(response_format={"type": "json_object"})`. Asserts that the result contains 5 questions and `quiz_attempts` is incremented to 1.

```python
state = {"topic": "math", "weak_topics": [], "quiz_attempts": 0}
result = quiz_generate_node(state)
assert len(result["quiz_questions"]) == 5
assert result["quiz_attempts"] == 1
```

**Class: TestGraphStructure**

`test_graph_compiles`: Calls `build_graph()` and asserts it returns a non-None result. This catches import errors, circular dependencies, and invalid graph definitions. If any node function has a syntax error, this test fails.

`test_graph_has_nodes`: Calls `build_graph()` and inspects the graph structure. Asserts that all 7 expected nodes exist: router, explain, quiz_generate, quiz_evaluate, study_plan, study_plan_save, rag_query.

```python
expected = {"router", "explain", "quiz_generate", "quiz_evaluate",
            "study_plan", "study_plan_save", "rag_query"}
assert expected.issubset(node_ids)
```

---

### test_memory.py (10 tests)

**Class: TestShortTermMemory**

`test_trim_under_limit`: Creates 5 messages and trims with limit 10. Asserts all 5 are kept. Verifies that trimming does not remove messages when below the limit.

`test_trim_over_limit`: Creates 15 messages and trims with limit 10. Asserts exactly 10 are kept. Also verifies the correct messages are kept -- the last 10 (messages 5-14), not the first 10.

```python
msgs = [HumanMessage(content=f"msg {i}") for i in range(15)]
trimmed = trim_messages(msgs, max_messages=10)
assert len(trimmed) == 10
assert trimmed[0].content == "msg 5"
assert trimmed[-1].content == "msg 14"
```

`test_trim_exact_limit`: Creates exactly 10 messages and trims with limit 10. Asserts all 10 are kept. Tests the boundary condition.

**Class: TestLongTermMemory**

Every test in this class uses a pytest fixture that creates a temporary database:

```python
@pytest.fixture(autouse=True)
def _setup_temp_db(self, tmp_path):
    self.db_path = tmp_path / "test_studymate.db"
    from db.sqlite_store import init_db
    init_db(self.db_path)
```

This ensures each test gets a clean database with no leftover data from other tests.

`test_create_and_get_session`: Creates a session for "Alice" and retrieves it. Asserts the session exists and the name is correct. Tests the most basic database operation.

`test_weak_topics_roundtrip`: Creates a session, saves weak topics ["algebra", "calculus"], and retrieves them. Asserts both topics are returned. Tests the write-read cycle for weak topics.

`test_weak_topics_replace`: Saves ["topic1", "topic2"], then saves ["topic3"]. Retrieves and asserts only ["topic3"] is returned. This verifies that `save_weak_topics` does a full replacement (DELETE then INSERT), not an append.

```python
save_weak_topics(sid, ["topic1", "topic2"], db_path=self.db_path)
save_weak_topics(sid, ["topic3"], db_path=self.db_path)
topics = get_weak_topics(sid, db_path=self.db_path)
assert topics == ["topic3"]
```

`test_quiz_score_save_and_retrieve`: Saves two quiz scores (physics 85%, chemistry 60% with weak area "acids"). Retrieves and asserts 2 results, most recent first (chemistry first), with correct score and weak areas parsed from JSON.

`test_study_plan_save_and_approve`: Saves a study plan with `approved=False`. Retrieves it and asserts `approved == 0`. Then calls `approve_study_plan` and retrieves again, asserting `approved == 1`. Tests the full approval workflow.

`test_load_session_memory`: Creates a session, saves weak topics and a quiz score. Then patches `DB_PATH` and verifies that the database functions return the correct data. This tests the integration between `memory.py` and `sqlite_store.py`.

`test_ensure_session_creates_new`: Creates a session for "Grace", verifies it exists, then tests name update by calling `update_session_name` and verifying the name changed to "Grace Updated". Tests the session re-use and update logic.

---

### test_rag.py (5 tests)

**Class: TestLoader**

`test_chunk_documents`: Creates a Document with ~600 characters of text ("A " repeated 300 times). Chunks it with `chunk_size=100, chunk_overlap=20`. Asserts that the result has more than 1 chunk and each chunk is at most 120 characters (allowing some variance from the overlap).

`test_chunk_preserves_metadata`: Creates a Document with metadata `{"source": "notes.pdf", "page": 3}`. Chunks it and asserts that every chunk inherits the same metadata. This is important because the retriever uses metadata to show source information.

```python
for chunk in chunks:
    assert chunk.metadata["source"] == "notes.pdf"
```

**Class: TestRetriever**

Each test uses a fixture that creates a temporary ChromaDB directory and resets the singleton vectorstore:

```python
@pytest.fixture(autouse=True)
def _setup_temp_chroma(self, tmp_path):
    self.chroma_dir = str(tmp_path / "test_chroma")
    os.makedirs(self.chroma_dir, exist_ok=True)
    from rag import retriever
    retriever.reset_vectorstore()
    retriever._vectorstore = None
    yield
    retriever.reset_vectorstore()
    retriever._vectorstore = None
```

`test_add_and_retrieve`: Adds 3 documents (photosynthesis, Newton's law, mitochondria). Queries "How do plants make food?" with k=2. Asserts that at most 2 results are returned. This tests the full add-embed-search pipeline.

`test_retrieve_empty_collection`: Queries an empty collection. Asserts the result is an empty list, not an error. Tests graceful handling of the "no documents uploaded" case.

`test_retrieve_as_text`: Adds 1 document about machine learning. Calls `retrieve_as_text` and asserts the result contains "machine learning". Tests the text formatting function that adds chunk headers and source metadata.

---

### Why tests use mocks for LLM calls

LLM calls are:
- **Non-deterministic**: The same prompt can produce different outputs. Tests need predictable results.
- **Slow**: Each call takes 0.5-2 seconds. A test suite with 23 tests would take over a minute.
- **Costly**: Even with Groq's free tier, excessive calls can hit rate limits.
- **Unreliable**: Network issues, API outages, or rate limiting can cause tests to fail for reasons unrelated to the code.

By mocking `_get_llm`, we replace the real LLM with a fake that returns predetermined responses. This makes tests fast (milliseconds), deterministic (same result every time), and offline (no network needed).

### How to run tests

```bash
cd e:\StudyMate
pytest tests/ -v
```

The `-v` flag shows verbose output with each test name and pass/fail status.

### What to do if a test fails

1. Read the assertion error message. It tells you what was expected and what was actually returned.
2. Check if the test is testing your code or a mock. If the mock is wrong, fix the mock.
3. If the test is testing your code, open the relevant source file and trace through the logic with the test's input.
4. Common causes: changed function signature, renamed state key, changed return dict structure.

---

## 10. Interview preparation -- complete Q&A

### Q1: Walk me through your StudyMate project.

StudyMate is an AI-powered personal tutor built with LangGraph, a framework for building stateful agent workflows. The student interacts through a Streamlit chat interface and can do four things: get topic explanations, take adaptive quizzes, generate study plans, and search their uploaded notes using RAG. When the student types a message, it enters a LangGraph StateGraph where a router node classifies the intent (explain, quiz, study_plan, or rag_query) and routes it to the appropriate workflow node. The quiz workflow includes a conditional loop that retries up to 3 times if the student scores below 70%, focusing on weak areas each time. The study plan workflow uses Human-in-the-Loop -- the graph literally pauses execution and waits for the student to approve or reject the plan before saving. Student progress (weak topics, quiz scores, study plans) is persisted in SQLite so it survives across sessions, and uploaded PDFs are indexed in ChromaDB for RAG retrieval.

### Q2: What is LangGraph and why did you use it?

LangGraph is a framework built on top of LangChain that lets you define AI agent workflows as directed graphs. Instead of writing linear chains, you define nodes (Python functions) and edges (connections between them, including conditional edges for branching and loops). I used it because StudyMate needs three things a simple chain cannot provide: conditional routing (4 different workflows based on user intent), iterative loops (quiz retry up to 3 times), and Human-in-the-Loop interrupts (study plan approval). With LangGraph, the quiz retry loop is just a conditional edge that points back to the quiz_generate node when the score is below 70%. Without it, I would have needed complex imperative control flow with manual state management, which is fragile and hard to maintain. LangGraph also provides a checkpointer system that saves graph state, which is essential for the HITL interrupt to work -- the graph state needs to persist while the student decides whether to approve the plan.

### Q3: How does your quiz retry loop work technically?

The quiz workflow starts when the router classifies a message as "quiz". It goes to `quiz_generate_node`, which generates 5 MCQ questions and increments `quiz_attempts` to 1. The questions are returned to the UI where the student answers them. Then `quiz_evaluate_node` runs, comparing each answer to the correct answer and computing a percentage score. After evaluation, a conditional edge function `_quiz_should_retry` checks two conditions: `score < 70 and attempts < 3`. If both are true, it returns `"quiz_generate"`, which loops execution back to generate a new quiz. Critically, the new quiz focuses on the student's weak areas by including them in the LLM prompt and explicitly telling the LLM not to repeat previous questions. If the score is 70% or above, or if the student has used all 3 attempts, the function returns `END` and the graph terminates. This loop mechanism is a first-class feature of LangGraph -- a conditional edge whose return value maps to either the same node (loop) or END (terminate).

### Q4: What is Human-in-the-Loop and how did you implement it?

Human-in-the-Loop means the agent pauses its execution and waits for a human decision before continuing. In StudyMate, when the student requests a study plan, the `study_plan_node` generates a 7-day plan and sets `plan_approved = None`. The graph is compiled with `interrupt_before=["study_plan_save"]`, which means LangGraph literally stops execution before the save node runs. The graph state is saved to a `MemorySaver` checkpointer, keyed by the session's `thread_id`. In the Streamlit UI, Approve and Reject buttons appear. When the student clicks Approve, the code calls `study_graph.update_state(config, {"plan_approved": True})` to inject the decision into the saved state, then calls `study_graph.invoke(None, config)` to resume the graph from the interrupt point. The `study_plan_save_node` then runs with `plan_approved = True` and saves the plan to SQLite. This is fundamentally different from just using UI buttons -- the graph's execution flow is genuinely paused and resumed, maintaining all state correctly.

### Q5: How does RAG work in your project?

RAG in StudyMate follows a standard pipeline. First, the student uploads a PDF through the Streamlit sidebar. The `ingest_pdf` function in `retriever.py` loads the PDF using pypdf, splits it into ~500-character chunks using `RecursiveCharacterTextSplitter`, and embeds each chunk using the all-MiniLM-L6-v2 sentence transformer model running locally on CPU. The embeddings are stored in ChromaDB, a vector database that persists to disk. When the student asks a question, the router classifies it as `rag_query`. The `rag_query_node` embeds the question using the same model, performs a similarity search in ChromaDB to find the top 3 most relevant chunks, and injects those chunks into the LLM's system prompt. The LLM is instructed to answer primarily from the retrieved context and to be transparent if the answer is not found in the notes. This grounds the LLM's response in the student's actual study material rather than relying on general training data.

### Q6: How does memory persist across sessions?

StudyMate has two layers of memory. Short-term memory is the `messages` list in `AgentState`, which stores the conversation history and is trimmed to the last 10 messages to keep the LLM's context window manageable. This lives in memory and is lost when the server restarts. Long-term memory uses SQLite with four tables: `sessions` (student name and session UUID), `quiz_scores` (topic, score, attempt count, and weak areas as JSON), `weak_topics` (topics the student struggled with), and `study_plans` (plan text and approval status). When a student enters their name, the app calls `ensure_session` to either load an existing session or create a new one, then calls `load_session_memory` to retrieve their saved weak topics and latest quiz score. This data is merged into the `AgentState` so the LLM has access to the student's history from day one.

### Q7: How did you prevent hallucination in your agent?

I applied several anti-hallucination measures. First, all quantitative calculations (quiz scores) are done in Python code, not by the LLM -- the LLM is explicitly told "DO NOT output a final score." Second, the RAG prompt instructs the LLM to answer from retrieved context first and to clearly disclose when it falls back to general knowledge. Third, the router uses two different prompts depending on whether notes are uploaded, preventing it from routing to RAG when there are no documents. Fourth, the quiz generator uses `response_format={"type": "json_object"}` to force structured JSON output, reducing malformed responses. Fifth, I added a `_parse_quiz_json` function that tries three different JSON extraction strategies as fallbacks. Sixth, every node wraps LLM calls in `_safe_invoke`, which catches exceptions and returns a graceful fallback message instead of crashing.

### Q8: What is LangSmith and what did you observe with it?

LangSmith is an observability platform built by the LangChain team. It automatically traces every LLM call, graph node execution, and state transition in my application. I enabled it by setting three environment variables: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT=studymate`. In the LangSmith dashboard, I can see each graph invocation as a tree of operations with latency, token counts, and input/output for every step. I used it to debug routing issues -- I could see the exact prompt sent to the router and the exact classification it returned, which helped me fix cases where the router was misclassifying academic questions as "explain" instead of "rag_query." I also used it to measure latency and found that the router's topic extraction was adding an unnecessary second LLM call, which I optimized.

### Q9: Why Groq over OpenAI?

Groq offers a free tier with generous rate limits, which is critical for a fresher's portfolio project where budget is zero. OpenAI charges per token for both input and output, and costs add up quickly during development and testing. Groq runs inference on custom LPU (Language Processing Unit) hardware, which provides extremely fast response times -- often under 500ms for the models available. The model I use, llama-3.1-8b-instant, is an open-source Llama model that performs well for classification, explanation, and quiz generation tasks. The trade-off is that Groq's model selection is more limited than OpenAI's, and the free tier has rate limits that would not work for production traffic. But for a portfolio project, the combination of free + fast is ideal.

### Q10: What would you improve if you had more time?

Five things. First, I would add multi-PDF support so students can upload multiple documents and the RAG pipeline searches across all of them. Currently, each upload replaces the previous one. Second, I would add streaming token-by-token output using LangGraph's streaming API for all routes, not just explain and rag_query. Third, I would replace MemorySaver with a persistent checkpointer (like SqliteSaver) so HITL state survives server restarts. Fourth, I would add authentication so multiple students can use the same deployment with isolated data. Fifth, I would add a progress dashboard showing quiz score trends over time, topics mastered vs. topics still weak, using charts in the Streamlit sidebar.

### Q11: How did you test your project?

I wrote 23 unit tests across three test files. `test_graph.py` tests the AgentState schema (all keys present, total=False), the router's classification logic with mocked LLM calls, the explain node's output structure, quiz generation with mocked JSON responses, and the graph's compilation and node presence. `test_memory.py` tests message trimming at, below, and above the limit, plus all SQLite operations: session creation, weak topic round-trip and replacement, quiz score storage and retrieval, study plan save and approve, and session memory loading. `test_rag.py` tests document chunking (splitting and metadata preservation) and ChromaDB operations (add/retrieve, empty collection, text formatting). All LLM calls are mocked using `unittest.mock.patch` so tests are fast, deterministic, and offline. I run them with `pytest tests/ -v`.

### Q12: What is the role of ChromaDB in your project?

ChromaDB is the vector database that stores embeddings of the student's uploaded PDF content. When a student uploads a PDF, it is split into ~500-character chunks and each chunk is converted into a 384-dimensional vector using the all-MiniLM-L6-v2 model. ChromaDB stores these vectors along with the original text and metadata (source file, page number). When the student asks a question, ChromaDB performs a cosine similarity search between the question's embedding and all stored chunk embeddings, returning the top-k most similar chunks. These chunks are then injected into the LLM prompt as context. I chose ChromaDB because it requires zero infrastructure -- no server, no configuration, just `pip install chromadb`. It persists to a local directory, which is perfect for a single-user study app.

### Q13: Explain AgentState and why it is important.

AgentState is a TypedDict that defines every piece of data that flows through the LangGraph workflow. It has 15 keys covering conversation messages, student identity, quiz state (questions, answers, score, attempts), study plan state (pending plan, approval status), RAG context, routing decisions, and error handling. It is defined with `total=False`, meaning nodes can return partial updates -- a node only needs to return the keys it changed, and LangGraph merges the update into the full state. The `messages` key uses the `add_messages` reducer annotation, which means new messages are appended rather than replaced. AgentState is important because it is the single source of truth for the entire workflow. Every node reads from it and writes to it. Without it, nodes would need to pass data through function arguments, which would create tight coupling and make the graph inflexible.

### Q14: What happens when the student scores below 70% three times?

When the student scores below 70% on the third attempt, the `_quiz_should_retry` function checks `score < 70 and attempts < 3`. Since attempts is now 3 (equal to the limit, not less than), the condition is false, and the function returns `END`. The `quiz_evaluate_node` detects this case and appends the message "You've used all 3 attempts. Consider reviewing the topic and trying again later." It also resets `quiz_questions` to an empty list and `quiz_attempts` to 0, clearing the quiz state. The failing topic is added to `weak_topics` and persisted to SQLite, so it will influence future study plans and quiz focus. The student can request another quiz later, which will start fresh with `quiz_attempts = 0`, but the weak topics will carry over, making the new quiz focus on the areas they struggled with.

### Q15: How is short-term memory different from long-term memory here?

Short-term memory is the `messages` list that holds the conversation history. It lives in Python memory (in `AgentState` and `st.session_state`), is trimmed to the last 10 messages to keep the LLM's context window efficient, and is lost when the server restarts. Its purpose is to give the LLM conversational context -- "what were we just talking about?" Long-term memory is the SQLite database that stores structured facts: student name, weak topics, quiz scores, and approved study plans. It lives on disk in `studymate.db`, is never trimmed, and survives restarts indefinitely. Its purpose is to remember facts about the student that matter across sessions -- "this student is weak at calculus." The two work together: long-term memory provides factual context (loaded at startup and injected into prompts), while short-term memory provides conversational context (maintained during the session).

---

## 11. Architecture -- text diagram

```
+========================================================================================+
|                                 STREAMLIT UI (app.py)                                  |
|                                                                                        |
|  +------------------+  +------------------+  +-------------------+  +-----------+      |
|  | Chat Input       |  | Quiz Form        |  | Approve / Reject  |  | PDF       |      |
|  | (st.chat_input)  |  | (st.form)        |  | Buttons (HITL)    |  | Uploader  |      |
|  +--------+---------+  +--------+---------+  +--------+----------+  +-----+-----+      |
|           |                      |                     |                    |            |
+===========|======================|=====================|====================|============+
            |                      |                     |                    |
            v                      v                     v                    v
+========================================================================================+
|                          SESSION STATE (st.session_state)                               |
|                                                                                        |
|  messages, student_name, topic, weak_topics, quiz_score, quiz_questions,                |
|  quiz_attempts, quiz_answers, pending_plan, plan_approved, session_id                   |
+===========+=============+====================================+=========================+
            |             |                                    |
            v             v                                    |
+========================================================================================+
|                        LANGGRAPH STATE GRAPH (agent/graph.py)                          |
|                                                                                        |
|  AgentState flows through all nodes. MemorySaver checkpoints state for HITL.           |
|                                                                                        |
|  +---> [ROUTER NODE] ----+---> "explain" ----> [EXPLAIN NODE] ---------> END           |
|  |     (router_node)     |                     (explain_node)                          |
|  |                       |                                                             |
|  |                       +---> "quiz" -------> [QUIZ GENERATE] ---------> END          |
|  |                       |                     (quiz_generate_node)    (returns Qs)     |
|  |                       |                                                             |
|  |                       |     [QUIZ EVALUATE] <--- student answers from UI            |
|  |                       |     (quiz_evaluate_node)                                    |
|  |                       |           |                                                 |
|  |                       |           +--- score < 70% AND attempts < 3 --> [QUIZ GEN]  |
|  |                       |           |                       (RETRY LOOP)               |
|  |                       |           +--- score >= 70% OR attempts >= 3 --> END         |
|  |                       |                                                             |
|  |                       +---> "study_plan" -> [STUDY PLAN NODE] --+                   |
|  |                       |                     (study_plan_node)   |                   |
|  |                       |                                         |                   |
|  |                       |                     ** INTERRUPT **      |                   |
|  |                       |                     (graph pauses)      |                   |
|  |                       |                          |              |                   |
|  |                       |                     [Approve/Reject]    |                   |
|  |                       |                          |              |                   |
|  |                       |                     [STUDY PLAN SAVE] ---> END               |
|  |                       |                     (study_plan_save_node)                   |
|  |                       |                                                             |
|  |                       +---> "rag_query" --> [RAG QUERY NODE] -------> END            |
|  |                                             (rag_query_node)                        |
|  |                                                    |                                |
+==|====================================================|================================+
   |                                                    |
   |                                                    v
   |                                    +===============================+
   |                                    |  RAG LAYER                    |
   |                                    |                               |
   |                                    |  rag/loader.py                |
   |                                    |    load_pdf() -> Documents    |
   |                                    |    chunk_documents() -> chunks|
   |                                    |                               |
   |                                    |  rag/retriever.py             |
   |                                    |    ingest_pdf() -> ChromaDB   |
   |                                    |    retrieve() -> top-k chunks |
   |                                    |    retrieve_as_text() -> str  |
   |                                    |                               |
   |                                    |  +-------------------------+  |
   |                                    |  |  ChromaDB               |  |
   |                                    |  |  (chroma_db/ directory) |  |
   |                                    |  |  all-MiniLM-L6-v2      |  |
   |                                    |  |  embeddings (384-dim)   |  |
   |                                    |  +-------------------------+  |
   |                                    +===============================+
   |
   v
+===================================+       +===================================+
|  PERSISTENCE LAYER                |       |  EXTERNAL SERVICES                |
|                                   |       |                                   |
|  db/sqlite_store.py               |       |  Groq API                         |
|    studymate.db                   |       |    llama-3.1-8b-instant           |
|                                   |       |    temperature: 0.0 - 0.7         |
|    +---------------------------+  |       |    JSON mode for quiz gen         |
|    | sessions                  |  |       |                                   |
|    |   session_id, student_name|  |       +-----------------------------------+
|    +---------------------------+  |
|    | quiz_scores               |  |       +===================================+
|    |   topic, score, attempts  |  |       |  OBSERVABILITY                    |
|    +---------------------------+  |       |                                   |
|    | weak_topics               |  |       |  LangSmith                        |
|    |   session_id, topic       |  |       |    LANGCHAIN_TRACING_V2=true      |
|    +---------------------------+  |       |    LANGCHAIN_API_KEY=lsv2_...     |
|    | study_plans               |  |       |    LANGCHAIN_PROJECT=studymate    |
|    |   plan_text, approved     |  |       |                                   |
|    +---------------------------+  |       |    Traces every LLM call,         |
|                                   |       |    node execution, and            |
+===================================+       |    state transition.              |
                                            +===================================+
```

**Reading the diagram**:

1. The student interacts through the Streamlit UI layer at the top. Four input mechanisms: chat text input, quiz answer form, HITL approve/reject buttons, and PDF uploader.

2. All UI state lives in `st.session_state`, which bridges the UI and the agent.

3. The LangGraph StateGraph is the core. Every message enters at the Router Node, which classifies intent and routes to one of four workflows.

4. The Explain and RAG Query workflows are simple: router -> node -> END.

5. The Quiz workflow has an iterative loop: quiz_generate -> (student answers) -> quiz_evaluate -> conditional check -> either loop back to quiz_generate or go to END.

6. The Study Plan workflow has a HITL interrupt: study_plan -> INTERRUPT -> (student approves/rejects) -> study_plan_save -> END.

7. The RAG Layer sits below the graph. It handles PDF loading, chunking, embedding, and retrieval. ChromaDB stores the vectors on disk.

8. The Persistence Layer (SQLite) stores long-term data: sessions, quiz scores, weak topics, and study plans.

9. External services: Groq API for LLM inference, LangSmith for observability. Both are accessed via API keys stored in `.env`.

---

*End of StudyMate Learning Guide. Last updated: 2026-05-28.*
