# StudyMate

A personalised AI study agent built with LangGraph. It explains topics,
quizzes students with adaptive retry logic, generates study plans with
human approval, and answers questions from uploaded notes.

## Features

- Topic explanation with context-aware responses
- Adaptive quiz generation with iterative retry until 70% score
- RAG-based Q&A from uploaded PDF notes
- Personalised study plan generation with human-in-the-loop approval
- Persistent memory -- weak topics and scores saved across sessions
- Real-time streaming responses
- LangSmith observability integration

## Tech Stack

- **LangGraph** -- agent framework and graph orchestration
- **LangChain** -- LLM integration and RAG pipeline
- **Groq API** -- LLM inference (llama-3.1-8b-instant)
- **ChromaDB** -- local vector store for PDF notes
- **sentence-transformers** -- local embeddings (all-MiniLM-L6-v2)
- **SQLite** -- persistent session and memory storage
- **Streamlit** -- user interface and deployment
- **LangSmith** -- agent observability and tracing

## Project Structure

```
studymate/
├── app.py
├── agent/
│   ├── __init__.py
│   ├── graph.py
│   ├── nodes.py
│   ├── state.py
│   └── memory.py
├── rag/
│   ├── __init__.py
│   ├── loader.py
│   └── retriever.py
├── db/
│   ├── __init__.py
│   └── sqlite_store.py
├── tests/
│   ├── __init__.py
│   ├── test_graph.py
│   ├── test_memory.py
│   └── test_rag.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/studymate.git
cd studymate
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Open the `.env` file and fill in your keys:

```
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=studymate
```

Get a Groq API key free at: https://console.groq.com

Get a LangSmith API key free at: https://smith.langchain.com

### 5. Run the app

```bash
streamlit run app.py
```

## How It Works

The agent uses a Router Node with conditional edges to classify user input
into four routes: explain, quiz, study_plan, or rag_query. Each route
triggers a different subgraph. Quiz uses an iterative workflow that retries
until the student scores above 70%. Study plans use LangGraph's
interrupt_before for human-in-the-loop approval before saving. All session
data persists in SQLite across restarts.

## Requirements

Python 3.10 or higher
