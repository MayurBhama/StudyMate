# 🎓 StudyMate — Personalized AI Study Agent

A fully-featured AI-powered study companion built with **LangGraph**, **Groq**, **ChromaDB**, and **Streamlit**. StudyMate can explain topics, quiz you, generate personalized study plans, and answer questions from your uploaded notes.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Smart Routing** | Automatically classifies your intent (explain, quiz, study plan, or notes search) |
| 💡 **Topic Explanations** | Clear, structured explanations at your level with RAG-enhanced context |
| 📝 **Interactive Quizzes** | 3-question MCQs with evaluation, retry logic (up to 3 attempts), and weak-area tracking |
| 📋 **Study Plans** | Personalized 7-day plans based on your weak topics with Human-in-the-Loop approval |
| 🔍 **RAG Note Search** | Upload PDFs and ask questions — answers are grounded in your own notes |
| 🧠 **Memory** | Short-term (last 10 messages) + long-term (SQLite: scores, weak topics, plans) |
| 📊 **Progress Tracking** | Weak topics tracker, quiz score history, session persistence |
| 🔭 **LangSmith Observability** | Every graph execution is traced and tagged |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │ Chat     │  │ Quiz     │  │ Sidebar            │ │
│  │ Interface│  │ Forms    │  │ • Profile           │ │
│  │          │  │ (Radio)  │  │ • PDF Upload        │ │
│  │          │  │          │  │ • Weak Topics       │ │
│  │          │  │          │  │ • Scores            │ │
│  │          │  │          │  │ • HITL Approval     │ │
│  └────┬─────┘  └────┬─────┘  └────────────────────┘ │
│       │              │                               │
└───────┼──────────────┼───────────────────────────────┘
        │              │
        ▼              ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Agent Graph                   │
│                                                      │
│  ┌────────┐                                          │
│  │ Router │──── explain ────► Explain Node ──► END   │
│  │  Node  │──── quiz ──────► Quiz Gen ──► END        │
│  │        │                   (answers)               │
│  │        │                  Quiz Eval ◄──┐           │
│  │        │                   │  retry?   │           │
│  │        │                   ├── yes ────┘           │
│  │        │                   └── no ──► END          │
│  │        │──── study_plan ─► Plan Node ──► END       │
│  │        │                   (HITL)                  │
│  │        │                  Plan Save ──► END        │
│  │        │──── rag_query ──► RAG Node ──► END        │
│  └────────┘                                          │
└─────────────────────────────────────────────────────┘
        │                    │
        ▼                    ▼
┌──────────────┐    ┌──────────────────┐
│   ChromaDB   │    │     SQLite       │
│  (Vectors)   │    │  (Persistence)   │
│              │    │  • Sessions      │
│  all-MiniLM  │    │  • Quiz Scores   │
│  -L6-v2      │    │  • Weak Topics   │
│              │    │  • Study Plans   │
└──────────────┘    └──────────────────┘
```

---

## 📂 Project Structure

```
studymate/
├── app.py                  # Streamlit UI entry point
├── agent/
│   ├── graph.py            # Main LangGraph graph definition
│   ├── nodes.py            # All node functions
│   ├── state.py            # AgentState TypedDict
│   └── memory.py           # Short + long term memory handlers
├── rag/
│   ├── loader.py           # PDF loader + chunker
│   └── retriever.py        # ChromaDB vector store setup
├── db/
│   └── sqlite_store.py     # SQLite session + quiz score storage
├── tests/
│   ├── test_graph.py       # LangGraph workflow tests
│   ├── test_rag.py         # RAG pipeline tests
│   └── test_memory.py      # Memory persistence tests
├── .env.example            # Template with key names only
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/StudyMate.git
cd StudyMate
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your actual API keys:

```
GROQ_API_KEY=your_groq_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=studymate
```

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. Launch the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🧪 Testing

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test files
pytest tests/test_graph.py -v
pytest tests/test_rag.py -v
pytest tests/test_memory.py -v
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| Tools + RAG | LangChain |
| LLM | Groq API (LLaMA 3.3 70B) |
| Vector Store | ChromaDB (local) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Persistence | SQLite |
| UI | Streamlit |
| Observability | LangSmith |
| PDF Parsing | pypdf |

---

## 📋 Usage Guide

1. **Enter your name** in the sidebar to start a session
2. **Upload a PDF** (optional) to enable RAG-powered note search
3. **Chat naturally** — StudyMate auto-routes your intent:
   - *"Explain photosynthesis"* → Topic explanation
   - *"Quiz me on calculus"* → Interactive MCQ quiz
   - *"Create a study plan"* → 7-day personalized plan
   - *"What do my notes say about X?"* → RAG note search
4. **Take quizzes** — answer via radio buttons; retry if score < 70%
5. **Approve study plans** — use the sidebar Approve/Reject buttons
6. **Track progress** — weak topics and scores are saved across sessions

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
