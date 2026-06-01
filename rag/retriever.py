"""ChromaDB vector-store setup and retrieval for the RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from rag.loader import load_and_chunk_pdf

# ── constants ────────────────────────────────────────────────────────────────

CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "studymate_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── singleton caches ─────────────────────────────────────────────────────────

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstores: dict[str, Chroma] = {}


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return (and lazily create) the shared HuggingFace embedding function."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore(persist_directory: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME) -> Chroma:
    """Return (and lazily create) the shared Chroma vector store for a specific collection."""
    global _vectorstores
    if collection_name not in _vectorstores:
        _vectorstores[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=persist_directory,
        )
    return _vectorstores[collection_name]


def reset_vectorstore() -> None:
    """Reset the cached vectorstores (useful for tests)."""
    global _vectorstores
    _vectorstores.clear()


# ── public API ───────────────────────────────────────────────────────────────

def sanitise_text(text: str) -> str:
    return text.encode(
        "utf-8", errors="replace"
    ).decode("utf-8", errors="replace")

def add_documents(docs: list[Document], persist_directory: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME) -> int:
    """Add pre-chunked documents to the vector store.

    Returns the number of chunks added.
    """
    vs = get_vectorstore(persist_directory, collection_name)
    
    batch_size = 150
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        for doc in batch:
            doc.page_content = sanitise_text(doc.page_content)
        vs.add_documents(batch)
        
    return len(docs)


def ingest_pdf(file: str | BinaryIO, persist_directory: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME) -> int:
    """Load, chunk, embed and store a PDF in one call.

    Returns the number of chunks indexed.
    """
    vs = get_vectorstore(persist_directory, collection_name)
    try:
        vs.delete_collection()
    except Exception:
        pass
    
    # We must reset the specific collection so it creates a fresh one
    global _vectorstores
    if collection_name in _vectorstores:
        del _vectorstores[collection_name]
    
    chunks = load_and_chunk_pdf(file)
    if not chunks:
        return 0
    return add_documents(chunks, persist_directory, collection_name)


def retrieve(query: str, k: int = 10, persist_directory: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME) -> list[Document]:
    """Return the top-*k* most relevant document chunks for *query*."""
    vs = get_vectorstore(persist_directory, collection_name)
    try:
        results = vs.similarity_search(query, k=k)
    except Exception:
        # Collection might be empty
        results = []
    return results


def retrieve_as_text(query: str, k: int = 10, persist_directory: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME) -> str:
    """Retrieve top-*k* chunks and concatenate them into a single context string."""
    docs = retrieve(query, k=k, persist_directory=persist_directory, collection_name=collection_name)
    if not docs:
        return ""
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i} — {source} p.{page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def collection_count(persist_directory: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME) -> int:
    """Return the number of documents currently in the collection."""
    vs = get_vectorstore(persist_directory, collection_name)
    try:
        return vs._collection.count()
    except Exception:
        return 0

# -- Raw ChromaDB Client for retrieve_context ---------------------------------
import chromadb

IS_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") is not None

if IS_CLOUD:
    client = chromadb.Client()
else:
    client = chromadb.PersistentClient(path=CHROMA_DIR)

def retrieve_context(query: str, n_results: int = 3, collection_name: str = COLLECTION_NAME) -> list[str]:
    col = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    results = col.query(
        query_texts=[query],
        n_results=n_results
    )
    chunks = results["documents"][0] if results["documents"] else []
    return [sanitise_text(c) for c in chunks]

