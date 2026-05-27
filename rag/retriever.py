"""ChromaDB vector-store setup and retrieval for the RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from rag.loader import load_and_chunk_pdf

# ── constants ────────────────────────────────────────────────────────────────

CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "studymate_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── singleton caches ─────────────────────────────────────────────────────────

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None


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


def get_vectorstore(persist_directory: str = CHROMA_DIR) -> Chroma:
    """Return (and lazily create) the shared Chroma vector store."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=persist_directory,
        )
    return _vectorstore


def reset_vectorstore() -> None:
    """Reset the cached vectorstore (useful for tests)."""
    global _vectorstore
    _vectorstore = None


# ── public API ───────────────────────────────────────────────────────────────

def add_documents(docs: list[Document], persist_directory: str = CHROMA_DIR) -> int:
    """Add pre-chunked documents to the vector store.

    Returns the number of chunks added.
    """
    vs = get_vectorstore(persist_directory)
    
    batch_size = 150
    for i in range(0, len(docs), batch_size):
        vs.add_documents(docs[i : i + batch_size])
        
    return len(docs)


def ingest_pdf(file: str | BinaryIO, persist_directory: str = CHROMA_DIR) -> int:
    """Load, chunk, embed and store a PDF in one call.

    Returns the number of chunks indexed.
    """
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


def retrieve(query: str, k: int = 3, persist_directory: str = CHROMA_DIR) -> list[Document]:
    """Return the top-*k* most relevant document chunks for *query*."""
    vs = get_vectorstore(persist_directory)
    try:
        results = vs.similarity_search(query, k=k)
    except Exception:
        # Collection might be empty
        results = []
    return results


def retrieve_as_text(query: str, k: int = 3, persist_directory: str = CHROMA_DIR) -> str:
    """Retrieve top-*k* chunks and concatenate them into a single context string."""
    docs = retrieve(query, k=k, persist_directory=persist_directory)
    if not docs:
        return ""
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i} — {source} p.{page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def collection_count(persist_directory: str = CHROMA_DIR) -> int:
    """Return the number of documents currently in the collection."""
    vs = get_vectorstore(persist_directory)
    try:
        return vs._collection.count()
    except Exception:
        return 0
