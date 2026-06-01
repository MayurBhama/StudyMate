"""Tests for the RAG pipeline (loader + retriever)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document


class TestLoader:
    """Test PDF loading and chunking."""

    def test_chunk_documents(self):
        """chunk_documents should split documents into smaller pieces."""
        from rag.loader import chunk_documents

        docs = [
            Document(
                page_content="A " * 300,  # ~600 chars
                metadata={"source": "test.pdf", "page": 1},
            )
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1, "Should split a long document into multiple chunks"
        for chunk in chunks:
            assert len(chunk.page_content) <= 120  # allow some variance

    def test_chunk_preserves_metadata(self):
        """Chunks should inherit source metadata."""
        from rag.loader import chunk_documents

        docs = [
            Document(
                page_content="Hello world. " * 100,
                metadata={"source": "notes.pdf", "page": 3},
            )
        ]
        chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata["source"] == "notes.pdf"


class TestRetriever:
    """Test ChromaDB vector store operations."""

    @pytest.fixture(autouse=True)
    def _setup_temp_chroma(self, tmp_path):
        """Create a temporary ChromaDB directory for each test."""
        self.chroma_dir = str(tmp_path / "test_chroma")
        os.makedirs(self.chroma_dir, exist_ok=True)

        # Reset the singleton so each test gets a fresh store
        from rag import retriever
        retriever.reset_vectorstore()
        retriever._vectorstore = None
        yield
        retriever.reset_vectorstore()
        retriever._vectorstore = None

    def test_add_and_retrieve(self):
        """Adding documents then retrieving should return relevant results."""
        from rag.retriever import add_documents, retrieve

        docs = [
            Document(page_content="Photosynthesis converts light energy into chemical energy in plants.",
                     metadata={"source": "bio.pdf", "page": 1}),
            Document(page_content="Newton's second law states F equals m times a.",
                     metadata={"source": "physics.pdf", "page": 5}),
            Document(page_content="The mitochondria is the powerhouse of the cell.",
                     metadata={"source": "bio.pdf", "page": 2}),
        ]

        count = add_documents(docs, persist_directory=self.chroma_dir)
        assert count == 3

        results = retrieve("How do plants make food?", k=2, persist_directory=self.chroma_dir)
        assert len(results) <= 2

    def test_retrieve_empty_collection(self):
        """Retrieving from an empty collection should return an empty list."""
        from rag.retriever import retrieve

        results = retrieve("anything", k=3, persist_directory=self.chroma_dir)
        assert results == []

    def test_retrieve_as_text(self):
        """retrieve_as_text should return a formatted string."""
        from rag.retriever import add_documents, retrieve_as_text

        docs = [
            Document(page_content="Machine learning is a subset of AI.",
                     metadata={"source": "ml.pdf", "page": 1}),
        ]
        add_documents(docs, persist_directory=self.chroma_dir)

        text = retrieve_as_text("What is machine learning?", k=1, persist_directory=self.chroma_dir)
        assert "machine learning" in text.lower() or "Machine learning" in text
