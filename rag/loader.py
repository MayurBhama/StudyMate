"""PDF loader and text chunker for the RAG pipeline."""

from __future__ import annotations

import io
from typing import BinaryIO

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(file: str | BinaryIO) -> list[Document]:
    """Read a PDF and return one ``Document`` per page.

    Parameters
    ----------
    file : str | BinaryIO
        A file path *or* an in-memory file object (e.g. from Streamlit's
        ``st.file_uploader``).
    """
    if isinstance(file, str):
        reader = PdfReader(file)
        source = file
    else:
        # Streamlit UploadedFile → BytesIO wrapper
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


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split documents into smaller chunks for embedding.

    Uses ``RecursiveCharacterTextSplitter`` with the specified ``chunk_size``
    and ``chunk_overlap``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def load_and_chunk_pdf(
    file: str | BinaryIO,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Convenience wrapper: load a PDF then chunk it in one call."""
    docs = load_pdf(file)
    return chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
