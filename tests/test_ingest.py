import warnings
from pathlib import Path

import pytest
from langchain_core.documents import Document

from docagent.ingest import load_documents, split_documents
from docagent.config import Settings


def test_load_documents_skips_hidden_files(tmp_path):
    (tmp_path / ".hidden.md").write_text("secret")
    (tmp_path / "visible.md").write_text("# Hello\nContent here.")
    docs = load_documents(tmp_path)
    sources = [d.metadata["source"] for d in docs]
    assert all(".hidden" not in s for s in sources)
    assert any("visible.md" in s for s in sources)


def test_load_documents_skips_unsupported_extensions(tmp_path):
    (tmp_path / "notes.docx").write_bytes(b"fake docx")
    (tmp_path / "readme.md").write_text("# Readme")
    docs = load_documents(tmp_path)
    sources = [d.metadata["source"] for d in docs]
    assert not any("notes.docx" in s for s in sources)
    assert any("readme.md" in s for s in sources)


def test_load_documents_returns_empty_for_empty_dir(tmp_path):
    assert load_documents(tmp_path) == []


def test_split_documents_assigns_sequential_chunk_ids():
    docs = [
        Document(page_content="A" * 2000, metadata={"source": "a.md"}),
        Document(page_content="B" * 2000, metadata={"source": "b.md"}),
    ]
    config = Settings(chunk_size=500, chunk_overlap=50)
    chunks = split_documents(docs, config)
    chunk_ids = [c.metadata["chunk_id"] for c in chunks]
    assert chunk_ids == list(range(len(chunks)))


def test_split_documents_respects_chunk_size():
    long_text = "内容。" * 500
    docs = [Document(page_content=long_text, metadata={"source": "x.md"})]
    config = Settings(chunk_size=200, chunk_overlap=20)
    chunks = split_documents(docs, config)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 300


def test_load_documents_recurses_into_subdirectories(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text("nested content")
    docs = load_documents(tmp_path)
    sources = [d.metadata["source"] for d in docs]
    assert any("nested.md" in s for s in sources)
