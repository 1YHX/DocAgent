from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from docagent.config import Settings, settings
from docagent.vectorstore import get_vectorstore


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def _load_pdf(path: Path) -> list[Document]:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path))
    docs = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(Document(
            page_content=text,
            metadata={"source": str(path), "page": page_num},
        ))
    return docs


def _load_text(path: Path) -> list[Document]:
    return [Document(
        page_content=path.read_text(encoding="utf-8"),
        metadata={"source": str(path)},
    )]


def load_documents(source_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        if suffix == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            documents.extend(_load_text(path))
    return documents


def split_documents(documents: list[Document], config: Settings = settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks


def ingest(config: Settings = settings, reset: bool = False) -> int:
    config.source_dir.mkdir(parents=True, exist_ok=True)
    documents = load_documents(config.source_dir)
    if not documents:
        raise RuntimeError(f"No supported documents found in {config.source_dir}. Add .md, .txt, or .pdf files.")

    chunks = split_documents(documents, config)
    vectorstore = get_vectorstore(config)
    if reset:
        vectorstore.reset_collection()
    else:
        existing = vectorstore.get()
        if existing and existing.get("ids"):
            warnings.warn(
                f"Adding {len(chunks)} chunks to an existing collection with "
                f"{len(existing['ids'])} entries. Run with reset=True (--reset) "
                "to avoid duplicate chunks.",
                stacklevel=2,
            )
    vectorstore.add_documents(chunks)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the DocAgent Chroma index.")
    parser.add_argument("--reset", action="store_true", help="Clear the existing Chroma collection before ingesting.")
    args = parser.parse_args()
    count = ingest(reset=args.reset)
    print(f"Ingested {count} chunks into {settings.persist_dir}.")


if __name__ == "__main__":
    main()
