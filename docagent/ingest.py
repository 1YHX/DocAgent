from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docagent.config import Settings, settings
from docagent.vectorstore import get_vectorstore


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_documents(source_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        if suffix == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
        else:
            documents.extend(TextLoader(str(path), encoding="utf-8").load())
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
            import warnings
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
