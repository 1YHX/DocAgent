from __future__ import annotations

import shutil
import sys
from pathlib import Path

from docagent.config import Settings, settings
from docagent.ingest import ingest
from docagent.state import AgentState


DEMO_SOURCE = Path("examples/mini_knowledge_base.md")
DEMO_TARGET = Path("data/docagent_demo.md")
DEMO_QUESTION = "DocAgent 的核心流程是什么？"


def run_demo() -> AgentState:
    if not DEMO_SOURCE.exists():
        raise RuntimeError(f"Demo source file not found: {DEMO_SOURCE}")

    DEMO_TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEMO_SOURCE, DEMO_TARGET)
    chunk_count = ingest(reset=True)
    print(f"Demo knowledge base ready: {chunk_count} chunks ingested.")
    print(f"Question: {DEMO_QUESTION}\n")

    from docagent.main import ask, print_result

    result = ask(DEMO_QUESTION, baseline=False)
    print_result(result, show_trace=True)
    return result


def run_doctor(config: Settings = settings) -> int:
    checks = [
        ("CHAT_API_KEY", bool(config.chat_api_key), "required for grade/rewrite/generate/self-check"),
        ("CHAT_BASE_URL", bool(config.chat_base_url), f"current: {config.chat_base_url or '<empty>'}"),
        ("CHAT_MODEL", bool(config.chat_model), f"current: {config.chat_model or '<empty>'}"),
        ("EMBEDDING_API_KEY", bool(config.embedding_api_key), "required for ingest/retrieve"),
        (
            "EMBEDDING_BASE_URL",
            bool(config.embedding_base_url),
            f"current: {config.embedding_base_url or '<empty>'}",
        ),
        ("EMBEDDING_MODEL", bool(config.embedding_model), f"current: {config.embedding_model or '<empty>'}"),
    ]

    has_error = False
    print("DocAgent configuration check")
    for name, ok, detail in checks:
        marker = "OK" if ok else "MISSING"
        print(f"- {marker:7} {name}: {detail}")
        has_error = has_error or not ok

    print(f"\nKnowledge base directory: {config.source_dir}")
    print(f"Vector store directory: {config.persist_dir}")
    print(f"Collection: {config.collection_name}")

    if has_error:
        print("\nFix .env first. Start from .env.example and fill the missing values.", file=sys.stderr)
        return 1
    return 0


def print_cli_error(error: Exception) -> int:
    message = str(error)
    print(f"Error: {message}", file=sys.stderr)

    if "CHAT_API_KEY" in message or "EMBEDDING_API_KEY" in message:
        print("Run `python -m docagent.main doctor` to check your .env.", file=sys.stderr)
    elif "No supported documents" in message:
        print("Put .md, .txt, or .pdf files in data/, or run `python -m docagent.main demo`.", file=sys.stderr)
    elif "Connection" in message or "connect" in message.lower():
        print("Check your API base URL, network, and proxy settings.", file=sys.stderr)

    return 1
