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


def run_compare(question: str, show_trace: bool = True) -> tuple[AgentState, AgentState]:
    from docagent.main import ask, print_result

    print("Baseline RAG")
    print("=" * 12)
    baseline_result = ask(question, baseline=True)
    print_result(baseline_result, show_trace=show_trace)

    print("\nDocAgent")
    print("=" * 8)
    agent_result = ask(question, baseline=False)
    print_result(agent_result, show_trace=show_trace)

    print("\nComparison")
    print("=" * 10)
    print("- Baseline RAG: retrieves once, then generates directly.")
    print("- DocAgent: grades retrieved evidence before deciding whether to answer, rewrite, or fallback.")
    return baseline_result, agent_result


def run_chat(show_trace: bool = False, baseline: bool = False) -> None:
    from docagent.main import ask, print_result

    trace_enabled = show_trace
    baseline_enabled = baseline

    print("DocAgent chat")
    print("Type a question and press Enter. Commands: /help, /trace on|off, /baseline on|off, /compare <question>, /exit")

    while True:
        try:
            user_input = input("\ndocagent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not user_input:
            continue
        if user_input in {"/exit", "/quit", "exit", "quit"}:
            print("bye")
            return
        if user_input == "/help":
            print("Commands:")
            print("- /trace on|off       Show or hide retrieve/rewrite trace.")
            print("- /baseline on|off    Switch between baseline RAG and DocAgent.")
            print("- /compare <question> Compare baseline RAG with DocAgent for one question.")
            print("- /exit               Leave chat.")
            continue
        if user_input.startswith("/trace"):
            trace_enabled = _parse_toggle(user_input, trace_enabled, "/trace")
            print(f"trace: {'on' if trace_enabled else 'off'}")
            continue
        if user_input.startswith("/baseline"):
            baseline_enabled = _parse_toggle(user_input, baseline_enabled, "/baseline")
            print(f"baseline: {'on' if baseline_enabled else 'off'}")
            continue
        if user_input.startswith("/compare "):
            question = user_input.removeprefix("/compare ").strip()
            if not question:
                print("Usage: /compare <question>")
                continue
            run_compare(question, show_trace=trace_enabled)
            continue

        try:
            result = ask(user_input, baseline=baseline_enabled)
            print_result(result, show_trace=trace_enabled)
        except Exception as error:
            print_cli_error(error)


def _parse_toggle(command: str, current: bool, prefix: str) -> bool:
    parts = command.split()
    if len(parts) == 1:
        return not current
    if len(parts) == 2 and parts[1] in {"on", "true", "1"}:
        return True
    if len(parts) == 2 and parts[1] in {"off", "false", "0"}:
        return False
    print(f"Usage: {prefix} on|off")
    return current


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
