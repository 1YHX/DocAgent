from __future__ import annotations

import shutil
import sys
from pathlib import Path

from docagent.config import Settings, settings
from docagent.ingest import SUPPORTED_SUFFIXES, ingest
from docagent.state import AgentState
from docagent.vectorstore import get_vectorstore


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


def run_status(config: Settings = settings) -> dict[str, object]:
    data_files = list_data_files(config)
    index = get_index_summary(config)

    print("DocAgent status")
    print(f"- Data directory: {config.source_dir}")
    print(f"- Vector store: {config.persist_dir}")
    print(f"- Collection: {config.collection_name}")
    print(f"- Data files: {len(data_files)}")
    print(f"- Indexed chunks: {index['chunk_count']}")
    print(f"- Indexed sources: {len(index['sources'])}")

    missing = sorted(set(data_files) - set(index["sources"]))
    stale = sorted(set(index["sources"]) - set(data_files))

    if data_files:
        print("\nData files:")
        for source in data_files:
            marker = "indexed" if source in index["sources"] else "not indexed"
            print(f"- {source} ({marker})")

    if index["sources"]:
        print("\nIndexed sources:")
        for source, count in index["source_counts"].items():
            print(f"- {source}: {count} chunks")

    if missing:
        print("\nNot indexed yet:")
        for source in missing:
            print(f"- {source}")
        print("Run `docagent ingest --reset` to update the vector store.")

    if stale:
        print("\nIndexed but missing from data/:")
        for source in stale:
            print(f"- {source}")
        print("Run `docagent ingest --reset` to remove stale entries.")

    return {"data_files": data_files, **index, "missing": missing, "stale": stale}


def run_sources(config: Settings = settings) -> dict[str, object]:
    index = get_index_summary(config)
    print("Indexed sources")
    if not index["sources"]:
        print("- <empty>")
        print("Run `docagent ingest --reset` after adding documents to data/.")
        return index

    for source, count in index["source_counts"].items():
        print(f"- {source}: {count} chunks")
    return index


def list_data_files(config: Settings = settings) -> list[str]:
    if not config.source_dir.exists():
        return []
    files = []
    for path in sorted(config.source_dir.rglob("*")):
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(str(path))
    return files


def get_index_summary(config: Settings = settings) -> dict[str, object]:
    vectorstore = get_vectorstore(config)
    data = vectorstore.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    source_counts: dict[str, int] = {}
    for metadata in metadatas:
        source = str((metadata or {}).get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "chunk_count": len(data.get("ids", [])),
        "sources": sorted(source_counts),
        "source_counts": dict(sorted(source_counts.items())),
    }


def run_chat(show_trace: bool = False, baseline: bool = False) -> None:
    from docagent.main import ask, print_result

    trace_enabled = show_trace
    baseline_enabled = baseline
    last_question: str | None = None
    last_answer: str | None = None

    print("DocAgent chat")
    print(
        "Type a question and press Enter. Commands: /help, /trace on|off, "
        "/baseline on|off, /compare <question>, /status, /reset, /exit"
    )

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
            print("- /status             Show indexed files and chunks.")
            print("- /reset              Clear chat context.")
            print("- /exit               Leave chat.")
            continue
        if user_input == "/status":
            run_status()
            continue
        if user_input == "/reset":
            last_question = None
            last_answer = None
            print("chat context cleared")
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
            contextual_query = build_contextual_query(user_input, last_question, last_answer)
            result = ask(user_input, baseline=baseline_enabled, query=contextual_query)
            print_result(result, show_trace=trace_enabled)
            last_question = user_input
            last_answer = result.get("answer", "")
        except Exception as error:
            print_cli_error(error)


def build_contextual_query(question: str, last_question: str | None, _last_answer: str | None) -> str:
    if not last_question:
        return question

    query = f"{last_question}；{question}"
    if any(keyword in question for keyword in ["几个", "多少", "三个", "第三", "还有", "不是"]):
        query += "；核实项目列表、项目数量、是否存在补充项目"
    return query


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
            True,
            f"optional; current: {config.embedding_base_url or '<not set>'}",
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
