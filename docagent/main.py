from __future__ import annotations

import argparse
from collections.abc import Callable

from docagent.cli import print_cli_error, run_chat, run_compare, run_demo, run_doctor, run_sources, run_status
from docagent.graph import build_graph
from docagent.ingest import ingest
from docagent.nodes import generate_baseline, retrieve
from docagent.resume import answer_resume_project_question
from docagent.state import AgentState


def ask(
    question: str,
    baseline: bool = False,
    query: str | None = None,
    on_token: Callable[[str], None] | None = None,
) -> AgentState:
    search_query = query or question
    history = [] if query is None else [f"contextual_query: {search_query}"]
    initial: AgentState = {
        "question": question,
        "standalone_question": search_query,
        "query": search_query,
        "retry_count": 0,
        "history": history,
    }
    if baseline:
        retrieved = retrieve(initial)
        answer = generate_baseline({**initial, **retrieved})
        return {**initial, **retrieved, **answer}
    structured_answer = answer_resume_project_question(question, query=search_query)
    if structured_answer:
        return structured_answer
    return build_graph(on_token=on_token).invoke(initial)


def print_result(result: AgentState, show_trace: bool = False) -> None:
    if not result.get("streamed"):
        print(result.get("answer", ""))
    if result.get("self_check"):
        print(f"\nSelf-check: {result['self_check']}")
        if str(result["self_check"]).lower().startswith("unsupported"):
            print("Warning: self-check marked this answer as unsupported by the retrieved evidence.")
    if show_trace and result.get("history"):
        print("\nTrace:")
        for item in result["history"]:
            print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocAgent CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("doctor", help="Check .env and local configuration.")
    subparsers.add_parser("demo", help="Run the bundled mini knowledge-base demo.")
    subparsers.add_parser("status", help="Show data files and indexed Chroma sources.")
    subparsers.add_parser("sources", help="List sources currently stored in Chroma.")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive DocAgent chat session.")
    chat_parser.add_argument("--show-trace", action="store_true", help="Show retrieve/rewrite trace by default.")
    chat_parser.add_argument("--baseline", action="store_true", help="Start in baseline RAG mode.")

    compare_parser = subparsers.add_parser("compare", help="Compare baseline RAG with DocAgent.")
    compare_parser.add_argument("question")
    compare_parser.add_argument("--no-trace", action="store_true", help="Hide retrieve/rewrite traces.")

    ingest_parser = subparsers.add_parser("ingest", help="Load data/ documents into Chroma.")
    ingest_parser.add_argument("--reset", action="store_true", help="Clear the existing Chroma collection first.")

    ask_parser = subparsers.add_parser("ask", help="Ask a question.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--baseline", action="store_true", help="Run one-shot baseline RAG without reflection.")
    ask_parser.add_argument("--show-trace", action="store_true", help="Print retrieve/rewrite trace.")

    args = parser.parse_args()

    try:
        if args.command is None:
            run_chat()
            return

        if args.command == "doctor":
            raise SystemExit(run_doctor())

        if args.command == "demo":
            run_demo()
            return

        if args.command == "status":
            run_status()
            return

        if args.command == "sources":
            run_sources()
            return

        if args.command == "chat":
            run_chat(show_trace=args.show_trace, baseline=args.baseline)
            return

        if args.command == "compare":
            run_compare(args.question, show_trace=not args.no_trace)
            return

        if args.command == "ingest":
            count = ingest(reset=args.reset)
            print(f"Ingested {count} chunks.")
            return

        result = ask(args.question, baseline=args.baseline)
        print_result(result, show_trace=args.show_trace)
    except Exception as error:
        raise SystemExit(print_cli_error(error)) from error


if __name__ == "__main__":
    main()
