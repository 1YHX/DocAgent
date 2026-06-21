from __future__ import annotations

import argparse

from docagent.cli import print_cli_error, run_chat, run_compare, run_demo, run_doctor
from docagent.graph import build_graph
from docagent.ingest import ingest
from docagent.nodes import generate_baseline, retrieve
from docagent.state import AgentState


def ask(question: str, baseline: bool = False) -> AgentState:
    initial: AgentState = {"question": question, "query": question, "retry_count": 0, "history": []}
    if baseline:
        retrieved = retrieve(initial)
        answer = generate_baseline({**initial, **retrieved})
        return {**initial, **retrieved, **answer}
    return build_graph().invoke(initial)


def print_result(result: AgentState, show_trace: bool = False) -> None:
    print(result.get("answer", ""))
    if result.get("self_check"):
        print(f"\nSelf-check: {result['self_check']}")
    if show_trace and result.get("history"):
        print("\nTrace:")
        for item in result["history"]:
            print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocAgent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check .env and local configuration.")
    subparsers.add_parser("demo", help="Run the bundled mini knowledge-base demo.")

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
        if args.command == "doctor":
            raise SystemExit(run_doctor())

        if args.command == "demo":
            run_demo()
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
