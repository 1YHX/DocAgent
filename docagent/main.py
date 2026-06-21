from __future__ import annotations

import argparse

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


def main() -> None:
    parser = argparse.ArgumentParser(description="DocAgent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Load data/ documents into Chroma.")

    ask_parser = subparsers.add_parser("ask", help="Ask a question.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--baseline", action="store_true", help="Run one-shot baseline RAG without reflection.")
    ask_parser.add_argument("--show-trace", action="store_true", help="Print retrieve/rewrite trace.")

    args = parser.parse_args()

    if args.command == "ingest":
        count = ingest()
        print(f"Ingested {count} chunks.")
        return

    result = ask(args.question, baseline=args.baseline)
    print(result.get("answer", ""))
    if result.get("self_check"):
        print(f"\nSelf-check: {result['self_check']}")
    if args.show_trace and result.get("history"):
        print("\nTrace:")
        for item in result["history"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()
