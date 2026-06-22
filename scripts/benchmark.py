#!/usr/bin/env python3
"""
Benchmark: compare baseline RAG vs DocAgent on a fixed question set.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --questions questions.txt   # one question per line
    python scripts/benchmark.py --md                        # output Markdown table
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_QUESTIONS = [
    # Q1 - has an answer in the knowledge base
    "DocAgent 的核心流程是什么？",
    # Q2 - query wording differs from document wording (tests rewrite)
    "这个系统是怎么防止 AI 胡说八道的？",
    # Q3 - asks for something NOT in the knowledge base (tests fallback)
    "DocAgent 支持多少种语言？",
    # Q4 - counting / enumeration question (tests grade + rewrite loop)
    "知识库里提到了哪些 Agent 节点？",
]


def _truncate(text: str, max_chars: int = 160) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def run_benchmark(questions: list[str], markdown: bool = False) -> list[dict]:
    from docagent.main import ask

    rows = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}", file=sys.stderr)

        t0 = time.perf_counter()
        baseline = ask(question, baseline=True)
        t_baseline = time.perf_counter() - t0
        print(f"  baseline: {t_baseline:.1f}s", file=sys.stderr)

        t0 = time.perf_counter()
        agent = ask(question, baseline=False)
        t_agent = time.perf_counter() - t0
        print(f"  docagent: {t_agent:.1f}s", file=sys.stderr)

        rows.append(
            {
                "question": question,
                "baseline_answer": baseline.get("answer", ""),
                "agent_answer": agent.get("answer", ""),
                "agent_self_check": agent.get("self_check", ""),
                "agent_history": agent.get("history", []),
                "t_baseline": t_baseline,
                "t_agent": t_agent,
            }
        )

    return rows


def print_text_report(rows: list[dict]) -> None:
    sep = "=" * 72
    for i, row in enumerate(rows, 1):
        print(f"\n{sep}")
        print(f"Q{i}: {row['question']}")
        print(sep)
        print("\n[Baseline RAG]")
        print(textwrap.fill(row["baseline_answer"], 72))
        print(f"\n[DocAgent]  self-check: {row['agent_self_check']}")
        print(textwrap.fill(row["agent_answer"], 72))
        print(f"\nTrace: {' -> '.join(row['agent_history'])}")
        print(f"Time: baseline={row['t_baseline']:.1f}s  agent={row['t_agent']:.1f}s")


def print_markdown_table(rows: list[dict]) -> None:
    print("| # | 问题 | Baseline RAG | DocAgent | Self-check |")
    print("|---|------|-------------|---------|-----------|")
    for i, row in enumerate(rows, 1):
        q = row["question"]
        b = _truncate(row["baseline_answer"])
        a = _truncate(row["agent_answer"])
        sc = row["agent_self_check"].split("。")[0] if row["agent_self_check"] else "-"
        print(f"| {i} | {q} | {b} | {a} | {sc} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline RAG vs DocAgent")
    parser.add_argument("--questions", help="Path to question file (one per line)")
    parser.add_argument("--md", action="store_true", help="Output a Markdown table")
    args = parser.parse_args()

    if args.questions:
        questions = Path(args.questions).read_text(encoding="utf-8").splitlines()
        questions = [q.strip() for q in questions if q.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    print(f"Running benchmark on {len(questions)} questions...", file=sys.stderr)
    rows = run_benchmark(questions, markdown=args.md)

    if args.md:
        print_markdown_table(rows)
    else:
        print_text_report(rows)


if __name__ == "__main__":
    main()
