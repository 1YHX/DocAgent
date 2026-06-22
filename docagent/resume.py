from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader

from docagent.config import Settings, settings
from docagent.state import AgentState


PROJECT_COUNT_TERMS = ("多少", "几个", "数量", "不是", "还有", "别的", "第三", "三个")


def answer_resume_project_question(
    question: str,
    query: str | None = None,
    config: Settings = settings,
) -> AgentState | None:
    search_text = f"{question}\n{query or ''}"
    if "项目" not in search_text or not any(term in search_text for term in PROJECT_COUNT_TERMS):
        return None

    projects = []
    research_items = []
    sources = []
    for pdf_path in sorted(config.source_dir.rglob("*.pdf")):
        text = _load_pdf_text(pdf_path)
        extracted_projects = extract_project_titles(text)
        extracted_research = extract_research_titles(text)
        if extracted_projects:
            projects.extend(extracted_projects)
            research_items.extend(extracted_research)
            sources.append(str(pdf_path))

    projects = _dedupe(projects)
    research_items = _dedupe(research_items)
    if not projects:
        return None

    lines = [
        "我按简历里的栏目来数：在“项目经历”部分，明确列出的项目共有 "
        f"**{len(projects)} 个**。",
        "",
    ]
    for index, project in enumerate(projects, start=1):
        lines.append(f"{index}. **{project}**")

    if research_items:
        lines.extend(
            [
                "",
                "另外，简历里还有“科研经历”，但它不是“项目经历”，所以不计入项目数量：",
            ]
        )
        for item in research_items:
            lines.append(f"- {item}")

    lines.extend(["", f"依据：{', '.join(sources)}"])

    return {
        "question": question,
        "standalone_question": query or question,
        "query": query or question,
        "answer": "\n".join(lines),
        "self_check": "supported。答案由简历“项目经历”栏目中的标题确定性抽取得到。",
        "history": [
            f"structured_resume: extracted {len(projects)} project entries from {', '.join(sources)}",
        ],
    }


def _load_pdf_text(path: Path) -> str:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_project_titles(text: str) -> list[str]:
    section = _between(text, "项目经历", "竞赛与证书")
    if not section:
        return []

    titles = []
    for match in re.finditer(r"(?:^|[。•]\s*)([^。•｜]{4,90}?)\s*｜\s*[^。•]+?(?=\s*•)", section):
        title = match.group(1).strip()
        title = title.removeprefix("项目经历").strip()
        if title and not _is_pdf_header(title):
            titles.append(title)
    return _dedupe(titles)


def extract_research_titles(text: str) -> list[str]:
    section = _between(text, "科研经历", "项目经历")
    if not section:
        return []

    titles = []
    for match in re.finditer(r"(?:^|[。•]\s*)([^。•｜]{4,90}?)\s*｜\s*[^。•]+?(?=\s*•)", section):
        title = match.group(1).strip()
        title = title.removeprefix("科研经历").strip()
        if title and not _is_pdf_header(title):
            titles.append(title)
    return _dedupe(titles)


def _between(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    if start_index == -1:
        return ""
    end_index = text.find(end, start_index + len(start))
    if end_index == -1:
        return text[start_index:]
    return text[start_index:end_index]


def _dedupe(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        normalized = " ".join(item.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _is_pdf_header(title: str) -> bool:
    return title == "易海祥" or "简历" in title
