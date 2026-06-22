from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document


Route = Literal["generate", "rewrite", "fallback"]


class Grade(TypedDict):
    doc_index: int
    relevant: bool
    reason: str


class AgentState(TypedDict, total=False):
    question: str
    standalone_question: str
    query: str
    retry_count: int
    documents: list[Document]
    grades: list[Grade]
    relevant_documents: list[Document]
    answer: str
    self_check: str
    revised: bool
    streamed: bool
    route: Route
    history: list[str]
