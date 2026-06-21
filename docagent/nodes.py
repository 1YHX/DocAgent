from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from docagent.config import Settings, settings
from docagent.models import build_chat_model
from docagent.state import AgentState, Grade, Route
from docagent.vectorstore import get_vectorstore


def format_documents(documents: list[Document]) -> str:
    blocks = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        blocks.append(f"[{index}] source={source} chunk={chunk_id}\n{doc.page_content}")
    return "\n\n".join(blocks)


def retrieve(state: AgentState, app_settings: Settings = settings) -> AgentState:
    query = state.get("query") or state["question"]
    retriever = get_vectorstore(app_settings).as_retriever(search_kwargs={"k": app_settings.top_k})
    documents = retriever.invoke(query)
    history = state.get("history", []) + [f"retrieve: {query} -> {len(documents)} docs"]
    return {"query": query, "documents": documents, "history": history}


def generate_baseline(state: AgentState) -> AgentState:
    docs = state.get("documents", [])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个严谨的文档问答助手。只能基于给定资料回答；资料不足时直接说明资料中未找到相关信息。"),
            ("human", "问题：{question}\n\n资料：\n{context}\n\n请给出中文答案，并在末尾列出依据编号。"),
        ]
    )
    chain = prompt | build_chat_model()
    response = chain.invoke({"question": state["question"], "context": format_documents(docs)})
    return {"answer": response.content}


def grade_documents(state: AgentState, app_settings: Settings = settings) -> AgentState:
    documents = state.get("documents", [])
    if not documents:
        return {"grades": [], "relevant_documents": []}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 RAG 检索质量评估器。判断每个片段是否包含回答问题所需的信息。"
                "只输出 JSON 数组，不要输出 Markdown。每项格式："
                '{{"doc_index": 0, "relevant": true, "reason": "简短原因"}}',
            ),
            ("human", "问题：{question}\n\n候选片段：\n{context}"),
        ]
    )
    response = (prompt | build_chat_model()).invoke(
        {"question": state["question"], "context": format_documents(documents)}
    )
    grades = _parse_grades(str(response.content), len(documents))
    relevant_docs = [documents[item["doc_index"]] for item in grades if item["relevant"]]
    return {"grades": grades, "relevant_documents": relevant_docs}


def _parse_grades(raw: str, doc_count: int) -> list[Grade]:
    try:
        data = json.loads(_extract_json_payload(raw))
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        data = data.get("grades") or data.get("documents") or data.get("items")

    grades: list[Grade] = []
    if not isinstance(data, list):
        return grades

    seen_indexes: set[int] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        index = item.get("doc_index")
        if not isinstance(index, int) or index < 0 or index >= doc_count or index in seen_indexes:
            continue
        seen_indexes.add(index)
        grades.append(
            {
                "doc_index": index,
                "relevant": bool(item.get("relevant")),
                "reason": str(item.get("reason", "")),
            }
        )
    return grades


def _extract_json_payload(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        return cleaned[array_start : array_end + 1]

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        return cleaned[object_start : object_end + 1]

    return cleaned


def decide(state: AgentState, app_settings: Settings = settings) -> Route:
    relevant_count = len(state.get("relevant_documents", []))
    if relevant_count >= app_settings.min_relevant_docs:
        return "generate"
    if state.get("retry_count", 0) < app_settings.max_retries:
        return "rewrite"
    return "fallback"


def rewrite_query(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是检索 query 改写助手。基于原问题和失败原因，改写成更适合文档检索的中文查询。"
                "只输出改写后的 query。",
            ),
            (
                "human",
                "原问题：{question}\n上一次 query：{query}\n评估结果：{grades}\n请改写 query：",
            ),
        ]
    )
    response = (prompt | build_chat_model()).invoke(
        {
            "question": state["question"],
            "query": state.get("query", state["question"]),
            "grades": json.dumps(state.get("grades", []), ensure_ascii=False),
        }
    )
    retry_count = state.get("retry_count", 0) + 1
    history = state.get("history", []) + [f"rewrite[{retry_count}]: {response.content}"]
    return {"query": str(response.content).strip(), "retry_count": retry_count, "history": history}


def generate(state: AgentState) -> AgentState:
    docs = state.get("relevant_documents") or state.get("documents", [])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 DocAgent，一个严谨的文档问答 Agent。必须只基于资料回答，不要补充资料外事实。"
                "如果资料不足，明确说资料中未找到相关信息。答案末尾用“依据：”列出来源。",
            ),
            ("human", "问题：{question}\n\n可用资料：\n{context}\n\n请回答："),
        ]
    )
    response = (prompt | build_chat_model()).invoke(
        {"question": state["question"], "context": format_documents(docs)}
    )
    return {"answer": str(response.content)}


def fallback(state: AgentState) -> AgentState:
    return {
        "answer": "资料中未找到相关信息。为了避免编造答案，DocAgent 不会基于当前知识库继续推测。",
    }


def self_check(state: AgentState) -> AgentState:
    docs = state.get("relevant_documents") or state.get("documents", [])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是答案事实性检查器。判断答案是否完全由资料支持。"
                "只输出 supported 或 unsupported，并附一句简短原因。",
            ),
            ("human", "问题：{question}\n答案：{answer}\n资料：\n{context}"),
        ]
    )
    response = (prompt | build_chat_model()).invoke(
        {
            "question": state["question"],
            "answer": state.get("answer", ""),
            "context": format_documents(docs),
        }
    )
    return {"self_check": str(response.content)}
