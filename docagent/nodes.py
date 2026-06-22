from __future__ import annotations

import json
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from docagent.config import Settings, settings
from docagent.models import build_chat_model
from docagent.state import AgentState, Grade, Route
from docagent.vectorstore import get_vectorstore

TokenCallback = Callable[[str], None]


def format_documents(documents: list[Document]) -> str:
    blocks = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        blocks.append(f"[{index}] source={source} chunk={chunk_id}\n{doc.page_content}")
    return "\n\n".join(blocks)


def question_for_prompt(state: AgentState) -> str:
    return state.get("standalone_question") or state["question"]


def retrieve(state: AgentState, app_settings: Settings = settings) -> AgentState:
    query = state.get("query") or state["question"]
    search_k = search_k_for_query(query, app_settings)
    retriever = get_vectorstore(app_settings).as_retriever(search_kwargs={"k": search_k})
    documents = retriever.invoke(query)
    history = state.get("history", []) + [f"retrieve: {query} -> {len(documents)} docs (k={search_k})"]
    return {"query": query, "documents": documents, "history": history}


def search_k_for_query(query: str, app_settings: Settings = settings) -> int:
    if any(keyword in query for keyword in ["项目", "经历", "列表", "几个", "多少", "第三", "还有"]):
        return max(app_settings.top_k, 8)
    return app_settings.top_k


def generate_baseline(state: AgentState) -> AgentState:
    docs = state.get("documents", [])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
        "system",
        "你是 DocAgent，一个 AI 文档问答助手，不是知识库中任何文档所描述的人物或实体。"
        "资料是你回答用户问题的参考来源，不代表你自身的身份。"
        "如果被问及“你是谁”或“你是什么”，请介绍自己是 DocAgent AI 文档问答助手。"
        "只能基于给定资料回答其他问题；资料不足时直接说明资料中未找到相关信息。",
    ),
            ("human", "问题：{question}\n\n资料：\n{context}\n\n请给出中文答案，并在末尾列出依据编号。"),
        ]
    )
    chain = prompt | build_chat_model()
    response = chain.invoke({"question": question_for_prompt(state), "context": format_documents(docs)})
    history = state.get("history", []) + [f"baseline_generate: using {len(docs)} retrieved docs"]
    return {"answer": response.content, "history": history}


def grade_documents(state: AgentState, app_settings: Settings = settings) -> AgentState:
    documents = state.get("documents", [])
    if not documents:
        history = state.get("history", []) + ["grade: 0/0 relevant docs"]
        return {"grades": [], "relevant_documents": [], "history": history}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 RAG 检索质量评估器。判断每个片段是否包含回答问题所需的信息。"
                "如果问题在核实列表、数量或是否遗漏，包含候选条目、项目细节或经历条目的片段也算相关，"
                "即使片段没有直接写出总数。"
                "只输出 JSON 对象，格式为 {{\"grades\": [...]}}，不要输出 Markdown。每项格式："
                '{{"doc_index": 0, "relevant": true, "reason": "简短原因"}}',
            ),
            ("human", "问题：{question}\n\n候选片段：\n{context}"),
        ]
    )
    # Bind JSON mode to eliminate markdown-wrapping and prose around the array.
    grader = build_chat_model().bind(response_format={"type": "json_object"})
    response = (prompt | grader).invoke(
        {"question": question_for_prompt(state), "context": format_documents(documents)}
    )
    grades = _parse_grades(str(response.content), len(documents))
    relevant_docs = [documents[item["doc_index"]] for item in grades if item["relevant"]]
    history = state.get("history", []) + [
        f"grade: {len(relevant_docs)}/{len(documents)} relevant docs",
        *_format_grade_reasons(grades),
    ]
    return {"grades": grades, "relevant_documents": relevant_docs, "history": history}


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


def decide_node(state: AgentState, app_settings: Settings = settings) -> AgentState:
    route = decide(state, app_settings)
    relevant_count = len(state.get("relevant_documents", []))
    retry_count = state.get("retry_count", 0)
    history = state.get("history", []) + [
        f"decide: {route} (relevant={relevant_count}, retry={retry_count}/{app_settings.max_retries})"
    ]
    return {"route": route, "history": history}


def route_from_state(state: AgentState) -> Route:
    return state.get("route", "fallback")


def _format_grade_reasons(grades: list[Grade]) -> list[str]:
    reasons = []
    for grade in grades:
        verdict = "relevant" if grade["relevant"] else "irrelevant"
        reason = grade["reason"].strip()
        if reason:
            reasons.append(f"grade_doc[{grade['doc_index']}]: {verdict} - {reason}")
        else:
            reasons.append(f"grade_doc[{grade['doc_index']}]: {verdict}")
    return reasons


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
            "question": question_for_prompt(state),
            "query": state.get("query", state["question"]),
            "grades": json.dumps(state.get("grades", []), ensure_ascii=False),
        }
    )
    retry_count = state.get("retry_count", 0) + 1
    rewritten = str(response.content).strip()
    if not rewritten:
        rewritten = state.get("query") or state["question"]
    history = state.get("history", []) + [f"rewrite[{retry_count}]: {rewritten}"]
    return {"query": rewritten, "retry_count": retry_count, "history": history}


def make_generate_node(on_token: TokenCallback | None = None) -> Callable[[AgentState], AgentState]:
    def _generate(state: AgentState) -> AgentState:
        docs = state.get("relevant_documents") or state.get("documents", [])
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 DocAgent，一个 AI 文档问答助手，不是知识库中任何文档所描述的人物或实体。"
                    "资料是你回答用户问题的参考来源，不代表你自身的身份。"
                    "如果被问及“你是谁”或“你是什么”，请介绍自己是 DocAgent AI 文档问答助手，"
                    "并说明当前知识库包含的文档内容供用户参考。"
                    "其他问题必须只基于资料回答，不要补充资料外事实；资料不足时明确说资料中未找到相关信息。"
                    "当问题要求统计、核实数量或列举条目时，可以基于资料中明确出现的不同条目名称计数，"
                    "但必须说明计数口径。处理简历时，要区分“项目经历”和“科研经历”："
                    "除非用户明确要求广义经历，否则科研经历不要计入项目数量。"
                    "答案末尾用“依据：”列出来源。",
                ),
                ("human", "问题：{question}\n\n可用资料：\n{context}\n\n请回答："),
            ]
        )
        chain = prompt | build_chat_model()
        input_dict = {"question": question_for_prompt(state), "context": format_documents(docs)}
        if on_token:
            chunks = [chunk.content for chunk in chain.stream(input_dict) if chunk.content]
            for chunk in chunks:
                on_token(chunk)
            answer = "".join(chunks)
        else:
            answer = str(chain.invoke(input_dict).content)
        history = state.get("history", []) + [f"generate: using {len(docs)} relevant docs"]
        return {"answer": answer, "streamed": on_token is not None, "history": history}

    return _generate


def make_revise_node(on_token: TokenCallback | None = None) -> Callable[[AgentState], AgentState]:
    def _revise(state: AgentState) -> AgentState:
        docs = state.get("relevant_documents") or state.get("documents", [])
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 DocAgent 的答案纠错器，是一个 AI 助手，不是知识库中描述的任何人物。"
                    "上一版答案已被 self-check 判定为 unsupported。"
                    "请根据资料和 self-check 的批评重写答案。必须修正遗漏、误数、误分类等问题。"
                    "如果资料支持通过枚举条目得到数量，可以明确给出数量，并说明计数口径。"
                    "处理简历时，科研经历不要计入项目数量；“项目经历”下的独立项目和明确写作“补充项目”的条目可以计入项目/补充项目。"
                    "不要保留上一版答案中的错误说法。",
                ),
                (
                    "human",
                    "问题：{question}\n\n资料：\n{context}\n\n上一版答案：\n{answer}\n\nSelf-check 批评：\n{self_check}\n\n请给出修正版答案：",
                ),
            ]
        )
        chain = prompt | build_chat_model()
        input_dict = {
            "question": question_for_prompt(state),
            "context": format_documents(docs),
            "answer": state.get("answer", ""),
            "self_check": state.get("self_check", ""),
        }
        if on_token:
            on_token("\n\n[答案已修正]\n")
            chunks = [chunk.content for chunk in chain.stream(input_dict) if chunk.content]
            for chunk in chunks:
                on_token(chunk)
            answer = "".join(chunks)
        else:
            answer = str(chain.invoke(input_dict).content)
        history = state.get("history", []) + ["revise: regenerated answer after unsupported self-check"]
        return {"answer": answer, "revised": True, "streamed": on_token is not None, "history": history}

    return _revise


generate = make_generate_node()
revise_answer = make_revise_node()


def fallback(state: AgentState) -> AgentState:
    history = state.get("history", []) + ["fallback: no sufficiently relevant evidence after retries"]
    return {
        "answer": "资料中未找到相关信息。为了避免编造答案，DocAgent 不会基于当前知识库继续推测。",
        "history": history,
    }


def self_check(state: AgentState) -> AgentState:
    docs = state.get("relevant_documents") or state.get("documents", [])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是答案事实性检查器。判断答案是否完全由资料支持。"
                "如果答案中的数量是通过枚举资料里明确出现的不同条目得到的，且条目名称都能在资料中找到，"
                "应视为 supported。处理简历时，要区分“项目经历”和“科研经历”。"
                "只输出 supported 或 unsupported，并附一句简短原因。",
            ),
            ("human", "问题：{question}\n答案：{answer}\n资料：\n{context}"),
        ]
    )
    response = (prompt | build_chat_model()).invoke(
        {
            "question": question_for_prompt(state),
            "answer": state.get("answer", ""),
            "context": format_documents(docs),
        }
    )
    history = state.get("history", []) + [f"self_check: {response.content}"]
    return {"self_check": str(response.content), "history": history}


def should_revise(state: AgentState) -> str:
    self_check_result = str(state.get("self_check", "")).lower()
    if self_check_result.startswith("unsupported") and not state.get("revised", False):
        return "revise"
    return "end"
