from docagent.config import Settings
from docagent.nodes import (
    _extract_json_payload,
    _parse_grades,
    decide,
    decide_node,
    fallback,
    format_documents,
    route_from_state,
    search_k_for_query,
    should_revise,
)
from langchain_core.documents import Document


def test_parse_grades_filters_invalid_items():
    raw = '[{"doc_index": 0, "relevant": true, "reason": "命中"}, {"doc_index": 9, "relevant": true}]'

    assert _parse_grades(raw, doc_count=1) == [
        {"doc_index": 0, "relevant": True, "confidence": 1.0, "reason": "命中"}
    ]


def test_parse_grades_accepts_fenced_json():
    raw = """```json
[
  {"doc_index": 0, "relevant": false, "reason": "只提到背景"},
  {"doc_index": 1, "relevant": true, "reason": "包含答案"}
]
```"""

    assert _parse_grades(raw, doc_count=2) == [
        {"doc_index": 0, "relevant": False, "confidence": 0.0, "reason": "只提到背景"},
        {"doc_index": 1, "relevant": True, "confidence": 1.0, "reason": "包含答案"},
    ]


def test_parse_grades_accepts_wrapped_json_object():
    raw = '{"grades": [{"doc_index": 0, "relevant": true, "confidence": 0.82, "reason": "命中"}]}'

    assert _parse_grades(raw, doc_count=1) == [
        {"doc_index": 0, "relevant": True, "confidence": 0.82, "reason": "命中"}
    ]


def test_parse_grades_ignores_duplicate_indexes():
    raw = '[{"doc_index": 0, "relevant": true}, {"doc_index": 0, "relevant": false}]'

    assert _parse_grades(raw, doc_count=1) == [
        {"doc_index": 0, "relevant": True, "confidence": 1.0, "reason": ""}
    ]


def test_parse_grades_clamps_confidence():
    raw = '[{"doc_index": 0, "relevant": true, "confidence": 2}, {"doc_index": 1, "relevant": true, "confidence": -1}]'

    assert _parse_grades(raw, doc_count=2) == [
        {"doc_index": 0, "relevant": True, "confidence": 1.0, "reason": ""},
        {"doc_index": 1, "relevant": True, "confidence": 0.0, "reason": ""},
    ]


def test_decide_generates_when_relevant_docs_are_enough():
    config = Settings(min_relevant_docs=1, max_retries=2)

    assert decide({"relevant_documents": [object()], "retry_count": 0}, config) == "generate"


def test_decide_rewrites_before_retry_limit():
    config = Settings(min_relevant_docs=1, max_retries=2)

    assert decide({"relevant_documents": [], "retry_count": 1}, config) == "rewrite"


def test_decide_falls_back_after_retry_limit():
    config = Settings(min_relevant_docs=1, max_retries=2)

    assert decide({"relevant_documents": [], "retry_count": 2}, config) == "fallback"


def test_decide_node_records_route_and_history():
    config = Settings(min_relevant_docs=1, max_retries=2)

    result = decide_node(
        {
            "relevant_documents": [],
            "grades": [{"doc_index": 0, "relevant": True, "confidence": 0.4, "reason": "弱相关"}],
            "retry_count": 0,
            "history": ["grade: 0/1 relevant docs"],
        },
        config,
    )

    assert result["route"] == "rewrite"
    assert result["history"][-1] == "decide: rewrite (relevant=0, avg_confidence=0.40, retry=0/2)"
    assert route_from_state(result) == "rewrite"


def test_project_queries_expand_retrieval_k():
    config = Settings(top_k=4)

    assert search_k_for_query("张三有哪些项目", config) == 8
    assert search_k_for_query("DocAgent 的核心流程是什么", config) == 4


def test_should_revise_only_once():
    assert should_revise({"self_check": "unsupported。遗漏证据"}) == "revise"
    assert should_revise({"self_check": "unsupported。仍有问题", "revised": True}) == "end"
    assert should_revise({"self_check": "supported。"}) == "end"


def test_extract_json_payload_strips_markdown_fence():
    raw = "```json\n[{\"a\": 1}]\n```"
    assert _extract_json_payload(raw) == '[{"a": 1}]'


def test_extract_json_payload_extracts_array_from_prose():
    raw = 'here is the result: [{"doc_index": 0, "relevant": true}] done.'
    assert _extract_json_payload(raw) == '[{"doc_index": 0, "relevant": true}]'


def test_extract_json_payload_prefers_inner_array():
    # The function finds the outermost [ ] first; for {"grades": []} it extracts []
    raw = '{"grades": []}'
    assert _extract_json_payload(raw) == "[]"


def test_extract_json_payload_extracts_pure_object_when_no_array():
    raw = '{"key": "value"}'
    assert _extract_json_payload(raw) == '{"key": "value"}'


def test_extract_json_payload_returns_cleaned_on_no_match():
    raw = "  no json here  "
    assert _extract_json_payload(raw) == "no json here"


def test_parse_grades_returns_empty_list_on_malformed_json():
    assert _parse_grades("not json at all", doc_count=3) == []


def test_format_documents_includes_source_and_chunk_id():
    docs = [
        Document(page_content="Hello world", metadata={"source": "a.md", "chunk_id": 0}),
        Document(page_content="Second chunk", metadata={"source": "b.md", "chunk_id": 1}),
    ]
    output = format_documents(docs)
    assert "[1] source=a.md chunk=0" in output
    assert "[2] source=b.md chunk=1" in output
    assert "Hello world" in output


def test_format_documents_includes_one_based_pdf_page():
    docs = [Document(page_content="PDF text", metadata={"source": "resume.pdf", "page": 0, "chunk_id": 3})]
    output = format_documents(docs)

    assert "[1] source=resume.pdf page=1 chunk=3" in output


def test_format_documents_keeps_non_numeric_page_metadata():
    docs = [Document(page_content="PDF text", metadata={"source": "resume.pdf", "page": "cover", "chunk_id": 3})]
    output = format_documents(docs)

    assert "page=cover" in output


def test_format_documents_handles_missing_metadata():
    docs = [Document(page_content="bare")]
    output = format_documents(docs)
    assert "source=unknown" in output
    assert "chunk=?" in output


def test_fallback_returns_fixed_answer():
    result = fallback({"question": "test", "history": []})
    assert "资料中未找到" in result["answer"]
    assert result["history"][-1].startswith("fallback:")
