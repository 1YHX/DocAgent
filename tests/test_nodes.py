from docagent.config import Settings
from docagent.nodes import _parse_grades, decide


def test_parse_grades_filters_invalid_items():
    raw = '[{"doc_index": 0, "relevant": true, "reason": "命中"}, {"doc_index": 9, "relevant": true}]'

    assert _parse_grades(raw, doc_count=1) == [{"doc_index": 0, "relevant": True, "reason": "命中"}]


def test_decide_generates_when_relevant_docs_are_enough():
    config = Settings(min_relevant_docs=1, max_retries=2)

    assert decide({"relevant_documents": [object()], "retry_count": 0}, config) == "generate"


def test_decide_rewrites_before_retry_limit():
    config = Settings(min_relevant_docs=1, max_retries=2)

    assert decide({"relevant_documents": [], "retry_count": 1}, config) == "rewrite"


def test_decide_falls_back_after_retry_limit():
    config = Settings(min_relevant_docs=1, max_retries=2)

    assert decide({"relevant_documents": [], "retry_count": 2}, config) == "fallback"
