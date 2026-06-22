from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from docagent import nodes


def _make_fake_grader(content: str):
    """Return a LangChain-compatible fake that ignores input and returns fixed content."""
    class _Response:
        pass

    def respond(_input, **_kwargs):
        r = _Response()
        r.content = content
        return r

    model = RunnableLambda(respond)
    # bind() must return a Runnable; RunnableLambda.bind() is supported natively
    return model


def test_grade_returns_structured_grades(monkeypatch):
    grader_content = '{"grades": [{"doc_index": 0, "relevant": true, "reason": "命中"}]}'
    monkeypatch.setattr(
        nodes,
        "build_chat_model",
        lambda *_a, **_kw: _make_fake_grader(grader_content),
    )

    result = nodes.grade_documents(
        {
            "question": "DocAgent 的核心流程是什么？",
            "documents": [Document(page_content="DocAgent 的核心流程是 Retrieve、Grade、Decide。")],
        }
    )

    assert result["grades"] == [{"doc_index": 0, "relevant": True, "reason": "命中"}]
    assert len(result["relevant_documents"]) == 1


def test_grade_falls_back_when_json_is_array(monkeypatch):
    """_parse_grades handles bare arrays too (e.g. from older provider output)."""
    grader_content = '[{"doc_index": 0, "relevant": false, "reason": "无关"}]'
    monkeypatch.setattr(
        nodes,
        "build_chat_model",
        lambda *_a, **_kw: _make_fake_grader(grader_content),
    )

    result = nodes.grade_documents(
        {
            "question": "随便一个问题",
            "documents": [Document(page_content="不相关的内容")],
        }
    )

    assert result["grades"] == [{"doc_index": 0, "relevant": False, "reason": "无关"}]
    assert result["relevant_documents"] == []
