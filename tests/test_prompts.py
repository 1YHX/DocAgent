from langchain_core.documents import Document

from docagent import nodes


class FakeChatModel:
    def __call__(self, _messages):
        class Response:
            content = '[{"doc_index": 0, "relevant": true, "reason": "命中"}]'

        return Response()


def test_grade_prompt_escapes_json_example(monkeypatch):
    monkeypatch.setattr(nodes, "build_chat_model", lambda: FakeChatModel())

    result = nodes.grade_documents(
        {
            "question": "DocAgent 的核心流程是什么？",
            "documents": [Document(page_content="DocAgent 的核心流程是 Retrieve、Grade、Decide。")],
        }
    )

    assert result["grades"] == [{"doc_index": 0, "relevant": True, "reason": "命中"}]
