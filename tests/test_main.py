from docagent import main


def test_ask_uses_contextual_query_for_initial_state(monkeypatch):
    captured = {}

    class FakeGraph:
        def invoke(self, state):
            captured.update(state)
            return state

    monkeypatch.setattr(main, "build_graph", lambda **_kw: FakeGraph())
    monkeypatch.setattr(main, "answer_resume_project_question", lambda *_args, **_kwargs: None)

    result = main.ask("不是三个项目嘛？", query="张三项目列表；不是三个项目嘛？")

    assert result["question"] == "不是三个项目嘛？"
    assert result["standalone_question"] == "张三项目列表；不是三个项目嘛？"
    assert result["query"] == "张三项目列表；不是三个项目嘛？"
    assert result["history"] == ["contextual_query: 张三项目列表；不是三个项目嘛？"]
