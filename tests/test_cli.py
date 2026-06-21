from docagent import cli


def test_run_compare_calls_baseline_and_agent(monkeypatch, capsys):
    calls = []

    def fake_ask(question, baseline=False):
        calls.append((question, baseline))
        if baseline:
            return {"answer": "baseline answer", "history": ["retrieve once"]}
        return {"answer": "agent answer", "history": ["retrieve", "grade"]}

    def fake_print_result(result, show_trace=False):
        print(result["answer"])
        if show_trace:
            print("trace shown")

    monkeypatch.setattr("docagent.main.ask", fake_ask)
    monkeypatch.setattr("docagent.main.print_result", fake_print_result)

    baseline_result, agent_result = cli.run_compare("问题", show_trace=True)

    assert calls == [("问题", True), ("问题", False)]
    assert baseline_result["answer"] == "baseline answer"
    assert agent_result["answer"] == "agent answer"

    output = capsys.readouterr().out
    assert "Baseline RAG" in output
    assert "DocAgent" in output
    assert "Comparison" in output


def test_parse_toggle():
    assert cli._parse_toggle("/trace on", False, "/trace") is True
    assert cli._parse_toggle("/trace off", True, "/trace") is False
    assert cli._parse_toggle("/trace", False, "/trace") is True
    assert cli._parse_toggle("/trace maybe", True, "/trace") is True


def test_build_contextual_query_without_history_returns_question():
    assert cli.build_contextual_query("不是三个项目嘛？", None, None) == "不是三个项目嘛？"


def test_build_contextual_query_includes_previous_turn():
    query = cli.build_contextual_query(
        "不是三个项目嘛？",
        "易海祥手里有什么项目，都介绍一下我看看",
        "根据资料，易海祥手中有以下两个项目：日程助手和 Novel2Script。",
    )

    assert "易海祥手里有什么项目，都介绍一下我看看；不是三个项目嘛？" in query
    assert "核实项目列表、项目数量、是否存在补充项目" in query
