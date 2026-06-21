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
