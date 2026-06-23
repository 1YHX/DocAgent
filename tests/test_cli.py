import sys
from docagent import cli
from docagent.config import Settings


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


def test_normalize_chat_command_accepts_bare_cli_commands():
    assert cli._normalize_chat_command("status") == "/status"
    assert cli._normalize_chat_command("docagent status") == "/status"
    assert cli._normalize_chat_command("sources") == "/sources"
    assert cli._normalize_chat_command("docagent sources") == "/sources"
    assert cli._normalize_chat_command("doctor") == "/doctor"
    assert cli._normalize_chat_command("docagent doctor") == "/doctor"
    assert cli._normalize_chat_command("ingest") == "/ingest"
    assert cli._normalize_chat_command("ingest --reset") == "/ingest"
    assert cli._normalize_chat_command("reindex") == "/ingest"
    assert cli._normalize_chat_command("docagent ingest --reset") == "/ingest"
    assert cli._normalize_chat_command("docagent reindex") == "/ingest"
    assert cli._normalize_chat_command("张三是谁") == "张三是谁"


def test_run_chat_handles_ingest_alias(monkeypatch, capsys):
    calls = []
    inputs = iter(["docagent ingest --reset", "/exit"])

    monkeypatch.setattr(cli, "_read_chat_input", lambda: next(inputs))
    monkeypatch.setattr(cli, "ingest", lambda reset=False: calls.append(reset) or 3)

    cli.run_chat()

    output = capsys.readouterr().out
    assert calls == [True]
    assert "Ingested 3 chunks. Chat context cleared." in output


def test_run_chat_handles_doctor_alias(monkeypatch, capsys):
    calls = []
    inputs = iter(["docagent doctor", "/exit"])

    monkeypatch.setattr(cli, "_read_chat_input", lambda: next(inputs))
    monkeypatch.setattr(cli, "run_doctor", lambda: calls.append("doctor") or 0)

    cli.run_chat()

    assert calls == ["doctor"]
    assert "DocAgent chat" in capsys.readouterr().out


def test_build_contextual_query_without_history_returns_question():
    assert cli.build_contextual_query("不是三个项目嘛？", []) == "不是三个项目嘛？"


def test_build_contextual_query_includes_previous_turn():
    history = [
        {
            "question": "张三手里有什么项目，都介绍一下我看看",
            "answer": "根据资料，张三手中有以下两个项目：日程助手和 Novel2Script。",
        }
    ]
    query = cli.build_contextual_query("不是三个项目嘛？", history)

    assert "张三手里有什么项目，都介绍一下我看看；不是三个项目嘛？" in query
    assert "核实项目列表、项目数量、是否存在补充项目" in query


def test_build_contextual_query_plain_follow_up_no_count_keywords():
    history = [{"question": "张三有哪些项目", "answer": ""}]
    query = cli.build_contextual_query("再介绍一下第一个", history)
    assert query == "张三有哪些项目；再介绍一下第一个"
    assert "核实" not in query


def test_build_contextual_query_uses_only_last_two_turns():
    history = [
        {"question": "第一问", "answer": "答1"},
        {"question": "第二问", "answer": "答2"},
        {"question": "第三问", "answer": "答3"},
    ]
    query = cli.build_contextual_query("第四问", history)
    assert "第一问" not in query
    assert "第二问；第三问；第四问" == query


def test_run_doctor_ok_when_all_set(capsys):
    config = Settings(
        chat_api_key="k1",
        chat_base_url="https://api.deepseek.com",
        chat_model="deepseek-chat",
        embedding_api_key="k2",
        embedding_base_url=None,
        embedding_model="text-embedding-3-small",
    )
    exit_code = cli.run_doctor(config)
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "MISSING" not in output


def test_run_doctor_fails_when_chat_key_missing(capsys):
    config = Settings(
        chat_api_key=None,
        chat_model="deepseek-chat",
        embedding_api_key="k2",
        embedding_model="text-embedding-3-small",
    )
    exit_code = cli.run_doctor(config)
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "MISSING" in out
    assert "CHAT_API_KEY" in out


def test_run_doctor_does_not_fail_when_embedding_base_url_missing(capsys):
    config = Settings(
        chat_api_key="k1",
        chat_model="deepseek-chat",
        embedding_api_key="k2",
        embedding_base_url=None,
        embedding_model="text-embedding-3-small",
    )
    exit_code = cli.run_doctor(config)
    assert exit_code == 0


def test_list_data_files_only_supported_non_hidden(tmp_path):
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF")
    (tmp_path / ".hidden.md").write_text("hidden")
    (tmp_path / "ignore.docx").write_text("ignore")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    config = Settings(source_dir=tmp_path)

    assert cli.list_data_files(config) == [
        str(tmp_path / "a.md"),
        str(tmp_path / "b.pdf"),
        str(sub / "c.txt"),
    ]


def test_run_status_reports_missing_and_stale(monkeypatch, tmp_path, capsys):
    (tmp_path / "a.md").write_text("a")
    config = Settings(source_dir=tmp_path, persist_dir=tmp_path / ".chroma")

    monkeypatch.setattr(
        cli,
        "get_index_summary",
        lambda _config: {
            "chunk_count": 2,
            "sources": ["old.md"],
            "source_counts": {"old.md": 2},
        },
    )

    result = cli.run_status(config)
    output = capsys.readouterr().out

    assert result["missing"] == [str(tmp_path / "a.md")]
    assert result["stale"] == ["old.md"]
    assert "Not indexed yet" in output
    assert "Indexed but missing" in output


def test_run_sources_prints_empty(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "get_index_summary",
        lambda _config: {"chunk_count": 0, "sources": [], "source_counts": {}},
    )

    result = cli.run_sources(Settings())
    output = capsys.readouterr().out

    assert result["chunk_count"] == 0
    assert "<empty>" in output
