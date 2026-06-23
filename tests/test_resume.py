from docagent.resume import (
    _between,
    _dedupe,
    _is_pdf_header,
    extract_project_titles,
    extract_research_titles,
)


RESUME_TEXT = """
科研经历 LLM 辅助的视觉枢轴多模态机器翻译研究 ｜ 机器翻译 / 多模态 / LLM / 论文投稿中
• 研究内容。
项目经历 SightMate Web 端 AI 视觉对话助手 ｜ Qwen-Omni-Realtime / WebSocket / FastAPI
• 项目内容。
VoiceCalendar 语音交互式智能日历 ｜ DeepSeek / ASR / TTS
• 项目内容。
Novel2Script AI 小说转剧本工作台 ｜ LLM Workflow / FastAPI
• 项目内容。
计算机视觉模型训练补充项目：蓝莓成熟度检测 ｜ Python / YOLOv8n / SimCLR
• 项目内容。
竞赛与证书 蓝桥杯。
"""


def test_extract_project_titles_from_resume_sections():
    assert extract_project_titles(RESUME_TEXT) == [
        "SightMate Web 端 AI 视觉对话助手",
        "VoiceCalendar 语音交互式智能日历",
        "Novel2Script AI 小说转剧本工作台",
        "计算机视觉模型训练补充项目：蓝莓成熟度检测",
    ]


def test_extract_research_titles_separately():
    assert extract_research_titles(RESUME_TEXT) == ["LLM 辅助的视觉枢轴多模态机器翻译研究"]


def test_extract_project_titles_returns_empty_when_no_section():
    assert extract_project_titles("没有项目经历栏目的文本") == []


def test_extract_research_titles_returns_empty_when_no_section():
    assert extract_research_titles("没有科研经历栏目的文本") == []


def test_between_extracts_text_between_markers():
    text = "前言 开始内容 结束 后续"
    assert _between(text, "开始", "结束") == "开始内容 "


def test_between_returns_empty_when_start_not_found():
    assert _between("hello world", "missing", "world") == ""


def test_between_returns_to_end_when_end_not_found():
    result = _between("A 开始 内容 到最后", "开始", "不存在")
    assert result == "开始 内容 到最后"


def test_dedupe_preserves_order_and_removes_duplicates():
    items = ["B", "A", "B", "C", "A"]
    assert _dedupe(items) == ["B", "A", "C"]


def test_dedupe_normalizes_whitespace():
    items = ["hello  world", "hello world"]
    assert _dedupe(items) == ["hello world"]


def test_is_pdf_header_filters_name_and_resume():
    assert _is_pdf_header("个人简历") is True
    assert _is_pdf_header("张三") is True
    assert _is_pdf_header("SightMate Web 端 AI 视觉对话助手") is False
