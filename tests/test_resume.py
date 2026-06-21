from docagent.resume import extract_project_titles, extract_research_titles


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
