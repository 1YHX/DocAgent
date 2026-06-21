# DocAgent

DocAgent 是一个带自我反思的文档问答 Agent。它不是检索一次就生成的普通 RAG，而是在回答前先评估检索片段是否足够；不够就改写 query 重新检索，仍然不足时明确说「资料中未找到相关信息」。

## 核心流程

```text
用户提问
  -> Retrieve 检索候选片段
  -> Grade 评估片段是否能回答问题
  -> Decide 条件分叉
      -> 够用：Generate 基于资料生成答案并引用来源
      -> 不够且未达上限：Rewrite query 后回到 Retrieve
      -> 达到上限：Fallback 拒绝编造
  -> Self-check 检查答案是否被资料支持
```

这个流程对应 Corrective RAG / Self-RAG 的工程思路：把 Agent 决策循环套在 RAG 外面，用状态机处理「答、重查、认不知道」三种动作。

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

填好 `.env` 后，把 `.md`、`.txt` 或 `.pdf` 放进 `data/`，然后执行：

```bash
python -m docagent.main ingest
python -m docagent.main ask "你的问题" --show-trace
```

运行普通 RAG 对照基线：

```bash
python -m docagent.main ask "你的问题" --baseline
```

## 设计取舍

- 编排使用 LangGraph：DocAgent 需要条件分叉和循环，`retrieve -> grade -> decide -> rewrite -> retrieve` 用状态图比线性 chain 更清楚。
- 评估使用 LLM：相似度高不等于能回答问题，所以 `grade` 节点让模型判断片段是否真正支持问题，代价是多一次模型调用。
- 重试有上限：默认最多改写检索 2 次，避免查不到时无限循环和浪费 token。
- 兜底优先：知识库没有答案时明确拒绝编造，这是项目区别于普通 RAG 的核心价值。

## 当前进度

- [x] 项目骨架、配置与 CLI
- [x] 文档加载、切分、Chroma 入库
- [x] 普通 RAG baseline
- [x] LangGraph 版 Retrieve / Grade / Decide / Rewrite / Generate / Fallback / Self-check
- [ ] 对照 demo 与 README 实验结果
- [ ] 更完整的异常处理和单元测试

## 已知局限

- 当前默认使用 OpenAI-compatible chat 与 embedding 接口；如果使用通义 embedding，需要补充对应 provider 适配。
- `grade` 依赖 LLM 输出 JSON，已做基础解析，但后续可换成结构化输出接口增强稳定性。
- 引用粒度目前是文件与 chunk id，尚未精确到 PDF 页码或 Markdown 标题层级。
