# DocAgent

DocAgent 是一个带自我反思的文档问答 Agent。它不是检索一次就生成的普通 RAG，而是在回答前先评估检索片段是否足够；不够就改写 query 重新检索，仍然不足时明确说「资料中未找到相关信息」。

## 核心流程

```
用户提问
  │
  ▼
[1] Retrieve  — 向量检索 top-k 候选片段
  │
  ▼
[2] Grade     — LLM 评估每个片段与问题的相关性（输出 JSON）
  │
  ▼
[3] Decide    — 条件分叉
      ├─ 相关片段够 ──────────────► [4] Generate 生成答案
      ├─ 不足且未超重试上限 ─────► [5] Rewrite query → 回到 [1]
      └─ 超上限仍不足 ──────────► [6] Fallback 拒绝编造
  │
  ▼
[4] Generate  — 基于通过评估的片段生成答案，附来源引用
  │
  ▼
[7] Self-check — 校验答案是否完全由检索内容支持
      ├─ supported  → 直接返回
      └─ unsupported（仅一次）→ [8] Revise 修正后再 self-check
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# 填好 .env 里的 API key
```

把 `.md`、`.txt` 或 `.pdf` 放进 `data/`，然后执行：

```bash
docagent doctor          # 检查配置
docagent status          # 查看已索引文件
docagent ingest --reset  # 向量化入库
docagent ask "你的问题" --show-trace
```

第一次试跑可以使用仓库里的迷你知识库：

```bash
docagent demo
```

进入交互式聊天（支持流式输出）：

```bash
docagent
```

在 `docagent>` 里可以直接执行常用维护命令：

```text
docagent status          # 查看 data/ 和向量库是否一致
docagent sources         # 查看当前已入库来源
docagent ingest --reset  # 重新读取 data/ 并重建向量库
docagent doctor          # 检查 .env 和本地配置
```

对比普通 RAG 和 DocAgent：

```bash
docagent compare "DocAgent 使用了哪家公司的线上向量数据库？"
```

## 对照实验

知识库：`data/docagent_demo.md`（DocAgent 介绍）+ `data/易海祥-简历.pdf`，共 5 chunks。

| # | 问题 | Baseline RAG | DocAgent | 关键差异 |
|---|------|-------------|---------|---------|
| 1 | DocAgent 的核心流程是什么？ | ✅ 正确（直接检索命中） | ✅ 正确，Grade 过滤掉 3 个简历无关片段，仅用 1 个相关片段生成 | Baseline 把简历 chunk 也喂进 prompt；DocAgent 精准过滤 |
| 2 | 这个系统是怎么防止 AI 胡说八道的？ | ✅ 正确（问法换了但向量还是命中） | ✅ 正确，Grade 识别出 `chunk[2]` 相关（描述兜底策略），过滤掉其余 3 个简历片段 | 两者均命中，DocAgent 多了 self-check 验证 |
| 3 | DocAgent 支持多少种语言？（知识库内无答案） | ⚠️ **回答"未提及"但措辞模糊**，没有明确拒绝 | ✅ **明确返回"资料中未找到相关信息"**，经历了 2 次 query rewrite 后 fallback | 核心差异：DocAgent 重查 2 轮（改写为"语言种类"→"编程语言或自然语言"）后确认无答案，主动拒绝编造 |
| 4 | 知识库里提到了哪些 Agent 节点？ | ✅ 正确列出 5 个节点 | ✅ 正确列出 5 个节点，self-check: supported | 两者均正确；DocAgent 多了 self-check 验证引用来源 |

**运行对照实验**（需先 `docagent ingest`）：

```bash
python scripts/benchmark.py           # 文本输出
python scripts/benchmark.py --md      # Markdown 表格
```

**结论**：DocAgent 相比 Baseline 的核心优势在第 3 题体现得最明显——对知识库外的问题，Baseline 给了模糊的"未提及"（用户可能误以为是答案），DocAgent 经过两轮 query rewrite 确认无法找到后，**明确拒绝回答**，不会引发幻觉。

## 设计取舍

### 为什么用 LangGraph 而不是线性 chain？

DocAgent 有条件分叉（生成 / 重查 / 兜底）和循环（rewrite → retrieve 最多 2 次），线性 chain 表达不了这种状态机，LangGraph 的 `StateGraph` 让图结构一目了然。

### Grade 为什么用 LLM 打分而不是相似度阈值？

向量相似度高 ≠ 能回答问题。例如「DocAgent 是什么语言写的？」的 embedding 与「DocAgent 使用 Python」很相近，但如果知识库里只有「DocAgent 使用 LangGraph」，相似度检索无法区分。LLM Grade 直接判断「这段话能回答这个问题吗」，召回精度更高；代价是多一次 API 调用。

Grade 节点使用 `response_format={"type": "json_object"}` 强制模型输出结构化 JSON，消除 Markdown 包裹导致的解析失败。

### 重试上限为什么是 2 次？

防止死循环消耗 token。实测第一次 rewrite 后命中率已经很高，第三次往往是信息本来就不在知识库里。上限可通过 `DOCAGENT_MAX_RETRIES` 调整。

### 流式输出

Chat 模式使用 `chain.stream()` 逐 token 打印，避免 LLM 响应延迟带来的空白等待。如果 self-check 判定答案需要修正，revise 节点同样流式输出，并显示 `[答案已修正]` 提示。

### 兜底优先于编造

检索 + 重查仍不足时，直接返回固定拒绝回答，不把劣质片段喂进生成器。这是项目与普通 RAG 最核心的区别。

## 已知局限

- `resume.py` 的项目计数逻辑与简历格式强耦合，换格式需要重写。
- 引用粒度目前是文件路径 + PDF 页码 + chunk 编号，尚未做到原文高亮定位。
- 多文档大知识库下 top-k 扩大策略（当前硬编码关键词判断）可改为动态计算。
- Chat 上下文拼接仅保留上一轮，多轮深度追问时上下文可能不足。

## 当前进度

- [x] 项目骨架、配置与 CLI
- [x] 文档加载（PDF/Markdown/TXT）、切分、Chroma 入库
- [x] 普通 RAG baseline
- [x] LangGraph：Retrieve / Grade / Decide / Rewrite / Generate / Fallback / Self-check / Revise
- [x] Chat 流式输出
- [x] Grade 节点 JSON 模式（消除解析失败）
- [x] LLM 实例缓存 + 超时重试
- [x] PDF 来源引用显示页码
- [x] 对照 demo 语料与演示说明
- [x] Benchmark 脚本（`scripts/benchmark.py`）
- [ ] README 实验截图/文本（跑完 `benchmark.py` 后填入）
- [ ] 来源引用原文高亮定位
