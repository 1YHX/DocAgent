# DocAgent Mini Knowledge Base

## 项目定位

DocAgent 是一个带自我反思的文档问答 Agent。它不是普通的“检索一次就回答”的 RAG，而是在生成答案前先评估检索片段是否足够支持问题。

## 核心流程

DocAgent 的核心流程是：Retrieve 检索候选片段，Grade 评估片段相关性，Decide 决定生成、改写查询或兜底，Generate 基于资料回答，Self-check 检查答案是否被资料支持。

## Query Rewriting

当检索片段不足以回答问题，并且重试次数没有超过上限时，DocAgent 会基于原问题、上一轮 query 和评估失败原因改写 query，再进行下一轮检索。

## Fallback Policy

如果达到最大重试次数后仍然没有足够相关的资料，DocAgent 必须回答“资料中未找到相关信息”，而不是根据常识或模型记忆编造答案。

## Retry Limit

DocAgent 默认最多重试 2 次。设置重试上限是为了避免知识库没有答案时陷入无限循环，也能控制外部模型调用成本。

## Baseline RAG

普通 baseline RAG 只执行一次检索和一次生成，不会评估检索片段是否足够，也不会主动改写 query 重新检索。

## 向量数据库选型

DocAgent 当前 MVP 使用 Chroma 作为本地持久化向量库，数据默认保存在项目目录下的 `.chroma/`。它没有使用 Pinecone、Milvus Cloud、Weaviate Cloud 或其他公司的线上向量数据库。

选择 Chroma 的原因是：本地开发零部署、便于 demo 复现、适合个人项目快速验证 Corrective RAG / Self-RAG 的核心流程。后续如果需要团队协作或生产部署，可以把向量库替换为 Milvus、pgvector、Pinecone 或 Weaviate。

## 模型与 API

DocAgent 的聊天模型和 embedding 模型通过 `.env` 配置。聊天模型可使用 DeepSeek API，embedding 可使用 OpenAI-compatible 的 embedding 服务。模型供应商和向量库是两层不同组件：embedding 负责把文本转成向量，Chroma 负责保存和检索这些向量。

## CLI 使用方式

DocAgent 提供 `doctor`、`demo`、`ingest`、`ask`、`compare` 和 `chat` 命令。`chat` 命令会进入交互式界面，用户可以连续提问，也可以用 `/trace on` 查看 Agent 的检索、评估、决策和改写过程。
