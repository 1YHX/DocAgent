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
