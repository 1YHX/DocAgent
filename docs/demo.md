# Demo Guide

这个 demo 使用 `examples/mini_knowledge_base.md` 作为小型知识库，用来展示普通 RAG 和 DocAgent 的差异。

## 准备语料

最简单方式：

```bash
python -m docagent.main demo
```

手动方式：

```bash
mkdir -p data
cp examples/mini_knowledge_base.md data/docagent_demo.md
python -m docagent.main ingest --reset
```

## 建议问题

进入交互式模式：

```bash
python -m docagent.main chat --show-trace
```

有明确答案的问题：

```bash
python -m docagent.main ask "DocAgent 的核心流程是什么？" --show-trace
```

需要体现拒绝编造的问题：

```bash
python -m docagent.main ask "DocAgent 使用了哪家公司的线上向量数据库？" --show-trace
```

对照普通 RAG：

```bash
python -m docagent.main compare "DocAgent 使用了哪家公司的线上向量数据库？"
```

## 预期观察点

| 问题类型 | Baseline RAG | DocAgent |
| --- | --- | --- |
| 文档中有答案 | 直接基于召回片段回答 | 先评估片段，再基于相关片段回答 |
| 文档中无答案 | 可能受 prompt 约束说不知道，但没有显式检索质量判断 | 会经过 grade、rewrite 和 retry 上限，最后 fallback |
| 问法和文档措辞不一致 | 只检索一次，召回失败就结束 | 可改写 query 再查一轮 |

实际输出会受模型和 embedding 服务影响。面试展示时重点看 `--show-trace` 输出里的 retrieve / rewrite 轨迹，以及最终是否拒绝资料外推测。
