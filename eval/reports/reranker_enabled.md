# Eval Report — memocortex_zh_v1_reranker

- Timestamp: `2026-06-14T12:46:36.094463`
- Weights: config 默认
- Total entries: 80
- Failed: 0

## Overall

| Metric | Value |
|---|---:|
| recall@1 | 0.9688 |
| recall@3 | 1.0000 |
| recall@5 | 1.0000 |
| recall@10 | 1.0000 |
| ndcg@1 | 1.0000 |
| ndcg@3 | 1.0000 |
| ndcg@5 | 1.0000 |
| ndcg@10 | 1.0000 |
| hit@1 | 1.0000 |
| hit@3 | 1.0000 |
| hit@5 | 1.0000 |
| mrr | 1.0000 |

## Latency

| Stat | ms |
|---|---:|
| p50_ms | 313.90 |
| p95_ms | 436.30 |
| mean_ms | 2224.16 |
| max_ms | 151807.94 |

## By Scenario

| Scenario | n | recall@5 | ndcg@5 | ndcg@10 | mrr | hit@1 |
|---|---:|---:|---:|---:|---:|---:|
| conflict_latest | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| episodic_temporal | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_recall | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| mixed_types | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| negation | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| paraphrase | 15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| procedural | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| temporal_window | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
