# Eval Report — memocortex_zh_v1

- Timestamp: `2026-06-14T11:08:04.117240`
- Weights: config 默认
- Total entries: 80
- Failed: 0

## Overall

| Metric | Value |
|---|---:|
| recall@1 | 0.9187 |
| recall@3 | 0.9688 |
| recall@5 | 1.0000 |
| recall@10 | 1.0000 |
| ndcg@1 | 0.9500 |
| ndcg@3 | 0.9593 |
| ndcg@5 | 0.9730 |
| ndcg@10 | 0.9730 |
| hit@1 | 0.9500 |
| hit@3 | 0.9750 |
| hit@5 | 1.0000 |
| mrr | 0.9667 |

## Latency

| Stat | ms |
|---|---:|
| p50_ms | 23.65 |
| p95_ms | 32.23 |
| mean_ms | 25.38 |
| max_ms | 36.34 |

## By Scenario

| Scenario | n | recall@5 | ndcg@5 | ndcg@10 | mrr | hit@1 |
|---|---:|---:|---:|---:|---:|---:|
| conflict_latest | 10 | 1.0000 | 0.8931 | 0.8931 | 0.8583 | 0.8000 |
| episodic_temporal | 10 | 1.0000 | 0.9431 | 0.9431 | 0.9250 | 0.9000 |
| exact_recall | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| mixed_types | 5 | 1.0000 | 0.9701 | 0.9701 | 1.0000 | 1.0000 |
| negation | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| paraphrase | 15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| procedural | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| temporal_window | 10 | 1.0000 | 0.9631 | 0.9631 | 0.9500 | 0.9000 |
