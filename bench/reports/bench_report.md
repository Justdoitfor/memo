# MemoCortex Benchmark Report

- Timestamp: `2026-06-14T15:23:20.972680`
- Platform: CPU-only (no GPU), bge-small-zh-v1.5 + bge-reranker-v2-m3

## 1. Recall Latency vs Data Scale (一阶段)

| Scale | n_queries | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 50 | 81.00 | 80.33 | 85.53 | 90.00 | 95.00 |
| 1000 | 50 | 115.00 | 113.77 | 122.89 | 130.00 | 135.00 |
| 10000 | 50 | 155.00 | 152.55 | 174.28 | 180.00 | 195.00 |

## 2. Storage Upsert: SQLite vs PostgreSQL

| Backend | n | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| sqlite | 100 | 12.53 | 12.51 | 13.64 | 16.30 |
| postgres | 100 | 6.91 | 6.79 | 7.56 | 13.46 |

## 3. Reranker Overhead

| Config | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| No reranker | 160.00 | 155.00 | 166.82 | 175.00 |
| + Reranker | 1900.00 | 1850.00 | 2128.34 | 2300.00 |

**P50 slowdown**: 11.94x  | **P95 overhead**: +1961.52ms

## 图表

`bench/reports/latency_curves.png`