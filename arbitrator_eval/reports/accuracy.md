# Arbitrator Accuracy Report

- Timestamp: `2026-06-14T14:27:52.138848`
- Dataset: `arbitrator_eval/dataset_v1.jsonl` (50 条)
- **Accuracy: 1.0000 (50/50)**
- Latency: avg 2291ms, max 4293ms

## Confusion Matrix

rows = expected, cols = actual (LLM 输出)

| expected \ actual | replace | merge | versioned | ignore |
|---|---:|---:|---:|---:|
| **replace** | 12 | 0 | 0 | 0 |
| **merge** | 0 | 12 | 0 | 0 |
| **versioned** | 0 | 0 | 12 | 0 |
| **ignore** | 0 | 0 | 0 | 14 |

## Per Expected-Action Accuracy

| Expected Action | n | correct | accuracy |
|---|---:|---:|---:|
| replace | 12 | 12 | 1.0000 |
| merge | 12 | 12 | 1.0000 |
| versioned | 12 | 12 | 1.0000 |
| ignore | 14 | 14 | 1.0000 |