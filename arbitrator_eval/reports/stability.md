# Arbitrator Stability Report

- Timestamp: `2026-06-14T14:37:41.637900`
- Setup: 50 条 × 5 次 = 250 次 LLM 调用

## 关键指标

- **Average mode rate (每条 case 多数派 action 占比的均值)**: `0.9840`
- **Perfect consistency (多次跑 action 完全一致的 case 比例)**: `94.00%`
- **Majority action matches expected (多数派 action 与 ground truth 吻合)**: `98.00%`

## Unstable Cases

共 3 条 (mode_rate < 1.0):

| ID | Category | Expected | Runs | Mode | Mode Rate |
|---|---|---|---|---|---:|
| arb-006 | replace_marital | replace | replace / replace / replace / replace / versioned | replace | 0.8000 |
| arb-015 | merge_likes_overlap | merge | ignore / ignore / merge / ignore / ignore | ignore | 0.8000 |
| arb-023 | merge_overlap_three | merge | ignore / merge / merge / merge / ignore | merge | 0.6000 |