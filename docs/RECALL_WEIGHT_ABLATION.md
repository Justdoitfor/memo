# 4 信号召回融合 — 权重调优全报告

> **数据**: 80 条中文长期记忆评测集 (`eval/datasets/memocortex_zh_v1.jsonl`)
> **方法**: 单轴消融 (Phase A) + 粗筛 grid (Phase B) + 邻域精扫 (Phase C)
> **主排序指标**: nDCG@10

## 摘要 (TL;DR)

| 项 | 值 |
|---|---|
| **Baseline 权重 `(0.40, 0.20, 0.20, 0.20)`** | nDCG@10 = 0.9730, MRR = 0.9667, P50 = 24ms |
| **单轴最优组合** | `(vec=0.6, temp=0.2, kw=0.2, imp=0.2)` → nDCG@10 = 0.9919 (+0.019) |
| **关停 importance 信号** | `(0.4, 0.2, 0.2, 0.0)` → nDCG@10 = **1.0000** (+0.027) |
| **Grid search 最优 (待精扫确认)** | 见 `grid_search_results.md` |

---

## Phase A: 单轴消融 — 4 个信号的边际贡献

### 实验方法

固定其他三维 = baseline 0.2, 单独 sweep 第 i 维 ∈ {0.0, 0.2, 0.4, 0.6, 0.8}, 共 4 × 5 = 20 组.
每组在完整 80 条评测集上跑, 记录 nDCG@10 / MRR / hit@1 / P95 latency.

### 关键发现

#### ✅ Vector signal — 最重要的信号

| `vec` | (vec, 0.2, 0.2, 0.2) | nDCG@10 | hit@1 |
|---:|---|---:|---:|
| 0.0 | (0.0, 0.2, 0.2, 0.2) | 0.9418 | 0.900 |
| 0.2 | (0.2, 0.2, 0.2, 0.2) | 0.9659 | 0.938 |
| 0.4 | (0.4, 0.2, 0.2, 0.2) ← baseline | 0.9730 | 0.950 |
| **0.6** | **(0.6, 0.2, 0.2, 0.2)** | **0.9919** | **0.988** |
| 0.8 | (0.8, 0.2, 0.2, 0.2) | 0.9812 | 0.975 |

- **关掉 vec (=0.0) 损失 -3.1% nDCG@10** — 是 4 个信号里影响最大的
- **vec=0.6 比 baseline 0.4 高 +0.019 nDCG@10** — baseline 给得偏低
- vec=0.8 反而下降 — 过度依赖向量会让 BM25 命中的关键词被压住

#### ⚠️ Temporal decay signal — 边际贡献为负

| `temp` | (0.4, temp, 0.2, 0.2) | nDCG@10 | hit@1 |
|---:|---|---:|---:|
| **0.0** | **(0.4, 0.0, 0.2, 0.2)** | **0.9862** | **0.963** |
| 0.2 | (0.4, 0.2, 0.2, 0.2) ← baseline | 0.9730 | 0.950 |
| 0.4 | (0.4, 0.4, 0.2, 0.2) | 0.9659 | 0.938 |
| 0.6 | (0.4, 0.6, 0.2, 0.2) | 0.9552 | 0.913 |
| 0.8 | (0.4, 0.8, 0.2, 0.2) | 0.9382 | 0.888 |

- **单调递减**: temp 权重越大, 召回质量越差
- **关掉 temp (=0.0) 提升 +1.4% nDCG@10**
- **解读**: 时间衰减信号在评测集里反而干扰主信号 — 这暴露了一个真问题:
  - 旧的 SEMANTIC 事实 (e.g. "我对花生过敏" 写于 30 天前) 被时间衰减压低
  - 但用户问"花生过敏"时, 这条事实仍然完全有效
  - **temporal_decay 的语义"越新越相关"对 stable facts 不成立**, 它更适合 episodic memory

#### ⚠️ Importance signal — 同样负贡献

| `imp` | (0.4, 0.2, 0.2, imp) | nDCG@10 | hit@1 |
|---:|---|---:|---:|
| **0.0** | **(0.4, 0.2, 0.2, 0.0)** | **1.0000** | **1.000** |
| 0.2 | (0.4, 0.2, 0.2, 0.2) ← baseline | 0.9730 | 0.950 |
| 0.4 | (0.4, 0.2, 0.2, 0.4) | 0.9722 | 0.950 |
| 0.6 | (0.4, 0.2, 0.2, 0.6) | 0.9705 | 0.950 |
| 0.8 | (0.4, 0.2, 0.2, 0.8) | 0.9705 | 0.950 |

- **关掉 imp (=0.0) 直接 nDCG@10 = 1.0** — 完美召回
- **解读**: 当前 `compute_importance` 用的是 `effective_strength` (Ebbinghaus + 复习 + source weight + staleness)
  - 评测集里所有记忆都是新写入的 → effective_strength 主要由 confidence_score (0.7-0.8) 决定
  - 不同 candidates 的 importance 差异很小, **加进 final_score 主要起到引入随机性的作用**, 反而让 vector_sim 信号被稀释

#### ✓ Keyword (BM25) signal — 中性

| `kw` | (0.4, 0.2, kw, 0.2) | nDCG@10 |
|---:|---|---:|
| 0.0 | (0.4, 0.2, 0.0, 0.2) | 0.9730 |
| 0.2 | (0.4, 0.2, 0.2, 0.2) ← baseline | 0.9730 |
| 0.4 | (0.4, 0.2, 0.4, 0.2) | 0.9693 |
| 0.6 | (0.4, 0.2, 0.6, 0.2) | 0.9728 |
| 0.8 | (0.4, 0.2, 0.8, 0.2) | 0.9728 |

- **几乎无影响** — 中文 query 在 bge-small-zh 向量层已经覆盖了大部分关键词匹配
- BM25 的边际价值会在 **长 query / 罕见专有名词** 场景出现, 当前评测集没设计这类 case
- 后续 v2 评测集可以补 "包含产品名/型号/术语" 的 case 测 BM25

### 信号重要性排序

```
vec:   ████████████████████   ↑ 主信号, 0.6 最优
imp:   ████████████           ↓ 关掉后召回完美 (评测集场景下)
temp:  ███████                ↓ 单调递减, 不适合 SEMANTIC stable facts
kw:    ▓                       ⊙ 评测集场景下中性
```

### 这告诉我们的事情

1. **README 当前权重 `(0.4, 0.2, 0.2, 0.2)` 偏保守** — 给非主信号 (temp + imp) 加了过多权重
2. **温度衰减不应统一应用所有 memory_type** — 对 EPISODIC 合理, 对 SEMANTIC 是干扰
3. **Importance 信号需要重新设计** —
   当前 `effective_strength` 在新写入数据上方差小, 召回时区分度不够; 应该考虑:
   - 仅在 stale (staleness=True) 时启用 importance 罚分, 平时只看 vector
   - 或换成 "用户主动 confirm 加分" 的二元 boost, 而非连续值

---

## Phase B & C: Grid Search

完整 grid search 报告见: [grid_search_results.md](../eval/reports/grid_search_results.md)
Pareto 散点图: [grid_search_pareto.png](../eval/reports/grid_search_pareto.png)

### 关键发现

**裁剪 grid 设计** (基于 Phase A 消融结论):
- vec ∈ {0.4, 0.5, 0.6, 0.7} (4 值)
- temp ∈ {0.0, 0.1, 0.2} (3 值)
- kw ∈ {0.0, 0.2} (2 值)
- imp ∈ {0.0, 0.2} (2 值)

总搜索空间: 4 × 3 × 2 × 2 = **48 组** (剪枝后), 跑约 26 分钟.

**Top 7 组合 (并列 nDCG@10 = 1.0000)**:

| Rank | weights | nDCG@10 | MRR | hit@1 | conflict_latest |
|---:|---|---:|---:|---:|---:|
| 1 | (0.4, 0.1, 0.0, 0.0) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | (0.4, 0.1, 0.2, 0.0) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | (0.4, 0.2, 0.2, 0.0) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | (0.5, 0.2, 0.2, 0.0) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 | (0.6, 0.2, 0.2, 0.0) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 6 | (0.7, 0.2, 0.0, 0.0) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 7 | (0.7, 0.2, 0.2, 0.0) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

- **所有 Top 7 组合 imp=0.0** — 印证消融分析: importance 信号在评测集上是负贡献
- **conflict_latest 场景从 0.8931 → 1.0000** — 这是最关键的提升 (问题最严重场景的解)
- **Baseline `(0.4, 0.2, 0.2, 0.2)` 排名 40 / 48** — 当前 README 默认权重在 48 组中只比 8 组好

### 最优组合 vs Baseline 提升幅度

- nDCG@10 Δ = +0.0270 (+2.77%)
- MRR Δ = +0.0333

---

## 决策建议

基于消融分析, 给生产环境的权重选择提 **3 个候选方案**:

| 方案 | 权重 (vec, temp, kw, imp) | nDCG@10 (评测集) | 取舍 |
|---|---|---|---|
| **A. 评测集最优** | `(0.4, 0.2, 0.2, 0.0)` | 1.0000 | 关掉 importance, 但失去 staleness 软废弃的召回压制能力 |
| **B. Conservative 优化** | `(0.6, 0.2, 0.2, 0.2)` | 0.9919 | 提高 vec 权重, 保留所有信号 (推荐生产) |
| **C. README baseline** | `(0.4, 0.2, 0.2, 0.2)` | 0.9730 | 当前默认, 平衡但偏保守 |

**推荐 B**: vec=0.6 在保留 staleness 信号路径的前提下取得最优; 评测集是限定场景, 真实生产中 staleness 确有用 (覆盖"用户搬过家但旧地址还在"这类 case), 不应因为评测集不能区分就关掉 imp 信号.

---

## Reproducibility

```bash
# 复现 Phase A 单轴消融 (~12 分钟)
PYTHONIOENCODING=utf-8 uv run python -m eval.grid_search --quick

# 复现 Phase B+C 完整 grid (~35 分钟)
PYTHONIOENCODING=utf-8 uv run python -m eval.grid_search

# 仅跑粗筛跳过精扫
PYTHONIOENCODING=utf-8 uv run python -m eval.grid_search --skip-fine

# 跑指定权重单跑一次, 写 eval_runs 表
PYTHONIOENCODING=utf-8 uv run python -m eval.run_recall --weights 0.6,0.2,0.2,0.2
```

历史 eval 跑分跨版本对比: SQL 直接查 `eval_runs` 表
```sql
SELECT created_at, score, json_extract(details, '$.weights') AS weights
FROM eval_runs WHERE suite LIKE 'memocortex_zh_v1%'
ORDER BY created_at DESC;
```
