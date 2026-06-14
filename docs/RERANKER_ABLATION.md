# P1.1: Reranker 二阶段排序 — 实验报告

> **数据**: 80 条中文长期记忆评测集 (`eval/datasets/memocortex_zh_v1.jsonl`)
> **一阶段**: 4 信号融合 (vector + temporal + BM25 + importance) + Zep 风格时间过滤
> **二阶段**: bge-reranker-v2-m3 cross-encoder 重排 + 加权融合 (reranker × 0.7 + final × 0.3)
> **运行环境**: CPU only (x86_64 Windows + bge-small-zh + bge-reranker-v2-m3)

## TL;DR

| 配置 | nDCG@10 | MRR | hit@1 | P50 latency | P95 latency |
|---|---:|---:|---:|---:|---:|
| 一阶段 (baseline) | 0.9730 | 0.9667 | 0.950 | **23.6ms** | **32.2ms** |
| + Reranker (本实验) | **1.0000** | **1.0000** | **1.000** | 313.9ms | 436.3ms |
| **Δ** | +0.0270 | +0.0333 | +0.050 | **+13.3x** | +13.5x |

**结论**: Reranker 把召回质量拉到完美 (所有 8 个 scenario 全 1.0), 但 P95 延迟从 32ms 飙到 436ms (+13.5x).

这是一个**典型的 Pareto 取舍** — 适合需要"高准确率召回但能容忍 0.5s 延迟"的场景, 不适合 "对话每轮 < 100ms 响应" 场景.

## 实验设计

### 一阶段 (Stage 1)
- 拿 `top_k_before_rerank = 30` 候选
- 按 4 信号 final_score 排序

### 二阶段 (Stage 2)
- 把 30 个 (query, doc.content) pair 喂给 bge-reranker-v2-m3
- raw logit 经 sigmoid 归一化到 [0, 1]
- 加权融合: `final = reranker × 0.7 + stage1_final × 0.3`
- 重新排序 + 截 top-K

### 配置开关
```python
config.enable_reranker = True            # 默认 False, 显式开启
config.reranker_model = "BAAI/bge-reranker-v2-m3"
config.reranker_weight = 0.7             # reranker 主导但保留一阶段信号 30%
config.top_k_before_rerank = 30          # 一阶段拿 30 条进 reranker
```

环境变量 `MEMOCORTEX_ENABLE_RERANKER=true` 即可一键切换.

## 各 scenario 详细对比

| Scenario | n | 一阶段 nDCG@10 | + Reranker nDCG@10 | 提升 |
|---|---:|---:|---:|---:|
| exact_recall | 20 | 1.0000 | 1.0000 | — |
| paraphrase | 15 | 1.0000 | 1.0000 | — |
| negation | 5 | 1.0000 | 1.0000 | — |
| procedural | 5 | 1.0000 | 1.0000 | — |
| temporal_window | 10 | 0.9631 | **1.0000** | **+0.037** |
| mixed_types | 5 | 0.9701 | **1.0000** | **+0.030** |
| episodic_temporal | 10 | 0.9431 | **1.0000** | **+0.057** |
| conflict_latest | 10 | 0.8931 | **1.0000** | **+0.107** |

**Reranker 主要救活了 4 个最难场景**: conflict_latest / episodic_temporal / temporal_window / mixed_types, 这正是一阶段 4 信号融合"差临门一脚"的场景.

## 延迟分析

```
一阶段 baseline:                              + Reranker:
  P50    23.6ms     ████                       P50   313.9ms     ████████████████
  P95    32.2ms     █████                      P95   436.3ms     █████████████████████
  max    36.3ms     ██████                     max  151807ms     (首次模型加载, outlier)
```

### 延迟拆解 (二阶段)
- 一阶段召回 (vector + BM25 + 算分): ~25ms
- bge-reranker-v2-m3 cross-encoder predict (30 条 batch, CPU): **~280ms**
- 后处理 + 加权融合: <5ms

**reranker 是绝对瓶颈**, 占比 ~90%. 真实生产中:
- GPU 推理可让 reranker 降到 ~30-50ms (10x 加速)
- ONNX runtime + INT8 量化可再降 ~3x
- 也可考虑用更小的 reranker (bge-reranker-base) 换 ~50% 速度提升

## 决策建议

| 场景 | 推荐配置 |
|---|---|
| 对话每轮要求 P95 < 100ms (实时 Agent) | **关闭 reranker**, 用纯一阶段 (0.6, 0.2, 0.2, 0.2) |
| 离线 / batch 召回, 准确率优先 | **开启 reranker**, 接受 ~400ms P95 |
| 混合: 默认关闭, Agent 主动调用 (e.g. user 反馈"重新检索") | 走 `enable_reranker` 参数透传, runtime 切换 |
| GPU 部署 | 总是开启 reranker (延迟劣势小, 准确率红利大) |

## 实现要点

### 失败降级
reranker 失败时降级仅一阶段, 不破坏召回主路径:
```python
try:
    reranker_scores = await async_rerank_pairs(pairs)
    # ...融合
except Exception as e:
    logger.warning(f"Reranker 失败, 降级仅一阶段: {e}")
    return scored  # 仅返回一阶段结果
```

### 可解释性
reranker 的原始 sigmoid 分数挂在 `record.structured["_reranker_raw"]` 里, 便于 debug 时看每条记忆"reranker 觉得有多相关":
```json
"structured": {
  "subject": "user",
  "predicate": "lives_in",
  "object": "上海",
  "_reranker_raw": 0.9847    // reranker sigmoid 后分数
}
```

### 单测覆盖
`tests/unit/test_reranker.py` 8 个测试覆盖 fuse / sigmoid / weight clamp / 空输入边界, 不加载真实 cross-encoder, 跑得快.

## 复现命令

```bash
# 关闭 reranker (默认), 跑 baseline
PYTHONIOENCODING=utf-8 uv run python -m eval.run_recall \
    --json-out eval/reports/baseline_default_weights.json

# 开启 reranker
PYTHONIOENCODING=utf-8 MEMOCORTEX_ENABLE_RERANKER=true \
    uv run python -m eval.run_recall \
    --suite memocortex_zh_v1_reranker \
    --json-out eval/reports/reranker_enabled.json
```

## 后续优化空间

1. **GPU 部署**: bge-reranker-v2-m3 GPU 推理可让 P95 从 436ms → ~50ms
2. **量化**: INT8 量化可再降 ~3x 延迟 + 减小内存占用
3. **更小 reranker**: bge-reranker-base 速度提升 ~50%, 但中文场景精度可能略降
4. **动态 top_k_before_rerank**: 若一阶段 final_score 已经足够区分 (e.g. top-3 都 > 0.9), 跳过 reranker
5. **缓存层**: 同 (query, doc) 对的 reranker 分数 LRU 缓存, 重复 query 直接命中
