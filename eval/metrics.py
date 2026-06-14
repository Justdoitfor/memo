"""信息检索评测指标 — Recall@K / Precision@K / nDCG@K / MRR / Hit@K.

设计原则:
  - 纯函数, 完全独立, 无外部依赖, 单测覆盖必须 100%
  - relevant 是 set / list (binary 相关性, 命中=1 不命中=0)
  - retrieved 是有序列表 (rank 0 是最相关)

参考公式:
  Recall@K = |retrieved[:K] ∩ relevant| / |relevant|
  Precision@K = |retrieved[:K] ∩ relevant| / K
  Hit@K = 1 if retrieved[:K] ∩ relevant else 0
  MRR = mean(1 / first_relevant_rank), 没命中按 0
  nDCG@K = DCG@K / IDCG@K
    DCG = Σ rel_i / log2(i + 2)   (i 从 0 开始, log2(2) = 1 是 rank 0)
"""
from __future__ import annotations

import math
from typing import Iterable


def _coerce(items) -> list:
    """统一接受 list / tuple / set, 不破坏顺序当传入为 list 时."""
    if isinstance(items, (list, tuple)):
        return list(items)
    return list(items)


# ────────────────────────────────────────────────────────────────────────
#  单 query 指标 (binary relevance)
# ────────────────────────────────────────────────────────────────────────


def hit_at_k(retrieved: Iterable, relevant: Iterable, k: int) -> float:
    """Hit@K — top-K 内是否有任意一条相关项. 值 ∈ {0, 1}."""
    rel = set(relevant)
    if not rel:
        return 0.0
    top = _coerce(retrieved)[:k]
    return 1.0 if any(r in rel for r in top) else 0.0


def recall_at_k(retrieved: Iterable, relevant: Iterable, k: int) -> float:
    """Recall@K — top-K 中相关项数量 / 总相关项数量."""
    rel = set(relevant)
    if not rel:
        return 0.0
    top = _coerce(retrieved)[:k]
    hits = sum(1 for r in top if r in rel)
    return hits / len(rel)


def precision_at_k(retrieved: Iterable, relevant: Iterable, k: int) -> float:
    """Precision@K — top-K 中相关项数量 / K."""
    if k <= 0:
        return 0.0
    rel = set(relevant)
    top = _coerce(retrieved)[:k]
    hits = sum(1 for r in top if r in rel)
    return hits / k


def reciprocal_rank(retrieved: Iterable, relevant: Iterable) -> float:
    """RR — 1 / 第一条相关项的 1-indexed rank, 不命中返回 0.

    e.g. 命中在 rank 0 (top1) → RR = 1.0
         命中在 rank 4 (top5) → RR = 0.2
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    for i, r in enumerate(_coerce(retrieved)):
        if r in rel:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved: Iterable, relevant: Iterable, k: int) -> float:
    """DCG@K (binary relevance) = Σ rel_i / log2(i + 2), i 从 0 开始."""
    rel = set(relevant)
    score = 0.0
    for i, r in enumerate(_coerce(retrieved)[:k]):
        if r in rel:
            score += 1.0 / math.log2(i + 2)
    return score


def ndcg_at_k(retrieved: Iterable, relevant: Iterable, k: int) -> float:
    """nDCG@K = DCG@K / IDCG@K.

    IDCG: 理想排序下 (所有相关项排在最前面) 的 DCG.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = dcg_at_k(retrieved, relevant, k)
    # 理想 DCG: min(|relevant|, k) 个 1 排在最前
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ────────────────────────────────────────────────────────────────────────
#  聚合指标 — 一个 dataset 上的多 query 平均
# ────────────────────────────────────────────────────────────────────────


def mean(values: Iterable[float]) -> float:
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0


def percentile(values: Iterable[float], p: float) -> float:
    """简单 percentile (线性插值). p ∈ [0, 100]."""
    vs = sorted(values)
    if not vs:
        return 0.0
    if p <= 0:
        return vs[0]
    if p >= 100:
        return vs[-1]
    k = (len(vs) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vs[int(k)]
    return vs[f] + (vs[c] - vs[f]) * (k - f)


# ────────────────────────────────────────────────────────────────────────
#  Per-query 一站式打包
# ────────────────────────────────────────────────────────────────────────


def per_query_metrics(
    retrieved: list,
    relevant: list,
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """对一条 query 算所有指标."""
    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        out[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    out["mrr"] = reciprocal_rank(retrieved, relevant)
    return out
