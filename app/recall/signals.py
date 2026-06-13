"""4 信号召回打分实现 — 每个函数独立可测."""

from __future__ import annotations

import math
from datetime import datetime

from app.config import config
from app.models import MemoryRecord


def compute_vector_sim(raw_similarity: float) -> float:
    """ChromaDB cosine similarity 已经在 [0, 1], 直接返回."""
    return max(0.0, min(1.0, raw_similarity))


def compute_temporal_decay(
    created_at: datetime,
    now: datetime | None = None,
    tau_days: float | None = None,
) -> float:
    """指数时间衰减: f(t) = exp(-Δt / τ), 越新越接近 1.

    tau_days 控制衰减速度: 默认 30 天 → 一个月前的记忆衰减到 e^{-1} ≈ 0.37.
    """
    now = now or datetime.now()
    tau = tau_days if tau_days is not None else config.temporal_tau_days
    delta_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    if tau <= 0:
        return 1.0
    return math.exp(-delta_days / tau)


def compute_keyword_match(
    record: MemoryRecord,
    query_entities: set[str],
    user_neighbors: set[str],
) -> float:
    """记忆与查询的实体重叠度 (保留供 KG 场景未来扩展, 当前未使用).

    简化规则:
      - 查询中的实体在记忆的 structured.subject/object 出现 → +0.5
      - 记忆的实体在 user 的 KG 邻居中 → +0.3
      - 全无 → 0
    """
    score = 0.0
    record_entities: set[str] = set()
    if record.structured:
        for k in ("subject", "object"):
            v = record.structured.get(k)
            if v:
                record_entities.add(str(v))

    if record_entities & query_entities:
        score += 0.5
    if record_entities & user_neighbors:
        score += 0.3
    return min(1.0, score)


def compute_importance(record: MemoryRecord) -> float:
    """重要度信号 — Phase 1 起改用 effective_strength (含 Ebbinghaus 衰减 +
    复习提升 + 来源权重 + Staleness 软废弃).

    Staleness 标记的旧记忆 effective_strength × 0.2, 召回时自动被压下去.
    """
    from app.lifecycle import compute_effective_strength

    # effective_strength 范围 [0, ~1.5], 归一化到 [0, 1]
    strength = compute_effective_strength(record)
    return min(1.0, strength)


def fuse_signals(
    vector_sim: float,
    temporal_decay: float,
    keyword_match: float,
    importance: float,
    weights: tuple[float, float, float, float] | None = None,
) -> float:
    """加权融合 4 信号. 默认权重从 config 读, 可被业务覆盖.

    所有权重为 0 时直接返回 0 (语义: 无任何信号被采用 → 不应给分).
    """
    if weights is None:
        w = (
            config.recall_w_vector,
            config.recall_w_temporal,
            config.recall_w_keyword,
            config.recall_w_importance,
        )
    else:
        w = weights
    total_w = sum(w)
    if total_w <= 0:
        return 0.0
    return (
        w[0] * vector_sim
        + w[1] * temporal_decay
        + w[2] * keyword_match
        + w[3] * importance
    ) / total_w
