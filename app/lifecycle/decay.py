"""Ebbinghaus 遗忘曲线 + recall-driven 复习提升

公式:
  effective_strength(t) = confidence_score
                        × e^(-decay_rate × active_days)
                        × (1 + 0.15 × log(recall_count + 1))
                        × SOURCE_WEIGHTS[source_type]
                        × (0.2 if staleness_signal else 1.0)

设计目标:
  - 新写入的 explicit_statement: 1.0
  - 30 天未召回的 explicit_statement (λ=0.01): exp(-0.3) ≈ 0.74
  - 30 天未召回 + staleness=True: 0.74 × 0.2 ≈ 0.15 (基本被压下去)
  - 频繁召回的旧记忆: 加 0.15 × log(50+1) ≈ 加 0.59 ≈ 翻倍后再衰减
"""

from __future__ import annotations

import math
from datetime import datetime

from app.models import SOURCE_WEIGHTS, MemoryRecord


def compute_effective_strength(
    record: MemoryRecord,
    now: datetime | None = None,
) -> float:
    """根据 Ebbinghaus 公式 + 复习提升 + 来源权重 + staleness 罚分计算有效强度.

    返回值范围: [0.0, ~1.5] (CORRECTED 来源可超 1.0)
    """
    now = now or datetime.now()

    # 1. 基础置信度
    strength = record.confidence_score

    # 2. Ebbinghaus 衰减 — 以 last_recalled_at 为锚点 (复习重置遗忘曲线)
    anchor = record.last_recalled_at or record.created_at
    active_days = max(0.0, (now - anchor).total_seconds() / 86400.0)
    if record.decay_rate > 0:
        strength *= math.exp(-record.decay_rate * active_days)

    # 3. 复习提升 — 召回次数对数加成 (防止线性爆炸)
    if record.recall_count > 0:
        strength *= 1.0 + 0.15 * math.log(record.recall_count + 1)

    # 4. 来源权重 — explicit > corrected > inferred
    weight = SOURCE_WEIGHTS.get(record.source_type, 1.0)
    strength *= weight

    # 5. Staleness 罚分 — 软废弃
    if record.staleness_signal:
        strength *= 0.2

    return max(0.0, strength)
