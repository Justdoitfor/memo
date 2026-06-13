"""生命周期管理 — 置信度衰减 + Staleness Detection

核心公式 (MemoryMesh §5.1):

    effective_strength(t) = confidence_base
                          × e^(-λ × active_days)               # Ebbinghaus 衰减
                          × (1 + 0.15 × log(recall_count + 1)) # 复习提升
                          × SOURCE_WEIGHTS[source_type]        # 来源权重
                          × (0.2 if staleness_signal else 1.0) # 软废弃罚分

- staleness 不直接覆盖, 而是软废弃 (effective_strength × 0.2)
- 旧记忆仍保留, 在审计 / 时间溯源场景仍可访问
- 用户后续可主动恢复 (manage_memory mark_active)
"""

from app.lifecycle.decay import compute_effective_strength
from app.lifecycle.staleness import (
    apply_staleness,
    detect_stale_records,
)

__all__ = [
    "compute_effective_strength",
    "apply_staleness",
    "detect_stale_records",
]
