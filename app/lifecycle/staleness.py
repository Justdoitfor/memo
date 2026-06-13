"""Staleness Detection — 软废弃机制 (与 LLM Arbitrator 共存的兜底方案)

核心思路 (MemoryMesh §5.1):
  当新记忆与现有高置信度记忆在语义上矛盾时, 不直接覆盖, 而是:
    1. 新建一条修订记忆 (source_type=CORRECTED, 权重 1.2)
    2. 旧记忆 staleness_signal=True (effective_strength × 0.2 软废弃)
    3. 双向链接: 旧.superseded_by = 新.id, 新.structured["supersedes"] = 旧.id
    4. 召回时两条都呈现, 但旧的被排到后面 (effective_strength 低)
    5. 旧记忆在 N 次召回中从未被 Agent 选用 → 自动归档 cold

与 LLM Arbitrator 的关系:
  - Arbitrator 是主策略 — LLM 决策 REPLACE/MERGE/VERSIONED/IGNORE
  - Staleness 是兜底 — 当 Arbitrator 失败 / 无 LLM Key / 高频写入避免 LLM 调用
  - 业务方可在 WriteRequest 显式选 strategy = 'arbitrator' | 'staleness' | 'auto'
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger

from app.lifecycle.decay import compute_effective_strength
from app.models import MemoryRecord, SourceType
from app.storage import get_metadata, get_vector_store


async def apply_staleness(
    new_record: MemoryRecord,
    old_records: list[MemoryRecord],
) -> dict:
    """对一组冲突旧记忆执行软废弃, 把新记忆标 CORRECTED.

    Args:
        new_record: 新写入的记忆 (会被原地修改 source_type=CORRECTED + structured.supersedes)
        old_records: 与 new_record 冲突的旧记忆列表

    Returns:
        {"superseded": [old_id, ...], "new_id": new.id, "action": "stale_marked"}
    """
    if not old_records:
        return {"superseded": [], "new_id": new_record.id, "action": "no_conflict"}

    meta = get_metadata()
    vec = get_vector_store()

    # 1. 新记忆标 CORRECTED (用户主动修正, 权重最高)
    new_record.source_type = SourceType.CORRECTED.value
    new_record.confidence_score = max(new_record.confidence_score, 0.85)
    structured = dict(new_record.structured)
    structured["supersedes"] = [r.id for r in old_records]
    new_record.structured = structured

    # 2. 旧记忆批量软废弃
    superseded_ids = []
    for old in old_records:
        old.staleness_signal = True
        old.superseded_by = new_record.id
        # 持久化到 SQLite
        await meta.upsert_memory(old)
        # 同步 Chroma metadata
        try:
            await vec.update_metadata(
                old.id, old.user_id,
                {
                    "staleness_signal": 1,
                    "superseded_by": new_record.id,
                },
            )
        except Exception as e:
            logger.warning(f"Chroma staleness 更新失败 {old.id[:8]}: {e}")
        superseded_ids.append(old.id)

    logger.info(
        f"[Staleness] 软废弃 {len(superseded_ids)} 条旧记忆 → 新记忆 {new_record.id[:8]} "
        f"(标 CORRECTED, supersedes={[i[:8] for i in superseded_ids]})"
    )

    return {
        "superseded": superseded_ids,
        "new_id": new_record.id,
        "action": "stale_marked",
    }


async def detect_stale_records(
    user_id: str,
    strength_threshold: float = 0.15,
    days_unused: int = 60,
) -> list[MemoryRecord]:
    """扫描某用户全部记忆, 找出已软废弃且长期未被召回的 → 待归档为 cold tier.

    后台 Worker 定期调用.
    """
    meta = get_metadata()
    candidates = await meta.list_memories(user_id, limit=10000)
    now = datetime.now()

    stale: list[MemoryRecord] = []
    for r in candidates:
        if not r.staleness_signal:
            continue
        strength = compute_effective_strength(r, now=now)
        if strength > strength_threshold:
            continue
        anchor = r.last_recalled_at or r.created_at
        if (now - anchor).days < days_unused:
            continue
        stale.append(r)
    return stale
