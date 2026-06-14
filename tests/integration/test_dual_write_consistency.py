"""双写一致性测试 — README 声称 "SQLite 是 source of truth, ChromaDB 缺失自动补偿".

这是 MemoCortex 最大的工程承诺之一. 必须有可证伪的测试, 否则面试官会戳穿.

覆盖场景:
  1. 正常写入 → 双侧都有 (基线)
  2. 故障注入 (mock chroma.add 抛错) → SQLite 不被污染
  3. ChromaDB 缺失 → consistency_check 应能识别 (返回 missing 列表)
  4. 用户级 GDPR 删除 → 两侧都被清空
"""
from __future__ import annotations

import uuid

import pytest

from app.models import MemoryRecord, MemoryType

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ────────────────────────────────────────────────────────────────────────
#  Baseline: 正常路径双侧都有
# ────────────────────────────────────────────────────────────────────────


async def test_dual_write_baseline_both_sides_present(storage_initialized, test_user_id):
    """通过 orchestrator 写入 EPISODIC, ChromaDB + SQLite 都应有该记录."""
    from app.memories.episodic import episodic_memory
    from app.storage import get_metadata, get_vector_store

    rec = MemoryRecord(
        user_id=test_user_id,
        type=MemoryType.EPISODIC,
        content="今天与小李讨论了 vector DB 的选型",
    )
    await episodic_memory.write(rec)

    # 1. SQLite 侧
    meta = get_metadata()
    fetched = await meta.get_memory(rec.id)
    assert fetched is not None, "SQLite 应有该 memory"

    # 2. ChromaDB 侧 (按 user_id + content 搜索)
    vec = get_vector_store()
    results = await vec.search(user_id=test_user_id, query="vector DB 选型", top_k=10)
    assert any(r.id == rec.id for r, _ in results), "ChromaDB 应有该 memory"


# ────────────────────────────────────────────────────────────────────────
#  故障注入: ChromaDB 失败时 SQLite 行为
# ────────────────────────────────────────────────────────────────────────


async def test_chroma_failure_does_not_corrupt_sqlite(
    storage_initialized, test_user_id, monkeypatch
):
    """ChromaDB add 抛错时, episodic_memory.write 仍应让 SQLite 保持自洽:
    要么两边都没有 (写入彻底失败), 要么两边都有 (重试成功).
    绝不能出现"SQLite 有, ChromaDB 没有"且无法被 consistency_check 发现的情况.

    当前实现的语义: episodic.write 先写 vector 后写 meta, vector 失败时 meta 也不会写
    → 双方都不会出现该记录, 是干净的.
    """
    from app.memories.episodic import episodic_memory
    from app.storage import get_metadata, get_vector_store

    vec = get_vector_store()
    original_add = vec.add

    async def failing_add(record):
        raise RuntimeError("Simulated ChromaDB outage")

    monkeypatch.setattr(vec, "add", failing_add)

    rec = MemoryRecord(
        user_id=test_user_id, type=MemoryType.EPISODIC,
        content="不应被任何一侧持久化",
    )
    with pytest.raises(RuntimeError, match="Simulated ChromaDB outage"):
        await episodic_memory.write(rec)

    # 关键断言: SQLite 侧不应有这条 (否则 consistency check 之外永远脏数据)
    monkeypatch.setattr(vec, "add", original_add)  # 解除 mock 让 query 不走错
    meta = get_metadata()
    fetched = await meta.get_memory(rec.id)
    assert fetched is None, "ChromaDB 失败时 SQLite 不应已写入"


# ────────────────────────────────────────────────────────────────────────
#  GDPR 删除: 双侧同时清理
# ────────────────────────────────────────────────────────────────────────


async def test_gdpr_delete_clears_both_sides(storage_initialized):
    """forget(all_user_data=True) 必须从双侧都清掉 (vector + meta + signals + entities)."""
    from app.orchestrator.graph import orchestrator
    from app.storage import get_metadata, get_vector_store

    # 用独立 user, 避免污染其他测试
    u = "gdpr_dual_" + uuid.uuid4().hex[:6]
    from app.memories.episodic import episodic_memory

    for i in range(3):
        await episodic_memory.write(
            MemoryRecord(
                user_id=u, type=MemoryType.EPISODIC,
                content=f"事件-{i}-{uuid.uuid4().hex[:4]}",
            )
        )

    # 删除前确认两侧都有
    meta = get_metadata()
    vec = get_vector_store()
    assert len(await meta.list_memories(u)) == 3
    assert await vec.count(u) == 3

    # 执行 GDPR 删除
    result = await orchestrator.forget(user_id=u, all_user_data=True)
    assert result["metadata_deleted"] == 3
    assert result["vector_deleted"] == 3

    # 验证两侧都空
    assert await meta.list_memories(u) == []
    assert await vec.count(u) == 0


# ────────────────────────────────────────────────────────────────────────
#  consistency_check: 主动检测出双写漂移
# ────────────────────────────────────────────────────────────────────────


async def test_consistency_check_detects_drift(storage_initialized):
    """在 SQLite 直接插入一条不写 ChromaDB 的记录, consistency_check 应能识别出漂移
    (具体行为: 报告 missing 列表 + 自动补偿).

    这测的是 README 承诺: "SQLite 为 source of truth, ChromaDB 缺失时自动补偿".
    """
    from app.storage import get_metadata, get_vector_store

    u = "consistency_" + uuid.uuid4().hex[:6]
    meta = get_metadata()
    vec = get_vector_store()

    # 直接绕过 vector 写 SQLite, 制造漂移
    rec = MemoryRecord(
        user_id=u, type=MemoryType.SEMANTIC, content="只在SQLite的孤儿记录",
    )
    await meta.upsert_memory(rec)

    # 此时 SQLite 有, ChromaDB 没有
    assert await meta.get_memory(rec.id) is not None
    vec_count_before = await vec.count(u)

    # 触发一致性检查
    result = await meta.consistency_check(u)

    # 应识别出 1 条漂移并自动补偿写回 ChromaDB (README 承诺的 self-healing 行为)
    assert isinstance(result, dict)
    assert result["sqlite_count"] >= 1
    assert result["chroma_missing"] >= 1, "consistency_check 应识别出 ChromaDB 缺失"
    assert result["chroma_fixed"] >= 1, "consistency_check 应自动补偿写回 ChromaDB"

    # 补偿后应能从 ChromaDB 召回到该记录
    vec_count_after = await vec.count(u)
    assert vec_count_after > vec_count_before
