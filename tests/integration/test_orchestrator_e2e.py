"""Orchestrator end-to-end 测试 — write → recall → forget 全链路.

不调 LLM (走 EPISODIC 路径不强制抽取 semantic; 调用方主动用 SEMANTIC 时才会).
关键: 测召回结果的 4 信号都被填充, signals_used 字段诚实反映.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from app.models import MemoryRecord, MemoryType, WriteRequest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_episodic_write_then_recall(storage_initialized, test_user_id):
    """EPISODIC 写入后, 用关键词搜索能召回, 且返回的 signals 4 维都已填充."""
    from app.orchestrator.graph import orchestrator

    write_resp = await orchestrator.write(
        WriteRequest(
            user_id=test_user_id,
            content="昨天和小张讨论了 ChromaDB 的索引调优",
            type=MemoryType.EPISODIC,
        )
    )
    assert write_resp.memory_id
    assert write_resp.routed_type == MemoryType.EPISODIC

    # 等可能的后台 semantic 抽取结束, 让测试确定性
    await orchestrator.wait_pending(timeout=10.0)

    search = await orchestrator.search(
        user_id=test_user_id,
        query="ChromaDB 索引调优",
        types=[MemoryType.EPISODIC],
        top_k=5,
    )

    assert search.latency_ms > 0
    assert len(search.results) >= 1
    top = search.results[0]
    # final_score 必须是 4 信号融合的结果
    sig = top.signals
    assert 0.0 <= sig.vector_sim <= 1.0
    assert 0.0 <= sig.temporal_decay <= 1.0
    assert 0.0 <= sig.keyword_match <= 1.0
    assert 0.0 <= sig.importance <= 1.0
    assert sig.final_score > 0.0

    # signals_used 必须诚实列出 4 信号 (Mem0 v3 风格的可解释返回)
    assert "vector" in search.signals_used
    assert "temporal" in search.signals_used
    assert any("keyword" in s for s in search.signals_used)
    assert any("importance" in s for s in search.signals_used)


async def test_recall_user_isolation(storage_initialized):
    """两个 user 写同样 content, 互相不应召回到对方."""
    from app.orchestrator.graph import orchestrator

    u1 = "iso_e2e_" + uuid.uuid4().hex[:6]
    u2 = "iso_e2e_" + uuid.uuid4().hex[:6]

    await orchestrator.write(
        WriteRequest(user_id=u1, content="独特短语ABC123", type=MemoryType.EPISODIC)
    )
    await orchestrator.write(
        WriteRequest(user_id=u2, content="独特短语ABC123", type=MemoryType.EPISODIC)
    )
    await orchestrator.wait_pending(timeout=5.0)

    s1 = await orchestrator.search(user_id=u1, query="独特短语ABC123", top_k=10)
    s2 = await orchestrator.search(user_id=u2, query="独特短语ABC123", top_k=10)

    s1_users = {r.record.user_id for r in s1.results}
    s2_users = {r.record.user_id for r in s2.results}

    assert s1_users <= {u1}, f"u1 召回出现非 u1 数据: {s1_users}"
    assert s2_users <= {u2}, f"u2 召回出现非 u2 数据: {s2_users}"


async def test_forget_single_memory(storage_initialized, test_user_id):
    from app.orchestrator.graph import orchestrator

    resp = await orchestrator.write(
        WriteRequest(
            user_id=test_user_id,
            content="临时事件需要被删除" + uuid.uuid4().hex[:4],
            type=MemoryType.EPISODIC,
        )
    )
    mid = resp.memory_id

    forget = await orchestrator.forget(user_id=test_user_id, memory_id=mid)
    assert forget["deleted"] is True

    # 召回应不再返回此 ID
    s = await orchestrator.search(
        user_id=test_user_id, query="临时事件需要被删除", top_k=10,
    )
    assert all(r.record.id != mid for r in s.results)


async def test_score_threshold_zero_returns_all_candidates(
    storage_initialized, test_user_id
):
    """score_threshold=0.0 显式关闭过滤, 用于调试 / eval 完全召回."""
    from app.orchestrator.graph import orchestrator

    await orchestrator.write(
        WriteRequest(
            user_id=test_user_id, content="完全不相关的内容", type=MemoryType.EPISODIC,
        )
    )
    await orchestrator.wait_pending(timeout=5.0)

    s = await orchestrator.search(
        user_id=test_user_id,
        query="完全不相关的内容",
        types=[MemoryType.EPISODIC],
        top_k=20,
        score_threshold=0.0,  # 关闭过滤
    )
    assert len(s.results) >= 1
