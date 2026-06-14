"""Storage Protocol 契约测试.

任何 VectorStore / MetadataStore / KnowledgeGraph 实现都必须通过这套测试.
用途:
  1. 确保 ChromaDB 实现没有 regression
  2. 后续如果加 PostgreSQL 实现 (P1), 直接复用此契约即可

设计原则:
  - 不依赖具体实现细节 (e.g. SQLAlchemy session)
  - 不假设全局状态 (每个测试用独立 user_id)
  - 仅断言 Protocol 中声明的语义
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest

from app.models import Entity, MemoryRecord, MemoryType, SourceType, Triple

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ────────────────────────────────────────────────────────────────────────
#  MetadataStore 契约
# ────────────────────────────────────────────────────────────────────────


class TestMetadataStoreContract:
    async def test_upsert_then_get(self, storage_initialized, test_user_id):
        from app.storage import get_metadata

        meta = get_metadata()
        rec = MemoryRecord(
            user_id=test_user_id,
            type=MemoryType.SEMANTIC,
            content="住在北京",
            structured={"subject": "user", "predicate": "lives_in", "object": "北京"},
            importance=0.9,
        )
        await meta.upsert_memory(rec)
        fetched = await meta.get_memory(rec.id)

        assert fetched is not None
        assert fetched.id == rec.id
        assert fetched.user_id == test_user_id
        assert fetched.content == "住在北京"
        assert fetched.structured == rec.structured
        assert fetched.importance == 0.9
        assert fetched.type == MemoryType.SEMANTIC

    async def test_upsert_idempotent(self, storage_initialized, test_user_id):
        """同 ID 二次写入应 update 而非 insert (双写一致性的基石)."""
        from app.storage import get_metadata

        meta = get_metadata()
        rec = MemoryRecord(
            user_id=test_user_id,
            type=MemoryType.SEMANTIC,
            content="原始内容",
            importance=0.5,
        )
        await meta.upsert_memory(rec)

        rec.content = "更新后"
        rec.importance = 0.8
        await meta.upsert_memory(rec)

        fetched = await meta.get_memory(rec.id)
        assert fetched is not None
        assert fetched.content == "更新后"
        assert fetched.importance == 0.8

    async def test_get_missing_returns_none(self, storage_initialized):
        from app.storage import get_metadata

        meta = get_metadata()
        result = await meta.get_memory("does_not_exist_" + uuid.uuid4().hex)
        assert result is None

    async def test_list_by_type(self, storage_initialized, test_user_id):
        from app.storage import get_metadata

        meta = get_metadata()
        # 写 2 条 SEMANTIC + 1 条 EPISODIC
        for content in ["住北京", "对花生过敏"]:
            await meta.upsert_memory(
                MemoryRecord(user_id=test_user_id, type=MemoryType.SEMANTIC, content=content)
            )
        await meta.upsert_memory(
            MemoryRecord(user_id=test_user_id, type=MemoryType.EPISODIC, content="今天加班")
        )

        sem = await meta.list_memories(test_user_id, memory_type=MemoryType.SEMANTIC.value)
        epi = await meta.list_memories(test_user_id, memory_type=MemoryType.EPISODIC.value)
        all_mem = await meta.list_memories(test_user_id)

        assert len(sem) == 2
        assert len(epi) == 1
        assert len(all_mem) == 3

    async def test_list_by_since(self, storage_initialized, test_user_id):
        """since 时间过滤应排除早于该时间的记录."""
        from app.storage import get_metadata

        meta = get_metadata()
        old = MemoryRecord(
            user_id=test_user_id, type=MemoryType.SEMANTIC, content="老记忆",
            created_at=datetime.now() - timedelta(days=10),
        )
        new = MemoryRecord(
            user_id=test_user_id, type=MemoryType.SEMANTIC, content="新记忆",
        )
        await meta.upsert_memory(old)
        await meta.upsert_memory(new)

        recent = await meta.list_memories(
            test_user_id, since=datetime.now() - timedelta(days=1),
        )
        contents = {r.content for r in recent}
        assert "新记忆" in contents
        assert "老记忆" not in contents

    async def test_list_user_isolation(self, storage_initialized):
        """user_id 维度应严格隔离 — 多租户安全的基础."""
        from app.storage import get_metadata

        meta = get_metadata()
        u1 = "iso_user_" + uuid.uuid4().hex[:6]
        u2 = "iso_user_" + uuid.uuid4().hex[:6]

        await meta.upsert_memory(
            MemoryRecord(user_id=u1, type=MemoryType.SEMANTIC, content="u1的事实")
        )
        await meta.upsert_memory(
            MemoryRecord(user_id=u2, type=MemoryType.SEMANTIC, content="u2的事实")
        )

        u1_mems = await meta.list_memories(u1)
        u2_mems = await meta.list_memories(u2)

        assert all(r.user_id == u1 for r in u1_mems)
        assert all(r.user_id == u2 for r in u2_mems)
        assert {r.content for r in u1_mems} == {"u1的事实"}
        assert {r.content for r in u2_mems} == {"u2的事实"}

    async def test_batch_get(self, storage_initialized, test_user_id):
        """batch_get_memories 是召回路径 BM25 补候选用的, 必须严格保护."""
        from app.storage import get_metadata

        meta = get_metadata()
        ids = []
        for i in range(5):
            rec = MemoryRecord(
                user_id=test_user_id, type=MemoryType.SEMANTIC, content=f"事实-{i}",
            )
            await meta.upsert_memory(rec)
            ids.append(rec.id)

        # 包含真实 + 不存在的 ID, 不存在的应被静默过滤
        ids_with_missing = ids + ["does_not_exist_" + uuid.uuid4().hex]
        batch = await meta.batch_get_memories(ids_with_missing)
        assert len(batch) == 5
        assert {r.id for r in batch} == set(ids)

    async def test_batch_get_empty_input(self, storage_initialized):
        from app.storage import get_metadata

        meta = get_metadata()
        assert await meta.batch_get_memories([]) == []

    async def test_delete_memory(self, storage_initialized, test_user_id):
        from app.storage import get_metadata

        meta = get_metadata()
        rec = MemoryRecord(user_id=test_user_id, type=MemoryType.EPISODIC, content="x")
        await meta.upsert_memory(rec)

        ok = await meta.delete_memory(rec.id)
        assert ok is True
        assert await meta.get_memory(rec.id) is None

    async def test_delete_missing_returns_false(self, storage_initialized):
        from app.storage import get_metadata

        meta = get_metadata()
        ok = await meta.delete_memory("never_existed_" + uuid.uuid4().hex)
        assert ok is False

    async def test_delete_all_user_data(self, storage_initialized):
        """GDPR 删除 — 所有数据按 user_id 维度被清干净."""
        from app.storage import get_metadata

        meta = get_metadata()
        u = "gdpr_test_" + uuid.uuid4().hex[:8]
        for i in range(3):
            await meta.upsert_memory(
                MemoryRecord(user_id=u, type=MemoryType.SEMANTIC, content=f"事实-{i}")
            )

        deleted = await meta.delete_all_memories(u)
        assert deleted == 3
        assert await meta.list_memories(u) == []

    async def test_arbitration_log(self, storage_initialized, test_user_id):
        from app.storage import get_metadata

        meta = get_metadata()
        await meta.log_arbitration({
            "user_id": test_user_id,
            "subject": "user",
            "predicate": "lives_in",
            "old_value": "[\"北京\"]",
            "new_value": "上海",
            "action": "replace",
            "reasoning": "用户主动告知搬家",
            "confidence": 0.95,
        })
        logs = await meta.list_arbitrations(test_user_id, limit=10)
        assert any(l["new_value"] == "上海" for l in logs)


# ────────────────────────────────────────────────────────────────────────
#  VectorStore 契约
# ────────────────────────────────────────────────────────────────────────


class TestVectorStoreContract:
    async def test_add_then_search(self, storage_initialized, test_user_id):
        """写入后语义搜索能召回."""
        from app.storage import get_vector_store

        vec = get_vector_store()
        rec = MemoryRecord(
            user_id=test_user_id,
            type=MemoryType.SEMANTIC,
            content="我对花生过敏",
        )
        await vec.add(rec)

        results = await vec.search(
            user_id=test_user_id, query="花生过敏", top_k=5,
        )
        assert len(results) >= 1
        assert any(r.id == rec.id for r, _ in results)

    async def test_user_isolation_in_vector(self, storage_initialized):
        """向量召回严格按 user_id 过滤, 不能跨用户泄漏."""
        from app.storage import get_vector_store

        vec = get_vector_store()
        u1 = "vec_iso_" + uuid.uuid4().hex[:6]
        u2 = "vec_iso_" + uuid.uuid4().hex[:6]
        rec_u1 = MemoryRecord(user_id=u1, type=MemoryType.SEMANTIC, content="u1的秘密内容")
        rec_u2 = MemoryRecord(user_id=u2, type=MemoryType.SEMANTIC, content="u2的秘密内容")
        await vec.add(rec_u1)
        await vec.add(rec_u2)

        # u1 查询不应召回 u2 的记忆
        results_u1 = await vec.search(user_id=u1, query="秘密内容", top_k=10)
        ids_u1 = {r.id for r, _ in results_u1}
        assert rec_u2.id not in ids_u1

    async def test_count_after_add(self, storage_initialized, test_user_id):
        from app.storage import get_vector_store

        vec = get_vector_store()
        before = await vec.count(test_user_id)
        for i in range(3):
            await vec.add(
                MemoryRecord(user_id=test_user_id, type=MemoryType.EPISODIC, content=f"事件-{i}")
            )
        after = await vec.count(test_user_id)
        assert after - before == 3

    async def test_delete_removes_from_search(self, storage_initialized, test_user_id):
        from app.storage import get_vector_store

        vec = get_vector_store()
        rec = MemoryRecord(
            user_id=test_user_id, type=MemoryType.SEMANTIC, content="独特内容ABCXYZ",
        )
        await vec.add(rec)

        ok = await vec.delete(rec.id, test_user_id)
        assert ok is True

        # 删除后再搜索不应命中
        results = await vec.search(user_id=test_user_id, query="独特内容ABCXYZ", top_k=10)
        assert all(r.id != rec.id for r, _ in results)

    async def test_update_metadata(self, storage_initialized, test_user_id):
        """recall 路径会调 update_metadata 异步更新 last_recalled_at / recall_count."""
        from app.storage import get_vector_store

        vec = get_vector_store()
        rec = MemoryRecord(
            user_id=test_user_id, type=MemoryType.SEMANTIC, content="x",
        )
        await vec.add(rec)
        ok = await vec.update_metadata(
            rec.id, test_user_id,
            {"recall_count": 5, "staleness_signal": 1},
        )
        assert ok is True


# ────────────────────────────────────────────────────────────────────────
#  KnowledgeGraph 契约
# ────────────────────────────────────────────────────────────────────────


class TestKnowledgeGraphContract:
    async def test_add_and_find_triple(self, storage_initialized, test_user_id):
        from app.storage import get_kg

        kg = get_kg()
        t = Triple(subject="user", predicate="lives_in", object="北京")
        await kg.add_triple(test_user_id, t)

        found = await kg.find_triples(test_user_id, subject="user", predicate="lives_in")
        objects = {tt.object for tt in found}
        assert "北京" in objects

    async def test_find_with_partial_pattern(self, storage_initialized, test_user_id):
        from app.storage import get_kg

        kg = get_kg()
        await kg.add_triple(
            test_user_id, Triple(subject="user", predicate="likes", object="编程")
        )
        await kg.add_triple(
            test_user_id, Triple(subject="user", predicate="likes", object="跑步")
        )

        # 任意维度可省略
        likes = await kg.find_triples(test_user_id, predicate="likes")
        assert len(likes) >= 2

    async def test_neighbors_two_hops(self, storage_initialized, test_user_id):
        """user → works_at → 字节, 字节 → has_employee → Alice
        2-hop neighbors of user 应包含字节 + Alice.
        """
        from app.storage import get_kg

        kg = get_kg()
        await kg.add_triple(test_user_id, Triple(subject="user", predicate="works_at", object="字节"))
        await kg.add_triple(test_user_id, Triple(subject="字节", predicate="has_employee", object="Alice"))

        neighbors = await kg.neighbors(test_user_id, "user", max_hops=2)
        assert "字节" in neighbors
        assert "Alice" in neighbors

    async def test_kg_user_isolation(self, storage_initialized):
        from app.storage import get_kg

        kg = get_kg()
        u1 = "kg_iso_" + uuid.uuid4().hex[:6]
        u2 = "kg_iso_" + uuid.uuid4().hex[:6]
        await kg.add_triple(u1, Triple(subject="user", predicate="lives_in", object="北京"))
        await kg.add_triple(u2, Triple(subject="user", predicate="lives_in", object="上海"))

        u1_triples = await kg.find_triples(u1)
        u2_triples = await kg.find_triples(u2)

        assert all(t.object == "北京" for t in u1_triples if t.predicate == "lives_in")
        assert all(t.object == "上海" for t in u2_triples if t.predicate == "lives_in")
