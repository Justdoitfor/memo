"""PostgreSQL MetadataStore 契约测试 — 证明 Storage Protocol 抽象闭合.

设计原则:
  - 这是一份"shadow contract test"—— 测试覆盖的方法和 test_storage_protocol.py
    里 TestMetadataStoreContract 类完全平行, 只是把后端从 SQLite 切到 PG
  - 同一份 ORM 模型 (MemoryORM, EntityORM, ArbitrationLogORM, ...) 在两个 dialect 下都能工作
  - 这就是 "MetadataStore Protocol 抽象层 + ORM 跨 dialect 复用" 的工程证据

跑测试需要:
  - docker run -d --name memocortex_pg_test -e POSTGRES_USER=test \\
      -e POSTGRES_PASSWORD=test -e POSTGRES_DB=memocortex_test \\
      -p 5433:5432 postgres:16-alpine
  - 设置环境变量 MEMOCORTEX_TEST_PG_URL=postgresql+asyncpg://test:test@localhost:5433/memocortex_test

CI 上不需要 PG 时 (默认), 这些测试自动 skip.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

import pytest

from app.models import MemoryRecord, MemoryType


# 跑 PG 契约测试需要这个环境变量, 没设就全部 skip
PG_URL = os.getenv(
    "MEMOCORTEX_TEST_PG_URL",
    "",
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not PG_URL,
        reason="MEMOCORTEX_TEST_PG_URL 未设置 — 需要本地起 PG 容器才能跑此契约测试",
    ),
]


@pytest.fixture
async def pg_store():
    """每个 test 一个 PG store, 避免 pytest-asyncio + module-scoped fixture
    "Event loop is closed" 经典坑 — engine 绑定的 event loop 在测试间会变.

    schema init 是 IF NOT EXISTS, 幂等; 单个 test ~150ms init 开销可接受.
    每个测试用 uuid user_id 隔离, 不需要清表.
    """
    from app.storage.pg_store import PostgresMetadataStore

    store = PostgresMetadataStore(url=PG_URL)
    await store.init_schema()
    try:
        yield store
    finally:
        # 释放 engine 持有的连接池, 避免 "Event loop is closed" 警告
        await store._engine.dispose()


# ────────────────────────────────────────────────────────────────────────
#  与 test_storage_protocol.py 的 TestMetadataStoreContract 平行
# ────────────────────────────────────────────────────────────────────────


class TestPostgresMetadataStoreContract:
    """跟 SQLiteMetadataStore 完全相同的契约 — 证明 Protocol 抽象闭合."""

    async def test_upsert_then_get(self, pg_store, test_user_id):
        rec = MemoryRecord(
            user_id=test_user_id,
            type=MemoryType.SEMANTIC,
            content="我住在北京海淀区",
            structured={"subject": "user", "predicate": "lives_in", "object": "北京海淀区"},
            importance=0.9,
        )
        await pg_store.upsert_memory(rec)
        fetched = await pg_store.get_memory(rec.id)

        assert fetched is not None
        assert fetched.id == rec.id
        assert fetched.user_id == test_user_id
        assert fetched.content == "我住在北京海淀区"
        assert fetched.structured == rec.structured
        assert fetched.importance == 0.9
        assert fetched.type == MemoryType.SEMANTIC

    async def test_upsert_idempotent(self, pg_store, test_user_id):
        rec = MemoryRecord(
            user_id=test_user_id, type=MemoryType.SEMANTIC,
            content="原始", importance=0.5,
        )
        await pg_store.upsert_memory(rec)

        rec.content = "更新后"
        rec.importance = 0.85
        await pg_store.upsert_memory(rec)

        fetched = await pg_store.get_memory(rec.id)
        assert fetched.content == "更新后"
        assert fetched.importance == 0.85

    async def test_get_missing_returns_none(self, pg_store):
        result = await pg_store.get_memory("missing_" + uuid.uuid4().hex)
        assert result is None

    async def test_list_by_type(self, pg_store, test_user_id):
        for content in ["住北京", "对花生过敏"]:
            await pg_store.upsert_memory(
                MemoryRecord(
                    user_id=test_user_id, type=MemoryType.SEMANTIC, content=content,
                )
            )
        await pg_store.upsert_memory(
            MemoryRecord(user_id=test_user_id, type=MemoryType.EPISODIC, content="今天加班")
        )

        sem = await pg_store.list_memories(test_user_id, memory_type=MemoryType.SEMANTIC.value)
        epi = await pg_store.list_memories(test_user_id, memory_type=MemoryType.EPISODIC.value)
        assert len(sem) >= 2
        assert len(epi) >= 1

    async def test_list_by_since(self, pg_store, test_user_id):
        old = MemoryRecord(
            user_id=test_user_id, type=MemoryType.SEMANTIC, content="老记忆",
            created_at=datetime.now() - timedelta(days=10),
        )
        new = MemoryRecord(
            user_id=test_user_id, type=MemoryType.SEMANTIC, content="新记忆",
        )
        await pg_store.upsert_memory(old)
        await pg_store.upsert_memory(new)

        recent = await pg_store.list_memories(
            test_user_id, since=datetime.now() - timedelta(days=1),
        )
        contents = {r.content for r in recent}
        assert "新记忆" in contents
        assert "老记忆" not in contents

    async def test_user_isolation(self, pg_store):
        u1 = "pg_iso_" + uuid.uuid4().hex[:6]
        u2 = "pg_iso_" + uuid.uuid4().hex[:6]
        await pg_store.upsert_memory(
            MemoryRecord(user_id=u1, type=MemoryType.SEMANTIC, content="u1的事实")
        )
        await pg_store.upsert_memory(
            MemoryRecord(user_id=u2, type=MemoryType.SEMANTIC, content="u2的事实")
        )

        u1_mems = await pg_store.list_memories(u1)
        u2_mems = await pg_store.list_memories(u2)

        assert all(r.user_id == u1 for r in u1_mems)
        assert all(r.user_id == u2 for r in u2_mems)

    async def test_batch_get(self, pg_store, test_user_id):
        ids = []
        for i in range(5):
            rec = MemoryRecord(
                user_id=test_user_id, type=MemoryType.SEMANTIC, content=f"事实-{i}",
            )
            await pg_store.upsert_memory(rec)
            ids.append(rec.id)

        ids_with_missing = ids + ["does_not_exist_" + uuid.uuid4().hex]
        batch = await pg_store.batch_get_memories(ids_with_missing)
        assert len(batch) == 5
        assert {r.id for r in batch} == set(ids)

    async def test_batch_get_empty_input(self, pg_store):
        assert await pg_store.batch_get_memories([]) == []

    async def test_delete_memory(self, pg_store, test_user_id):
        rec = MemoryRecord(user_id=test_user_id, type=MemoryType.EPISODIC, content="x")
        await pg_store.upsert_memory(rec)

        ok = await pg_store.delete_memory(rec.id)
        assert ok is True
        assert await pg_store.get_memory(rec.id) is None

    async def test_delete_missing_returns_false(self, pg_store):
        ok = await pg_store.delete_memory("never_existed_" + uuid.uuid4().hex)
        assert ok is False

    async def test_delete_all_user_data(self, pg_store):
        u = "pg_gdpr_" + uuid.uuid4().hex[:8]
        for i in range(3):
            await pg_store.upsert_memory(
                MemoryRecord(user_id=u, type=MemoryType.SEMANTIC, content=f"事实-{i}")
            )

        deleted = await pg_store.delete_all_memories(u)
        assert deleted == 3
        assert await pg_store.list_memories(u) == []

    async def test_arbitration_log(self, pg_store, test_user_id):
        await pg_store.log_arbitration({
            "user_id": test_user_id,
            "subject": "user",
            "predicate": "lives_in",
            "old_value": "[\"北京\"]",
            "new_value": "上海",
            "action": "replace",
            "reasoning": "用户主动告知搬家",
            "confidence": 0.95,
        })
        logs = await pg_store.list_arbitrations(test_user_id, limit=10)
        assert any(l["new_value"] == "上海" for l in logs)

    async def test_eval_run(self, pg_store):
        """eval_runs 表 — 跨版本回归对比的核心 (P0 stage 3 已实现)."""
        suite_name = "pg_test_" + uuid.uuid4().hex[:6]
        await pg_store.save_eval_run(
            suite=suite_name,
            score=0.87,
            details={"weights": [0.4, 0.2, 0.2, 0.2], "n_total": 80},
        )
        last = await pg_store.last_eval(suite_name)
        assert last is not None
        assert last["score"] == 0.87
        assert last["details"]["n_total"] == 80


# ────────────────────────────────────────────────────────────────────────
#  Sanitize URL — 安全特性测试
# ────────────────────────────────────────────────────────────────────────


def test_sanitize_url_hides_password():
    from app.storage.pg_store import _sanitize_url

    sanitized = _sanitize_url("postgresql+asyncpg://alice:secretpass@localhost:5432/db")
    assert "secretpass" not in sanitized
    assert "alice" in sanitized
    assert "***" in sanitized


def test_sanitize_url_handles_no_password():
    from app.storage.pg_store import _sanitize_url

    # 没有 @ 的 URL 应原样返回
    assert _sanitize_url("notaurl") == "notaurl"
    assert _sanitize_url("postgresql://user@host/db") == "postgresql://user@host/db"
