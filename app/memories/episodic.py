"""Episodic Memory — 时序事件 ("X 时间发生了 Y")

特征:
  - 写入即向量化 → ChromaDB (per-user filter)
  - SQLite 持久化备份 (元数据真源)
  - 按时间窗口 + 语义相似度混合查询
  - 30 天后由 reflection worker 考虑提炼为 Semantic 或归档冷存储
"""

from __future__ import annotations

from datetime import datetime, timedelta

from loguru import logger

from app.models import MemoryRecord, MemoryType
from app.storage import get_metadata, get_vector_store


class EpisodicMemory:
    """情景记忆 — 双写 ChromaDB (向量召回) + SQLite (元数据真源)."""

    def __init__(self) -> None:
        self._vector = get_vector_store()
        self._meta = get_metadata()
        logger.info("EpisodicMemory 初始化")

    async def write(self, record: MemoryRecord) -> str:
        """写一条 episodic. 自动设置 type=EPISODIC, 双写 vector + meta."""
        if record.type != MemoryType.EPISODIC:
            record = record.model_copy(update={"type": MemoryType.EPISODIC})
        await self._vector.add(record)
        await self._meta.upsert_memory(record)
        logger.debug(f"Episodic write: {record.id} user={record.user_id}")
        return record.id

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        since: datetime | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """语义召回 + 可选时间窗过滤."""
        results = await self._vector.search(
            user_id=user_id,
            query=query,
            memory_types=[MemoryType.EPISODIC.value],
            top_k=top_k,
        )
        if since:
            results = [(r, s) for r, s in results if r.created_at >= since]
        return results

    async def list_recent(
        self,
        user_id: str,
        days: int = 7,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        """直接从 SQLite 取最近 N 天的事件 (不走向量召回)."""
        since = datetime.now() - timedelta(days=days)
        return await self._meta.list_memories(
            user_id, memory_type=MemoryType.EPISODIC.value, since=since, limit=limit
        )

    async def delete(self, memory_id: str, user_id: str) -> bool:
        ok_vec = await self._vector.delete(memory_id, user_id)
        ok_meta = await self._meta.delete_memory(memory_id)
        return ok_vec and ok_meta


episodic_memory = EpisodicMemory()
