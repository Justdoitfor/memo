"""Working Memory — 当前会话的短期上下文

特征:
  - 容量极小 (默认 20 条/session), FIFO 淘汰
  - 内存 LRU (Python OrderedDict), SQLite 持久化备份
  - TTL 24h, 会话结束自动失效
  - 不进 ChromaDB, 不进 KG (短期, 不值得向量化成本)
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

from loguru import logger

from app.config import config
from app.models import MemoryRecord, MemoryType
from app.storage import get_metadata


class WorkingMemory:
    """per-session FIFO 缓存, 进程内存 + SQLite 持久化备份.

    设计取舍:
      - 不用 functools.lru_cache: 需要 per-(user, session) 隔离 + 手动淘汰
      - 不用 Redis: MVP 单进程, 避免外部依赖
      - 写穿 (write-through) SQLite: 进程重启不丢
    """

    def __init__(self, capacity: int | None = None) -> None:
        self._capacity = capacity or config.working_capacity
        # key: (user_id, session_id) → OrderedDict[memory_id, MemoryRecord]
        self._buckets: dict[tuple[str, str], OrderedDict[str, MemoryRecord]] = {}
        self._lock = Lock()
        self._meta = get_metadata()
        logger.info(f"WorkingMemory 初始化 — capacity={self._capacity}/session")

    @staticmethod
    def _bucket_key(user_id: str, session_id: str | None) -> tuple[str, str]:
        return (user_id, session_id or "_default_")

    def _get_bucket(self, user_id: str, session_id: str | None) -> OrderedDict[str, MemoryRecord]:
        key = self._bucket_key(user_id, session_id)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = OrderedDict()
            self._buckets[key] = bucket
        return bucket

    async def write(self, record: MemoryRecord) -> str:
        """写入一条 working memory. 超容量自动 FIFO 淘汰."""
        if record.type != MemoryType.WORKING:
            record = record.model_copy(update={"type": MemoryType.WORKING})
        if record.ttl_at is None:
            record.ttl_at = datetime.now() + timedelta(hours=24)

        with self._lock:
            bucket = self._get_bucket(record.user_id, record.session_id)
            bucket[record.id] = record
            bucket.move_to_end(record.id)
            evicted: list[str] = []
            while len(bucket) > self._capacity:
                old_id, _ = bucket.popitem(last=False)
                evicted.append(old_id)

        # 持久化 (写穿)
        await self._meta.upsert_memory(record)
        for old_id in evicted:
            await self._meta.delete_memory(old_id)
            logger.debug(f"WorkingMemory FIFO 淘汰: {old_id}")

        return record.id

    async def read(
        self,
        user_id: str,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        """按时间倒序读最近 N 条 working memory."""
        with self._lock:
            bucket = self._get_bucket(user_id, session_id)
            records = list(bucket.values())
        # 过滤 TTL 过期
        now = datetime.now()
        records = [r for r in records if not r.ttl_at or r.ttl_at > now]
        records.reverse()
        if limit:
            records = records[:limit]
        return records

    async def read_all_sessions(
        self,
        user_id: str,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        """跨 session 取该用户的所有 working memory, 按 created_at 倒序.

        用途: Hybrid Recall 在没有 session_id 时也能召回 Working Memory
        (Playground / 跨 session 查询场景). 生产推荐显式传 session_id 走 read().
        """
        with self._lock:
            all_records: list[MemoryRecord] = []
            for (uid, _sid), bucket in self._buckets.items():
                if uid != user_id:
                    continue
                all_records.extend(bucket.values())
        now = datetime.now()
        all_records = [r for r in all_records if not r.ttl_at or r.ttl_at > now]
        all_records.sort(key=lambda r: r.created_at, reverse=True)
        if limit:
            all_records = all_records[:limit]
        return all_records

    async def clear(self, user_id: str, session_id: str | None = None) -> int:
        """清空一个会话的 working memory (会话结束时调用)."""
        with self._lock:
            key = self._bucket_key(user_id, session_id)
            bucket = self._buckets.pop(key, None)
            count = len(bucket) if bucket else 0
        if bucket:
            for mid in list(bucket.keys()):
                await self._meta.delete_memory(mid)
        logger.info(f"WorkingMemory clear: user={user_id} session={session_id} count={count}")
        return count

    async def restore_from_db(self, user_id: str, session_id: str | None = None) -> int:
        """进程重启时从 SQLite 恢复. lifespan 启动调用."""
        records = await self._meta.list_memories(
            user_id, memory_type=MemoryType.WORKING.value, limit=self._capacity * 2
        )
        now = datetime.now()
        with self._lock:
            bucket = self._get_bucket(user_id, session_id)
            for r in sorted(records, key=lambda x: x.created_at):
                if r.ttl_at and r.ttl_at <= now:
                    continue
                if session_id is not None and r.session_id != session_id:
                    continue
                bucket[r.id] = r
                if len(bucket) > self._capacity:
                    bucket.popitem(last=False)
        return len(records)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "buckets": len(self._buckets),
                "total_items": sum(len(b) for b in self._buckets.values()),
                "capacity_per_session": self._capacity,
            }


# 全局单例
working_memory = WorkingMemory()
