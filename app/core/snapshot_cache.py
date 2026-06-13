"""热记忆 Snapshot 缓存 — 参考 Letta Core Memory 设计

核心洞察: Agent 每轮对话都需要用户的核心事实和偏好, 但每次调 recall 走完整向量检索
太慢. Letta 的 core memory 将 500-2000 token 的关键信息始终保持在 context window 中,
实现零检索延迟.

本模块:
  - MemorySnapshotCache 按 user_id 缓存紧凑的"热记忆快照" (< 500 tokens)
  - 快照内容: Semantic 前 10 条事实 + Reflective 画像 + Implicit 偏好
  - 命中时延迟 <1ms, 远快于 recall (50-200ms)
  - 缓存失效: 新的 Semantic/Reflective 写入时自动 invalidate

Agent 使用方式:
  1. 每轮对话开始时读 memory://snapshot/{user_id} Resource (< 1ms)
  2. 需要更多细节时再调 recall tool (50-200ms)
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from app.models import MemoryType


class MemorySnapshotCache:
    """LRU 缓存 — 紧凑热记忆快照, 命中时 < 1ms."""

    def __init__(self, max_users: int = 100, ttl_seconds: float = 300.0) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, float] = {}
        self._max_users = max_users
        self._ttl = ttl_seconds

    def get(self, user_id: str) -> dict[str, Any] | None:
        """读缓存, 过期则返回 None."""
        if user_id not in self._cache:
            return None
        if time.monotonic() - self._timestamps[user_id] > self._ttl:
            # TTL 过期, 主动淘汰
            del self._cache[user_id]
            del self._timestamps[user_id]
            return None
        return self._cache[user_id]

    def set(self, user_id: str, snapshot: dict[str, Any]) -> None:
        """写入缓存, 超容量时淘汰最早的."""
        # 容量控制: 淘汰最旧条目
        if len(self._cache) >= self._max_users and user_id not in self._cache:
            oldest = min(self._timestamps, key=self._timestamps.get)
            del self._cache[oldest]
            del self._timestamps[oldest]
        self._cache[user_id] = snapshot
        self._timestamps[user_id] = time.monotonic()

    def invalidate(self, user_id: str) -> None:
        """失效缓存 — Semantic/Reflective/Implicit 写入时调用."""
        if user_id in self._cache:
            del self._cache[user_id]
            del self._timestamps[user_id]

    def invalidate_all(self) -> None:
        """全量失效 — consolidate 或 profile refresh 后."""
        self._cache.clear()
        self._timestamps.clear()


# 全局单例
snapshot_cache = MemorySnapshotCache()


async def build_snapshot(user_id: str) -> dict[str, Any]:
    """构建用户的热记忆快照 — 从 SQLite 直接拉取, 不走向量检索.

    返回结构:
      {
        "user_id": "alice",
        "facts": ["住在北京", "对花生过敏", ...],  # Semantic 前 10 条
        "profile": {"one_liner": "...", "preferences": [...], "constraints": [...]},
        "preferences": ["偏好简洁回答", ...],  # Implicit 偏好
        "updated_at": "...",
      }
    """
    from app.storage import get_metadata
    from app.memories.reflective import reflective_memory

    meta = get_metadata()

    # 1. Semantic 核心事实 (前 10, 排除 stale)
    semantic_records = await meta.list_memories(
        user_id, memory_type=MemoryType.SEMANTIC.value, limit=10,
    )
    facts = [
        r.content for r in semantic_records
        if not r.staleness_signal
    ]

    # 2. Reflective 画像
    profile_data = await reflective_memory.get(user_id)
    profile = profile_data.get("profile", {}) if profile_data else {}

    # 3. Implicit 偏好 (前 5 条)
    implicit_records = await meta.list_memories(
        user_id, memory_type=MemoryType.IMPLICIT.value, limit=5,
    )
    preferences = [r.content for r in implicit_records]

    snapshot = {
        "user_id": user_id,
        "facts": facts,
        "profile": profile,
        "preferences": preferences,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    snapshot_cache.set(user_id, snapshot)
    return snapshot


async def get_snapshot(user_id: str) -> dict[str, Any]:
    """读快照 — 缓存命中时 < 1ms, 未命中时构建 (~10ms SQLite 查询)."""
    cached = snapshot_cache.get(user_id)
    if cached is not None:
        return cached
    return await build_snapshot(user_id)