"""热记忆 Snapshot 缓存 — 参考 Letta Core Memory 设计

核心洞察: Agent 每轮对话都需要用户的核心事实和偏好, 但每次调 recall 走完整向量检索
太慢. Letta 的 core memory 将 500-2000 token 的关键信息始终保持在 context window 中,
实现零检索延迟.

本模块:
  - MemorySnapshotCache 按 user_id 缓存紧凑的"热记忆快照" (< 500 tokens)
  - 快照内容: Semantic 前 10 条事实 + Reflective 画像 + Implicit 偏好
  - 命中时延迟 <1ms, 远快于 recall (50-200ms)
  - 缓存失效: 新的 Semantic/Reflective 写入时调 invalidate(user_id)

设计 (v2 — 版本号):
  - 用单调递增版本号代替 TTL 时间戳
  - 任何 invalidate 让该 user 的版本号 +1, cache miss 走 build
  - 优点: 无 TTL 漂移窗口 ("写入后 5 分钟内可能拿旧数据" 这种问题不再有);
    并发 set / invalidate 不会 KeyError
  - 容量保护用 LRU 淘汰 (单调访问计数器); 不受 TTL 干扰

Agent 使用方式:
  1. 每轮对话开始时读 memory://snapshot/{user_id} Resource (< 1ms)
  2. 需要更多细节时再调 recall tool (50-200ms)
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

from loguru import logger

from app.models import MemoryType


class MemorySnapshotCache:
    """版本号 + LRU 缓存 — 命中时 < 1ms, 写入立即一致.

    线程安全: asyncio.Lock 保护 _cache / _versions / _access 三个 dict.
    构建快照本身在锁外, 不阻塞其他 user 的读.
    """

    def __init__(self, max_users: int = 100) -> None:
        # user_id → (cached_version, snapshot_dict)
        self._cache: dict[str, tuple[int, dict[str, Any]]] = {}
        # user_id → 当前版本号 (单调递增, invalidate 时 +1)
        self._versions: dict[str, int] = defaultdict(int)
        # user_id → LRU 访问计数 (容量淘汰用)
        self._access: dict[str, int] = {}
        self._counter = 0
        self._max_users = max_users
        self._lock = asyncio.Lock()

    async def get_or_build(
        self,
        user_id: str,
        builder: Callable[[str], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """读 cache, 命中且版本一致直接返回; 否则用 builder 构建后写回.

        builder 在锁外执行, 避免阻塞其它 user 的读取.
        构建期间被 invalidate → 写回时检测到版本不一致, 不写 (避免覆盖更新版本).
        """
        async with self._lock:
            cur_version = self._versions[user_id]
            cached = self._cache.get(user_id)
            if cached is not None:
                v, snap = cached
                if v == cur_version:
                    self._counter += 1
                    self._access[user_id] = self._counter
                    return snap
            # 即将走 build, 在锁内提前记录目标版本号
            target_version = cur_version

        # 锁外构建 (不阻塞其它 user)
        snap = await builder(user_id)

        async with self._lock:
            # 容量保护: 淘汰最久未访问的
            if len(self._cache) >= self._max_users and user_id not in self._cache:
                if self._access:
                    oldest = min(self._access, key=lambda k: self._access[k])
                    self._cache.pop(oldest, None)
                    self._access.pop(oldest, None)
            # 仅在版本仍是 target_version 时才写回 (期间被 invalidate 则放弃, 下次重建)
            if self._versions[user_id] == target_version:
                self._cache[user_id] = (target_version, snap)
                self._counter += 1
                self._access[user_id] = self._counter
            else:
                logger.debug(
                    f"snapshot_cache: build {user_id} 期间被 invalidate, "
                    f"放弃写回 (target_v={target_version}, cur_v={self._versions[user_id]})"
                )
        return snap

    def invalidate(self, user_id: str) -> None:
        """让该 user 下次读必 miss. 单调递增版本号, 不删 cache 项 (容量自然淘汰).

        同步函数 — 业务路径调用方便. 内部仅一次 dict 写, 与 asyncio.Lock 不冲突
        (defaultdict 的 __setitem__ 是原子的, asyncio 单线程不会切换).
        """
        self._versions[user_id] += 1

    def invalidate_all(self) -> None:
        """全量失效 — consolidate 或 profile refresh 后."""
        # 给所有已知 user 的版本 +1; 未知 user 的 defaultdict 默认 0, 下次访问取 0,
        # cache 中没有 → miss, 自动 build, 行为一致
        for uid in list(self._versions.keys()):
            self._versions[uid] += 1


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
    return snapshot


async def get_snapshot(user_id: str) -> dict[str, Any]:
    """读快照 — 缓存命中时 < 1ms, 未命中时构建 (~10ms SQLite 直查, 不走向量)."""
    return await snapshot_cache.get_or_build(user_id, build_snapshot)
