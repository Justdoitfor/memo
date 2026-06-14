"""MemorySnapshotCache 单元测试 — 验证版本号失效 + LRU 容量保护 + 并发安全.

核心场景:
  1. 命中: build 一次后再读不再调 builder
  2. invalidate 后下次读必 miss
  3. 并发 invalidate during build → 不写脏数据 (该轮 build 结果丢弃)
  4. LRU 容量满时淘汰最久未访问的
  5. 单调访问计数器: 读访问刷新 LRU 位置
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.core.snapshot_cache import MemorySnapshotCache


pytestmark = pytest.mark.asyncio


class _Counter:
    """记录 builder 被调用的次数, 用于断言缓存命中."""

    def __init__(self, payload: Any = None):
        self.count = 0
        self.payload = payload or {"facts": []}

    async def build(self, user_id: str) -> dict:
        self.count += 1
        return {**self.payload, "user_id": user_id, "build_seq": self.count}


# ────────────────────────────────────────────────────────────────────────
#  基本命中 / 失效
# ────────────────────────────────────────────────────────────────────────


async def test_miss_then_hit():
    cache = MemorySnapshotCache(max_users=10)
    counter = _Counter()

    snap1 = await cache.get_or_build("alice", counter.build)
    snap2 = await cache.get_or_build("alice", counter.build)

    assert counter.count == 1, "第二次读应命中缓存, 不再调 builder"
    assert snap1 == snap2


async def test_invalidate_forces_rebuild():
    cache = MemorySnapshotCache(max_users=10)
    counter = _Counter()

    await cache.get_or_build("alice", counter.build)
    cache.invalidate("alice")
    snap_after = await cache.get_or_build("alice", counter.build)

    assert counter.count == 2, "invalidate 后下次读应重新 build"
    assert snap_after["build_seq"] == 2


async def test_per_user_isolation():
    cache = MemorySnapshotCache(max_users=10)
    counter = _Counter()

    snap_a = await cache.get_or_build("alice", counter.build)
    snap_b = await cache.get_or_build("bob", counter.build)

    assert counter.count == 2
    assert snap_a["user_id"] == "alice"
    assert snap_b["user_id"] == "bob"
    # alice 的 invalidate 不应影响 bob
    cache.invalidate("alice")
    await cache.get_or_build("bob", counter.build)
    assert counter.count == 2, "bob 的读应仍命中缓存"


async def test_invalidate_all():
    cache = MemorySnapshotCache(max_users=10)
    counter = _Counter()

    await cache.get_or_build("alice", counter.build)
    await cache.get_or_build("bob", counter.build)
    cache.invalidate_all()
    await cache.get_or_build("alice", counter.build)
    await cache.get_or_build("bob", counter.build)

    assert counter.count == 4, "invalidate_all 后两个 user 都应重 build"


# ────────────────────────────────────────────────────────────────────────
#  并发: invalidate during build
# ────────────────────────────────────────────────────────────────────────


async def test_invalidate_during_build_discards_stale_result():
    """关键不变性: build 期间被 invalidate, 该轮结果不应写回 (会覆盖更新版本).

    复现:
      1. 启动 build(alice), 它会等 build_event 才完成
      2. build 进行中调 invalidate("alice")
      3. 释放 build_event, build 完成尝试写回, 但版本不一致 → 放弃
      4. 下次读 → 必 miss → 重新 build (拿到第二次)
    """
    cache = MemorySnapshotCache(max_users=10)
    build_event = asyncio.Event()
    build_count = 0

    async def slow_builder(user_id: str) -> dict:
        nonlocal build_count
        build_count += 1
        my_seq = build_count
        if my_seq == 1:
            await build_event.wait()  # 卡住第一次 build
        return {"user_id": user_id, "build_seq": my_seq}

    # 启动第一次 build (会卡住)
    task = asyncio.create_task(cache.get_or_build("alice", slow_builder))
    await asyncio.sleep(0.05)  # 让 task 进入 builder

    # 在 build 进行中 invalidate
    cache.invalidate("alice")

    # 释放 build, 让第一次完成
    build_event.set()
    snap1 = await task
    assert snap1["build_seq"] == 1  # 第一次 build 仍然返回了它的结果

    # 下次读 — cache 应该被丢弃 → 触发第二次 build
    snap2 = await cache.get_or_build("alice", slow_builder)
    assert snap2["build_seq"] == 2, "build1 期间被 invalidate, 结果不应写回缓存"
    assert build_count == 2


async def test_concurrent_reads_dont_double_build():
    """同时多个 read 命中同一个 user → 至多 build 一次后续都拿缓存.

    注: 当前实现是 'first builder wins', 后续 read 看到 cached 直接返回.
    """
    cache = MemorySnapshotCache(max_users=10)
    counter = _Counter()

    # 串行 (asyncio 单线程, 第一个 await 完后, 第二个一定能读到 cache)
    await cache.get_or_build("alice", counter.build)
    # 接下来 5 个并发读
    results = await asyncio.gather(
        *[cache.get_or_build("alice", counter.build) for _ in range(5)]
    )
    assert counter.count == 1, "5 个后续读不应再 build"
    assert all(r["build_seq"] == 1 for r in results)


# ────────────────────────────────────────────────────────────────────────
#  LRU 容量保护
# ────────────────────────────────────────────────────────────────────────


async def test_lru_eviction_when_capacity_exceeded():
    """容量 = 2: 写 3 个 user, 最早访问的应被淘汰."""
    cache = MemorySnapshotCache(max_users=2)
    counter = _Counter()

    await cache.get_or_build("alice", counter.build)   # access_seq = 1
    await cache.get_or_build("bob", counter.build)     # access_seq = 2
    await cache.get_or_build("carol", counter.build)   # 触发淘汰: alice 最久未访问

    assert counter.count == 3

    # alice 应被淘汰 → 再读必 miss
    await cache.get_or_build("alice", counter.build)
    assert counter.count == 4, "alice 被淘汰后再读应重新 build"


async def test_lru_access_refreshes_position():
    """命中读应把 user 刷到 LRU 队尾, 让它在下一次淘汰中幸免."""
    cache = MemorySnapshotCache(max_users=2)
    counter = _Counter()

    await cache.get_or_build("alice", counter.build)  # build, access=1
    await cache.get_or_build("bob", counter.build)    # build, access=2
    # 关键: 命中读 alice, 把它从 access=1 刷到 access=3 (新于 bob 的 2)
    snap_alice = await cache.get_or_build("alice", counter.build)  # hit
    assert snap_alice["build_seq"] == 1
    assert counter.count == 2, "alice 第二次读应命中, 不再 build"

    # 此时 cache: {alice: access=3, bob: access=2}, max=2
    # 写 carol → 必须淘汰一个; 期望淘汰 bob (access=2, 比 alice 旧)
    await cache.get_or_build("carol", counter.build)  # build, access=4
    assert counter.count == 3

    # alice 应仍在缓存 (访问刷新生效)
    snap_alice2 = await cache.get_or_build("alice", counter.build)
    assert counter.count == 3, "alice 应仍命中 (LRU 刷新到队尾, 没被淘汰)"
    assert snap_alice2["build_seq"] == 1
