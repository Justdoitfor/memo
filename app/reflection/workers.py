"""Reflection Workers — APScheduler 周期任务

4 个任务 (现在全部自动遍历活跃用户):
  1. distill_episodic_to_semantic   — Episodic → Semantic 提炼
  2. merge_duplicates              — 相似 Semantic 三元组合并
  3. decay_importance              — 长期未召回的记忆 importance 衰减
  4. refresh_reflective_profile    — 重新生成用户画像 (run_all_for_user 内调用)

MCP consolidate 工具可手动触发单个用户 (绕过 scheduler).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from app.config import config
from app.memories.reflective import reflective_memory
from app.memories.semantic import semantic_memory
from app.models import MemoryType
from app.storage import get_metadata, get_vector_store
from app.utils.metrics import metrics

_scheduler: AsyncIOScheduler | None = None


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         活跃用户查询                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝


async def _get_active_users(days: int = 7) -> list[str]:
    """查询最近 N 天有记忆活动的用户列表."""
    meta = get_metadata()
    return await meta.list_active_users(days=days)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         Task 1: Distillation                         ║
# ╚══════════════════════════════════════════════════════════════════════╝


async def distill_episodic_to_semantic(user_id: str | None = None) -> dict[str, Any]:
    """从最近 N 天 Episodic 中, LLM 提炼出可结构化的事实写入 Semantic.

    user_id=None 时自动遍历所有活跃用户 (scheduler 调用).
    """
    if user_id is None:
        return await _run_for_all_active(distill_episodic_to_semantic)

    meta = get_metadata()
    since = datetime.now() - timedelta(days=7)

    records = await meta.list_memories(
        user_id, memory_type=MemoryType.EPISODIC.value, since=since, limit=50
    )
    processed = 0
    for r in records:
        # 只补偿没被抽取过的 (source != distilled)
        if r.source == "distilled":
            continue
        try:
            await semantic_memory.write_from_text(
                user_id=user_id, text=r.content, source_memory_id=r.id
            )
            processed += 1
        except Exception as e:
            logger.warning(f"distill 失败 {r.id}: {e}")
    logger.info(f"[Reflection] distill: user={user_id} processed={processed}/{len(records)}")
    metrics.incr("reflection.distill.runs")
    return {"user_id": user_id, "scanned": len(records), "distilled": processed}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         Task 2: Merge Duplicates                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


async def merge_duplicates(user_id: str | None = None) -> dict[str, Any]:
    """查找近似重复的 Semantic 三元组并合并.

    user_id=None 时自动遍历所有活跃用户 (scheduler 调用).
    """
    if user_id is None:
        return await _run_for_all_active(merge_duplicates)

    meta = get_metadata()
    records = await meta.list_memories(
        user_id, memory_type=MemoryType.SEMANTIC.value, limit=500
    )
    # 按 content 分桶
    buckets: dict[str, list] = {}
    for r in records:
        buckets.setdefault(r.content, []).append(r)

    removed = 0
    vec = get_vector_store()
    for _content, group in buckets.items():
        if len(group) <= 1:
            continue
        # 留 importance 最高 + recall_count 最多的, 其他删除
        keep = max(group, key=lambda r: (r.importance, r.recall_count))
        for r in group:
            if r.id == keep.id:
                continue
            try:
                await vec.delete(r.id, user_id)
                await meta.delete_memory(r.id)
                removed += 1
            except Exception as e:
                logger.warning(f"merge 删除失败 {r.id}: {e}")

    logger.info(f"[Reflection] merge_duplicates: user={user_id} removed={removed}")
    metrics.incr("reflection.merge.runs")
    return {"user_id": user_id, "scanned": len(records), "removed": removed}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         Task 3: Importance Decay                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


async def decay_importance(user_id: str | None = None) -> dict[str, Any]:
    """对长期未召回的记忆做 importance 指数衰减, 低于阈值标记为冷数据.

    user_id=None 时自动遍历所有活跃用户 (scheduler 调用).
    """
    if user_id is None:
        return await _run_for_all_active(decay_importance)

    meta = get_metadata()
    vec = get_vector_store()
    now = datetime.now()
    records = await meta.list_memories(user_id, limit=1000)
    decayed = 0
    cooled = 0
    for r in records:
        last = r.last_recalled_at or r.created_at
        days_silent = (now - last).total_seconds() / 86400.0
        # 60 天未召回 → 半衰一次
        new_imp = r.importance * math.exp(-days_silent / 120.0)
        if abs(new_imp - r.importance) < 0.01:
            continue
        r.importance = round(new_imp, 4)
        try:
            await vec.update_metadata(r.id, user_id, {"importance": r.importance})
            await meta.upsert_memory(r)
            decayed += 1
            # importance < 0.1 且 30 天未召回 → 标 cold
            if r.importance < 0.1 and days_silent > 30 and r.tier == "hot":
                r.tier = "cold"
                await meta.upsert_memory(r)
                cooled += 1
        except Exception as e:
            logger.warning(f"decay 更新失败 {r.id}: {e}")

    logger.info(f"[Reflection] decay: user={user_id} decayed={decayed} cooled={cooled}")
    metrics.incr("reflection.decay.runs")
    return {"user_id": user_id, "decayed": decayed, "cooled": cooled}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         Task 4: Profile Refresh                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


async def refresh_reflective_profile(user_id: str) -> dict[str, Any]:
    """重新生成用户画像."""
    profile = await reflective_memory.refresh(user_id)
    return {"user_id": user_id, "profile": profile}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         通用: 遍历活跃用户                              ║
# ╚══════════════════════════════════════════════════════════════════════╝


async def _run_for_all_active(task_fn) -> dict[str, Any]:
    """遍历活跃用户列表, 对每个用户执行 task_fn.

    返回汇总统计: 执行了多少用户, 成功/失败数.
    """
    users = await _get_active_users(days=14)
    if not users:
        logger.debug(f"[Reflection] {task_fn.__name__}: 无活跃用户, 跳过")
        return {"skipped": True, "reason": "no_active_users"}

    logger.info(f"[Reflection] {task_fn.__name__}: 开始遍历 {len(users)} 个活跃用户")
    success = 0
    failed = 0
    for uid in users:
        try:
            await task_fn(uid)
            success += 1
        except Exception as e:
            logger.warning(f"[Reflection] {task_fn.__name__}: user={uid} 失败: {e}")
            failed += 1

    return {"users_total": len(users), "success": success, "failed": failed}


async def _mine_all_users() -> None:
    """Pattern Miner — 扫所有有 signals 的活跃用户."""
    from app.pattern import mine_patterns_for_user

    users = await _get_active_users(days=14)
    if not users:
        return
    logger.info(f"[PatternMiner cron] 开始挖掘 {len(users)} 个活跃用户的隐式模式")
    for uid in users:
        try:
            await mine_patterns_for_user(uid)
        except Exception as e:
            logger.warning(f"[PatternMiner cron] user={uid} 失败: {e}")


async def _flush_graph() -> None:
    """定时刷盘 — 将 NetworkX dirty 用户图持久化到磁盘."""
    from app.storage import get_kg

    kg = get_kg()
    try:
        await kg.flush()
    except Exception as e:
        logger.warning(f"NetworkX flush 失败: {e}")


async def _consistency_check_all() -> None:
    """定时一致性检查 — 比对所有活跃用户的 SQLite/ChromaDB 数据, 补偿缺失."""
    meta = get_metadata()
    users = await _get_active_users(days=30)
    if not users:
        return
    total_fixed = 0
    total_cleaned = 0
    for uid in users:
        try:
            result = await meta.consistency_check(uid)
            total_fixed += result.get("chroma_fixed", 0)
            total_cleaned += result.get("chroma_cleaned", 0)
        except Exception as e:
            logger.warning(f"[Consistency] user={uid} 检查失败: {e}")
    if total_fixed > 0 or total_cleaned > 0:
        logger.info(f"[Consistency cron] 修复 {total_fixed} 条, 清理 {total_cleaned} 条")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         Scheduler 控制                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


def start_scheduler() -> AsyncIOScheduler:
    """启动调度器 (MCP Server lifespan 调用).

    所有 Worker 传入 user_id=None, 内部自动遍历活跃用户列表.
    """
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    _scheduler = AsyncIOScheduler()

    _scheduler.add_job(
        lambda: distill_episodic_to_semantic(None),
        "interval",
        seconds=config.reflect_distill_interval_sec,
        id="distill",
    )
    _scheduler.add_job(
        lambda: merge_duplicates(None),
        "interval",
        seconds=config.reflect_merge_interval_sec,
        id="merge",
    )
    _scheduler.add_job(
        lambda: decay_importance(None),
        "interval",
        seconds=config.reflect_decay_interval_sec,
        id="decay",
    )
    # Pattern Miner — 跨所有活跃用户定期扫描
    _scheduler.add_job(
        _mine_all_users,
        "interval",
        seconds=config.pattern_mine_interval_sec,
        id="pattern_mine",
    )
    # NetworkX 定时刷盘 — 每 60s 将 dirty 用户图写入磁盘
    _scheduler.add_job(
        _flush_graph,
        "interval",
        seconds=60,
        id="graph_flush",
    )
    # 一致性检查 — 每 5 分钟比对 SQLite/ChromaDB 数据
    _scheduler.add_job(
        _consistency_check_all,
        "interval",
        seconds=300,
        id="consistency_check",
    )
    # 过期 VERSIONED triple 归档 — 每 1 小时检查并归档过期 >30 天的 triple
    _scheduler.add_job(
        lambda: archive_expired_versioned(None),
        "interval",
        seconds=3600,
        id="archive_expired",
    )
    _scheduler.start()
    logger.info(
        f"Reflection scheduler 启动: distill={config.reflect_distill_interval_sec}s "
        f"merge={config.reflect_merge_interval_sec}s decay={config.reflect_decay_interval_sec}s "
        f"pattern_mine={config.pattern_mine_interval_sec}s graph_flush=60s "
        f"consistency=300s archive=3600s"
    )
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Reflection scheduler 已停止")


async def run_all_for_user(user_id: str) -> dict[str, Any]:
    """便捷入口: 给定 user_id, 跑全套 5 个任务 (供 MCP consolidate 工具调用)."""
    return {
        "distill": await distill_episodic_to_semantic(user_id),
        "merge": await merge_duplicates(user_id),
        "decay": await decay_importance(user_id),
        "profile": await refresh_reflective_profile(user_id),
        "archive_expired": await archive_expired_versioned(user_id),
    }


async def archive_expired_versioned(user_id: str | None = None) -> dict[str, Any]:
    """将 valid_until 过期 >30 天的 VERSIONED triple 归档到 Cold Storage (Zep 风格, P3-2).

    这些 triple 已失效很长时间, 对当前召回无价值, 但历史审计可能需要.
    归档而非删除, 保证可追溯.
    """
    if user_id is None:
        return await _run_for_all_active(archive_expired_versioned)

    meta = get_metadata()
    vec = get_vector_store()
    cold = None
    try:
        from app.storage import get_cold
        cold = get_cold()
    except Exception:
        pass  # Cold Storage 未配置 → 仅从 ChromaDB 删除, 不归档

    now = datetime.now()
    cutoff = now - timedelta(days=30)  # 过期 >30 天才归档
    records = await meta.list_memories(
        user_id, memory_type=MemoryType.SEMANTIC.value, limit=500
    )

    archived = 0
    for r in records:
        structured = r.structured or {}
        valid_until_str = structured.get("valid_until")
        if not valid_until_str:
            continue
        try:
            valid_until = datetime.fromisoformat(str(valid_until_str))
        except (TypeError, ValueError):
            continue

        # 只归档 valid_until < cutoff (过期 >30 天) 的 triple
        if valid_until >= cutoff:
            continue

        # 归档到 Cold Storage
        if cold:
            try:
                import json
                uri = await cold.archive(
                    f"versioned/{r.id}",
                    json.dumps(r.model_dump(mode="json"), ensure_ascii=False),
                )
                logger.debug(f"[Archive] 归档 {r.id} → {uri}")
            except Exception as e:
                logger.warning(f"[Archive] 归档失败 {r.id}: {e}")
                continue

        # 从 ChromaDB + SQLite 删除 (KG triple 保留用于审计)
        try:
            await vec.delete(r.id, user_id)
            await meta.delete_memory(r.id)
            archived += 1
        except Exception as e:
            logger.warning(f"[Archive] 删除失败 {r.id}: {e}")

    if archived > 0:
        logger.info(f"[Archive] user={user_id} 归档 {archived} 条过期 VERSIONED triple")
    metrics.incr("reflection.archive.runs")
    return {"user_id": user_id, "archived": archived}
