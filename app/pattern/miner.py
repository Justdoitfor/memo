"""Pattern Miner — 把行为信号聚合为 Implicit 偏好记忆

触发时机:
  1. 后台 Cron (APScheduler) — 每 N 次会话或每天定时
  2. 手动 MCP reflect tool / REST POST /admin/mine_patterns/{user_id}

挖掘策略:
  1. 拉取该 user 最近 X 天的所有 behavior_signals
  2. 按 (signal_type, context_tags 排序后拼接) 分组
  3. 每组若出现次数 >= MIN_OCCURRENCES, 视为"重复模式"
  4. 用 LLM 把信号序列归纳成一句自然语言偏好
  5. 写入 type=IMPLICIT, source_type=INFERRED, confidence=0.55 的记忆
  6. 重复挖掘时若已有相同偏好, 仅更新 last_recalled_at 不重复入库
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.core.llm_factory import llm_factory
from app.models import MemoryRecord, MemoryType, SourceType
from app.storage import get_metadata, get_vector_store
from app.utils.metrics import metrics

# 同一模式出现 >= 此次数才挖掘
MIN_OCCURRENCES = 3

# 默认拉取最近多少天的信号
DEFAULT_WINDOW_DAYS = 14


class _ImplicitInsight(BaseModel):
    """LLM 输出 schema."""

    insight: str = Field(description="一句话总结的用户隐式偏好, 中文, < 60 字")
    confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list, description="该偏好涉及的关键场景词")


_MINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是一个用户行为偏好分析师. 给定一段时间内用户的重复行为信号,\n"
                "用一句话总结这种行为隐含的偏好.\n"
                "\n"
                "原则:\n"
                "- insight 必须以'用户...'开头, < 60 字, 中文\n"
                "- 仅描述明确出现频次 >= 3 次的模式, 不要从单次行为推断\n"
                "- 如果同时涉及多个场景, 在 keywords 列出 (e.g. ['code_review', 'writing'])\n"
                "- confidence: 信号一致性高 → 0.65; 较模糊 → 0.45\n"
                "- 仅输出 JSON, 字段: insight / confidence / keywords"
            ),
        ),
        ("human", "信号类型: {signal_type}\n上下文场景: {context_tags}\n累计出现: {count} 次\n\n请归纳."),
    ]
)


def _group_signals(signals: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """按 (signal_type, 排序后的 context_tags 拼串) 分组."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in signals:
        tags_key = "+".join(sorted(s.get("context_tags") or []))
        key = (s["signal_type"], tags_key)
        buckets[key].append(s)
    return buckets


async def mine_patterns_for_user(
    user_id: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_occurrences: int = MIN_OCCURRENCES,
) -> list[MemoryRecord]:
    """对单个 user 跑挖掘. 返回新增的 Implicit 记忆列表."""
    meta = get_metadata()
    vec = get_vector_store()

    since = datetime.now() - timedelta(days=window_days)
    signals = await meta.list_signals(user_id=user_id, since=since, limit=2000)
    logger.info(
        f"[PatternMiner] user={user_id} 拉取 {len(signals)} 条信号 "
        f"(最近 {window_days} 天)"
    )

    if not signals:
        return []

    buckets = _group_signals(signals)
    new_records: list[MemoryRecord] = []

    # 一次性拉取所有 Implicit 记忆, 避免 N 个 signal group 各查一次 (P2-5)
    existing_implicit = await meta.list_memories(
        user_id, memory_type=MemoryType.IMPLICIT.value, limit=200
    )
    # 构建 (signal_type, sorted_keywords) → record 的索引, O(1) 去重
    implicit_index: dict[tuple[str, tuple[str, ...]], MemoryRecord] = {}
    for ex in existing_implicit:
        ex_keywords = tuple(sorted(ex.structured.get("keywords", []) if ex.structured else []))
        ex_stype = ex.structured.get("signal_type", "") if ex.structured else ""
        implicit_index[(ex_stype, ex_keywords)] = ex

    for (signal_type, tags_key), items in buckets.items():
        if len(items) < min_occurrences:
            continue
        tag_list = tags_key.split("+") if tags_key else []
        # 查重: O(1) 查 implicit_index
        lookup_key = (signal_type, tuple(sorted(tag_list)))
        if lookup_key in implicit_index:
            logger.debug(f"[PatternMiner] 已存在该模式记忆, 跳过 {signal_type}+{tag_list}")
            continue

        # LLM 归纳
        try:
            with metrics.timer("pattern.mine.latency"):
                result = await llm_factory.structured_invoke(
                    _MINE_PROMPT, _ImplicitInsight,
                    {
                        "signal_type": signal_type,
                        "context_tags": ", ".join(tag_list) or "(无标签)",
                        "count": len(items),
                    },
                    temperature=0.3,
                    purpose="pattern_miner",
                )
            if result is None:
                logger.warning(f"[PatternMiner] LLM 归纳失败 {signal_type}+{tag_list}")
                continue
        except Exception as e:
            logger.warning(f"[PatternMiner] LLM 调用异常: {e}")
            continue

        # 入库 — keywords 用 tag_list (去重 key 基础) + LLM 补充词, 保证去重一致性
        llm_keywords = result.keywords or []
        # 去重关键词必须包含原始 tag_list, LLM 补充词作为额外标签
        merged_keywords = list(set(tag_list + llm_keywords))
        record = MemoryRecord(
            user_id=user_id,
            type=MemoryType.IMPLICIT,
            content=result.insight,
            structured={
                "signal_type": signal_type,
                "keywords": merged_keywords,
                "evidence_count": len(items),
                "window_days": window_days,
            },
            importance=0.55,
            confidence_score=result.confidence,
            source_type=SourceType.INFERRED.value,
            source="inferred",
            tags=["implicit", "pattern"] + tag_list,
        )
        await vec.add(record)
        await meta.upsert_memory(record)
        new_records.append(record)
        metrics.incr("pattern.implicit_created")
        logger.info(
            f"[PatternMiner] 新增 Implicit: '{result.insight[:50]}' "
            f"(signal={signal_type}, count={len(items)}, conf={result.confidence:.2f})"
        )

    return new_records
