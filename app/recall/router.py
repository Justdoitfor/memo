"""Hybrid Recall Router — 4 信号融合 + 实体感知统一召回入口

流程:
  1. 并行执行: (a) query 向量化 + ChromaDB 召回  (b) BM25 FTS5 召回
  2. 合并候选池, BM25 补充向量未命中的记录
  3. P0: 实体感知 — 识别查询中的已知实体, 对包含这些实体的记忆加权
  4. 对每个候选计算 4 信号, 加权融合 → final_score
  5. 阈值过滤 + 重排 + 截 Top-K
  6. fire-and-forget 异步更新 last_recalled_at / recall_count
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger

from app.config import config
from app.models import (
    MemoryRecord,
    MemoryType,
    RecallResult,
    RecallSignals,
)
from app.recall.signals import (
    compute_importance,
    compute_temporal_decay,
    compute_vector_sim,
    fuse_signals,
)
from app.storage import get_kg, get_vector_store
from app.utils.metrics import metrics

# 召回候选数 = top_k × OVERSAMPLE, 大网捞了再重排, 提高最终质量
_OVERSAMPLE = 3

# 召回质量阈值 — 基于纯 vector_sim (非 final_score, 因后者被 temporal/importance 稀释)
# bge-small-zh 中文嵌入基线相似度普遍偏高 (0.55+), 0.65 是个经验拐点:
#   - vector_sim >= 0.65 → 大概率真相关
#   - vector_sim <  0.65 → "找不到强相关, 但硬要的话最像的是这条"
# 业务方可在 search() 显式传 score_threshold=0.0 关闭过滤拿到全部候选 (调试用).
_DEFAULT_VECTOR_THRESHOLD = 0.65

# P0: 实体感知加权系数 — 当查询中识别到已知实体, 且记忆的 structured 中包含
# 该实体 (subject/object), 额外加分. 值较低避免过度压制纯语义匹配.
_ENTITY_BOOST = 0.08


class HybridRecallRouter:
    """统一召回入口 — 业务方应只调用 search(), 不要直接访问 VectorStore."""

    def __init__(self) -> None:
        self._vector = get_vector_store()
        self._kg = get_kg()
        logger.info(
            f"HybridRecall 权重: vec={config.recall_w_vector} "
            f"temp={config.recall_w_temporal} keyword={config.recall_w_keyword} "
            f"imp={config.recall_w_importance}"
        )

    async def search(
        self,
        user_id: str,
        query: str,
        memory_types: list[MemoryType] | None = None,
        top_k: int | None = None,
        weights: tuple[float, float, float, float] | None = None,
        score_threshold: float | None = None,
        valid_at: datetime | None = None,
    ) -> list[RecallResult]:
        """主入口.

        Args:
            score_threshold: final_score 低于此值的结果被过滤. None 用 config 默认,
                显式传 0.0 可拿到所有候选 (调试用).
            valid_at: 时间过滤 — 仅返回该时刻有效的 Semantic 事实
                (要求 valid_from <= valid_at <= valid_until).
                None 时返回所有候选 (默认行为, 与历史兼容).
                适用场景: "用户 2024 住哪？" — 业务方传 valid_at=datetime(2024,1,1).
                只对 SEMANTIC 类型生效, Episodic 是历史事件本身, 不参与时间过滤.
        """
        top_k = top_k or config.default_top_k
        threshold = score_threshold if score_threshold is not None else _DEFAULT_VECTOR_THRESHOLD
        type_strs = [t.value for t in memory_types] if memory_types else None

        with metrics.timer("recall.total.latency"):
            # 1. 并行召回: 向量 + BM25 (embedding 已异步化, 可真正并行)
            async def _vector_search():
                return await self._vector.search(
                    user_id=user_id,
                    query=query,
                    memory_types=type_strs,
                    top_k=top_k * _OVERSAMPLE,
                )

            async def _bm25_search():
                bm25_results = []
                try:
                    from app.storage.fts_store import get_fts_store
                    fts = get_fts_store()
                    bm25_results = fts.search(
                        user_id=user_id, query=query,
                        memory_types=type_strs, top_k=top_k * _OVERSAMPLE,
                    )
                except Exception as e:
                    logger.debug(f"BM25 召回失败 (降级仅向量): {e}")
                return bm25_results

            candidates, bm25_results = await asyncio.gather(
                _vector_search(), _bm25_search(),
            )
            bm25_map: dict[str, float] = {mid: sim for mid, sim in bm25_results}

            # 补 BM25 找到但向量未召回的记忆
            if bm25_map:
                seen_ids = {r.id for r, _ in candidates}
                missing_ids = [mid for mid in bm25_map if mid not in seen_ids]
                if missing_ids:
                    extra = await self._fetch_records_by_ids(user_id, missing_ids)
                    candidates.extend([(r, 0.3) for r in extra])

            if not candidates:
                return []

            # 2. P0: 实体感知 — 识别查询中的已知实体, 获取其 KG 邻居
            query_entity_names = await self._resolve_query_entities(user_id, query)
            user_neighbors: set[str] = set()
            for ent in query_entity_names:
                user_neighbors |= await self._kg.neighbors(user_id, ent, max_hops=2)

            # 3. 算分 + 融合 — 4 信号: 向量 + 时间 + BM25 + importance(含 effective_strength)
            #    + P0 实体加权
            #    同时过滤已失效的 VERSIONED triple (Zep 风格时间窗口, P3-2)
            now = datetime.now()
            scored: list[RecallResult] = []
            for record, raw_sim in candidates:
                bm25_score = bm25_map.get(record.id, 0.0)

                # Zep 风格: valid_until < now 的 VERSIONED triple 大幅降权而非硬删
                #   — 硬删会导致"当时住在北京"这类有价值的历史事实完全消失
                #   — 降权到 importance × 0.1 让它在"真的找不到其他信息时"仍可被召回
                #   — Agent 可通过 recall(query, score_threshold=0.0) 显式请求历史数据
                if self._is_expired_versioned(record, now):
                    bm25_score *= 0.1  # 大幅降低关键词匹配分数
                    # importance 也降权 (compute_importance 内 staleness × 0.2 会叠加)

                sig = RecallSignals(
                    vector_sim=compute_vector_sim(raw_sim),
                    temporal_decay=compute_temporal_decay(record.created_at, now=now),
                    keyword_match=bm25_score,  # BM25 关键词匹配分数
                    importance=compute_importance(record),
                )
                sig.final_score = fuse_signals(
                    sig.vector_sim,
                    sig.temporal_decay,
                    sig.keyword_match,
                    sig.importance,
                    weights=weights,
                )

                # P0: 实体感知加权 — 如果查询包含已知实体, 且该记忆的 structured
                # 中包含对应实体 (subject/object), 额外加分
                if query_entity_names:
                    entity_boost = self._compute_entity_boost(
                        record, query_entity_names, user_neighbors,
                    )
                    sig.final_score += entity_boost

                scored.append(RecallResult(record=record, signals=sig))

            # 4. 阈值过滤 (基于 vector_sim) + 重排 + 截断
            #    用 vector_sim 而非 final_score 做阈值, 避免 importance/temporal 加权
            #    把"语义相似度=0.5 但 importance=1.0"的不相关高分项误留.
            if threshold > 0:
                before = len(scored)
                scored = [r for r in scored if r.signals.vector_sim >= threshold]
                if not scored and before > 0:
                    logger.debug(
                        f"召回全部 {before} 条 vector_sim 均低于 {threshold:.2f}, 返回空"
                    )

            # 4.5. valid_at 时间过滤 (Graphiti 风格) — 仅 Semantic 事实参与.
            #      Agent 用例: "用户 2024 住哪？" → valid_at=2024-01-01
            if valid_at is not None:
                before = len(scored)
                scored = [r for r in scored if self._is_valid_at(r.record, valid_at)]
                if before and not scored:
                    logger.debug(
                        f"valid_at={valid_at.isoformat()} 过滤后无结果 "
                        f"(候选 {before} 条都不在该时间窗口)"
                    )

            scored.sort(key=lambda r: r.signals.final_score, reverse=True)
            top = scored[:top_k]
            for i, r in enumerate(top, 1):
                r.rank = i

        # 5. fire-and-forget 异步更新 last_recalled_at / recall_count (不阻塞返回)
        if top:
            asyncio.create_task(
                self._batch_update_recall_meta(top, user_id, now)
            )

        metrics.incr("recall.invocations")
        return top

    async def _batch_update_recall_meta(
        self, results: list[RecallResult], user_id: str, now: datetime
    ) -> None:
        """后台批量更新召回元数据 — fire-and-forget, 不阻塞返回."""
        for r in results:
            try:
                await self._vector.update_metadata(
                    r.record.id,
                    user_id,
                    {
                        "recall_count": r.record.recall_count + 1,
                        "last_recalled_at_iso": now.isoformat(),
                    },
                )
            except Exception:
                pass

    async def _fetch_records_by_ids(
        self, user_id: str, memory_ids: list[str]
    ) -> list[MemoryRecord]:
        """BM25 命中但向量未召回的记录 — batch 查询, 避免 N+1."""
        from app.storage import get_metadata
        meta = get_metadata()
        records = await meta.batch_get_memories(memory_ids)
        return [r for r in records if r.user_id == user_id]

    async def _resolve_query_entities(
        self, user_id: str, query: str,
    ) -> set[str]:
        """P0: 从查询中识别已知实体名 (含别名).

        优先走 Entity Store 精确匹配, 匹配到的实体名 (规范名 + 别名) 都返回.
        降级: 空格切分 + 过滤短词 (原有逻辑).
        """
        words = {w.strip("，。!?,.!?;:") for w in query.split() if len(w.strip("，。!?,.!?;:")) >= 2}
        if not words:
            return set()

        resolved_names: set[str] = set()
        try:
            from app.storage import get_metadata
            meta = get_metadata()
            for word in words:
                entity = await meta.find_entity_by_name(user_id, word)
                if entity:
                    # 加入规范名和所有别名, 扩大匹配面
                    resolved_names.add(entity.name)
                    resolved_names.update(entity.aliases)
        except Exception as e:
            logger.debug(f"Entity resolution in recall failed: {e}")

        # 始终包含原始词 (兼容无 Entity 的场景)
        resolved_names.update(words)
        return resolved_names

    @staticmethod
    def _compute_entity_boost(
        record: MemoryRecord,
        query_entity_names: set[str],
        user_neighbors: set[str],
    ) -> float:
        """P0: 计算实体感知加权.

        - 记忆的 structured 中包含查询实体 → +ENTITY_BOOST
        - 记忆与查询实体在 KG 中 2-hop 相邻 → +ENTITY_BOOST × 0.5
        """
        structured = record.structured or {}
        subj = structured.get("subject", "")
        obj = structured.get("object", "")

        # 直接包含查询中的实体
        if subj in query_entity_names or obj in query_entity_names:
            return _ENTITY_BOOST

        # KG 邻居实体 (2-hop)
        if subj in user_neighbors or obj in user_neighbors:
            return _ENTITY_BOOST * 0.5

        return 0.0

    @staticmethod
    def _is_valid_at(record: MemoryRecord, valid_at: datetime) -> bool:
        """判断记忆在 valid_at 时刻是否有效 (Graphiti 双时间轴语义).

        规则:
          - 仅 SEMANTIC 类型参与时间过滤. Episodic/Procedural 等是历史事件本身,
            不存在"那时候是否成立"的问题
          - valid_from 缺省视为 -∞, valid_until 缺省视为 +∞
          - 时间戳格式损坏时保守视为有效 (不漏掉数据是默认)
        """
        if record.type != MemoryType.SEMANTIC:
            return True
        s = record.structured or {}
        try:
            vf = s.get("valid_from")
            if vf and datetime.fromisoformat(str(vf)) > valid_at:
                return False
            vu = s.get("valid_until")
            if vu and datetime.fromisoformat(str(vu)) < valid_at:
                return False
        except (TypeError, ValueError):
            return True
        return True

    @staticmethod
    def _is_expired_versioned(record: MemoryRecord, now: datetime) -> bool:
        """判断 Semantic 记录是否是已失效的 VERSIONED triple (Zep 风格时间窗口).

        valid_until < now 的 triple 表示"从某时间起此事实不再成立", 默认大幅降权.
        """
        if record.type != MemoryType.SEMANTIC:
            return False
        structured = record.structured or {}
        # ChromaDB mirror 的 structured 中存储了 triple 的 valid_until
        valid_until_str = structured.get("valid_until")
        if not valid_until_str:
            return False
        try:
            valid_until = datetime.fromisoformat(str(valid_until_str))
            return valid_until < now
        except (TypeError, ValueError):
            return False


# 全局单例
recall_router = HybridRecallRouter()
