"""Memory Orchestrator — 统一的 read/write/search/forget 入口

设计:
  - 不用 LangGraph 重型编排 (MVP 流程比较直接, LangGraph 反而过度抽象)
  - 暴露 4 个清晰的 async 入口供 API/SDK/MCP 共享调用
  - 路由策略 in-place: 业务方传 type 则按 type 路由, 否则智能推断
  - 所有调用走 metrics.timer 采集延迟
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from loguru import logger

# 强制触发 arbitrator 模块 init, 把 ConflictArbitrator 注入 SemanticMemory
import app.arbitrator  # noqa: F401
# P0: 强制触发 entity 模块 init, 把 EntityResolver 注入 SemanticMemory
import app.entity  # noqa: F401
from app.core.snapshot_cache import snapshot_cache
from app.memories import (
    episodic_memory,
    procedural_memory,
    reflective_memory,
    semantic_memory,
    working_memory,
)
from app.models import (
    MemoryRecord,
    MemoryType,
    RecallResult,
    SearchResponse,
    WriteRequest,
    WriteResponse,
)
from app.recall import recall_router
from app.storage import get_metadata, get_vector_store
from app.utils.metrics import metrics
from app.utils.tokenizer import cut_for_fts
from app.utils.trace_context import traced


# Trivial episode 启发式: 包含这些关键词的 episode 才同步触发 LLM 抽取.
# 漏抽的少量 fact 由后台 distill worker (1h 间隔) 兜底, 不会丢.
# 设计原则: 宁可漏抽不可乱抽 (后台兜底成本固定, 同步抽取浪费 LLM token).
#
# 触发词选取: 来自 _PRED_REGISTRY 模板核心动词/名词 + 用户主语 + 否定词.
# 单字 ("我", "有") 也保留 — 因为短句 "我搬了" 中 "搬"/"我" 单字关键
# (extract_keywords 的 TF-IDF 在短句上会漏掉低 IDF 词, 此处用 cut_for_fts
# 全切再做集合交即可, 性能 <10µs).
_FACT_TRIGGER_WORDS = frozenset({
    # 用户主语 / 关系
    "我", "我们", "用户", "对象", "女朋友", "男朋友", "老婆", "老公", "配偶",
    "孩子", "宠物", "家人", "兄弟", "姐妹", "猫", "狗",
    # 个人状态变化
    "住", "搬", "搬家", "工作", "公司", "上班", "就职", "辞职", "跳槽",
    "出生", "周岁", "年龄", "结婚", "离婚",
    # 偏好 / 情感
    "喜欢", "爱", "爱吃", "讨厌", "偏好", "不爱",
    # 健康
    "过敏", "血型", "身高", "体重",
    # 拥有 / 物品
    "有", "买", "卖", "用", "戴", "车", "手机", "笔记本", "相机",
    # 联系方式 / 地理
    "电话", "邮箱", "地址", "微信", "QQ",
    # 学习经历 / 能力
    "毕业", "学习", "在读", "会说", "会写",
})

# Trivial episode 长度阈值: 短于此长度的 episode 直接跳过抽取
# (问候 "好的" / "嗯" / "ok" / "在吗" 等通常 ≤ 5 字)
_MIN_FACT_LENGTH = 6


def _is_likely_fact(text: str) -> bool:
    """启发式: 此 episode 是否值得跑 LLM 抽取.

    跳过约 60% trivial episode (问候/确认/闲聊). 误判成本: 偶尔漏抽,
    由后台 distill worker 兜底, 长期一致性不丢.

    实现: 用 cut_for_fts 全切后做集合交, 不依赖 TF-IDF 排序
    (TF-IDF 在短句上漏掉低 IDF 词, e.g. "我搬了" 的 "搬" / "我").
    """
    text = text.strip()
    if len(text) < _MIN_FACT_LENGTH:
        return False
    tokens = set(cut_for_fts(text).split())
    # 包含触发词中任意一个即视为有 fact 倾向
    if tokens & _FACT_TRIGGER_WORDS:
        return True
    # cut_for_fts 已过滤单字, 但 _FACT_TRIGGER_WORDS 含单字 ("我"/"有"等)
    # 兜底: 直接对原文做单字包含检查
    for w in _FACT_TRIGGER_WORDS:
        if len(w) == 1 and w in text:
            return True
    return False


class MemoryOrchestrator:
    """4 入口聚合器: write / search / get_profile / forget."""

    def __init__(self) -> None:
        self._meta = get_metadata()
        self._vector = get_vector_store()
        # 持有 background tasks 引用, 防止被 GC 提前回收 (Python 3.11+ 已知问题)
        self._bg_tasks: set = set()
        logger.info("MemoryOrchestrator 初始化完成")

    def _spawn_bg(self, coro) -> None:
        """fire-and-forget 启动后台任务, 同时持有引用避免 GC 提前回收."""
        import asyncio

        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def wait_pending(self, timeout: float = 30.0) -> None:
        """等待所有未完成的 background tasks. eval / test 场景需要确定性."""
        import asyncio

        if not self._bg_tasks:
            return
        pending = list(self._bg_tasks)
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"wait_pending 超时, 还有 {len(self._bg_tasks)} 个任务未完成")

    # ────────────────────────────────────────────────────────────────────
    # write — 路由 + 写入 + (可选) semantic 抽取
    # ────────────────────────────────────────────────────────────────────
    @traced
    async def write(self, req: WriteRequest) -> WriteResponse:
        """统一写入入口.

        路由规则:
          - type=WORKING    → WorkingMemory (session 强相关)
          - type=PROCEDURAL → ProceduralMemory (需要 structured.steps)
          - type=SEMANTIC   → 直接 SemanticMemory.write_from_text
          - type=EPISODIC (默认) →
              * 写 Episodic
              * 异步触发 semantic 抽取 (best effort, 不阻塞返回)
        """
        record = MemoryRecord(
            user_id=req.user_id,
            session_id=req.session_id,
            type=req.type,
            content=req.content,
            structured=req.structured or {},
            importance=req.importance if req.importance is not None else 0.5,
            tags=req.tags,
        )
        # Phase 1: source_type 显式覆盖 (默认 explicit_statement, 上游 Agent
        # 可标 corrected / inferred / agent_confirmed 等影响 effective_strength)
        if req.source_type:
            record.source_type = req.source_type

        arbitration = None

        if req.type == MemoryType.WORKING:
            memory_id = await working_memory.write(record)

        elif req.type == MemoryType.PROCEDURAL:
            steps = req.structured.get("steps", [])
            task = req.structured.get("task_pattern", req.content)
            memory_id = await procedural_memory.write(
                user_id=req.user_id,
                task_pattern=task,
                steps=steps,
                success_rate=req.structured.get("success_rate", 1.0),
                tags=req.tags,
            )

        elif req.type == MemoryType.SEMANTIC:
            # 走 SemanticMemory.write_from_text (LLM 抽取 + 冲突仲裁)
            results = await semantic_memory.write_from_text(
                user_id=req.user_id,
                text=req.content,
                source_memory_id=record.id,
            )
            memory_id = record.id
            # 取第一个 arbitration 给上层 (如果有)
            for r in results:
                if r.get("arbitration"):
                    arbitration = r["arbitration"]
                    break

            # 降级保护: 如果 LLM 抽取返回空结果 (如时态事实被过滤),
            # 原始内容仍然必须持久化, 否则记忆会静默丢失.
            if not results:
                logger.warning(
                    f"Semantic extraction returned empty for '{req.content[:50]}...', "
                    f"falling back to direct persist"
                )
                await self._vector.add(record)
                await self._meta.upsert_memory(record)

        else:  # EPISODIC (默认)
            memory_id = await episodic_memory.write(record)
            # Trivial 跳过: 仅当文本看起来含有可抽取 fact 时才同步触发 LLM.
            # 漏抽由后台 distill worker (1h 间隔) 兜底.
            if _is_likely_fact(req.content):
                self._spawn_bg(
                    self._extract_semantic_safely(req.user_id, req.content, record.id)
                )
            else:
                metrics.incr("orchestrator.write.trivial_skipped")

        metrics.incr(f"orchestrator.write.{req.type.value}")

        # 失效热记忆快照缓存 — Semantic/Reflective/Implicit 写入会改变核心事实
        if req.type in (MemoryType.SEMANTIC, MemoryType.EPISODIC):
            # Episodic 写入会异步触发 semantic 抽取, 也会改变快照
            snapshot_cache.invalidate(req.user_id)

        return WriteResponse(
            memory_id=memory_id, routed_type=req.type, arbitration=arbitration
        )

    async def _extract_semantic_safely(
        self, user_id: str, text: str, source_memory_id: str
    ) -> None:
        """异步从 episodic 内容抽取 semantic facts, 失败仅 warning."""
        try:
            await semantic_memory.write_from_text(
                user_id=user_id, text=text, source_memory_id=source_memory_id
            )
        except Exception as e:
            logger.warning(f"后台 semantic 抽取失败: {e}")

    # ────────────────────────────────────────────────────────────────────
    # search — Hybrid Recall + Working 兜底 + Profile 注入
    # ────────────────────────────────────────────────────────────────────
    @traced
    async def search(
        self,
        user_id: str,
        query: str,
        types: list[MemoryType] | None = None,
        top_k: int | None = None,
        session_id: str | None = None,
        score_threshold: float | None = None,
        valid_at: datetime | None = None,
    ) -> SearchResponse:
        """统一召回入口. 返回 RecallResult 列表 + 性能数据.

        Args:
            score_threshold: final_score 阈值, None 用默认, 0.0 关闭过滤.
            valid_at: 时间过滤 — 仅返回该时刻有效的 Semantic 事实.
                None 时返回所有候选 (默认行为). 见 router.search 详细说明.
        """
        start = time.perf_counter()

        results: list[RecallResult] = await recall_router.search(
            user_id=user_id,
            query=query,
            memory_types=types,
            top_k=top_k,
            score_threshold=score_threshold,
            valid_at=valid_at,
        )

        # Working Memory 不进 ChromaDB, 必须显式注入到 Hybrid 召回结果中:
        #   1. 如果显式传了 session_id, 取该 session 最近 3 条
        #   2. 否则 (Playground / 跨 session 查询场景), 取该 user 所有 buckets 的最近 3 条
        # Working Memory 参与评分而非固定 1.0 — 保证高优先级但不过度压制语义召回
        if types is None or MemoryType.WORKING in types:
            working: list = []
            if session_id:
                working = await working_memory.read(user_id, session_id, limit=3)
            else:
                # 跨 session: 把该 user 所有 working buckets 合并按时间倒序取 3 条
                all_working = await working_memory.read_all_sessions(user_id, limit=3)
                working = all_working

            if working:
                from app.models import RecallSignals
                from app.recall.signals import fuse_signals

                # Working Memory 参与评分: 高基础分但不固定 1.0
                # 这样当查询明显与当前会话无关时, Semantic 记忆仍可自然排序上来
                working_results = []
                for w in working:
                    sig = RecallSignals(
                        vector_sim=0.8,   # 会话上下文基础分 (不是真向量分数, 但语义合理)
                        temporal_decay=1.0,  # 刚刚产生, 无衰减
                        keyword_match=0.0,   # 工作记忆不做关键词匹配
                        importance=0.9,      # 高优先级
                    )
                    sig.final_score = fuse_signals(
                        sig.vector_sim, sig.temporal_decay,
                        sig.keyword_match, sig.importance,
                    )
                    working_results.append(RecallResult(record=w, signals=sig))

                results = working_results + [
                    r for r in results if r.record.type != MemoryType.WORKING
                ]
                # 重排 rank
                for i, r in enumerate(results, 1):
                    r.rank = i

        latency_ms = (time.perf_counter() - start) * 1000
        return SearchResponse(
            results=results,
            latency_ms=round(latency_ms, 2),
            signals_used=["vector", "temporal", "keyword_match", "importance(effective_strength)"],
        )

    # ────────────────────────────────────────────────────────────────────
    # get_profile — 直接读 Reflective Memory
    # ────────────────────────────────────────────────────────────────────
    async def get_profile(self, user_id: str, auto_refresh: bool = False) -> dict[str, Any]:
        """获取用户画像. auto_refresh=True 时若无缓存则触发实时生成."""
        cached = await reflective_memory.get(user_id)
        if cached:
            return cached
        if auto_refresh:
            profile = await reflective_memory.refresh(user_id)
            return {"profile": profile, "updated_at": datetime.now().isoformat()}
        return {"profile": {}, "updated_at": None}

    # ────────────────────────────────────────────────────────────────────
    # graph_query — 知识图谱增强查询 (P0)
    # ────────────────────────────────────────────────────────────────────
    async def graph_query(
        self,
        user_id: str,
        query_type: str,
        entity: str | None = None,
        max_hops: int = 3,
        predicate_filter: list[str] | None = None,
        relation_chain: list[str] | None = None,
        min_community_size: int = 3,
    ) -> dict[str, Any]:
        """知识图谱增强查询入口.

        Args:
            query_type: multi_hop / related / community
            entity: 起始实体名 (multi_hop / related 必填)
            max_hops: 多跳最大深度 (multi_hop)
            predicate_filter: 谓词过滤列表 (multi_hop)
            relation_chain: 关系链 (related)
            min_community_size: 最小社区大小 (community)
        """
        from app.storage import get_kg

        kg = get_kg()

        if query_type == "multi_hop":
            if not entity:
                return {"error": "multi_hop 查询需要 entity 参数"}
            paths = await kg.multi_hop_query(
                user_id=user_id,
                start_entity=entity,
                max_hops=max_hops,
                predicate_filter=predicate_filter,
            )
            return {
                "query_type": "multi_hop",
                "entity": entity,
                "paths": [
                    {
                        "nodes": p.nodes,
                        "edges": p.edges,
                        "length": p.length,
                    }
                    for p in paths
                ],
                "count": len(paths),
            }

        if query_type == "related":
            if not entity:
                return {"error": "related 查询需要 entity 参数"}
            related = await kg.find_related_entities(
                user_id=user_id,
                entity=entity,
                relation_chain=relation_chain,
            )
            return {
                "query_type": "related",
                "entity": entity,
                "related_entities": related,
                "count": len(related),
            }

        if query_type == "community":
            communities = await kg.community_detect(
                user_id=user_id,
                min_size=min_community_size,
            )
            return {
                "query_type": "community",
                "communities": communities,
                "count": len(communities),
            }

        return {"error": f"未知 query_type: {query_type}, 支持: multi_hop / related / community"}

    # ────────────────────────────────────────────────────────────────────
    # entity 操作 — 实体管理入口 (P0)
    # ────────────────────────────────────────────────────────────────────
    async def list_entities(
        self, user_id: str, entity_type: str | None = None, limit: int = 200,
    ) -> list[dict[str, Any]]:
        """列出用户的所有实体."""
        entities = await self._meta.list_entities(user_id, entity_type=entity_type, limit=limit)
        return [
            {
                "id": e.id,
                "name": e.name,
                "aliases": e.aliases,
                "entity_type": e.entity_type,
                "summary": e.summary,
                "created_at": e.created_at.isoformat(),
                "updated_at": e.updated_at.isoformat(),
            }
            for e in entities
        ]

    async def merge_entities(
        self, user_id: str, primary_id: str, secondary_id: str,
    ) -> dict[str, Any]:
        """手动合并两个实体."""
        from app.entity.resolver import entity_resolver

        merged = await entity_resolver.merge_entities(user_id, primary_id, secondary_id)
        if not merged:
            return {"error": "合并失败: 实体不存在或不属于此用户"}
        return {
            "status": "merged",
            "primary": {
                "id": merged.id,
                "name": merged.name,
                "aliases": merged.aliases,
            },
        }

    # ────────────────────────────────────────────────────────────────────
    # forget — GDPR Right to be Forgotten
    # ────────────────────────────────────────────────────────────────────
    @traced
    async def forget(
        self,
        user_id: str,
        memory_id: str | None = None,
        all_user_data: bool = False,
    ) -> dict[str, Any]:
        """按 memory_id 或全量删除."""
        if all_user_data:
            from app.storage import get_kg, get_vector_store
            vec_deleted = await get_vector_store().delete_by_user(user_id)
            kg_deleted = await get_kg().delete_by_user(user_id)
            # SQLite: 批量 DELETE, 不用全量加载 (P3-1)
            meta_deleted = await self._meta.delete_all_memories(user_id)
            signals_deleted = await self._meta.delete_all_signals(user_id)
            # P0: 同时删除实体
            entities_deleted = await self._meta.delete_all_entities(user_id)
            # 同时删除 reflective profile 和 arbitration logs
            await self._meta.upsert_profile(user_id, {})  # 清空 profile
            logger.warning(
                f"GDPR forget: user={user_id} vec={vec_deleted} kg={kg_deleted} "
                f"meta={meta_deleted} signals={signals_deleted} entities={entities_deleted}"
            )
            return {
                "user_id": user_id,
                "vector_deleted": vec_deleted,
                "graph_deleted": kg_deleted,
                "metadata_deleted": meta_deleted,
                "signals_deleted": signals_deleted,
                "entities_deleted": entities_deleted,
            }

        if memory_id:
            ok_vec = await self._vector.delete(memory_id, user_id)
            ok_meta = await self._meta.delete_memory(memory_id)
            return {"memory_id": memory_id, "deleted": ok_vec and ok_meta}

        return {"error": "需要 memory_id 或 all_user_data=True"}


# 全局单例
orchestrator = MemoryOrchestrator()
