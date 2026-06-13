"""Entity Resolver — 实体消解器

流程:
  1. 精确名/别名匹配 — 快速路径, O(N) 扫 SQLite aliases
  2. 向量相似检索 — 在 ChromaDB 中搜索相似实体名 (threshold=0.85)
  3. LLM 判断 — 给 LLM 候选列表 + 上下文, 判断是否同一实体
  4. 合并或创建 — 匹配则合并别名; 不匹配则新建实体

参考 Zep/Graphiti 的 Entity Resolution Prompt, 但简化为单次 LLM 调用.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.core.embedder import async_embed_text
from app.core.llm_factory import llm_factory
from app.models import Entity, ResolvedEntity
from app.storage import get_metadata, get_vector_store
from app.utils.metrics import metrics


# ── LLM 判断 Schema ────────────────────────────────────────────────────


class _ResolutionDecision(BaseModel):
    """LLM 判断新实体是否与已有候选匹配."""

    matches_existing: bool = Field(
        description="是否与某个候选实体匹配 (同一人/地点/组织)"
    )
    matched_entity_name: str | None = Field(
        default=None,
        description="匹配的候选实体规范名称 (matches_existing=True 时必填)",
    )
    reasoning: str = Field(
        description="判断理由 (一句话)",
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="匹配置信度",
    )


_RESOLVE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是一个实体消解器. 判断新出现的实体名是否与已有实体是同一个.\n"
                "\n"
                "判断原则:\n"
                "- 同一人的不同称呼 (昵名/全名/英文名/职称) → 匹配\n"
                "  e.g. '小明' vs '张小明' → 匹配 (同一人)\n"
                "  e.g. 'Alice' vs 'Alice Wang' → 匹配\n"
                "- 同一地点的不同表述 → 匹配\n"
                "  e.g. '上海' vs '魔都' → 不匹配 (不同名, 需更多上下文)\n"
                "  e.g. '北京' vs 'BJ' → 可能匹配\n"
                "- 不同的人/地点/组织 → 不匹配\n"
                "  e.g. '小明' vs '小红' → 不匹配\n"
                "  e.g. '字节' vs '腾讯' → 不匹配\n"
                "- 实体类型不同 → 不匹配\n"
                "  e.g. '小明' (person) vs '小明咖啡' (organization) → 不匹配\n"
                "\n"
                "上下文信息很重要:\n"
                "- 如果上下文暗示它们指同一对象, 可以匹配\n"
                "- 如果缺乏足够信息, 宁可不匹配 (保守原则)\n"
                "\n"
                "返回 JSON: {\"matches_existing\": bool, \"matched_entity_name\": str|null, "
                "\"reasoning\": \"一句话\", \"confidence\": 0.0-1.0}"
            ),
        ),
        (
            "human",
            (
                "新实体: {new_name} (类型: {new_type})\n"
                "上下文: {context}\n"
                "\n"
                "已有候选实体:\n{candidates}"
                "\n请判断新实体是否与某个候选匹配."
            ),
        ),
    ]
)


class EntityResolver:
    """实体消解器 — 判断新实体名是否与已有实体相同."""

    def __init__(self) -> None:
        self._meta = get_metadata()
        self._vector = get_vector_store()
        logger.info("EntityResolver 初始化")

    async def resolve(
        self,
        user_id: str,
        new_name: str,
        entity_type: str = "person",
        context: str = "",
    ) -> ResolvedEntity:
        """消解一个新实体名.

        Args:
            user_id: 用户标识
            new_name: 新出现的实体名 (e.g. "小明")
            entity_type: 实体类型 (person/location/organization/product/concept)
            context: 上下文描述 (e.g. "用户说'我和小明一起吃饭'")

        Returns:
            ResolvedEntity — 消解后的实体引用
        """
        # ── Phase 1: 精确名/别名匹配 (快速路径) ──
        with metrics.timer("entity.resolve.fast_path"):
            exact_match = await self._meta.find_entity_by_name(user_id, new_name)
            if exact_match:
                # 别名已存在, 直接匹配
                logger.debug(
                    f"Entity exact match: '{new_name}' → {exact_match.name} (id={exact_match.id})"
                )
                metrics.incr("entity.resolve.exact_match")
                # 如果新名不在别名列表中, 添加
                if new_name != exact_match.name and new_name not in exact_match.aliases:
                    exact_match.aliases.append(new_name)
                    await self._meta.upsert_entity(exact_match)
                return ResolvedEntity(
                    entity_id=exact_match.id,
                    name=new_name,
                    is_new=False,
                    canonical_name=exact_match.name,
                )

        # ── Phase 2: 向量相似检索 + LLM 判断 ──
        candidates = await self._find_similar_candidates(user_id, new_name)

        if not candidates:
            # 无相似候选, 创建新实体
            new_entity = Entity(
                user_id=user_id,
                name=new_name,
                aliases=[new_name],
                entity_type=entity_type,
            )
            await self._meta.upsert_entity(new_entity)
            # 写入向量库供后续相似检索
            await self._index_entity_vector(new_entity)
            metrics.incr("entity.resolve.new_entity")
            logger.info(f"Entity new: '{new_name}' (id={new_entity.id}, type={entity_type})")
            return ResolvedEntity(
                entity_id=new_entity.id,
                name=new_name,
                is_new=True,
                canonical_name=new_name,
            )

        # ── Phase 3: LLM 判断是否匹配 ──
        with metrics.timer("entity.resolve.llm_judge"):
            decision = await self._llm_judge(new_name, entity_type, context, candidates)

        if decision.matches_existing and decision.matched_entity_name:
            # 匹配: 找到对应实体, 合并别名
            matched = None
            for c in candidates:
                if c.name == decision.matched_entity_name:
                    matched = c
                    break
            if matched is None:
                # LLM 给的名称可能与候选不完全一致, 取第一个
                matched = candidates[0]

            # 合并别名
            if new_name != matched.name and new_name not in matched.aliases:
                matched.aliases.append(new_name)
            await self._meta.upsert_entity(matched)
            metrics.incr("entity.resolve.llm_match")
            logger.info(
                f"Entity resolved: '{new_name}' → {matched.name} "
                f"(id={matched.id}, conf={decision.confidence:.2f})"
            )
            return ResolvedEntity(
                entity_id=matched.id,
                name=new_name,
                is_new=False,
                canonical_name=matched.name,
            )

        # 不匹配: 创建新实体
        new_entity = Entity(
            user_id=user_id,
            name=new_name,
            aliases=[new_name],
            entity_type=entity_type,
        )
        await self._meta.upsert_entity(new_entity)
        await self._index_entity_vector(new_entity)
        metrics.incr("entity.resolve.llm_no_match")
        logger.info(f"Entity new (after LLM no-match): '{new_name}' (id={new_entity.id})")
        return ResolvedEntity(
            entity_id=new_entity.id,
            name=new_name,
            is_new=True,
            canonical_name=new_name,
        )

    async def merge_entities(
        self, user_id: str, primary_id: str, secondary_id: str
    ) -> Entity | None:
        """合并两个实体: 将 secondary 的别名和三元组指向 primary.

        用于手动合并或后台 consolidation.
        """
        primary = await self._meta.get_entity(primary_id)
        secondary = await self._meta.get_entity(secondary_id)
        if not primary or not secondary:
            return None
        if primary.user_id != user_id or secondary.user_id != user_id:
            return None

        # 合并别名
        for alias in secondary.aliases:
            if alias not in primary.aliases and alias != primary.name:
                primary.aliases.append(alias)
        if secondary.name not in primary.aliases and secondary.name != primary.name:
            primary.aliases.append(secondary.name)

        # 更新 primary
        await self._meta.upsert_entity(primary)

        # 删除 secondary
        await self._meta.delete_entity(secondary_id)

        logger.info(
            f"Entity merge: secondary '{secondary.name}' (id={secondary_id}) "
            f"→ primary '{primary.name}' (id={primary_id})"
        )
        return primary

    async def _find_similar_candidates(
        self, user_id: str, new_name: str, top_k: int = 5
    ) -> list[Entity]:
        """向量相似检索 — 找名称相似但非精确匹配的实体候选."""
        try:
            # 在 ChromaDB 中搜索相似实体名
            results = await self._vector.search(
                user_id=user_id,
                query=new_name,
                memory_types=["semantic"],
                top_k=top_k,
                score_threshold=0.75,
            )
            # 从结果中提取实体名, 查对应 Entity
            candidate_names: set[str] = set()
            for record, sim in results:
                structured = record.structured or {}
                # triple mirror 的 structured 中有 subject/object
                subj = structured.get("subject", "")
                obj = structured.get("object", "")
                if subj != "user":
                    candidate_names.add(subj)
                if obj and obj != "user":
                    candidate_names.add(obj)

            if not candidate_names:
                return []

            # 查 Entity 表
            entities = await self._meta.list_entities(user_id)
            return [e for e in entities if e.name in candidate_names or
                    any(a in candidate_names for a in e.aliases)]

        except Exception as e:
            logger.warning(f"Entity vector search failed: {e}")
            return []

    async def _llm_judge(
        self,
        new_name: str,
        new_type: str,
        context: str,
        candidates: list[Entity],
    ) -> _ResolutionDecision:
        """LLM 判断新实体是否与候选匹配."""
        candidates_text = "\n".join(
            f"  - {c.name} (类型: {c.entity_type}, 别名: {', '.join(c.aliases[:5])})"
            for c in candidates[:5]  # 限制候选数量
        )

        try:
            result = await llm_factory.structured_invoke(
                _RESOLVE_PROMPT,
                _ResolutionDecision,
                {
                    "new_name": new_name,
                    "new_type": new_type,
                    "context": context or "(无额外上下文)",
                    "candidates": candidates_text,
                },
                temperature=0,
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"Entity Resolution LLM failed: {e}")

        # LLM 失败时保守不匹配
        return _ResolutionDecision(
            matches_existing=False,
            matched_entity_name=None,
            reasoning="LLM 不可用, 保守创建新实体",
            confidence=0.3,
        )

    async def _index_entity_vector(self, entity: Entity) -> None:
        """将实体名写入向量库, 供后续相似检索."""
        from app.models import MemoryRecord, MemoryType

        # 用实体名作为向量索引内容, type 标记为 semantic (借用语义记忆的向量库)
        record = MemoryRecord(
            id=f"entity_{entity.id}",
            user_id=entity.user_id,
            type=MemoryType.SEMANTIC,
            content=entity.name,
            structured={
                "entity_id": entity.id,
                "entity_type": entity.entity_type,
                "aliases": entity.aliases,
            },
            importance=0.6,
            source="entity_resolution",
        )
        try:
            await self._vector.add(record)
            await self._meta.upsert_memory(record)
        except Exception as e:
            logger.warning(f"Entity vector indexing failed: {e}")


# 全局单例
entity_resolver = EntityResolver()
