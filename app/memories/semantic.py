"""Semantic Memory — 事实知识 (用户偏好/属性, 三元组)

双索引设计:
  - NetworkX 知识图谱: (subject, predicate, object) triple → BFS/Cypher 风格查询
  - ChromaDB 向量库: 自然语言原文 → 语义召回 (兜底召回)

核心流程 (write):
  1. LLM Entity Extractor: 自然语言 → List[Triple]
  2. 对每个 triple, 查 KG 是否已有 (subject, predicate, *) → 检测冲突
  3. 有冲突 → 调 Arbitrator 决策 (REPLACE/MERGE/VERSIONED/IGNORE)
  4. 按决策应用到 KG + 双写 ChromaDB
  5. 全程日志到 arbitration_logs

冲突 Arbitrator 在 Phase 3 实现, 这里先实现 Entity Extractor + KG 同步.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.core.llm_factory import llm_factory
from app.models import MemoryRecord, MemoryType, Triple
from app.storage import get_kg, get_metadata, get_vector_store
from app.utils.metrics import metrics

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       LLM Entity Extractor                           ║
# ╚══════════════════════════════════════════════════════════════════════╝


class _ExtractedTriple(BaseModel):
    """LLM 输出的单个 triple (内部用)."""

    subject: str = Field(description="实体名, 用户相关事实通常是 'user'")
    predicate: str = Field(description="谓词, 小写下划线, e.g. lives_in / allergic_to / likes")
    object: str = Field(description="值, 实体名/字面值")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class _ExtractResult(BaseModel):
    """LLM 输出的完整结构: 多个 triples."""

    triples: list[_ExtractedTriple] = Field(default_factory=list)


_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是一个事实抽取器. 把用户对话中的客观事实抽取为 (subject, predicate, object) 三元组.\n"
                "\n"
                "规则:\n"
                "- subject: 与用户相关的事实统一用 'user'\n"
                "- predicate: 必须小写下划线英文. 常用 predicate 列表 (优先复用):\n"
                "  * 基本属性: age / occupation / gender / married_to / weight_kg / height_cm\n"
                "  * 地理: lives_in / works_at / hometown / visited\n"
                "  * 过去经历: worked_at / worked_in / lived_in / studied_at\n"
                "  * 未来计划: planned_move_to / will_work_at / will_visit / planned_event\n"
                "  * 关系: has_pet / has_child / girlfriend / boyfriend / spouse / sibling\n"
                "  * 偏好: likes / dislikes / favorite_food / favorite_color / hobby\n"
                "  * 健康: allergic_to / blood_type\n"
                "  * 物品: owns_car / owns_phone / owns_laptop / uses_camera\n"
                "  * 能力: speaks_language / educational_background\n"
                "  * 未在列表中的概念也可以新建 predicate, 保持英文小写下划线即可\n"
                "- object: 实体名或字面值 (单位用统一格式, e.g. '70 公斤' / 'iPhone 16 Pro')\n"
                "- 只抽取明确的事实, 不抽取猜测/疑问\n"
                "- 过去经历和未来计划都要抽取, 用对应时态 predicate 区分:\n"
                "  * 过去: worked_in / worked_at / lived_in / studied_at 等\n"
                "  * 未来: planned_move_to / will_work_at / will_visit 等\n"
                "  * 当前: lives_in / works_at 等\n"
                "- 否定偏好用 dislikes, 不要跳过 (e.g. '不爱吃芒果' → dislikes 芒果)\n"
                "- 同一句话可能产出多个三元组\n"
                "- 如果没有可抽取的事实, 返回空 triples 列表\n"
                "\n"
                "示例:\n"
                "  '我对花生过敏'              → [(user, allergic_to, 花生)]\n"
                "  '我搬家了, 现在住北京'       → [(user, lives_in, 北京)]\n"
                "  '我家有只叫小白的猫'         → [(user, has_pet, 小白)]\n"
                "  '我会说中文和英语'           → [(user, speaks_language, 中文), (user, speaks_language, 英语)]\n"
                "  '我女朋友叫小雪'            → [(user, girlfriend, 小雪)]\n"
                "  '我的车是大众朗逸'           → [(user, owns_car, 大众朗逸)]\n"
                "  '我手机是 iPhone 14'        → [(user, owns_phone, iPhone 14)]\n"
                "  '我体重 80 公斤'            → [(user, weight_kg, 80)]\n"
                "  '我跳槽到字节做基础架构'      → [(user, works_at, 字节), (user, occupation, 基础架构)]\n"
                "  '我之前在北京工作过'         → [(user, worked_in, 北京)]\n"
                "  '我下个月要回成都工作'       → [(user, planned_move_to, 成都), (user, will_work_at, 成都)]\n"
                "  '我以前在上海住过'           → [(user, lived_in, 上海)]\n"
                "  '我不爱吃芒果'              → [(user, dislikes, 芒果)]\n"
                "  '今天天气真好'              → []  (无可结构化事实)\n"
                "\n"
                "返回 JSON, 格式: "
                '{{"triples": [{{"subject": "user", "predicate": "lives_in", "object": "北京", "confidence": 0.95}}]}}'
            ),
        ),
        ("human", "{text}"),
    ]
)


async def extract_triples(text: str) -> list[Triple]:
    """从自然语言抽取三元组列表. LLM 失败时返回空列表 (降级)."""
    try:
        with metrics.timer("semantic.extract.latency"):
            result = await llm_factory.structured_invoke(
                _EXTRACT_PROMPT, _ExtractResult, {"text": text}, temperature=0
            )
        if result is None:
            return []
        triples = [
            Triple(
                subject=t.subject.strip(),
                predicate=t.predicate.strip().lower(),
                object=t.object.strip(),
                confidence=t.confidence,
            )
            for t in result.triples
            if t.subject and t.predicate and t.object
        ]
        metrics.incr("semantic.triples_extracted", len(triples))
        return triples
    except Exception as e:
        logger.warning(f"Entity extraction 失败 (降级返回空): {e}")
        return []


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       Field Semantics Schema                         ║
# ╚══════════════════════════════════════════════════════════════════════╝


# 字段语义: 决定冲突时默认 action
# - unique: 同 (subject, predicate) 只能有一个 object → 倾向 REPLACE
# - list:   可以有多个 object → 倾向 MERGE
# - versioned: 时间相关, 保留历史 → 倾向 VERSIONED
#
# _PRED_REGISTRY 统一管理所有已知谓词的 semantics + 中文模板,
# 新增谓词只需在此处添加一行, 不再需要同步更新两个 dict.
_PRED_REGISTRY: dict[str, dict[str, str]] = {
    # ── unique (单值) ──
    "lives_in":       {"sem": "unique", "cn": "住在 {obj}"},
    "works_at":       {"sem": "unique", "cn": "在 {obj} 工作"},
    "age":            {"sem": "unique", "cn": "年龄是 {obj}"},
    "occupation":     {"sem": "unique", "cn": "职业是 {obj}"},
    "gender":         {"sem": "unique", "cn": "性别是 {obj}"},
    "married_to":     {"sem": "unique", "cn": "已婚, 配偶是 {obj}"},
    "favorite_food":  {"sem": "unique", "cn": "最喜欢的食物是 {obj}"},
    "favorite_color": {"sem": "unique", "cn": "最喜欢的颜色是 {obj}"},
    "hometown":       {"sem": "unique", "cn": "家乡是 {obj}"},
    "weight_kg":      {"sem": "unique", "cn": "体重 {obj}"},
    "height_cm":      {"sem": "unique", "cn": "身高 {obj}"},
    "blood_type":     {"sem": "unique", "cn": "血型是 {obj}"},
    "girlfriend":     {"sem": "unique", "cn": "女朋友是 {obj}"},
    "boyfriend":      {"sem": "unique", "cn": "男朋友是 {obj}"},
    "spouse":         {"sem": "unique", "cn": "配偶是 {obj}"},
    "owns_car":       {"sem": "unique", "cn": "车是 {obj}"},
    "owns_phone":     {"sem": "unique", "cn": "手机是 {obj}"},
    "owns_laptop":    {"sem": "unique", "cn": "笔记本电脑是 {obj}"},
    "uses_camera":    {"sem": "unique", "cn": "用的相机是 {obj}"},
    "educational_background": {"sem": "unique", "cn": "学历: {obj}"},
    # ── versioned (时态, 保留历史) ──
    "worked_in":      {"sem": "versioned", "cn": "之前在 {obj} 工作过"},
    "worked_at":      {"sem": "versioned", "cn": "之前在 {obj} 工作过"},
    "lived_in":       {"sem": "versioned", "cn": "之前在 {obj} 住过"},
    "studied_at":     {"sem": "versioned", "cn": "在 {obj} 学习过"},
    "planned_move_to":{"sem": "versioned", "cn": "计划搬去 {obj}"},
    "will_work_at":   {"sem": "versioned", "cn": "将在 {obj} 工作"},
    "will_visit":     {"sem": "versioned", "cn": "计划去 {obj}"},
    "planned_event":  {"sem": "versioned", "cn": "计划 {obj}"},
    # ── list (多值) ──
    "allergic_to":    {"sem": "list", "cn": "对 {obj} 过敏"},
    "likes":          {"sem": "list", "cn": "喜欢 {obj}"},
    "dislikes":       {"sem": "list", "cn": "不喜欢 {obj}"},
    "has_pet":        {"sem": "list", "cn": "养了宠物 {obj}"},
    "has_child":      {"sem": "list", "cn": "有孩子 {obj}"},
    "sibling":        {"sem": "list", "cn": "兄弟姐妹: {obj}"},
    "hobby":          {"sem": "list", "cn": "爱好是 {obj}"},
    "speaks_language":{"sem": "list", "cn": "会说 {obj}"},
    "visited":        {"sem": "list", "cn": "去过 {obj}"},
}


def get_field_semantics(predicate: str) -> str:
    """返回 'unique' / 'list' / 'versioned', 未知谓词默认 'unique'."""
    return _PRED_REGISTRY.get(predicate, {}).get("sem", "unique")


def get_pred_cn_template(predicate: str) -> str | None:
    """返回谓词的中文模板, 未知谓词返回 None."""
    return _PRED_REGISTRY.get(predicate, {}).get("cn")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       Semantic Memory 主类                            ║
# ╚══════════════════════════════════════════════════════════════════════╝


class SemanticMemory:
    """事实知识记忆 — 双索引 (KG + Vector).

    Note: 冲突仲裁逻辑在 Phase 3 的 arbitrator 模块, 这里通过 hook 接口注入.
    P0: Entity Resolution 集成到写入管线, 非 "user" 的 subject/object 实体名会自动消解.
    """

    def __init__(self) -> None:
        self._kg = get_kg()
        self._vector = get_vector_store()
        self._meta = get_metadata()
        # arbitrator 在 Phase 3 注入, 这里先用 None 占位
        self._arbitrator = None
        # P0: Entity Resolver (懒加载, 避免循环依赖)
        self._entity_resolver = None
        logger.info("SemanticMemory 初始化")

    def set_arbitrator(self, arbitrator: Any) -> None:
        """由 orchestrator 在启动时注入 ConflictArbitrator 实例 (避免循环依赖)."""
        self._arbitrator = arbitrator

    def set_entity_resolver(self, resolver: Any) -> None:
        """注入 EntityResolver (避免循环依赖)."""
        self._entity_resolver = resolver

    async def write_from_text(
        self,
        user_id: str,
        text: str,
        source_memory_id: str | None = None,
        conflict_strategy: str = "defer",
    ) -> list[dict[str, Any]]:
        """从自然语言抽取 + 写入. 返回每个 triple 的写入结果.

        Args:
            conflict_strategy: defer (默认, 启发式快速处理) / staleness / arbitrator
            conflict_strategy: arbitrator (LLM 决策, 默认) / staleness (软废弃) / auto

        每个返回元素结构:
          {
            "triple": Triple,
            "action": "added" | "replaced" | "merged" | "versioned" | "ignored" | "stale_marked",
            "arbitration": ArbitrationDecision | None,
            "entity_resolution": ResolvedEntity | None (P0 新增, 仅非 "user" 实体),
          }
        """
        triples = await extract_triples(text)
        if not triples:
            return []

        # P0: Entity Resolution — 对非 "user" 实体名做消解
        entity_resolutions: dict[str, Any] = {}
        if self._entity_resolver is not None:
            for triple in triples:
                # 对 subject (如果不是 "user")
                if triple.subject != "user" and triple.subject not in entity_resolutions:
                    resolved = await self._entity_resolver.resolve(
                        user_id=user_id,
                        new_name=triple.subject,
                        entity_type=_infer_entity_type(triple.predicate, "subject"),
                        context=text[:200],
                    )
                    entity_resolutions[triple.subject] = resolved
                    # 用消解后的规范名替换 triple 的 subject
                    if not resolved.is_new and resolved.canonical_name != triple.subject:
                        triple.subject = resolved.canonical_name

                # 对 object (如果看起来是实体而非字面值)
                if _is_entity_object(triple.object) and triple.object not in entity_resolutions:
                    resolved = await self._entity_resolver.resolve(
                        user_id=user_id,
                        new_name=triple.object,
                        entity_type=_infer_entity_type(triple.predicate, "object"),
                        context=text[:200],
                    )
                    entity_resolutions[triple.object] = resolved
                    # 用消解后的规范名替换 triple 的 object
                    if not resolved.is_new and resolved.canonical_name != triple.object:
                        triple.object = resolved.canonical_name

        results: list[dict[str, Any]] = []
        for triple in triples:
            triple.source_memory_id = source_memory_id
            res = await self.upsert_triple(user_id, triple, conflict_strategy=conflict_strategy)
            # 添加 Entity Resolution 信息到结果
            original_subj = triple.subject
            original_obj = triple.object
            if original_subj in entity_resolutions:
                res["entity_resolution_subject"] = entity_resolutions[original_subj].model_dump()
            if original_obj in entity_resolutions:
                res["entity_resolution_object"] = entity_resolutions[original_obj].model_dump()
            results.append(res)
        return results

    async def upsert_triple(
        self,
        user_id: str,
        triple: Triple,
        conflict_strategy: str = "defer",
    ) -> dict[str, Any]:
        """写入单个 triple, 冲突检测 + 快速处理.

        Args:
            conflict_strategy:
                defer (默认) — 用字段语义启发式快速处理, 标 pending_arbitration
                    供后台 consolidate 批量仲裁 (参考 Mem0 v3: 写入求快, 离线求质)
                staleness — 直接软废弃旧 triple (跳过 LLM, 无 Key / 批量写)
                arbitrator — LLM 决策 REPLACE/MERGE/VERSIONED/IGNORE (慢, 仅调试用)
                auto — LLM 失败时 fallback 到 staleness
        """
        # 检查冲突: 同 (subject, predicate) 已有 triple?
        existing = await self._kg.find_triples(
            user_id, subject=triple.subject, predicate=triple.predicate
        )
        existing = [t for t in existing if t.id != triple.id]  # 排除自身

        if not existing:
            # 无冲突, 直接添加
            await self._kg.add_triple(user_id, triple)
            await self._mirror_to_vector(user_id, triple)
            return {"triple": triple, "action": "added", "arbitration": None}

        # 已有相同 object → 幂等, 不重复添加
        for t in existing:
            if t.object == triple.object:
                logger.debug(f"Semantic upsert: 重复事实, 忽略 ({triple})")
                return {"triple": triple, "action": "duplicate", "arbitration": None}

        # ── staleness 策略: 跳过启发式, 直接软废弃 ──
        if conflict_strategy == "staleness":
            return await self._apply_staleness(user_id, triple, existing)

        # ── defer 策略 (默认): 用字段语义启发式快速处理, 标 pending ──
        if conflict_strategy == "defer":
            return await self._apply_deferred(user_id, triple, existing)

        # ── arbitrator 策略: LLM 决策 (慢, 仅调试 / 显式请求用) ──
        if self._arbitrator is None:
            # arbitrator 未注入时自动降级到 defer
            logger.info(f"Arbitrator 未注入, 降级到 defer 策略")
            return await self._apply_deferred(user_id, triple, existing)

        try:
            decision = await self._arbitrator.arbitrate(
                user_id=user_id,
                new_triple=triple,
                existing_triples=existing,
                field_semantics=get_field_semantics(triple.predicate),
            )
            action_str = await self._apply_decision(user_id, triple, existing, decision)
            return {"triple": triple, "action": action_str, "arbitration": decision}
        except Exception as e:
            logger.warning(f"Arbitrator 失败 ({e}), 降级到 defer")
            return await self._apply_deferred(user_id, triple, existing)

    async def _apply_deferred(
        self,
        user_id: str,
        new_triple: Triple,
        existing: list[Triple],
    ) -> dict[str, Any]:
        """延迟仲裁路径: 用字段语义启发式快速处理, 不调 LLM.

        参考 Mem0 v3: 写入路径只做 1 次 LLM call (实体抽取), 冲突用启发式处理.
        待处理的冲突标 pending_arbitration, 后台 consolidate 批量仲裁.
        """
        semantics = get_field_semantics(new_triple.predicate)

        # ── list 类型: 直接 append (无冲突) ──
        if semantics == "list":
            await self._kg.add_triple(user_id, new_triple)
            await self._mirror_to_vector(user_id, new_triple, pending=False)
            return {"triple": new_triple, "action": "merged_deferred", "arbitration": None}

        # ── versioned 类型: 同时保留 (标 pending, 后台处理时间窗口) ──
        if semantics == "versioned":
            await self._kg.add_triple(user_id, new_triple)
            await self._mirror_to_vector(user_id, new_triple, pending=True)
            # 旧 triple 也标 pending, 后台 consolidate 会设 valid_until
            vec = get_vector_store()
            for old_t in existing:
                try:
                    await vec.update_metadata(
                        old_t.id, user_id, {"pending_arbitration": 1},
                    )
                except Exception:
                    pass
            return {"triple": new_triple, "action": "versioned_deferred", "arbitration": None}

        # ── unique 类型: 新 triple 入库, 旧 triple 降权 (软废弃) ──
        return await self._apply_staleness(user_id, new_triple, existing)

    async def _apply_staleness(
        self,
        user_id: str,
        new_triple: Triple,
        existing: list[Triple],
    ) -> dict[str, Any]:
        """Staleness 路径: 旧 triple 不删, 但标 staleness → effective_strength × 0.2.

        新 triple 正常入库. 旧 episodic 不动 (审计可追溯).
        """
        from datetime import datetime

        from app.lifecycle.staleness import apply_staleness

        # 1. 新 triple 添加到 KG + 镜像 Chroma
        await self._kg.add_triple(user_id, new_triple)
        await self._mirror_to_vector(user_id, new_triple)

        # 2. 把所有旧 triple 关联的 source memory 一起软废弃
        meta = get_metadata()
        old_records: list[MemoryRecord] = []
        for old_t in existing:
            if not old_t.source_memory_id:
                continue
            old_rec = await meta.get_memory(old_t.source_memory_id)
            if old_rec:
                old_records.append(old_rec)

        # 3. 同时给旧 triple 自己 ("镜像在 Chroma 的 triple-mirror") 也降权
        #    旧 KG triple 不直接删除, 只在 Chroma metadata 标 staleness
        from app.storage import get_vector_store
        vec = get_vector_store()
        for old_t in existing:
            try:
                await vec.update_metadata(
                    old_t.id, user_id,
                    {"staleness_signal": 1, "superseded_by": new_triple.id},
                )
            except Exception:
                pass

        # 4. 创建新的 source record (CORRECTED 来源) 并软废弃旧 episodic
        new_record = MemoryRecord(
            id=new_triple.id,
            user_id=user_id,
            type=MemoryType.SEMANTIC,
            content=f"{new_triple.subject} {new_triple.predicate} {new_triple.object}",
            source_type="corrected",
            confidence_score=0.85,
            created_at=datetime.now(),
        )
        result = await apply_staleness(new_record, old_records)
        logger.info(
            f"[Staleness] Semantic 软废弃: 新 {new_triple.object}, 旧软废弃 "
            f"{len(result['superseded'])} 条 source memory"
        )

        return {
            "triple": new_triple,
            "action": "stale_marked",
            "arbitration": None,
            "superseded": result["superseded"],
        }

    async def _apply_decision(
        self,
        user_id: str,
        new_triple: Triple,
        existing: list[Triple],
        decision: Any,  # ArbitrationDecision, 避免循环依赖用 Any
    ) -> str:
        """执行 Arbitrator 决策."""
        from app.models import ConflictAction

        action = decision.action if hasattr(decision, "action") else ConflictAction(decision["action"])

        if action == ConflictAction.REPLACE:
            # 收集旧 triple 关联的 episodic source, 一起降权 (避免被 hybrid 召回顶上)
            stale_episodic_ids: set[str] = set()
            for t in existing:
                if t.source_memory_id:
                    stale_episodic_ids.add(t.source_memory_id)
                await self._kg.delete_triple(user_id, t.id)
                # 同时删除 Chroma 中的旧 mirror, 避免召回时混入过时事实
                await self._vector.delete(t.id, user_id)
            # 把对应的 Episodic 原文 importance 打到 0.05 (近似遗忘)
            for ep_id in stale_episodic_ids:
                try:
                    await self._vector.update_metadata(
                        ep_id, user_id, {"importance": 0.05, "tier": "cold"}
                    )
                except Exception:
                    pass
            await self._kg.add_triple(user_id, new_triple)
            await self._mirror_to_vector(user_id, new_triple)
            return "replaced"

        if action == ConflictAction.MERGE:
            # list 字段, 直接 append
            await self._kg.add_triple(user_id, new_triple)
            await self._mirror_to_vector(user_id, new_triple)
            return "merged"

        if action == ConflictAction.VERSIONED:
            from datetime import datetime

            now = datetime.now()
            # 旧 triple 设 valid_until = now
            for t in existing:
                t.valid_until = now
                await self._kg.delete_triple(user_id, t.id)
                await self._kg.add_triple(user_id, t)
            new_triple.valid_from = now
            await self._kg.add_triple(user_id, new_triple)
            await self._mirror_to_vector(user_id, new_triple)
            return "versioned"

        # IGNORE
        logger.info(f"Arbitrator 决定 IGNORE: {new_triple}")
        return "ignored"

    async def _mirror_to_vector(
        self, user_id: str, triple: Triple, pending: bool = False
    ) -> None:
        """把 triple 同步到 ChromaDB 作为兜底语义召回.

        Args:
            pending: True 时标记 pending_arbitration=1, 供后台 consolidate 批量仲裁.

        content 使用中文自然语言格式而非英文谓词格式, 提升 embedding 与中文查询的相似度.
        """
        # 从统一注册表取中文模板, 未知谓词用通用格式
        template = get_pred_cn_template(triple.predicate)
        if template:
            text = template.format(obj=triple.object)
        else:
            # 未知谓词: 用 "谓词描述: object" 的通用格式
            text = f"{triple.predicate.replace('_', ' ')}: {triple.object}"

        record = MemoryRecord(
            id=triple.id,  # 用 triple id 关联
            user_id=user_id,
            type=MemoryType.SEMANTIC,
            content=text,
            structured={
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "confidence": triple.confidence,
                "pending_arbitration": pending,  # 延迟仲裁标记
            },
            importance=0.7,  # semantic 默认高重要度
            source="distilled" if triple.source_memory_id else "explicit",
        )
        await self._vector.add(record)
        await self._meta.upsert_memory(record)

    # ── Query ──────────────────────────────────────────────────────────
    async def query_entity(
        self, user_id: str, subject: str = "user", predicate: str | None = None
    ) -> list[Triple]:
        """直接查 KG: e.g. user 的所有事实, 或 user.lives_in."""
        return await self._kg.find_triples(
            user_id, subject=subject, predicate=predicate
        )

    async def search(
        self, user_id: str, query: str, top_k: int = 10
    ) -> list[tuple[MemoryRecord, float]]:
        """向量召回 (兜底)."""
        return await self._vector.search(
            user_id=user_id,
            query=query,
            memory_types=[MemoryType.SEMANTIC.value],
            top_k=top_k,
        )

    async def export_for_profile(self, user_id: str) -> dict[str, Any]:
        """导出用户所有 semantic 事实, 用于 Reflective Profile 生成."""
        triples = await self._kg.find_triples(user_id, subject="user")
        # 按 predicate 聚合
        facts: dict[str, list[str]] = {}
        for t in triples:
            facts.setdefault(t.predicate, []).append(t.object)
        return {"user_id": user_id, "facts": facts, "triple_count": len(triples)}


# ── P0: Entity Resolution 辅助函数 ──────────────────────────────────────


# 谓词 → 实体类型映射: 用于推断非 "user" subject/object 的类型
_PRED_ENTITY_TYPE_MAP: dict[str, dict[str, str]] = {
    "works_at": {"subject": "person", "object": "organization"},
    "worked_at": {"subject": "person", "object": "organization"},
    "worked_in": {"subject": "person", "object": "location"},
    "lives_in": {"subject": "person", "object": "location"},
    "lived_in": {"subject": "person", "object": "location"},
    "hometown": {"subject": "person", "object": "location"},
    "visited": {"subject": "person", "object": "location"},
    "planned_move_to": {"subject": "person", "object": "location"},
    "will_work_at": {"subject": "person", "object": "location"},
    "will_visit": {"subject": "person", "object": "location"},
    "married_to": {"subject": "person", "object": "person"},
    "girlfriend": {"subject": "person", "object": "person"},
    "boyfriend": {"subject": "person", "object": "person"},
    "spouse": {"subject": "person", "object": "person"},
    "sibling": {"subject": "person", "object": "person"},
    "has_pet": {"subject": "person", "object": "concept"},
    "has_child": {"subject": "person", "object": "person"},
    "likes": {"subject": "person", "object": "concept"},
    "dislikes": {"subject": "person", "object": "concept"},
    "hobby": {"subject": "person", "object": "concept"},
    "speaks_language": {"subject": "person", "object": "concept"},
    "allergic_to": {"subject": "person", "object": "concept"},
    "owns_car": {"subject": "person", "object": "product"},
    "owns_phone": {"subject": "person", "object": "product"},
    "owns_laptop": {"subject": "person", "object": "product"},
    "uses_camera": {"subject": "person", "object": "product"},
    "studied_at": {"subject": "person", "object": "organization"},
}


def _infer_entity_type(predicate: str, role: str) -> str:
    """根据 predicate 和角色推断实体类型."""
    mapping = _PRED_ENTITY_TYPE_MAP.get(predicate, {})
    return mapping.get(role, "concept")


def _is_entity_object(obj: str) -> bool:
    """判断 triple 的 object 是否是一个实体 (而非字面值).

    字面值: 数字、日期、单位等 → 不做消解
    实体: 人名、地名、组织名 → 做消解
    """
    # 数字 (纯数字或带单位) → 字面值
    import re
    if re.match(r"^[\d.]+", obj):
        return False
    # 很短的无意义值 → 字面值
    if len(obj) <= 2 and not obj.isalpha():
        return False
    # 包含常见单位 → 字面值
    _units = {"公斤", "km", "cm", "kg", "米", "岁", "ml", "寸"}
    if any(u in obj for u in _units):
        return False
    # 其他 → 可能是实体
    return True


semantic_memory = SemanticMemory()
