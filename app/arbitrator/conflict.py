"""Conflict Arbitrator — LLM-as-Judge 自动消解记忆冲突

核心:
  - 检测同 (subject, predicate) 已有不同 object → 冲突
  - 4 种 action: REPLACE / MERGE / VERSIONED / IGNORE
  - LLM 用 with_structured_output 强约束输出
  - 写完整审计日志, 可回滚可解释
"""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from app.core.llm_factory import llm_factory
from app.memories.semantic import get_field_semantics, semantic_memory
from app.models import (
    ArbitrationDecision,
    ConflictAction,
    Triple,
)
from app.storage import get_metadata
from app.utils.metrics import metrics

_ARBITRATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是一个事实冲突仲裁器. 用户的旧事实与新事实出现矛盾时, 决定如何处理.\n"
                "\n"
                "你必须从以下 4 种 action 中选一个:\n"
                "\n"
                "1. REPLACE — 新事实完全取代旧事实, 旧事实归档.\n"
                "   场景: 用户搬家/换工作/年龄增长. 单值字段 (lives_in, works_at, age, occupation).\n"
                "\n"
                "2. MERGE — 新事实与旧事实并存 (多值合并).\n"
                "   场景: 用户提到新的过敏原/爱好/宠物. 多值字段 (allergic_to, likes, has_pet, hobby).\n"
                "   重要: action=MERGE 时, 必须返回 merged_value 字段, 形如 '乳糖,花生'.\n"
                "\n"
                "3. VERSIONED — 同时保留, 标记时间有效区间.\n"
                "   场景: 时间相关的事实链 (e.g. 用户上半年住北京, 下半年住上海, 历史很重要).\n"
                "\n"
                "4. IGNORE — 新事实可疑/不一致, 不写入 Semantic, 仅记 Episodic.\n"
                "   场景: 新事实置信度低、表述含糊、明显与历史矛盾.\n"
                "\n"
                "决策优先级:\n"
                "  字段语义是 'list' → 优先 MERGE\n"
                "  字段语义是 'unique' 且新事实置信度高 → 优先 REPLACE\n"
                "  时间敏感事实 → VERSIONED\n"
                "  其他不确定 → IGNORE\n"
                "\n"
                "必须给出 reasoning (一句话, 说明为什么) 和 confidence (0-1).\n"
                "\n"
                "返回 JSON, 字段: action (replace/merge/versioned/ignore 小写), "
                "reasoning (一句话), confidence (0-1), merged_value (action=merge 时填, 否则 null).\n"
                '示例: {{"action": "merge", "reasoning": "list 字段, 应保留所有过敏原", "confidence": 0.95, "merged_value": "花生,芝麻"}}'
            ),
        ),
        (
            "human",
            (
                "用户: {user_id}\n"
                "字段语义: {field_semantics}\n"
                "\n"
                "已有事实:\n"
                "{existing_facts}\n"
                "\n"
                "新事实:\n"
                "  ({subject}, {predicate}, {new_object})\n"
                "  置信度: {confidence}\n"
                "\n"
                "请决策."
            ),
        ),
    ]
)


class ConflictArbitrator:
    """LLM-as-Judge 冲突消解器."""

    def __init__(self) -> None:
        self._meta = get_metadata()
        logger.info("ConflictArbitrator 初始化")

    async def arbitrate(
        self,
        user_id: str,
        new_triple: Triple,
        existing_triples: list[Triple],
        field_semantics: str | None = None,
    ) -> ArbitrationDecision:
        """对一个 (subject, predicate) 已有冲突时, 决定如何处理.

        Args:
            user_id: 用户标识
            new_triple: 新写入的事实
            existing_triples: 已有的同 (subject, predicate) 事实列表
            field_semantics: 字段语义提示 ('unique' / 'list' / 'versioned'),
                             None 则由 semantic schema 推断
        """
        if field_semantics is None:
            field_semantics = get_field_semantics(new_triple.predicate)

        existing_summary = "\n".join(
            f"  - {t.object}  (置信度 {t.confidence:.2f}, 写入于 {t.created_at.strftime('%Y-%m-%d %H:%M')})"
            for t in existing_triples
        )

        decision: ArbitrationDecision | None = None
        try:
            with metrics.timer("arbitrator.latency"):
                decision = await llm_factory.structured_invoke(
                    _ARBITRATE_PROMPT,
                    ArbitrationDecision,
                    {
                        "user_id": user_id,
                        "field_semantics": field_semantics,
                        "existing_facts": existing_summary,
                        "subject": new_triple.subject,
                        "predicate": new_triple.predicate,
                        "new_object": new_triple.object,
                        "confidence": f"{new_triple.confidence:.2f}",
                    },
                    temperature=0,
                )
        except Exception as e:
            logger.warning(f"Arbitrator LLM 失败: {e}")
        if decision is None:
            logger.warning("Arbitrator 降级用启发式")
            decision = self._heuristic_fallback(new_triple, existing_triples, field_semantics)

        # 写审计日志
        await self._meta.log_arbitration(
            {
                "user_id": user_id,
                "subject": new_triple.subject,
                "predicate": new_triple.predicate,
                "old_value": json.dumps(
                    [t.object for t in existing_triples], ensure_ascii=False
                ),
                "new_value": new_triple.object,
                "action": decision.action.value,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
            }
        )
        metrics.incr(f"arbitrator.action.{decision.action.value}")
        logger.info(
            f"[Arbitrator] {decision.action.value} | "
            f"({new_triple.subject},{new_triple.predicate}): "
            f"{[t.object for t in existing_triples]} → {new_triple.object} | "
            f"reason={decision.reasoning[:50]}"
        )
        return decision

    @staticmethod
    def _heuristic_fallback(
        new_triple: Triple,
        existing_triples: list[Triple],
        field_semantics: str,
    ) -> ArbitrationDecision:
        """LLM 不可用时的启发式决策."""
        if field_semantics == "list":
            return ArbitrationDecision(
                action=ConflictAction.MERGE,
                reasoning="字段语义为 list, 启发式合并",
                confidence=0.7,
                merged_value=",".join(
                    list({t.object for t in existing_triples} | {new_triple.object})
                ),
            )
        if field_semantics == "versioned":
            return ArbitrationDecision(
                action=ConflictAction.VERSIONED,
                reasoning="字段语义为 versioned, 启发式版本化",
                confidence=0.6,
            )
        # 默认 unique → 新事实置信度更高才替换
        if new_triple.confidence >= max(t.confidence for t in existing_triples):
            return ArbitrationDecision(
                action=ConflictAction.REPLACE,
                reasoning="字段语义为 unique, 新事实置信度不低于旧, 启发式替换",
                confidence=0.6,
            )
        return ArbitrationDecision(
            action=ConflictAction.IGNORE,
            reasoning="新事实置信度低于历史, 启发式忽略",
            confidence=0.5,
        )


# 全局单例 + 自动注入到 SemanticMemory (打破循环依赖)
conflict_arbitrator = ConflictArbitrator()
semantic_memory.set_arbitrator(conflict_arbitrator)
