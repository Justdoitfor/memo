"""Conflict Arbitrator — LLM-as-Judge 自动消解记忆冲突

核心:
  - 检测同 (subject, predicate) 已有不同 object → 冲突
  - 4 种 action: REPLACE / MERGE / VERSIONED / IGNORE
  - LLM 用 with_structured_output 强约束输出
  - 写完整审计日志, 可回滚可解释

Prompt 版本化 (P1.2):
  - prompt 字符串已抽到 prompts/arbitrator/v1.yaml
  - 通过 ARBITRATOR_PROMPT_VERSION 环境变量切版本 (默认 v1)
  - 审计日志 prompt_version 字段记录用了哪个版本
"""

from __future__ import annotations

import json
import os

from loguru import logger

from app.core.llm_factory import llm_factory
from app.core.prompt_loader import load_prompt
from app.memories.semantic import get_field_semantics, semantic_memory
from app.models import (
    ArbitrationDecision,
    ConflictAction,
    Triple,
)
from app.storage import get_metadata
from app.utils.metrics import metrics

# 通过环境变量切 prompt 版本; 不设置时用 v1 (向后兼容)
_PROMPT_VERSION = os.getenv("ARBITRATOR_PROMPT_VERSION", "v1")
_ARBITRATE_PROMPT = load_prompt("arbitrator", version=_PROMPT_VERSION)
logger.info(f"Arbitrator 使用 prompt 版本: {_PROMPT_VERSION}")


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
