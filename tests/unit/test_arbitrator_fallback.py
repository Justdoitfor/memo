"""ConflictArbitrator 启发式 fallback 测试 + 字段语义 schema 测试.

LLM 不可达时的降级路径必须靠得住, 所以这层逻辑要单测保护.
LLM 调用本身的测试在 tests/integration/ 用 live_llm marker.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.arbitrator.conflict import ConflictArbitrator
from app.memories.semantic import get_field_semantics, get_pred_cn_template
from app.models import ArbitrationDecision, ConflictAction, Triple


def _t(obj: str, predicate: str = "lives_in", confidence: float = 0.8, days_ago: int = 0) -> Triple:
    return Triple(
        subject="user",
        predicate=predicate,
        object=obj,
        confidence=confidence,
        created_at=datetime.now() - timedelta(days=days_ago),
    )


# ────────────────────────────────────────────────────────────────────────
#  字段语义 schema
# ────────────────────────────────────────────────────────────────────────


class TestFieldSemantics:
    def test_unique_field(self):
        assert get_field_semantics("lives_in") == "unique"
        assert get_field_semantics("works_at") == "unique"
        assert get_field_semantics("age") == "unique"

    def test_list_field(self):
        assert get_field_semantics("allergic_to") == "list"
        assert get_field_semantics("likes") == "list"
        assert get_field_semantics("hobby") == "list"

    def test_versioned_field(self):
        assert get_field_semantics("worked_in") == "versioned"
        assert get_field_semantics("planned_move_to") == "versioned"

    def test_unknown_predicate_default_unique(self):
        """未注册 predicate 默认 'unique' (保守: 倾向 REPLACE 而非合并冲突值)."""
        assert get_field_semantics("never_seen_predicate_xyz") == "unique"

    def test_cn_template_unique_field(self):
        tmpl = get_pred_cn_template("lives_in")
        assert tmpl is not None
        assert "{obj}" in tmpl

    def test_cn_template_unknown_returns_none(self):
        assert get_pred_cn_template("never_seen_xyz") is None


# ────────────────────────────────────────────────────────────────────────
#  启发式 fallback (LLM 不可用降级路径)
# ────────────────────────────────────────────────────────────────────────


class TestHeuristicFallback:
    def test_list_field_merges_unique_values(self):
        """list 语义 → MERGE, 合并所有 object."""
        new = _t("芝麻", predicate="allergic_to")
        existing = [
            _t("花生", predicate="allergic_to"),
            _t("乳糖", predicate="allergic_to"),
        ]
        decision = ConflictArbitrator._heuristic_fallback(new, existing, "list")
        assert decision.action == ConflictAction.MERGE
        assert decision.merged_value is not None
        merged_set = set(decision.merged_value.split(","))
        assert merged_set == {"花生", "乳糖", "芝麻"}

    def test_list_field_dedups_duplicates(self):
        """新值已存在于旧值 → 合并去重."""
        new = _t("花生", predicate="allergic_to")
        existing = [_t("花生", predicate="allergic_to")]
        decision = ConflictArbitrator._heuristic_fallback(new, existing, "list")
        assert decision.action == ConflictAction.MERGE
        assert decision.merged_value == "花生"

    def test_versioned_field_keeps_history(self):
        new = _t("成都", predicate="worked_in")
        existing = [_t("北京", predicate="worked_in")]
        decision = ConflictArbitrator._heuristic_fallback(new, existing, "versioned")
        assert decision.action == ConflictAction.VERSIONED

    def test_unique_replace_when_new_confidence_higher(self):
        """unique 字段, 新事实置信度 ≥ 旧 → REPLACE."""
        new = _t("上海", confidence=0.9)
        existing = [_t("北京", confidence=0.8)]
        decision = ConflictArbitrator._heuristic_fallback(new, existing, "unique")
        assert decision.action == ConflictAction.REPLACE

    def test_unique_replace_when_confidences_equal(self):
        """新事实置信度 = 旧最大 → 仍 REPLACE (代码里是 >=)."""
        new = _t("上海", confidence=0.8)
        existing = [_t("北京", confidence=0.8)]
        decision = ConflictArbitrator._heuristic_fallback(new, existing, "unique")
        assert decision.action == ConflictAction.REPLACE

    def test_unique_ignore_when_new_lower_confidence(self):
        """unique + 新事实置信度更低 → IGNORE (不污染语义层)."""
        new = _t("上海", confidence=0.5)
        existing = [_t("北京", confidence=0.9)]
        decision = ConflictArbitrator._heuristic_fallback(new, existing, "unique")
        assert decision.action == ConflictAction.IGNORE

    def test_decision_carries_reasoning(self):
        new = _t("上海")
        existing = [_t("北京")]
        decision = ConflictArbitrator._heuristic_fallback(new, existing, "unique")
        assert decision.reasoning  # 非空字符串
        assert 0.0 <= decision.confidence <= 1.0


class TestArbitrationDecisionModel:
    """Pydantic 模型本身的契约 — JSON 序列化稳定."""

    def test_replace_no_merged_value(self):
        d = ArbitrationDecision(
            action=ConflictAction.REPLACE,
            reasoning="新事实更新",
            confidence=0.9,
        )
        assert d.merged_value is None

    def test_merge_with_merged_value(self):
        d = ArbitrationDecision(
            action=ConflictAction.MERGE,
            reasoning="list 字段合并",
            confidence=0.85,
            merged_value="花生,芝麻",
        )
        assert d.merged_value == "花生,芝麻"

    def test_confidence_must_be_in_range(self):
        with pytest.raises(Exception):
            ArbitrationDecision(
                action=ConflictAction.REPLACE,
                reasoning="x",
                confidence=1.5,  # 超界
            )
