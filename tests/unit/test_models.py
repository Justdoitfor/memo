"""MemoryRecord / Triple / RecallSignals 等核心模型的契约测试.

防回归: 字段约束 (importance ∈ [0,1]) / Chroma metadata 序列化 / 类型 lower-case 等
被改坏会立刻断, 而不是等到生产事故.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from app.models import (
    ArbitrationDecision,
    BehaviorSignal,
    ConflictAction,
    Entity,
    MemoryRecord,
    MemoryType,
    SignalType,
    SourceType,
    Triple,
)


class TestMemoryType:
    def test_lowercase_string_coerced(self):
        """模型 validator: 字符串大写也要 lower."""
        rec = MemoryRecord(user_id="u", type="EPISODIC", content="x")
        assert rec.type == MemoryType.EPISODIC

    def test_enum_passthrough(self):
        rec = MemoryRecord(user_id="u", type=MemoryType.SEMANTIC, content="x")
        assert rec.type == MemoryType.SEMANTIC

    def test_invalid_type_rejected(self):
        with pytest.raises(Exception):
            MemoryRecord(user_id="u", type="not_a_real_type", content="x")


class TestMemoryRecordConstraints:
    def test_importance_range(self):
        with pytest.raises(Exception):
            MemoryRecord(user_id="u", type=MemoryType.EPISODIC, content="x", importance=1.5)
        with pytest.raises(Exception):
            MemoryRecord(user_id="u", type=MemoryType.EPISODIC, content="x", importance=-0.1)

    def test_confidence_range(self):
        with pytest.raises(Exception):
            MemoryRecord(
                user_id="u", type=MemoryType.SEMANTIC, content="x", confidence_score=1.5
            )

    def test_empty_content_rejected(self):
        with pytest.raises(Exception):
            MemoryRecord(user_id="u", type=MemoryType.EPISODIC, content="")

    def test_default_id_unique(self):
        a = MemoryRecord(user_id="u", type=MemoryType.EPISODIC, content="x")
        b = MemoryRecord(user_id="u", type=MemoryType.EPISODIC, content="x")
        assert a.id != b.id

    def test_default_created_at_set(self):
        rec = MemoryRecord(user_id="u", type=MemoryType.EPISODIC, content="x")
        assert isinstance(rec.created_at, datetime)


class TestChromaMetadataSerialization:
    """ChromaDB 只能存原始类型, 复杂结构必须 JSON 化, 否则写入炸."""

    def test_only_primitives_in_metadata(self):
        rec = MemoryRecord(
            user_id="u",
            type=MemoryType.SEMANTIC,
            content="住北京",
            structured={"subject": "user", "predicate": "lives_in", "object": "北京"},
            tags=["geo", "user-attr"],
            importance=0.9,
        )
        meta = rec.to_chroma_metadata()
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool)), f"非原始类型: {k}={v!r}"

    def test_structured_json_round_trip(self):
        original = {"subject": "user", "valid_until": "2026-01-01T00:00:00"}
        rec = MemoryRecord(
            user_id="u", type=MemoryType.SEMANTIC, content="x", structured=original
        )
        meta = rec.to_chroma_metadata()
        restored = json.loads(meta["structured_json"])
        assert restored == original

    def test_empty_structured_serializes_empty(self):
        """空 structured 应产生空字符串而非 '{}'."""
        rec = MemoryRecord(user_id="u", type=MemoryType.EPISODIC, content="x")
        meta = rec.to_chroma_metadata()
        assert meta["structured_json"] == ""

    def test_tags_csv(self):
        rec = MemoryRecord(
            user_id="u", type=MemoryType.EPISODIC, content="x",
            tags=["a", "b", "c"],
        )
        assert rec.to_chroma_metadata()["tags_csv"] == "a,b,c"

    def test_staleness_signal_int_encoded(self):
        rec = MemoryRecord(
            user_id="u", type=MemoryType.SEMANTIC, content="x", staleness_signal=True
        )
        assert rec.to_chroma_metadata()["staleness_signal"] == 1
        rec2 = MemoryRecord(user_id="u", type=MemoryType.SEMANTIC, content="x")
        assert rec2.to_chroma_metadata()["staleness_signal"] == 0


class TestSourceWeightsCoverage:
    """SOURCE_WEIGHTS 必须覆盖所有 SourceType."""

    def test_all_source_types_have_weight(self):
        from app.models import SOURCE_WEIGHTS

        for st in SourceType:
            assert st.value in SOURCE_WEIGHTS, f"SourceType.{st.name} 缺少权重"
            assert 0.0 < SOURCE_WEIGHTS[st.value] <= 2.0, "权重应在合理范围"


class TestTripleAndEntity:
    def test_triple_confidence_clamped(self):
        with pytest.raises(Exception):
            Triple(subject="user", predicate="lives_in", object="北京", confidence=1.5)

    def test_triple_default_id(self):
        t1 = Triple(subject="user", predicate="lives_in", object="北京")
        t2 = Triple(subject="user", predicate="lives_in", object="上海")
        assert t1.id != t2.id

    def test_entity_aliases_default_empty(self):
        e = Entity(user_id="u", name="字节跳动")
        assert e.aliases == []
        assert e.entity_type == "person"  # 默认


class TestSignalEnums:
    def test_all_signal_types_strings(self):
        for st in SignalType:
            assert isinstance(st.value, str)

    def test_behavior_signal_default_tags_empty(self):
        s = BehaviorSignal(user_id="u", signal_type=SignalType.REGENERATE_REQUEST)
        assert s.context_tags == []
        assert s.memory_ids_in_context == []


class TestConflictActionEnum:
    def test_all_action_values_lowercase(self):
        for a in ConflictAction:
            assert a.value == a.value.lower()

    def test_arbitration_decision_action_str_coerced(self):
        """ConflictAction enum 接受小写字符串."""
        d = ArbitrationDecision(action="merge", reasoning="x", confidence=0.5)
        assert d.action == ConflictAction.MERGE
