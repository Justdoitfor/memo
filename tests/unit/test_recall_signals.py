"""4 信号召回打分函数的单元测试 — 纯函数, 无外部依赖.

覆盖:
  - compute_vector_sim: 边界值 + clamp
  - compute_temporal_decay: 30/60/365 天衰减曲线 + tau 边界
  - compute_keyword_match: 实体重叠 + 邻居加权
  - compute_importance: 调 effective_strength
  - fuse_signals: 加权融合 + 权重和=0 + 自定义权重
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from app.models import MemoryRecord, MemoryType, SourceType
from app.recall.signals import (
    compute_importance,
    compute_keyword_match,
    compute_temporal_decay,
    compute_vector_sim,
    fuse_signals,
)


# ────────────────────────────────────────────────────────────────────────
#  compute_vector_sim — clamp 到 [0, 1]
# ────────────────────────────────────────────────────────────────────────


class TestComputeVectorSim:
    def test_in_range_passthrough(self):
        assert compute_vector_sim(0.5) == 0.5
        assert compute_vector_sim(0.0) == 0.0
        assert compute_vector_sim(1.0) == 1.0

    def test_negative_clamped_to_zero(self):
        """ChromaDB cosine 偶尔返回负值 (浮点误差), 应 clamp 到 0."""
        assert compute_vector_sim(-0.1) == 0.0
        assert compute_vector_sim(-1e-9) == 0.0

    def test_above_one_clamped(self):
        """理论上不应 > 1, 但兜底防御."""
        assert compute_vector_sim(1.5) == 1.0
        assert compute_vector_sim(100.0) == 1.0


# ────────────────────────────────────────────────────────────────────────
#  compute_temporal_decay — exp(-Δt / tau)
# ────────────────────────────────────────────────────────────────────────


class TestComputeTemporalDecay:
    def test_just_now_is_one(self, fixed_now):
        assert compute_temporal_decay(fixed_now, now=fixed_now) == pytest.approx(1.0)

    def test_30_days_with_tau_30(self, fixed_now):
        """t = tau → 衰减到 e^-1 ≈ 0.3679 (设计文档承诺)."""
        thirty_days_ago = fixed_now - timedelta(days=30)
        score = compute_temporal_decay(thirty_days_ago, now=fixed_now, tau_days=30.0)
        assert score == pytest.approx(math.exp(-1.0), rel=1e-3)

    def test_60_days_with_tau_30(self, fixed_now):
        """t = 2 tau → e^-2 ≈ 0.1353"""
        sixty_days_ago = fixed_now - timedelta(days=60)
        score = compute_temporal_decay(sixty_days_ago, now=fixed_now, tau_days=30.0)
        assert score == pytest.approx(math.exp(-2.0), rel=1e-3)

    def test_one_year_decays_heavily(self, fixed_now):
        one_year_ago = fixed_now - timedelta(days=365)
        score = compute_temporal_decay(one_year_ago, now=fixed_now, tau_days=30.0)
        # 365/30 ≈ 12, e^-12 极小
        assert score < 0.01

    def test_tau_zero_returns_one(self, fixed_now):
        """tau<=0 表示禁用衰减, 永远返回 1.0."""
        old = fixed_now - timedelta(days=1000)
        assert compute_temporal_decay(old, now=fixed_now, tau_days=0.0) == 1.0
        assert compute_temporal_decay(old, now=fixed_now, tau_days=-5.0) == 1.0

    def test_future_timestamp_clamped_to_now(self, fixed_now):
        """记录创建时间在未来 (时钟漂移) → 视为 now, 衰减 = 1.0."""
        future = fixed_now + timedelta(days=10)
        score = compute_temporal_decay(future, now=fixed_now, tau_days=30.0)
        assert score == pytest.approx(1.0)


# ────────────────────────────────────────────────────────────────────────
#  compute_keyword_match — 实体重叠
# ────────────────────────────────────────────────────────────────────────


def _make_record_with_structured(subject="user", obj="北京") -> MemoryRecord:
    return MemoryRecord(
        user_id="u1",
        type=MemoryType.SEMANTIC,
        content=f"{subject} lives in {obj}",
        structured={"subject": subject, "object": obj},
    )


class TestComputeKeywordMatch:
    def test_no_overlap_returns_zero(self):
        rec = _make_record_with_structured(obj="北京")
        score = compute_keyword_match(rec, query_entities={"上海"}, user_neighbors=set())
        assert score == 0.0

    def test_query_entity_match_gets_half(self):
        rec = _make_record_with_structured(obj="北京")
        score = compute_keyword_match(
            rec, query_entities={"北京"}, user_neighbors=set()
        )
        assert score == 0.5

    def test_neighbor_overlap_gets_partial(self):
        rec = _make_record_with_structured(obj="北京")
        score = compute_keyword_match(
            rec, query_entities=set(), user_neighbors={"北京"}
        )
        assert score == pytest.approx(0.3)

    def test_both_match_capped_at_one(self):
        rec = _make_record_with_structured(obj="北京")
        score = compute_keyword_match(
            rec, query_entities={"北京"}, user_neighbors={"北京"}
        )
        # 0.5 + 0.3 = 0.8 (未到 1.0)
        assert score == pytest.approx(0.8)

    def test_empty_structured_returns_zero(self):
        rec = MemoryRecord(
            user_id="u1", type=MemoryType.EPISODIC, content="some event"
        )
        score = compute_keyword_match(
            rec, query_entities={"北京"}, user_neighbors=set()
        )
        assert score == 0.0


# ────────────────────────────────────────────────────────────────────────
#  compute_importance — 委托给 effective_strength, 测端到端范围
# ────────────────────────────────────────────────────────────────────────


class TestComputeImportance:
    def test_fresh_explicit_record_high(self):
        rec = MemoryRecord(
            user_id="u1",
            type=MemoryType.SEMANTIC,
            content="x",
            confidence_score=1.0,
            source_type=SourceType.EXPLICIT_STATEMENT.value,
        )
        score = compute_importance(rec)
        # 新写入 + explicit (权重 1.0) → 接近 1.0
        assert 0.9 <= score <= 1.0

    def test_stale_record_heavily_penalized(self):
        rec = MemoryRecord(
            user_id="u1",
            type=MemoryType.SEMANTIC,
            content="x",
            confidence_score=1.0,
            source_type=SourceType.EXPLICIT_STATEMENT.value,
            staleness_signal=True,
        )
        score = compute_importance(rec)
        # staleness × 0.2, 应远低于 fresh
        assert score < 0.25

    def test_clamped_at_one(self):
        """CORRECTED 来源权重 1.2 + recall_count 加成可能 > 1, 应被 clamp."""
        rec = MemoryRecord(
            user_id="u1",
            type=MemoryType.SEMANTIC,
            content="x",
            confidence_score=1.0,
            source_type=SourceType.CORRECTED.value,
            recall_count=100,
            decay_rate=0.0,  # 关闭衰减
        )
        score = compute_importance(rec)
        assert score == 1.0


# ────────────────────────────────────────────────────────────────────────
#  fuse_signals — 加权融合
# ────────────────────────────────────────────────────────────────────────


class TestFuseSignals:
    def test_default_weights_normalize_to_one(self):
        """4 个分量都为 1 时, 加权平均后仍为 1 (权重已归一化)."""
        score = fuse_signals(1.0, 1.0, 1.0, 1.0)
        assert score == pytest.approx(1.0)

    def test_all_zero_signals(self):
        score = fuse_signals(0.0, 0.0, 0.0, 0.0)
        assert score == 0.0

    def test_custom_weights(self):
        # 只看 vector_sim, 其它信号 0 权重不参与
        score = fuse_signals(
            1.0, 0.5, 0.3, 0.1,
            weights=(1.0, 0.0, 0.0, 0.0),
        )
        assert score == pytest.approx(1.0)

    def test_weights_sum_zero_returns_zero(self):
        """语义: 全 0 权重 → 没有信号被采用 → 不应给分."""
        score = fuse_signals(1.0, 1.0, 1.0, 1.0, weights=(0.0, 0.0, 0.0, 0.0))
        assert score == 0.0

    def test_weights_renormalized(self):
        """非归一化权重应被自动归一化, 保证 final_score ≤ 1.0."""
        # 总权重 = 4, 但每个分量都是 1 → 加权平均 = 1
        score = fuse_signals(1.0, 1.0, 1.0, 1.0, weights=(1.0, 1.0, 1.0, 1.0))
        assert score == pytest.approx(1.0)
        # 总权重 = 10, 分量都是 0.5 → 加权平均 = 0.5
        score2 = fuse_signals(0.5, 0.5, 0.5, 0.5, weights=(2.0, 3.0, 2.0, 3.0))
        assert score2 == pytest.approx(0.5)

    def test_dominant_vector_signal(self):
        """权重偏向 vector_sim 时, 它的值主导 final_score."""
        score = fuse_signals(
            1.0, 0.0, 0.0, 0.0,
            weights=(1.0, 0.0, 0.0, 0.0),
        )
        assert score == 1.0
        # 反之
        score2 = fuse_signals(
            1.0, 0.0, 0.0, 0.0,
            weights=(0.0, 1.0, 1.0, 1.0),
        )
        assert score2 == 0.0
