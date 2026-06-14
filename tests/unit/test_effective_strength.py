"""Ebbinghaus + 复习提升 + 来源权重 + staleness 罚分公式的单元测试.

公式:
  effective_strength(t) = confidence_score
                        × e^(-decay_rate × active_days)
                        × (1 + 0.15 × log(recall_count + 1))
                        × SOURCE_WEIGHTS[source_type]
                        × (0.2 if staleness_signal else 1.0)
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from app.lifecycle.decay import compute_effective_strength
from app.models import MemoryRecord, MemoryType, SourceType


def _make(
    *,
    confidence: float = 0.7,
    source: str = SourceType.EXPLICIT_STATEMENT.value,
    decay_rate: float = 0.01,
    recall_count: int = 0,
    staleness: bool = False,
    created_at: datetime,
    last_recalled: datetime | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        user_id="u1",
        type=MemoryType.SEMANTIC,
        content="x",
        confidence_score=confidence,
        source_type=source,
        decay_rate=decay_rate,
        recall_count=recall_count,
        staleness_signal=staleness,
        created_at=created_at,
        last_recalled_at=last_recalled,
    )


class TestFreshRecord:
    def test_just_written_explicit(self, fixed_now):
        """新写入 + explicit (权重 1.0) + decay=0 时应等于 confidence."""
        rec = _make(
            confidence=1.0, decay_rate=0.0, created_at=fixed_now,
        )
        assert compute_effective_strength(rec, now=fixed_now) == pytest.approx(1.0)

    def test_just_written_default_confidence(self, fixed_now):
        """默认 confidence=0.7, 无衰减 → 0.7."""
        rec = _make(decay_rate=0.0, created_at=fixed_now)
        assert compute_effective_strength(rec, now=fixed_now) == pytest.approx(0.7)


class TestEbbinghausDecay:
    def test_30_days_explicit(self, fixed_now):
        """λ=0.01 + 30 天 → exp(-0.3) ≈ 0.7408 → 0.7 × 0.7408 ≈ 0.5186"""
        rec = _make(
            confidence=0.7,
            decay_rate=0.01,
            created_at=fixed_now - timedelta(days=30),
        )
        score = compute_effective_strength(rec, now=fixed_now)
        assert score == pytest.approx(0.7 * math.exp(-0.3), rel=1e-3)

    def test_70_days_half_life(self, fixed_now):
        """λ=0.01 → 半衰期约 70 天 (ln 2 / 0.01 ≈ 69.3)."""
        rec = _make(
            confidence=1.0,
            decay_rate=0.01,
            created_at=fixed_now - timedelta(days=70),
        )
        score = compute_effective_strength(rec, now=fixed_now)
        # exp(-0.7) ≈ 0.4966, 接近一半
        assert 0.45 < score < 0.55

    def test_decay_zero_means_no_decay(self, fixed_now):
        rec = _make(
            confidence=1.0,
            decay_rate=0.0,
            created_at=fixed_now - timedelta(days=10000),
        )
        assert compute_effective_strength(rec, now=fixed_now) == pytest.approx(1.0)

    def test_last_recalled_resets_anchor(self, fixed_now):
        """复习重置遗忘曲线 — 用 last_recalled_at 而非 created_at 算 active_days."""
        rec = _make(
            confidence=1.0,
            decay_rate=0.01,
            created_at=fixed_now - timedelta(days=365),
            last_recalled=fixed_now - timedelta(days=1),  # 昨天召回
        )
        score = compute_effective_strength(rec, now=fixed_now)
        # 应接近 1, 因为 active_days = 1 而非 365
        # 但 recall_count=0 没加成, 所以纯衰减 exp(-0.01) ≈ 0.99
        assert score > 0.95


class TestRecallBoost:
    def test_zero_recall_no_boost(self, fixed_now):
        rec = _make(confidence=1.0, decay_rate=0.0, created_at=fixed_now)
        assert compute_effective_strength(rec, now=fixed_now) == pytest.approx(1.0)

    def test_recall_count_log_boost(self, fixed_now):
        """50 次召回 → 1 + 0.15 × log(51) ≈ 1.59"""
        rec = _make(
            confidence=1.0,
            decay_rate=0.0,
            recall_count=50,
            created_at=fixed_now,
        )
        score = compute_effective_strength(rec, now=fixed_now)
        expected = 1.0 + 0.15 * math.log(51)
        assert score == pytest.approx(expected, rel=1e-3)


class TestSourceWeight:
    def test_corrected_amplifies(self, fixed_now):
        """CORRECTED 权重 1.2, 应让有效强度超过 confidence."""
        rec = _make(
            confidence=1.0,
            decay_rate=0.0,
            source=SourceType.CORRECTED.value,
            created_at=fixed_now,
        )
        score = compute_effective_strength(rec, now=fixed_now)
        assert score == pytest.approx(1.2)

    def test_inferred_dampens(self, fixed_now):
        """INFERRED 权重 0.6."""
        rec = _make(
            confidence=1.0,
            decay_rate=0.0,
            source=SourceType.INFERRED.value,
            created_at=fixed_now,
        )
        score = compute_effective_strength(rec, now=fixed_now)
        assert score == pytest.approx(0.6)

    def test_unknown_source_default_one(self, fixed_now):
        """未知 source_type 默认权重 1.0, 不破坏函数."""
        rec = _make(
            confidence=0.8,
            decay_rate=0.0,
            source="unknown_source_type_xyz",
            created_at=fixed_now,
        )
        score = compute_effective_strength(rec, now=fixed_now)
        assert score == pytest.approx(0.8)


class TestStalenessSignal:
    def test_stale_multiplied_by_point_two(self, fixed_now):
        """staleness=True → 整个强度 × 0.2."""
        rec = _make(
            confidence=1.0,
            decay_rate=0.0,
            staleness=True,
            created_at=fixed_now,
        )
        score = compute_effective_strength(rec, now=fixed_now)
        assert score == pytest.approx(0.2)

    def test_stale_compounds_with_decay(self, fixed_now):
        """30 天 + stale: 0.7 × exp(-0.3) × 0.2 ≈ 0.1037"""
        rec = _make(
            confidence=0.7,
            decay_rate=0.01,
            staleness=True,
            created_at=fixed_now - timedelta(days=30),
        )
        score = compute_effective_strength(rec, now=fixed_now)
        expected = 0.7 * math.exp(-0.3) * 0.2
        assert score == pytest.approx(expected, rel=1e-3)


class TestMonotonicity:
    """关键不变性: 同其他参数下, 越新 / 召回越多 / 越权威 → 强度越高."""

    def test_newer_record_stronger(self, fixed_now):
        old = _make(
            confidence=1.0,
            decay_rate=0.01,
            created_at=fixed_now - timedelta(days=100),
        )
        new = _make(
            confidence=1.0,
            decay_rate=0.01,
            created_at=fixed_now - timedelta(days=1),
        )
        assert compute_effective_strength(new, now=fixed_now) > compute_effective_strength(old, now=fixed_now)

    def test_more_recalls_stronger(self, fixed_now):
        cold = _make(
            confidence=1.0, decay_rate=0.0, recall_count=0, created_at=fixed_now
        )
        hot = _make(
            confidence=1.0, decay_rate=0.0, recall_count=20, created_at=fixed_now
        )
        assert compute_effective_strength(hot, now=fixed_now) > compute_effective_strength(cold, now=fixed_now)

    def test_explicit_stronger_than_inferred(self, fixed_now):
        explicit = _make(
            confidence=1.0,
            decay_rate=0.0,
            source=SourceType.EXPLICIT_STATEMENT.value,
            created_at=fixed_now,
        )
        inferred = _make(
            confidence=1.0,
            decay_rate=0.0,
            source=SourceType.INFERRED.value,
            created_at=fixed_now,
        )
        assert compute_effective_strength(explicit, now=fixed_now) > compute_effective_strength(inferred, now=fixed_now)
