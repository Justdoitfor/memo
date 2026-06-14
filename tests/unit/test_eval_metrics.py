"""eval/metrics.py 单元测试 — 确保所有 IR 指标算分零盲点.

参考: TREC-eval / sklearn.metrics 的标准实现, 手算几个 case 对齐.
"""
from __future__ import annotations

import math

import pytest

from eval.metrics import (
    dcg_at_k,
    hit_at_k,
    mean,
    ndcg_at_k,
    per_query_metrics,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestHitAtK:
    def test_top1_hit(self):
        assert hit_at_k(["a", "b", "c"], {"a"}, 1) == 1.0

    def test_top1_miss(self):
        assert hit_at_k(["b", "a"], {"a"}, 1) == 0.0

    def test_top3_hit(self):
        assert hit_at_k(["b", "c", "a"], {"a"}, 3) == 1.0

    def test_no_relevant(self):
        assert hit_at_k(["a", "b"], set(), 5) == 0.0

    def test_empty_retrieved(self):
        assert hit_at_k([], {"a"}, 5) == 0.0


class TestRecallAtK:
    def test_all_hit(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_partial_hit(self):
        assert recall_at_k(["a", "b", "c"], {"a", "x"}, 3) == 0.5

    def test_top_k_window(self):
        # K=2 截断, "b" 应被排除
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5

    def test_no_relevant(self):
        assert recall_at_k(["a"], set(), 5) == 0.0

    def test_no_retrieved(self):
        assert recall_at_k([], {"a"}, 5) == 0.0


class TestPrecisionAtK:
    def test_basic(self):
        assert precision_at_k(["a", "b", "c"], {"a"}, 3) == pytest.approx(1 / 3)

    def test_full_precision(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_zero_k(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["b", "a"], {"a"}) == 0.5

    def test_fifth_position(self):
        assert reciprocal_rank(["x", "y", "z", "w", "a"], {"a"}) == pytest.approx(0.2)

    def test_no_hit(self):
        assert reciprocal_rank(["b", "c"], {"a"}) == 0.0

    def test_no_relevant(self):
        assert reciprocal_rank(["a", "b"], set()) == 0.0


class TestDCG:
    def test_first_position(self):
        # 单条命中在 rank 0 → 1 / log2(2) = 1.0
        assert dcg_at_k(["a"], {"a"}, 5) == pytest.approx(1.0)

    def test_two_positions(self):
        # ["a", "b"] both relevant: 1/log2(2) + 1/log2(3) = 1 + 0.6309
        expected = 1.0 + 1.0 / math.log2(3)
        assert dcg_at_k(["a", "b"], {"a", "b"}, 5) == pytest.approx(expected)

    def test_irrelevant_skipped(self):
        # ["x", "a"]: rank 1 命中 → 1/log2(3) ≈ 0.6309
        assert dcg_at_k(["x", "a"], {"a"}, 5) == pytest.approx(1.0 / math.log2(3))


class TestNDCG:
    def test_perfect_ranking(self):
        """所有相关项排在最前 → nDCG = 1.0."""
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """所有相关项排在 K 之外 → nDCG = 0."""
        assert ndcg_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_partial_ranking(self):
        """["x", "a"] vs 期望 ["a"] 排第一:
        DCG = 1/log2(3) ≈ 0.6309
        IDCG = 1.0
        nDCG ≈ 0.6309
        """
        assert ndcg_at_k(["x", "a"], {"a"}, 5) == pytest.approx(1.0 / math.log2(3))

    def test_no_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), 5) == 0.0


class TestMeanAndPercentile:
    def test_mean(self):
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_mean_empty(self):
        assert mean([]) == 0.0

    def test_percentile_p50_odd(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3

    def test_percentile_p95(self):
        # P95 of 0..100 should be 95
        assert percentile(list(range(101)), 95) == pytest.approx(95)

    def test_percentile_empty(self):
        assert percentile([], 50) == 0.0


class TestPerQueryMetrics:
    def test_full_pack(self):
        retrieved = ["a", "b", "x", "y"]
        relevant = ["a", "b"]
        m = per_query_metrics(retrieved, relevant, k_values=(1, 3, 5))

        # K=1: 命中 1 / 总相关 2 = 0.5
        assert m["recall@1"] == 0.5
        assert m["precision@1"] == 1.0
        assert m["hit@1"] == 1.0
        # K=3: 两条都命中 → recall=1, precision=2/3
        assert m["recall@3"] == 1.0
        assert m["precision@3"] == pytest.approx(2 / 3)
        # MRR = 1.0 (rank 0 命中)
        assert m["mrr"] == 1.0
        # nDCG@3: 理想 (a,b 在 0,1) DCG = 1 + 1/log2(3); 实际也是 → 1.0
        assert m["ndcg@3"] == pytest.approx(1.0)
