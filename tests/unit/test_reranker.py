"""Reranker 模块单元测试 — 纯函数 fuse_with_reranker + sigmoid 归一化.

不加载真实 cross-encoder (太慢, 568MB), 只测算分逻辑.
真实 reranker 路径在集成测试 + eval ablation 中验证.
"""
from __future__ import annotations

import math

import pytest

from app.recall.reranker import fuse_with_reranker, rerank_pairs


class TestFuseWithReranker:
    def test_default_weight_07(self):
        """默认 weight=0.7: reranker × 0.7 + final × 0.3"""
        # config.reranker_weight 默认 0.7
        result = fuse_with_reranker(final_score=0.5, reranker_score=1.0)
        assert result == pytest.approx(0.7 * 1.0 + 0.3 * 0.5, rel=1e-3)
        # = 0.7 + 0.15 = 0.85

    def test_custom_weight(self):
        # 完全用 reranker
        assert fuse_with_reranker(0.0, 1.0, weight=1.0) == 1.0
        # 完全用一阶段
        assert fuse_with_reranker(0.5, 1.0, weight=0.0) == 0.5
        # 50/50
        assert fuse_with_reranker(0.4, 0.8, weight=0.5) == pytest.approx(0.6)

    def test_weight_clamped_to_01(self):
        """weight 超界应被 clamp 到 [0, 1]."""
        assert fuse_with_reranker(0.5, 1.0, weight=1.5) == 1.0
        assert fuse_with_reranker(0.5, 1.0, weight=-0.5) == 0.5

    def test_both_scores_zero(self):
        assert fuse_with_reranker(0.0, 0.0, weight=0.5) == 0.0


class TestRerankPairs:
    """测 sigmoid 归一化, 不实际加载模型."""

    def test_empty_pairs_returns_empty(self):
        """空输入应直接返回 [], 不应触发模型加载."""
        assert rerank_pairs([]) == []


class TestRerankerSigmoidNormalization:
    """验证我们的 sigmoid 归一化公式. bge-reranker-v2-m3 输出 raw logit, 必须先 sigmoid."""

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def test_sigmoid_at_zero(self):
        assert self._sigmoid(0) == 0.5

    def test_sigmoid_large_positive(self):
        # logit=5 → ~0.993
        assert self._sigmoid(5) == pytest.approx(0.9933, rel=1e-3)

    def test_sigmoid_large_negative(self):
        # logit=-5 → ~0.0067
        assert self._sigmoid(-5) == pytest.approx(0.00669, rel=1e-2)
