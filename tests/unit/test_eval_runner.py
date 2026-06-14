"""eval/runner.py 聚合逻辑单元测试 — 不跑真 storage, 只测 _aggregate / render_markdown."""
from __future__ import annotations

import pytest

from eval.runner import QueryResult, RunReport, _aggregate, render_markdown


def _qr(eid: str, scenario: str, metrics: dict, latency: float = 20.0, error=None) -> QueryResult:
    return QueryResult(
        entry_id=eid,
        scenario=scenario,
        user_id=f"u_{eid}",
        query="?",
        expected_mids=["target"],
        retrieved_mids=["target"] if metrics.get("hit@1", 0) else [],
        retrieved_uuids=[],
        metrics=metrics,
        latency_ms=latency,
        error=error,
    )


class TestAggregate:
    def test_overall_mean(self):
        results = [
            _qr("a", "exact_recall", {"recall@5": 1.0, "ndcg@5": 1.0, "mrr": 1.0,
                                       "recall@1": 1.0, "ndcg@1": 1.0, "ndcg@3": 1.0,
                                       "recall@3": 1.0, "precision@1": 1.0, "precision@3": 0.33,
                                       "precision@5": 0.2, "hit@1": 1.0, "hit@3": 1.0,
                                       "hit@5": 1.0, "ndcg@10": 1.0, "recall@10": 1.0,
                                       "precision@10": 0.1, "hit@10": 1.0}),
            _qr("b", "paraphrase", {"recall@5": 0.5, "ndcg@5": 0.6, "mrr": 0.5,
                                     "recall@1": 0.0, "ndcg@1": 0.0, "ndcg@3": 0.5,
                                     "recall@3": 0.5, "precision@1": 0.0, "precision@3": 0.16,
                                     "precision@5": 0.1, "hit@1": 0.0, "hit@3": 1.0,
                                     "hit@5": 1.0, "ndcg@10": 0.6, "recall@10": 0.5,
                                     "precision@10": 0.05, "hit@10": 1.0}),
        ]
        report = _aggregate(results, k_values=(1, 3, 5, 10))
        # overall.recall@5 = (1.0 + 0.5) / 2 = 0.75
        assert report.overall["recall@5"] == 0.75
        assert report.overall["mrr"] == 0.75

    def test_by_scenario_isolation(self):
        results = [
            _qr("a", "exact_recall", {"recall@5": 1.0, "ndcg@5": 1.0, "mrr": 1.0,
                                       "recall@1": 1.0, "ndcg@1": 1.0, "ndcg@3": 1.0,
                                       "recall@3": 1.0, "precision@1": 1.0, "precision@3": 0.33,
                                       "precision@5": 0.2, "hit@1": 1.0, "hit@3": 1.0,
                                       "hit@5": 1.0, "ndcg@10": 1.0, "recall@10": 1.0,
                                       "precision@10": 0.1, "hit@10": 1.0}),
            _qr("b", "exact_recall", {"recall@5": 0.0, "ndcg@5": 0.0, "mrr": 0.0,
                                       "recall@1": 0.0, "ndcg@1": 0.0, "ndcg@3": 0.0,
                                       "recall@3": 0.0, "precision@1": 0.0, "precision@3": 0.0,
                                       "precision@5": 0.0, "hit@1": 0.0, "hit@3": 0.0,
                                       "hit@5": 0.0, "ndcg@10": 0.0, "recall@10": 0.0,
                                       "precision@10": 0.0, "hit@10": 0.0}),
            _qr("c", "paraphrase", {"recall@5": 1.0, "ndcg@5": 1.0, "mrr": 1.0,
                                     "recall@1": 1.0, "ndcg@1": 1.0, "ndcg@3": 1.0,
                                     "recall@3": 1.0, "precision@1": 1.0, "precision@3": 0.33,
                                     "precision@5": 0.2, "hit@1": 1.0, "hit@3": 1.0,
                                     "hit@5": 1.0, "ndcg@10": 1.0, "recall@10": 1.0,
                                     "precision@10": 0.1, "hit@10": 1.0}),
        ]
        report = _aggregate(results, k_values=(1, 3, 5, 10))
        assert report.by_scenario["exact_recall"]["n"] == 2
        assert report.by_scenario["exact_recall"]["recall@5"] == 0.5
        assert report.by_scenario["paraphrase"]["n"] == 1
        assert report.by_scenario["paraphrase"]["recall@5"] == 1.0

    def test_failed_entries_excluded(self):
        results = [
            _qr("a", "exact_recall", {"recall@5": 1.0, "ndcg@5": 1.0, "mrr": 1.0,
                                       "recall@1": 1.0, "ndcg@1": 1.0, "ndcg@3": 1.0,
                                       "recall@3": 1.0, "precision@1": 1.0, "precision@3": 0.33,
                                       "precision@5": 0.2, "hit@1": 1.0, "hit@3": 1.0,
                                       "hit@5": 1.0, "ndcg@10": 1.0, "recall@10": 1.0,
                                       "precision@10": 0.1, "hit@10": 1.0}),
            _qr("err", "exact_recall", {}, error="boom"),
        ]
        report = _aggregate(results, k_values=(1, 3, 5, 10))
        assert report.n_total == 2
        assert report.n_failed == 1
        # failed 不该影响均值
        assert report.overall["recall@5"] == 1.0

    def test_latency_percentiles(self):
        results = [
            _qr(str(i), "exact_recall", {"recall@5": 1.0, "ndcg@5": 1.0, "mrr": 1.0,
                                          "recall@1": 1.0, "ndcg@1": 1.0, "ndcg@3": 1.0,
                                          "recall@3": 1.0, "precision@1": 1.0, "precision@3": 0.33,
                                          "precision@5": 0.2, "hit@1": 1.0, "hit@3": 1.0,
                                          "hit@5": 1.0, "ndcg@10": 1.0, "recall@10": 1.0,
                                          "precision@10": 0.1, "hit@10": 1.0},
                latency=float(i * 10))
            for i in range(1, 11)  # 10, 20, ... 100
        ]
        report = _aggregate(results, k_values=(1, 3, 5, 10))
        assert report.latency["max_ms"] == 100.0
        # P50 of [10, 20, ... 100] should be ~ 55
        assert 50 <= report.latency["p50_ms"] <= 60


class TestRenderMarkdown:
    def test_basic_render(self):
        results = [
            _qr("a", "exact_recall", {"recall@5": 1.0, "ndcg@5": 1.0, "mrr": 1.0,
                                       "recall@1": 1.0, "ndcg@1": 1.0, "ndcg@3": 1.0,
                                       "recall@3": 1.0, "precision@1": 1.0, "precision@3": 0.33,
                                       "precision@5": 0.2, "hit@1": 1.0, "hit@3": 1.0,
                                       "hit@5": 1.0, "ndcg@10": 1.0, "recall@10": 1.0,
                                       "precision@10": 0.1, "hit@10": 1.0}),
        ]
        report = _aggregate(results, k_values=(1, 3, 5, 10))
        report.suite = "test_suite"
        report.weights = (0.4, 0.2, 0.2, 0.2)

        md = render_markdown(report)
        assert "# Eval Report — test_suite" in md
        assert "Weights" in md
        assert "Overall" in md
        assert "By Scenario" in md
        assert "exact_recall" in md
        assert "Latency" in md

    def test_render_no_weights(self):
        report = RunReport(
            suite="x",
            n_total=0, n_failed=0,
            overall={},
            by_scenario={},
            latency={"p50_ms": 0, "p95_ms": 0, "mean_ms": 0, "max_ms": 0},
            weights=None,
            timestamp="2026-06-14",
        )
        md = render_markdown(report)
        assert "config 默认" in md
