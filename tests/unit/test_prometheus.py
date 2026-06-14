"""Prometheus exporter 单元测试 — 验证文本格式正确性 + 指标命名规范."""
from __future__ import annotations

import pytest

from app.utils.metrics import metrics
from app.utils.prometheus import (
    PROMETHEUS_CONTENT_TYPE,
    _percentile,
    _sanitize_metric_name,
    render_prometheus_metrics,
)


@pytest.fixture(autouse=True)
def _reset_metrics():
    """每个测试前清空 metrics 状态, 避免串扰."""
    metrics._counters.clear()
    metrics._histograms.clear()
    metrics._gauges.clear()
    metrics._costs.clear()
    yield


class TestSanitizeMetricName:
    def test_dots_to_underscores(self):
        assert _sanitize_metric_name("recall.total.latency") == "recall_total_latency"

    def test_dashes_to_underscores(self):
        assert _sanitize_metric_name("foo-bar-baz") == "foo_bar_baz"

    def test_alphanumeric_passthrough(self):
        assert _sanitize_metric_name("write_episodic") == "write_episodic"

    def test_starts_with_digit_prefixed(self):
        # Prometheus metric 不能以数字开头
        assert _sanitize_metric_name("4_signals") == "_4_signals"

    def test_special_chars(self):
        # 任何非 alphanumeric 都换成 _
        assert _sanitize_metric_name("foo/bar:baz") == "foo_bar_baz"


class TestPercentile:
    def test_p50_odd(self):
        # 5 个值, P50 应是中间那个
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0

    def test_p95(self):
        values = sorted([1.0 * i for i in range(1, 101)])  # 1..100
        # 索引 95 对应值 96
        assert _percentile(values, 0.95) == 96.0

    def test_empty_returns_zero(self):
        assert _percentile([], 0.5) == 0.0


class TestRenderPrometheusMetrics:
    def test_counter_format(self):
        metrics.incr("recall.invocations", 5)
        metrics.incr("write.episodic", 10)
        body = render_prometheus_metrics()

        # 含正确前缀
        assert "memocortex_recall_invocations_total 5" in body
        assert "memocortex_write_episodic_total 10" in body
        # 含 HELP 和 TYPE 行
        assert "# HELP memocortex_recall_invocations_total" in body
        assert "# TYPE memocortex_recall_invocations_total counter" in body

    def test_histogram_summary_format(self):
        # 注入若干样本
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            metrics.observe("recall.latency", v)

        body = render_prometheus_metrics()
        assert "memocortex_recall_latency_milliseconds" in body
        assert 'quantile="0.5"' in body
        assert 'quantile="0.95"' in body
        assert 'quantile="0.99"' in body
        assert "_count 5" in body
        # sum 应是 150
        assert "_sum 150.00" in body

    def test_gauge_format(self):
        metrics.set_gauge("active_users", 42)
        body = render_prometheus_metrics()
        assert "memocortex_active_users 42" in body
        assert "# TYPE memocortex_active_users gauge" in body

    def test_cost_format(self):
        metrics.add_cost("llm.deepseek", 0.0042)
        body = render_prometheus_metrics()
        assert "memocortex_llm_deepseek_usd_total" in body

    def test_empty_metrics_returns_minimal(self):
        body = render_prometheus_metrics()
        # 空指标应返回有效格式 (空字符串或仅换行), 不应崩
        assert isinstance(body, str)

    def test_content_type_constant(self):
        # Prometheus 标准 content-type
        assert "text/plain" in PROMETHEUS_CONTENT_TYPE
        assert "version=0.0.4" in PROMETHEUS_CONTENT_TYPE

    def test_no_special_chars_in_metric_names(self):
        """Prometheus 指标名只允许 [a-zA-Z_:][a-zA-Z0-9_:]*"""
        metrics.incr("foo.bar.baz", 1)
        metrics.observe("hello-world.latency", 5.0)
        body = render_prometheus_metrics()

        # 提取所有指标名行 (非 #, 非空)
        for line in body.split("\n"):
            if not line or line.startswith("#"):
                continue
            # 拿 metric_name 部分 (去掉值和 labels)
            name = line.split("{")[0].split(" ")[0]
            if not name:
                continue
            # 不应含 . 或 -
            assert "." not in name, f"Metric name has '.': {name}"
            assert "-" not in name, f"Metric name has '-': {name}"
