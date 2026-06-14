"""Prometheus metrics 导出器 — 把 MetricsCollector 转成 Prometheus exposition format.

设计原则:
  - 不引入 prometheus_client 依赖, 手写 exposition format (足够简单)
  - 业务代码继续用现有 metrics.incr() / metrics.timer() API, 零改动
  - /metrics endpoint 暴露 Counter / Histogram (P50/P95) / Gauge

格式参考:
  https://prometheus.io/docs/instrumenting/exposition_formats/

例:
  # HELP memocortex_recall_invocations_total Total recall invocations
  # TYPE memocortex_recall_invocations_total counter
  memocortex_recall_invocations_total 1234

  # HELP memocortex_recall_total_latency_milliseconds Recall latency
  # TYPE memocortex_recall_total_latency_milliseconds summary
  memocortex_recall_total_latency_milliseconds{quantile="0.5"} 23.6
  memocortex_recall_total_latency_milliseconds{quantile="0.95"} 32.2
  memocortex_recall_total_latency_milliseconds_count 100

集成 Grafana / Prometheus:
  - prometheus.yml: scrape_configs 加 target memocortex_host:8766/metrics
  - Grafana 查询: rate(memocortex_recall_invocations_total[1m])
"""
from __future__ import annotations

import re

from app.utils.metrics import metrics


def _sanitize_metric_name(name: str) -> str:
    """把 'recall.total.latency' 转成 'recall_total_latency' (Prometheus 命名规范).

    Prometheus 指标名只允许 [a-zA-Z_:][a-zA-Z0-9_:]*
    """
    # 把点和短横线换成下划线
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # 防止以数字开头
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def _percentile(sorted_values: list[float], p: float) -> float:
    """计算百分位 (0-1). p=0.5 → P50, p=0.95 → P95."""
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
    return sorted_values[idx]


def render_prometheus_metrics() -> str:
    """渲染 Prometheus 文本格式. 调用 metrics.snapshot() 拿数据."""
    lines: list[str] = []
    snapshot = metrics.snapshot()
    prefix = "memocortex"

    # ── Counters ──
    for raw_name, value in sorted(snapshot["counters"].items()):
        sanitized = _sanitize_metric_name(raw_name)
        metric_name = f"{prefix}_{sanitized}_total"
        lines.append(f"# HELP {metric_name} Total {raw_name} count")
        lines.append(f"# TYPE {metric_name} counter")
        lines.append(f"{metric_name} {value}")
        lines.append("")

    # ── Histograms (用 summary 格式: count + 分位数) ──
    # 直接读底层 _histograms 拿原始样本, 算分位
    with metrics._lock:  # noqa: SLF001
        for raw_name, samples in sorted(metrics._histograms.items()):  # noqa: SLF001
            if not samples:
                continue
            sanitized = _sanitize_metric_name(raw_name)
            metric_name = f"{prefix}_{sanitized}_milliseconds"
            sorted_samples = sorted(samples)
            n = len(sorted_samples)

            lines.append(f"# HELP {metric_name} Latency for {raw_name} (ms)")
            lines.append(f"# TYPE {metric_name} summary")
            lines.append(
                f'{metric_name}{{quantile="0.5"}} {_percentile(sorted_samples, 0.5):.2f}'
            )
            lines.append(
                f'{metric_name}{{quantile="0.95"}} {_percentile(sorted_samples, 0.95):.2f}'
            )
            lines.append(
                f'{metric_name}{{quantile="0.99"}} {_percentile(sorted_samples, 0.99):.2f}'
            )
            lines.append(f"{metric_name}_count {n}")
            lines.append(f"{metric_name}_sum {sum(sorted_samples):.2f}")
            lines.append("")

    # ── Gauges ──
    for raw_name, value in sorted(snapshot["gauges"].items()):
        sanitized = _sanitize_metric_name(raw_name)
        metric_name = f"{prefix}_{sanitized}"
        lines.append(f"# HELP {metric_name} {raw_name}")
        lines.append(f"# TYPE {metric_name} gauge")
        lines.append(f"{metric_name} {value}")
        lines.append("")

    # ── Cost (累积) ──
    for raw_name, value in sorted(snapshot["costs_usd"].items()):
        sanitized = _sanitize_metric_name(raw_name)
        metric_name = f"{prefix}_{sanitized}_usd_total"
        lines.append(f"# HELP {metric_name} Accumulated cost in USD for {raw_name}")
        lines.append(f"# TYPE {metric_name} counter")
        lines.append(f"{metric_name} {value:.6f}")
        lines.append("")

    return "\n".join(lines) + "\n"


# Prometheus 内容类型 (text 格式 v0.0.4)
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
