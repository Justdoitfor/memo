"""轻量 metrics 收集器 — MVP 用进程内计数器, 生产可换 Prometheus client."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any


class MetricsCollector:
    """线程安全的进程内指标聚合器.

    采集 4 类指标:
      - counter:   累加 (e.g. memocortex.writes.episodic = 42)
      - histogram: 时延列表, /metrics 返回 P50/P95
      - gauge:     瞬时值
      - cost:      累积成本估算 (USD)
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._costs: dict[str, float] = defaultdict(float)
        self._lock = Lock()

    def incr(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            hist = self._histograms[name]
            hist.append(value)
            # 防爆内存: 保留最近 1000 个样本
            if len(hist) > 1000:
                del hist[: len(hist) - 1000]

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def add_cost(self, name: str, usd: float) -> None:
        with self._lock:
            self._costs[name] += usd

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """with metrics.timer('recall.latency'): ...  →  自动 observe."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000)  # ms

    def snapshot(self) -> dict[str, Any]:
        """供 /metrics 接口返回 — 计算 P50/P95."""
        with self._lock:
            histos = {}
            for k, vals in self._histograms.items():
                if not vals:
                    continue
                sorted_v = sorted(vals)
                n = len(sorted_v)
                histos[k] = {
                    "count": n,
                    "p50_ms": round(sorted_v[n // 2], 2),
                    "p95_ms": round(sorted_v[min(int(n * 0.95), n - 1)], 2),
                    "max_ms": round(sorted_v[-1], 2),
                    "avg_ms": round(sum(sorted_v) / n, 2),
                }
            return {
                "counters": dict(self._counters),
                "histograms": histos,
                "gauges": dict(self._gauges),
                "costs_usd": {k: round(v, 6) for k, v in self._costs.items()},
            }


# 全局单例
metrics = MetricsCollector()
