"""Pattern Miner 分组逻辑单元测试 — 不调 LLM, 只测信号 → bucket 映射.

挖掘策略关键:
  - 按 (signal_type, sorted(context_tags) 拼串) 分组
  - 同类型同标签集合视为同一模式 (顺序无关)
  - 出现次数 < min_occurrences 的不挖掘
"""
from __future__ import annotations

from app.pattern.miner import _group_signals


def _sig(signal_type: str, *tags: str) -> dict:
    return {"signal_type": signal_type, "context_tags": list(tags)}


class TestGroupSignals:
    def test_empty_input(self):
        assert _group_signals([]) == {}

    def test_single_signal(self):
        buckets = _group_signals([_sig("regenerate_request", "code", "python")])
        assert len(buckets) == 1
        key = next(iter(buckets))
        assert key == ("regenerate_request", "code+python")

    def test_tag_order_independent(self):
        """同样的标签不同顺序应进同一桶."""
        signals = [
            _sig("regenerate_request", "code", "python"),
            _sig("regenerate_request", "python", "code"),
            _sig("regenerate_request", "code", "python"),
        ]
        buckets = _group_signals(signals)
        assert len(buckets) == 1
        assert len(next(iter(buckets.values()))) == 3

    def test_different_signal_types_separate(self):
        signals = [
            _sig("regenerate_request", "code"),
            _sig("explicit_correction", "code"),
        ]
        buckets = _group_signals(signals)
        assert len(buckets) == 2

    def test_different_tags_separate(self):
        signals = [
            _sig("regenerate_request", "code"),
            _sig("regenerate_request", "writing"),
        ]
        buckets = _group_signals(signals)
        assert len(buckets) == 2

    def test_empty_tags_become_empty_string(self):
        signals = [
            _sig("regenerate_request"),
            _sig("regenerate_request"),
        ]
        buckets = _group_signals(signals)
        assert len(buckets) == 1
        key = next(iter(buckets))
        assert key == ("regenerate_request", "")

    def test_none_tags_treated_as_empty(self):
        """context_tags 显式 None 应等价于空列表."""
        signals = [
            {"signal_type": "regenerate_request", "context_tags": None},
            {"signal_type": "regenerate_request", "context_tags": []},
        ]
        buckets = _group_signals(signals)
        assert len(buckets) == 1
        assert len(next(iter(buckets.values()))) == 2

    def test_grouping_preserves_signals(self):
        """分组后每桶的总信号数应等于输入."""
        signals = [
            _sig("a", "x"),
            _sig("a", "y"),
            _sig("a", "x"),
            _sig("b", "x"),
        ]
        buckets = _group_signals(signals)
        total = sum(len(v) for v in buckets.values())
        assert total == 4
        assert len(buckets) == 3  # (a,x)/(a,y)/(b,x)
