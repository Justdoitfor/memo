"""cost_meter 单元测试 — 价格表查询 / 模型名归一化 / record_call / 上下文管理器."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.utils.cost_meter import (
    _normalize_model_name,
    estimate_cost_usd,
    extract_usage,
    llm_call_metering,
    record_call,
)
from app.utils.metrics import metrics


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics._counters.clear()
    metrics._histograms.clear()
    metrics._gauges.clear()
    metrics._costs.clear()
    yield


class TestNormalizeModelName:
    def test_passthrough_known(self):
        assert _normalize_model_name("deepseek-chat") == "deepseek-chat"
        assert _normalize_model_name("gpt-4o") == "gpt-4o"

    def test_strip_iso_date_suffix(self):
        # 各种 ISO 日期后缀应被去掉
        assert _normalize_model_name("gpt-4o-2024-08-06") == "gpt-4o"
        assert _normalize_model_name("claude-3-5-sonnet-20241022") == "claude-3-5-sonnet"

    def test_lowercase(self):
        assert _normalize_model_name("GPT-4o") == "gpt-4o"

    def test_empty(self):
        assert _normalize_model_name("") == "unknown"


class TestEstimateCostUsd:
    def test_known_model_deepseek(self):
        # deepseek-chat: input 0.14/M, output 0.28/M
        # 1M in + 1M out = 0.14 + 0.28 = 0.42 USD
        cost = estimate_cost_usd("deepseek-chat", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.42)

    def test_known_model_gpt4o(self):
        # gpt-4o: 2.50 in / 10.00 out per 1M
        # 1k in + 1k out = (2.5 + 10) / 1000 = 0.0125
        cost = estimate_cost_usd("gpt-4o", 1000, 1000)
        assert cost == pytest.approx(0.0125)

    def test_zero_tokens(self):
        assert estimate_cost_usd("deepseek-chat", 0, 0) == 0.0

    def test_unknown_model_uses_default(self):
        # 未知模型走 _DEFAULT_INPUT_PRICE / _DEFAULT_OUTPUT_PRICE
        cost = estimate_cost_usd("some-unknown-model", 1_000_000, 1_000_000)
        # 0.50 + 1.50 = 2.00 USD
        assert cost == pytest.approx(2.00)

    def test_iso_suffix_stripped(self):
        # gpt-4o-2024-08-06 应被识别为 gpt-4o
        assert estimate_cost_usd("gpt-4o-2024-08-06", 1000, 0) == pytest.approx(0.0025)


class TestRecordCall:
    def test_metrics_incremented(self):
        record_call(
            model="deepseek-chat",
            input_tokens=1000,
            output_tokens=500,
            duration_ms=2300.0,
            purpose="arbitrator",
        )
        snap = metrics.snapshot()
        # counter 增加
        assert snap["counters"].get("llm.calls.deepseek-chat") == 1
        assert snap["counters"].get("llm.calls.arbitrator") == 1
        assert snap["counters"].get("llm.tokens.input.deepseek-chat") == 1000
        assert snap["counters"].get("llm.tokens.output.deepseek-chat") == 500
        # cost 累加
        cost = snap["costs_usd"].get("llm.deepseek-chat", 0)
        assert cost > 0  # 1k in × 0.14 + 0.5k × 0.28 = 0.14 + 0.14 = 0.28 / 1000

    def test_error_increments_error_counter(self):
        record_call(
            model="deepseek-chat",
            input_tokens=0, output_tokens=0,
            duration_ms=100.0,
            purpose="arbitrator",
            error="API timeout",
        )
        snap = metrics.snapshot()
        assert snap["counters"].get("llm.errors.deepseek-chat") == 1


class TestLlmCallMetering:
    @pytest.mark.asyncio
    async def test_context_manager_records(self):
        async def fake_call():
            with llm_call_metering(model="deepseek-chat", purpose="testing") as ctx:
                ctx["input_tokens"] = 200
                ctx["output_tokens"] = 100
                # 不抛错就走完正常路径

        await fake_call()

        snap = metrics.snapshot()
        assert snap["counters"]["llm.calls.deepseek-chat"] == 1
        assert snap["counters"]["llm.tokens.input.deepseek-chat"] == 200

    @pytest.mark.asyncio
    async def test_context_manager_records_on_exception(self):
        with pytest.raises(ValueError):
            with llm_call_metering(model="gpt-4o", purpose="testing") as ctx:
                ctx["input_tokens"] = 50
                raise ValueError("simulated error")

        # 即使抛错也应该记录调用
        snap = metrics.snapshot()
        assert snap["counters"]["llm.calls.gpt-4o"] == 1
        assert snap["counters"].get("llm.errors.gpt-4o") == 1


class TestExtractUsage:
    def test_langchain_03_format(self):
        """LangChain 0.3+ 标准: message.usage_metadata = {"input_tokens": N, "output_tokens": N}"""
        msg = MagicMock()
        msg.usage_metadata = {"input_tokens": 200, "output_tokens": 100}
        msg.response_metadata = None
        in_t, out_t = extract_usage(msg)
        assert in_t == 200
        assert out_t == 100

    def test_legacy_response_metadata_format(self):
        """旧路径: message.response_metadata.token_usage."""
        msg = MagicMock()
        msg.usage_metadata = None
        msg.response_metadata = {
            "token_usage": {"prompt_tokens": 150, "completion_tokens": 75}
        }
        in_t, out_t = extract_usage(msg)
        assert in_t == 150
        assert out_t == 75

    def test_none_message(self):
        in_t, out_t = extract_usage(None)
        assert in_t == 0
        assert out_t == 0

    def test_no_usage_at_all(self):
        msg = MagicMock(spec=[])  # 空对象, 没有 usage_metadata 也没 response_metadata
        msg.usage_metadata = None
        msg.response_metadata = None
        in_t, out_t = extract_usage(msg)
        assert in_t == 0
        assert out_t == 0
