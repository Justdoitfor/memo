"""LLM 调用成本计量 (P2.3)  — 每次 LLM 调用记录 input/output tokens + 模型 + 耗时 + 估算 USD.

设计原则:
  - 不依赖 LangChain 私有 API. 利用 ChatOpenAI 调用返回的 usage_metadata (langchain 0.3+ 标准字段)
  - usage_metadata 是 BaseMessage 的属性, 通过 LangChain callback 拿到, 或直接 ainvoke 后检查 .usage_metadata
  - 优雅降级: 模型不返回 usage_metadata 时, 用字符级估算填充 (tokenizer 已实现)

成本估算:
  - 主流模型的 per-1k-tokens 价格写在 _MODEL_PRICES 里 (USD)
  - 输入和输出价不同 (e.g. DeepSeek-chat 输入 $0.14/M tokens, 输出 $0.28/M)
  - 未知模型默认走字符级估算 + 0.0014 USD/1k tokens 兜底

集成点:
  - llm_factory.structured_invoke() 末尾调 record_call(...)
  - 计入 metrics: llm.calls.{model}, llm.tokens.input.{model}, llm.tokens.output.{model}
  - 累加 cost: llm.{model}.usd

Prometheus 自动暴露:
  memocortex_llm_calls_deepseek_chat_total
  memocortex_llm_deepseek_chat_usd_total
  memocortex_llm_tokens_input_deepseek_chat_milliseconds (作为 summary)
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from typing import Any

from loguru import logger

from app.utils.metrics import metrics
from app.utils.token_meter import count_tokens
from app.utils.trace_context import get_trace_id


# 主流模型的 per-1M-tokens 价格 (USD), 输入 / 输出分开
# 数据采自各家官网公开价目, 用于成本可观测的近似估算
# 单位: USD per 1 Million tokens
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    # DeepSeek (China)
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.14, 2.19),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic Claude
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    # Qwen (Aliyun DashScope)
    "qwen-turbo": (0.05, 0.20),
    "qwen-plus": (0.40, 1.20),
}

# 未知模型的兜底价 (per 1M tokens)
_DEFAULT_INPUT_PRICE = 0.50
_DEFAULT_OUTPUT_PRICE = 1.50


def _normalize_model_name(model: str) -> str:
    """归一化模型名, 去掉版本后缀.

    e.g. "deepseek-chat-2024-08" → "deepseek-chat"
         "gpt-4o-2024-08-06"     → "gpt-4o"
         "claude-3-5-sonnet-20241022" → "claude-3-5-sonnet"
    """
    if not model:
        return "unknown"
    # 去掉 ISO date 后缀 (-YYYYMMDD or -YYYY-MM-DD)
    s = re.sub(r"-?\d{4}-?\d{2}-?\d{2}$", "", model.lower())
    s = re.sub(r"-?\d{4}-?\d{2}$", "", s)
    return s


def estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int,
) -> float:
    """根据模型 + token 数估算 USD 成本."""
    norm = _normalize_model_name(model)
    in_price, out_price = _MODEL_PRICES.get(
        norm, (_DEFAULT_INPUT_PRICE, _DEFAULT_OUTPUT_PRICE)
    )
    # per 1M tokens → per token
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000.0


def record_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: float,
    purpose: str = "unknown",
    error: str | None = None,
) -> None:
    """记录一次 LLM 调用的完整指标 + log.

    Args:
        model: 模型名 (会自动归一化)
        input_tokens: 输入 tokens (优先用 usage_metadata, 缺失时字符级估算)
        output_tokens: 输出 tokens
        duration_ms: 调用耗时
        purpose: 用途标签 (arbitrator / extractor / pattern_miner / ...), 用于分组分析
        error: 错误信息 (None 表示成功)
    """
    norm_model = _normalize_model_name(model)
    cost = estimate_cost_usd(norm_model, input_tokens, output_tokens)
    total_tokens = input_tokens + output_tokens

    # 计入 Prometheus 指标
    metrics.incr(f"llm.calls.{norm_model}")
    metrics.incr(f"llm.calls.{purpose}")
    if error:
        metrics.incr(f"llm.errors.{norm_model}")
    metrics.incr(f"llm.tokens.input.{norm_model}", input_tokens)
    metrics.incr(f"llm.tokens.output.{norm_model}", output_tokens)
    metrics.incr(f"llm.tokens.total.{norm_model}", total_tokens)
    metrics.observe(f"llm.duration.{norm_model}", duration_ms)
    metrics.add_cost(f"llm.{norm_model}", cost)
    # 全局 cost
    metrics.add_cost("llm.total", cost)

    # 结构化日志 (loguru 自动带 trace_id)
    logger.bind(
        purpose=purpose,
        model=norm_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=round(duration_ms, 2),
        cost_usd=round(cost, 6),
    ).info(
        f"[llm-call] {purpose} model={norm_model} "
        f"in={input_tokens}t out={output_tokens}t "
        f"latency={duration_ms:.0f}ms cost=${cost:.6f}"
        + (f" ERROR={error}" if error else "")
    )


@contextmanager
def llm_call_metering(model: str, purpose: str = "unknown"):
    """上下文管理器 — 调用 LLM 时包一下, 自动测时 + 记 token + 算 cost.

    用法:
        with llm_call_metering(model="deepseek-chat", purpose="arbitrator") as ctx:
            response = await chain.ainvoke(...)
            ctx["input_tokens"] = response.usage_metadata.get("input_tokens", 0)
            ctx["output_tokens"] = response.usage_metadata.get("output_tokens", 0)

    退出时自动调 record_call(...) 写指标 + log.
    异常情况下 ctx["error"] 应被设置.
    """
    ctx: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None,
    }
    start = time.perf_counter()
    try:
        yield ctx
    except Exception as e:
        ctx["error"] = str(e)
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        record_call(
            model=model,
            input_tokens=ctx["input_tokens"],
            output_tokens=ctx["output_tokens"],
            duration_ms=duration_ms,
            purpose=purpose,
            error=ctx["error"],
        )


def extract_usage(message: Any) -> tuple[int, int]:
    """从 LangChain BaseMessage / AIMessage 提取 token usage.

    LangChain 0.3+ 标准字段是 message.usage_metadata = {"input_tokens": N, "output_tokens": N}
    (取代了旧的 .response_metadata.usage 路径).

    返回: (input_tokens, output_tokens). 缺失时返回 (0, 0).
    """
    if message is None:
        return 0, 0

    # LangChain 0.3+ 标准
    usage = getattr(message, "usage_metadata", None)
    if usage:
        return (
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )

    # 旧路径 fallback
    response_metadata = getattr(message, "response_metadata", None) or {}
    usage = response_metadata.get("token_usage") or response_metadata.get("usage", {})
    if usage:
        return (
            int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)),
            int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)),
        )

    return 0, 0


def estimate_tokens_fallback(prompt_str: str, output_str: str) -> tuple[int, int]:
    """LLM 不返回 usage 时, 用字符级估算兜底."""
    return count_tokens(prompt_str), count_tokens(output_str)
