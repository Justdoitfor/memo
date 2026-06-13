"""字符级 token 估算 — 4 字符/token, 适用中英混合, 无外部依赖."""

from __future__ import annotations

from typing import Callable, TypeVar

_CHARS_PER_TOKEN = 4


def count_tokens(text: str | None) -> int:
    """估算 token 数 (中英混合约 3-4 字符/token).

    简化模型: 中文 1 字符 ≈ 1 token, 英文 ~4 字符/token. 平均 4 字符/token.
    误差对 token 预算控制场景 (Mem0 风格的上下文保护) 完全可接受.
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_cost(tokens: int, price_per_1k: float = 0.0014) -> float:
    """估算 USD 成本, 默认 DeepSeek-chat 输入价 $0.14/M tokens."""
    return tokens / 1000.0 * price_per_1k


T = TypeVar("T")


def truncate_to_budget(
    items: list[T],
    max_tokens: int,
    get_text: Callable[[T], str],
) -> list[T]:
    """按 token 预算保高分前缀截断.

    用法:
        truncate_to_budget(results, 6000, lambda r: r.record.content)

    语义:
      - 输入按相关度倒序 (调用方负责)
      - 累加 token, 超过预算前停止
      - 至少返回 1 条 (即使第一条就超预算; Agent 总要看到点东西, 截内容比静默丢更好)
      - max_tokens <= 0 时不截断, 返回原列表
    """
    if max_tokens <= 0 or not items:
        return items
    used, kept = 0, []
    for i, item in enumerate(items):
        t = count_tokens(get_text(item))
        if used + t > max_tokens and kept:
            break
        kept.append(item)
        used += t
    return kept
