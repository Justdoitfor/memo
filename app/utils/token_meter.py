"""字符级 token 估算 — 4 字符/token, 适用中英混合, 无外部依赖."""

from __future__ import annotations

_CHARS_PER_TOKEN = 4


def count_tokens(text: str | None) -> int:
    """估算 token 数 (中英混合约 3-4 字符/token)."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_cost(tokens: int, price_per_1k: float = 0.0014) -> float:
    """估算 USD 成本, 默认 DeepSeek-chat 输入价 $0.14/M tokens."""
    return tokens / 1000.0 * price_per_1k
