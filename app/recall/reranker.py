"""bge-reranker-v2-m3 二阶段重排.

设计:
  - 一阶段召回 (向量 + BM25 + 4 信号融合) 拿 top-30
  - 二阶段 reranker (cross-encoder) 重新打分, 加权融合最终排序
  - 默认关闭 (config.enable_reranker=False), 显式开启走二阶段路径

性能预算:
  - bge-reranker-v2-m3 cross-encoder, batch 30 条 ~ 100-300ms (CPU)
  - 加权融合: reranker × reranker_weight + final_score × (1 - reranker_weight)
  - reranker_weight 默认 0.7 (reranker 主导), 可通过 config 调

模型加载策略:
  - 懒加载, 第一次调 rerank 时初始化 (避免不开启场景下白白占内存)
  - 单例 + threading.Lock 防并发踩踏
"""
from __future__ import annotations

import asyncio
from threading import Lock
from typing import Any

from loguru import logger

from app.config import config


_reranker: Any | None = None
_lock = Lock()


def get_reranker() -> Any:
    """获取全局 Reranker 实例 — 单例, 线程安全, 懒加载.

    使用 sentence-transformers 的 CrossEncoder, model 从 config.reranker_model 配置.
    默认 BAAI/bge-reranker-v2-m3 (568MB, 多语言).
    """
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                # 延迟 import: sentence_transformers 是一阶段 embedder 的间接依赖,
                # 但 CrossEncoder 类需要显式 import. 不开 reranker 时不应该承担 import 开销.
                from sentence_transformers import CrossEncoder

                logger.info(f"加载 Reranker 模型: {config.reranker_model}")
                _reranker = CrossEncoder(
                    config.reranker_model,
                    max_length=512,
                    device="cpu",
                )
                logger.info("Reranker 模型加载完成")
    return _reranker


def rerank_pairs(
    pairs: list[tuple[str, str]],
) -> list[float]:
    """同步 rerank — pairs = [(query, doc_content), ...], 返回 raw 分数列表.

    bge-reranker-v2-m3 输出的是 logit 而非 [0,1], 需要 sigmoid 归一化.
    """
    if not pairs:
        return []
    reranker = get_reranker()
    scores = reranker.predict(pairs, show_progress_bar=False)
    # sigmoid 归一化到 [0, 1]
    import math
    return [1.0 / (1.0 + math.exp(-float(s))) for s in scores]


async def async_rerank_pairs(
    pairs: list[tuple[str, str]],
) -> list[float]:
    """异步 rerank — 通过 to_thread 不阻塞事件循环."""
    return await asyncio.to_thread(rerank_pairs, pairs)


def fuse_with_reranker(
    final_score: float,
    reranker_score: float,
    weight: float | None = None,
) -> float:
    """加权融合一阶段 final_score 和二阶段 reranker_score.

    Args:
        final_score: 一阶段 4 信号融合后的分数, [0, 1]
        reranker_score: bge-reranker-v2-m3 sigmoid 后的分数, [0, 1]
        weight: reranker 权重, None 用 config.reranker_weight (默认 0.7)
                等价于: result = reranker × weight + final × (1 - weight)
    """
    w = weight if weight is not None else config.reranker_weight
    w = max(0.0, min(1.0, w))
    return reranker_score * w + final_score * (1.0 - w)
