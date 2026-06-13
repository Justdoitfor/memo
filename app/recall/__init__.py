"""召回层入口"""

from app.recall.router import recall_router
from app.recall.signals import (
    compute_importance,
    compute_keyword_match,  # KG 实体重叠度 — 保留供 KG 场景扩展, 当前 Recall Router 用 BM25 score
    compute_temporal_decay,
    compute_vector_sim,
    fuse_signals,
)

__all__ = [
    "recall_router",
    "compute_vector_sim",
    "compute_temporal_decay",
    "compute_keyword_match",
    "compute_importance",
    "fuse_signals",
]
