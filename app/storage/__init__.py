"""存储层: Protocol + 4 个 MVP 实现 + 全局单例 getter

业务代码应只 import 这里的 getter (get_vector_store / get_kg / get_metadata / get_cold),
不要直接 import 具体实现类, 这样生产替换为 Milvus/Neo4j/PG 时不用动业务代码.

P1.3 添加: get_metadata() 根据 config.pg_url 是否设置, 自动切 PostgresMetadataStore.
"""

from __future__ import annotations

from threading import Lock

from app.config import config
from app.storage.base import ColdStorage, KnowledgeGraph, MetadataStore, VectorStore
from app.storage.chroma_store import ChromaVectorStore
from app.storage.fs_cold import FileSystemColdStorage
from app.storage.nx_graph import NetworkXGraph
from app.storage.sqlite_store import SQLiteMetadataStore

_lock = Lock()
_vector: VectorStore | None = None
_kg: KnowledgeGraph | None = None
_meta: MetadataStore | None = None
_cold: ColdStorage | None = None


def get_vector_store() -> VectorStore:
    global _vector
    if _vector is None:
        with _lock:
            if _vector is None:
                _vector = ChromaVectorStore()
    return _vector


def get_kg() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        with _lock:
            if _kg is None:
                _kg = NetworkXGraph()
    return _kg


def get_metadata() -> MetadataStore:
    """获取 MetadataStore 单例.

    P1.3: 自动根据 config.pg_url 切实现:
      - pg_url 已设置 → PostgresMetadataStore (生产)
      - pg_url 为空    → SQLiteMetadataStore (MVP / 默认)
    """
    global _meta
    if _meta is None:
        with _lock:
            if _meta is None:
                if config.pg_url:
                    # 延迟 import 避免无 PG 场景下引入 asyncpg 依赖
                    from app.storage.pg_store import PostgresMetadataStore

                    _meta = PostgresMetadataStore(url=config.pg_url)
                else:
                    _meta = SQLiteMetadataStore()
    return _meta


def get_cold() -> ColdStorage:
    global _cold
    if _cold is None:
        with _lock:
            if _cold is None:
                _cold = FileSystemColdStorage()
    return _cold


__all__ = [
    "VectorStore",
    "KnowledgeGraph",
    "MetadataStore",
    "ColdStorage",
    "get_vector_store",
    "get_kg",
    "get_metadata",
    "get_cold",
]
