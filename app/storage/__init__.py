"""存储层: Protocol + 4 个 MVP 实现 + 全局单例 getter

业务代码应只 import 这里的 getter (get_vector_store / get_kg / get_metadata / get_cold),
不要直接 import 具体实现类, 这样生产替换为 Milvus/Neo4j/PG 时不用动业务代码.
"""

from __future__ import annotations

from threading import Lock

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
    global _meta
    if _meta is None:
        with _lock:
            if _meta is None:
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
