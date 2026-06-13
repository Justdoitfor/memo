"""存储层 Protocol 抽象 — 业务代码只依赖这里的接口, 不依赖任何具体实现

MVP 实现:
  VectorStore     → chroma_store.ChromaVectorStore  (ChromaDB 内嵌)
  KnowledgeGraph  → nx_graph.NetworkXGraph          (NetworkX 内存图 + JSON)
  MetadataStore   → sqlite_store.SQLiteMetadataStore (SQLAlchemy + SQLite)
  ColdStorage     → fs_cold.FileSystemColdStorage   (本地目录)

生产替换:
  VectorStore     → langchain-milvus
  KnowledgeGraph  → neo4j-driver
  MetadataStore   → asyncpg + PostgreSQL
  ColdStorage     → boto3 / minio SDK
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from app.models import Entity, GraphPath, MemoryRecord, Triple


@runtime_checkable
class VectorStore(Protocol):
    """向量库 — per-user 隔离, 按 memory_type 分 collection."""

    async def add(self, record: MemoryRecord) -> None:
        """向量化并写入. 自动从 record.content 计算 embedding."""
        ...

    async def add_batch(self, records: list[MemoryRecord]) -> None:
        """批量写入 — 性能优化."""
        ...

    async def search(
        self,
        user_id: str,
        query: str,
        memory_types: list[str] | None = None,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """语义搜索, 返回 (record, similarity) 列表, similarity ∈ [0, 1] 越大越相似."""
        ...

    async def delete(self, memory_id: str, user_id: str) -> bool:
        """按 ID 删除, 返回是否成功."""
        ...

    async def delete_by_user(self, user_id: str) -> int:
        """删除用户所有记忆 — GDPR Right to be Forgotten."""
        ...

    async def update_metadata(
        self, memory_id: str, user_id: str, metadata_patch: dict[str, Any]
    ) -> bool:
        """更新元数据 (importance / last_recalled_at 等)."""
        ...

    async def count(self, user_id: str, memory_type: str | None = None) -> int:
        """统计某用户某类型的记忆数."""
        ...


@runtime_checkable
class KnowledgeGraph(Protocol):
    """知识图谱 — per-user 隔离, 存 (subject, predicate, object) triple.

    Semantic Memory 的"事实"侧索引, 与 VectorStore 双索引互补.
    """

    async def add_triple(self, user_id: str, triple: Triple) -> None:
        """新增 triple, 已存在则 merge metadata."""
        ...

    async def find_triples(
        self,
        user_id: str,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> list[Triple]:
        """模式查询 — 任意维度都可省略."""
        ...

    async def delete_triple(self, user_id: str, triple_id: str) -> bool:
        ...

    async def neighbors(
        self, user_id: str, entity: str, max_hops: int = 2
    ) -> set[str]:
        """BFS 找 entity 在 max_hops 内的所有相关实体 — 用于召回的实体加权信号."""
        ...

    async def delete_by_user(self, user_id: str) -> int:
        ...

    async def persist(self) -> None:
        """快照到磁盘 (NetworkX MVP 用; Neo4j 实现可空操作)."""
        ...

    async def flush(self) -> int:
        """将 dirty 用户的图刷盘 + 清 WAL, 返回刷盘的用户数."""
        ...

    # ── P0: 增强查询 ──────────────────────────────────────────────────

    async def multi_hop_query(
        self,
        user_id: str,
        start_entity: str,
        max_hops: int = 3,
        predicate_filter: list[str] | None = None,
    ) -> list[GraphPath]:
        """多跳路径查询 — 从 start_entity 出发, BFS 发现可达路径.

        e.g. user → works_at → 字节 → has_colleague → Alice
        """
        ...

    async def find_related_entities(
        self,
        user_id: str,
        entity: str,
        relation_chain: list[str] | None = None,
    ) -> list[str]:
        """关系链查询 — 按指定 predicate 链发现相关实体.

        Args:
            relation_chain: 可选 predicate 过滤列表, None 表示所有关系.
        """
        ...

    async def community_detect(
        self, user_id: str, min_size: int = 3
    ) -> list[dict[str, Any]]:
        """社区检测 — 识别强连通实体簇, 返回社区列表.

        每个社区: {community_id, entities, size, summary}
        """
        ...


@runtime_checkable
class MetadataStore(Protocol):
    """关系数据 — 记忆元数据 / 仲裁日志 / Reflective Profile / Eval 结果"""

    async def init_schema(self) -> None:
        """建表 — lifespan 启动时调用."""
        ...

    # ── MemoryRecord (持久化备份, ChromaDB 是 hot, 这里是 cold 元数据真源) ──
    async def upsert_memory(self, record: MemoryRecord) -> None: ...
    async def get_memory(self, memory_id: str) -> MemoryRecord | None: ...
    async def batch_get_memories(self, memory_ids: list[str]) -> list[MemoryRecord]:
        """批量查询记忆 — 用于 BM25 补充候选的一次性拉取."""
        ...
    async def list_memories(
        self,
        user_id: str,
        memory_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]: ...
    async def delete_memory(self, memory_id: str) -> bool: ...
    async def delete_all_memories(self, user_id: str) -> int:
        """批量删除某用户所有记忆 — 单条 DELETE, 不用全量加载."""
        ...
    async def delete_all_signals(self, user_id: str) -> int:
        """批量删除某用户所有行为信号."""
        ...

    # ── Reflective Profile ──
    async def upsert_profile(self, user_id: str, profile: dict[str, Any]) -> None: ...
    async def get_profile(self, user_id: str) -> dict[str, Any] | None: ...

    # ── Arbitration Log ──
    async def log_arbitration(self, entry: dict[str, Any]) -> None: ...
    async def list_arbitrations(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]: ...

    # ── Eval Runs ──
    async def save_eval_run(self, suite: str, score: float, details: dict[str, Any]) -> None: ...
    async def last_eval(self, suite: str) -> dict[str, Any] | None: ...
    async def list_eval_runs(self, suite: str, limit: int = 20) -> list[dict[str, Any]]: ...

    # ── Behavior Signals (Phase 2) ──
    async def add_signal(
        self, user_id: str, signal_type: str,
        context_tags: list[str] | None = None,
        memory_ids_in_context: list[str] | None = None,
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> int: ...
    async def list_signals(
        self, user_id: str, signal_type: str | None = None,
        since: datetime | None = None, limit: int = 200,
    ) -> list[dict[str, Any]]: ...
    async def count_signals(self, user_id: str) -> int: ...

    # ── Active Users (Reflection Worker) ──
    async def list_active_users(self, days: int = 7) -> list[str]:
        """查询最近 N 天有记忆活动的用户列表 (供 Reflection Worker 遍历)."""
        ...

    # ── Consistency Check ──
    async def consistency_check(self, user_id: str) -> dict[str, Any]:
        """比对 SQLite 和 ChromaDB 中同一 user_id 的记录一致性, 补偿缺失."""
        ...


@runtime_checkable
class ColdStorage(Protocol):
    """冷存储 — 长期未访问的记忆 / Reflection 归档的原始数据.

    MVP: 本地文件系统
    生产: S3 / MinIO / OSS
    """

    async def archive(self, key: str, content: str | bytes) -> str:
        """归档, 返回唯一 storage_uri (e.g. fs://cold/abc, s3://bucket/key)."""
        ...

    async def restore(self, storage_uri: str) -> bytes:
        """从 uri 恢复原始内容."""
        ...

    async def delete(self, storage_uri: str) -> bool:
        ...


@runtime_checkable
class EntityStore(Protocol):
    """实体存储 — Entity Resolution 的持久化层 (P0 新增)."""

    async def upsert_entity(self, entity: Entity) -> None: ...
    async def get_entity(self, entity_id: str) -> Entity | None: ...
    async def list_entities(
        self, user_id: str, entity_type: str | None = None, limit: int = 200,
    ) -> list[Entity]: ...
    async def find_entity_by_name(self, user_id: str, name: str) -> Entity | None: ...
    async def delete_entity(self, entity_id: str) -> bool: ...
    async def delete_all_entities(self, user_id: str) -> int: ...
