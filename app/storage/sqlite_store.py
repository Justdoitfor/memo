"""SQLAlchemy 2.0 + SQLite 异步实现 MetadataStore Protocol

4 张表:
  memories            — MemoryRecord 持久化备份 (真源)
  reflective_profiles — 用户画像 JSON Blob
  arbitration_logs    — 冲突仲裁审计
  eval_runs           — Eval 跑分历史 (跨版本回归对比)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    delete,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import config
from app.models import Entity, MemoryRecord, MemoryType


class Base(DeclarativeBase):
    pass


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          ORM Models                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


class MemoryORM(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(120), index=True)
    session_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    type: Mapped[str] = mapped_column(String(20), index=True)
    content: Mapped[str] = mapped_column(Text)
    structured: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    # Phase 1: 置信度生命周期
    confidence_score: Mapped[float] = mapped_column(Float, default=0.7)
    source_type: Mapped[str] = mapped_column(String(30), default="explicit_statement", index=True)
    staleness_signal: Mapped[bool] = mapped_column(Integer, default=0)  # SQLite 用 INT
    superseded_by: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    decay_rate: Mapped[float] = mapped_column(Float, default=0.01)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    last_recalled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    recall_count: Mapped[int] = mapped_column(Integer, default=0)
    ttl_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    tier: Mapped[str] = mapped_column(String(10), default="hot")
    storage_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    tags: Mapped[str] = mapped_column(String(500), default="")  # CSV
    source: Mapped[str] = mapped_column(String(30), default="explicit")


class ReflectiveProfileORM(Base):
    __tablename__ = "reflective_profiles"

    user_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    profile: Mapped[dict[str, Any]] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ArbitrationLogORM(Base):
    __tablename__ = "arbitration_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(120), index=True)
    subject: Mapped[str] = mapped_column(String(200))
    predicate: Mapped[str] = mapped_column(String(100))
    old_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_value: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(String(20))
    reasoning: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class EvalRunORM(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suite: Mapped[str] = mapped_column(String(80), index=True)
    score: Mapped[float] = mapped_column(Float)
    details: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class BehaviorSignalORM(Base):
    """Phase 2: 行为信号原始记录, Pattern Miner 聚合后挖掘为 Implicit Memory."""

    __tablename__ = "behavior_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(120), index=True)
    session_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    signal_type: Mapped[str] = mapped_column(String(40), index=True)
    context_tags: Mapped[str] = mapped_column(String(500), default="")  # CSV
    memory_ids_in_context: Mapped[str] = mapped_column(String(500), default="")  # CSV
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class EntityORM(Base):
    """P0: 实体节点 — 知识图谱中的消解后唯一实体."""

    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(120), index=True)
    name: Mapped[str] = mapped_column(String(200))
    aliases: Mapped[str] = mapped_column(String(1000), default="")  # CSV
    entity_type: Mapped[str] = mapped_column(String(40), default="person")
    summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      Helpers: ORM ↔ Pydantic                         ║
# ╚══════════════════════════════════════════════════════════════════════╝


def _orm_to_record(o: MemoryORM) -> MemoryRecord:
    return MemoryRecord(
        id=o.id,
        user_id=o.user_id,
        session_id=o.session_id,
        type=MemoryType(o.type),
        content=o.content,
        structured=o.structured or {},
        importance=o.importance,
        confidence_score=o.confidence_score,
        source_type=o.source_type,
        staleness_signal=bool(o.staleness_signal),
        superseded_by=o.superseded_by,
        decay_rate=o.decay_rate,
        created_at=o.created_at,
        last_recalled_at=o.last_recalled_at,
        recall_count=o.recall_count,
        ttl_at=o.ttl_at,
        tier=o.tier,
        storage_uri=o.storage_uri,
        tags=[t for t in (o.tags or "").split(",") if t],
        source=o.source,
    )


def _record_to_orm(r: MemoryRecord) -> MemoryORM:
    return MemoryORM(
        id=r.id,
        user_id=r.user_id,
        session_id=r.session_id,
        type=r.type.value,
        content=r.content,
        structured=r.structured,
        importance=r.importance,
        confidence_score=r.confidence_score,
        source_type=r.source_type,
        staleness_signal=1 if r.staleness_signal else 0,
        superseded_by=r.superseded_by,
        decay_rate=r.decay_rate,
        created_at=r.created_at,
        last_recalled_at=r.last_recalled_at,
        recall_count=r.recall_count,
        ttl_at=r.ttl_at,
        tier=r.tier,
        storage_uri=r.storage_uri,
        tags=",".join(r.tags),
        source=r.source,
    )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      SQLiteMetadataStore                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


class SQLiteMetadataStore:
    """MetadataStore 的 SQLite 实现 (异步)."""

    def __init__(self) -> None:
        config.ensure_dirs()
        self._engine = create_async_engine(
            config.sqlite_url,
            echo=False,
            future=True,
        )
        self._sessionmaker = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info(f"SQLiteMetadataStore 初始化 — url={config.sqlite_url}")

    async def init_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("SQLite schema 已就绪")

    # ── Memory ─────────────────────────────────────────────────────────
    async def upsert_memory(self, record: MemoryRecord) -> None:
        async with self._sessionmaker() as session:
            existing = await session.get(MemoryORM, record.id)
            if existing:
                for k, v in _record_to_orm(record).__dict__.items():
                    if k.startswith("_"):
                        continue
                    setattr(existing, k, v)
            else:
                session.add(_record_to_orm(record))
            await session.commit()

    async def get_memory(self, memory_id: str) -> MemoryRecord | None:
        async with self._sessionmaker() as session:
            orm = await session.get(MemoryORM, memory_id)
            return _orm_to_record(orm) if orm else None

    async def batch_get_memories(self, memory_ids: list[str]) -> list[MemoryRecord]:
        """批量查询记忆 — 用于 BM25 补充候选的一次性拉取 (替代逐条 N+1 查询)."""
        if not memory_ids:
            return []
        async with self._sessionmaker() as session:
            stmt = select(MemoryORM).where(MemoryORM.id.in_(memory_ids))
            result = await session.execute(stmt)
            return [_orm_to_record(o) for o in result.scalars().all()]

    async def list_memories(
        self,
        user_id: str,
        memory_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        async with self._sessionmaker() as session:
            stmt = select(MemoryORM).where(MemoryORM.user_id == user_id)
            if memory_type:
                stmt = stmt.where(MemoryORM.type == memory_type)
            if since:
                stmt = stmt.where(MemoryORM.created_at >= since)
            stmt = stmt.order_by(MemoryORM.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return [_orm_to_record(o) for o in result.scalars().all()]

    async def delete_memory(self, memory_id: str) -> bool:
        async with self._sessionmaker() as session:
            orm = await session.get(MemoryORM, memory_id)
            if not orm:
                return False
            await session.delete(orm)
            await session.commit()
            return True

    async def delete_all_memories(self, user_id: str) -> int:
        """批量删除某用户所有记忆 — 单条 DELETE WHERE, 不用全量加载."""
        async with self._sessionmaker() as session:
            stmt = delete(MemoryORM).where(MemoryORM.user_id == user_id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount

    async def delete_all_signals(self, user_id: str) -> int:
        """批量删除某用户所有行为信号."""
        async with self._sessionmaker() as session:
            stmt = delete(BehaviorSignalORM).where(BehaviorSignalORM.user_id == user_id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount

    # ── Reflective Profile ─────────────────────────────────────────────
    async def upsert_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        async with self._sessionmaker() as session:
            existing = await session.get(ReflectiveProfileORM, user_id)
            if existing:
                existing.profile = profile
                existing.updated_at = datetime.now()
            else:
                session.add(
                    ReflectiveProfileORM(
                        user_id=user_id, profile=profile, updated_at=datetime.now()
                    )
                )
            await session.commit()

    async def get_profile(self, user_id: str) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            orm = await session.get(ReflectiveProfileORM, user_id)
            if not orm:
                return None
            return {"profile": orm.profile, "updated_at": orm.updated_at.isoformat()}

    # ── Arbitration ────────────────────────────────────────────────────
    async def log_arbitration(self, entry: dict[str, Any]) -> None:
        async with self._sessionmaker() as session:
            session.add(
                ArbitrationLogORM(
                    user_id=entry["user_id"],
                    subject=entry["subject"],
                    predicate=entry["predicate"],
                    old_value=entry.get("old_value"),
                    new_value=entry["new_value"],
                    action=entry["action"],
                    reasoning=entry.get("reasoning", ""),
                    confidence=float(entry.get("confidence", 1.0)),
                )
            )
            await session.commit()

    async def list_arbitrations(
        self, user_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        async with self._sessionmaker() as session:
            stmt = (
                select(ArbitrationLogORM)
                .where(ArbitrationLogORM.user_id == user_id)
                .order_by(ArbitrationLogORM.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [
                {
                    "id": o.id,
                    "user_id": o.user_id,
                    "subject": o.subject,
                    "predicate": o.predicate,
                    "old_value": o.old_value,
                    "new_value": o.new_value,
                    "action": o.action,
                    "reasoning": o.reasoning,
                    "confidence": o.confidence,
                    "created_at": o.created_at.isoformat(),
                }
                for o in result.scalars().all()
            ]

    # ── Eval ───────────────────────────────────────────────────────────
    async def save_eval_run(
        self, suite: str, score: float, details: dict[str, Any]
    ) -> None:
        async with self._sessionmaker() as session:
            session.add(EvalRunORM(suite=suite, score=score, details=details))
            await session.commit()

    async def last_eval(self, suite: str) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            stmt = (
                select(EvalRunORM)
                .where(EvalRunORM.suite == suite)
                .order_by(EvalRunORM.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            orm = result.scalars().first()
            if not orm:
                return None
            return {
                "suite": orm.suite,
                "score": orm.score,
                "details": orm.details,
                "created_at": orm.created_at.isoformat(),
            }

    async def list_eval_runs(self, suite: str, limit: int = 20) -> list[dict[str, Any]]:
        """查询某 suite 的历史跑分 (按时间倒序)."""
        async with self._sessionmaker() as session:
            stmt = (
                select(EvalRunORM)
                .where(EvalRunORM.suite == suite)
                .order_by(EvalRunORM.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [
                {
                    "suite": o.suite,
                    "score": o.score,
                    "details": o.details,
                    "created_at": o.created_at.isoformat(),
                }
                for o in result.scalars().all()
            ]

    # ── Behavior Signals (Phase 2) ─────────────────────────────────────
    async def add_signal(
        self,
        user_id: str,
        signal_type: str,
        context_tags: list[str] | None = None,
        memory_ids_in_context: list[str] | None = None,
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> int:
        async with self._sessionmaker() as session:
            row = BehaviorSignalORM(
                user_id=user_id,
                session_id=session_id,
                signal_type=signal_type,
                context_tags=",".join(context_tags or []),
                memory_ids_in_context=",".join(memory_ids_in_context or []),
                extra=extra or {},
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row.id

    async def list_signals(
        self,
        user_id: str,
        signal_type: str | None = None,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        async with self._sessionmaker() as session:
            stmt = select(BehaviorSignalORM).where(BehaviorSignalORM.user_id == user_id)
            if signal_type:
                stmt = stmt.where(BehaviorSignalORM.signal_type == signal_type)
            if since:
                stmt = stmt.where(BehaviorSignalORM.created_at >= since)
            stmt = stmt.order_by(BehaviorSignalORM.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return [
                {
                    "id": o.id,
                    "user_id": o.user_id,
                    "session_id": o.session_id,
                    "signal_type": o.signal_type,
                    "context_tags": [t for t in (o.context_tags or "").split(",") if t],
                    "memory_ids_in_context": [t for t in (o.memory_ids_in_context or "").split(",") if t],
                    "extra": o.extra or {},
                    "created_at": o.created_at.isoformat(),
                }
                for o in result.scalars().all()
            ]

    async def count_signals(self, user_id: str) -> int:
        async with self._sessionmaker() as session:
            stmt = select(func.count()).select_from(BehaviorSignalORM).where(
                BehaviorSignalORM.user_id == user_id
            )
            result = await session.execute(stmt)
            return result.scalar() or 0

    async def list_active_users(self, days: int = 7) -> list[str]:
        """查询最近 N 天有记忆活动的用户 (created_at 或 last_recalled_at 在窗口内)."""
        from datetime import datetime, timedelta

        since = datetime.now() - timedelta(days=days)
        async with self._sessionmaker() as session:
            stmt = (
                select(MemoryORM.user_id)
                .where(
                    (MemoryORM.created_at >= since)
                    | (MemoryORM.last_recalled_at >= since)
                )
                .distinct()
            )
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]

    async def consistency_check(self, user_id: str) -> dict[str, Any]:
        """比对 SQLite 和 ChromaDB 中同一 user_id 的记录一致性.

        SQLite 是真源 (source of truth), ChromaDB 缺失的记录将从 SQLite 补偿写入.
        返回统计: {sqlite_count, chroma_missing, chroma_fixed, chroma_only}
        """
        from app.storage import get_vector_store

        vec = get_vector_store()

        # 1. SQLite 侧: 查所有记忆 id
        async with self._sessionmaker() as session:
            stmt = select(MemoryORM.id).where(MemoryORM.user_id == user_id)
            result = await session.execute(stmt)
            sqlite_ids = {row[0] for row in result.all()}

        # 2. ChromaDB 侧: 查所有记忆 id (用 count 确认非空后, get 全量)
        chroma_ids: set[str] = set()
        try:
            # Chroma 没有直接 list_ids(where) API, 用 get 拿全量
            vec_client = vec  # ChromaVectorStore instance
            if hasattr(vec_client, "_collection") and vec_client._collection.count() > 0:
                res = vec_client._collection.get(
                    where={"user_id": user_id}, include=[]
                )
                chroma_ids = set(res.get("ids", []))
        except Exception as e:
            logger.warning(f"consistency_check ChromaDB 查询失败: {e}")

        # 3. 找出差异
        missing_in_chroma = sqlite_ids - chroma_ids  # SQLite 有但 ChromaDB 没有 → 需补偿
        only_in_chroma = chroma_ids - sqlite_ids  # ChromaDB 有但 SQLite 没有 → 需清理

        # 4. 补偿: 将 ChromaDB 缺失的记录从 SQLite 读出并写入 ChromaDB
        fixed = 0
        if missing_in_chroma:
            records = await self.batch_get_memories(list(missing_in_chroma))
            for r in records:
                try:
                    await vec.add(r)
                    fixed += 1
                except Exception as e:
                    logger.warning(f"consistency 补偿 ChromaDB 写入失败 {r.id}: {e}")

        # 5. 清理: ChromaDB 中多余的记录从 ChromaDB 删除 (SQLite 为真源)
        cleaned = 0
        for mid in only_in_chroma:
            try:
                await vec.delete(mid, user_id)
                cleaned += 1
            except Exception as e:
                logger.warning(f"consistency 清理 ChromaDB 失败 {mid}: {e}")

        result = {
            "user_id": user_id,
            "sqlite_count": len(sqlite_ids),
            "chroma_missing": len(missing_in_chroma),
            "chroma_fixed": fixed,
            "chroma_only": len(only_in_chroma),
            "chroma_cleaned": cleaned,
        }
        if fixed > 0 or cleaned > 0:
            logger.info(f"[Consistency] {result}")
        return result

    # ── Entity Store (P0) ──────────────────────────────────────────────

    async def upsert_entity(self, entity: Entity) -> None:
        async with self._sessionmaker() as session:
            existing = await session.get(EntityORM, entity.id)
            if existing:
                existing.name = entity.name
                existing.aliases = ",".join(entity.aliases)
                existing.entity_type = entity.entity_type
                existing.summary = entity.summary
                existing.updated_at = datetime.now()
            else:
                session.add(EntityORM(
                    id=entity.id,
                    user_id=entity.user_id,
                    name=entity.name,
                    aliases=",".join(entity.aliases),
                    entity_type=entity.entity_type,
                    summary=entity.summary,
                    created_at=entity.created_at,
                    updated_at=entity.updated_at,
                ))
            await session.commit()

    async def get_entity(self, entity_id: str) -> Entity | None:
        async with self._sessionmaker() as session:
            orm = await session.get(EntityORM, entity_id)
            if not orm:
                return None
            return Entity(
                id=orm.id,
                user_id=orm.user_id,
                name=orm.name,
                aliases=[a for a in (orm.aliases or "").split(",") if a],
                entity_type=orm.entity_type,
                summary=orm.summary,
                created_at=orm.created_at,
                updated_at=orm.updated_at,
            )

    async def list_entities(
        self,
        user_id: str,
        entity_type: str | None = None,
        limit: int = 200,
    ) -> list[Entity]:
        async with self._sessionmaker() as session:
            stmt = select(EntityORM).where(EntityORM.user_id == user_id)
            if entity_type:
                stmt = stmt.where(EntityORM.entity_type == entity_type)
            stmt = stmt.order_by(EntityORM.updated_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return [
                Entity(
                    id=o.id,
                    user_id=o.user_id,
                    name=o.name,
                    aliases=[a for a in (o.aliases or "").split(",") if a],
                    entity_type=o.entity_type,
                    summary=o.summary,
                    created_at=o.created_at,
                    updated_at=o.updated_at,
                )
                for o in result.scalars().all()
            ]

    async def find_entity_by_name(self, user_id: str, name: str) -> Entity | None:
        """精确名/别名匹配查找实体."""
        async with self._sessionmaker() as session:
            stmt = select(EntityORM).where(EntityORM.user_id == user_id)
            result = await session.execute(stmt)
            for o in result.scalars().all():
                if o.name == name or name in [a for a in (o.aliases or "").split(",") if a]:
                    return Entity(
                        id=o.id,
                        user_id=o.user_id,
                        name=o.name,
                        aliases=[a for a in (o.aliases or "").split(",") if a],
                        entity_type=o.entity_type,
                        summary=o.summary,
                        created_at=o.created_at,
                        updated_at=o.updated_at,
                    )
            return None

    async def delete_entity(self, entity_id: str) -> bool:
        async with self._sessionmaker() as session:
            orm = await session.get(EntityORM, entity_id)
            if not orm:
                return False
            await session.delete(orm)
            await session.commit()
            return True

    async def delete_all_entities(self, user_id: str) -> int:
        """批量删除某用户所有实体."""
        async with self._sessionmaker() as session:
            stmt = delete(EntityORM).where(EntityORM.user_id == user_id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount

