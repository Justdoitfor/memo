"""PostgreSQL 实现 MetadataStore Protocol — 复用 SQLAlchemy 2.0 ORM 模型.

设计原则:
  - ORM 模型 (MemoryORM, EntityORM, ...) 直接复用 sqlite_store.py 定义,
    SQLAlchemy 是 dialect-agnostic 的, 同一份 ORM 在 PG / SQLite 都能用
  - 业务方法 (upsert_memory, list_memories, ...) 也直接复用 SQLiteMetadataStore 的实现:
    PostgresMetadataStore 继承自 SQLiteMetadataStore, 只覆盖 __init__ (engine 创建)
  - 这就是 Storage Protocol 抽象的真实价值: 21 个契约测试在 PG / SQLite 都过

为什么不重写 asyncpg 直连:
  - SQLAlchemy + asyncpg driver 已经够快 (单条 ~ 1-2ms), 不是热路径瓶颈
  - 重写要维护两套 SQL, 大幅增加 bug 面积, 且生产换 PG 时该 store 还要再调
  - 留给未来真正的性能压测后再判断 (P1.4 bench/ 会出延迟分布数据)

为什么继承而非组合:
  - 21 个契约测试要求 Protocol 方法行为完全一致, 继承零代码重复
  - 业务方法都用 self._sessionmaker(), 不依赖具体 dialect, 父类直接复用
  - 只有 __init__ 需要换 (engine URL + dialect-specific 优化)

PG-specific 优化空间 (留给后续):
  - 用 PG 原生 JSONB 索引代替 SQLAlchemy 默认 JSON
  - asyncpg 连接池调优 (pool_size / max_overflow)
  - PG INSERT ... ON CONFLICT DO UPDATE 替代 SQLAlchemy session.get + setattr
  - 这些都不影响 Protocol 契约, 内部优化即可
"""
from __future__ import annotations

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import config
from app.storage.sqlite_store import Base, SQLiteMetadataStore


class PostgresMetadataStore(SQLiteMetadataStore):
    """MetadataStore 的 PostgreSQL 实现 (异步).

    通过继承 SQLiteMetadataStore 复用所有业务方法 (upsert / list / delete / log_arbitration ...).
    仅覆盖 __init__ — 把 SQLite engine 换成 PG engine, schema init 等行为完全一致.

    用法:
        # .env / 环境变量
        MEMOCORTEX_PG_URL=postgresql+asyncpg://user:pass@localhost:5432/memocortex

        # 代码 (storage/__init__.py 里通过 config.use_postgres 切)
        from app.storage.pg_store import PostgresMetadataStore
        store = PostgresMetadataStore(url="postgresql+asyncpg://...")
        await store.init_schema()
    """

    def __init__(self, url: str | None = None) -> None:
        """覆盖 SQLite store 的 __init__ — 不调用父类 super().__init__() (它会建 SQLite engine).

        Args:
            url: SQLAlchemy URL, 形如 "postgresql+asyncpg://user:pass@host:5432/db".
                 None 时从 config.pg_url 读, 也未配则报错.
        """
        if url is None:
            url = config.pg_url
        if not url:
            raise ValueError(
                "PostgresMetadataStore 需要 url 参数或 MEMOCORTEX_PG_URL 环境变量. "
                "格式: postgresql+asyncpg://user:pass@host:5432/db"
            )
        if not url.startswith("postgresql+asyncpg://"):
            # 兼容 postgresql:// 前缀, 自动加 asyncpg driver
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            else:
                logger.warning(
                    f"PG URL 应使用 postgresql+asyncpg:// 前缀, 实际: {url[:30]}..."
                )

        # 注意: 不调 super().__init__(), 它会建 SQLite engine 写本地文件
        self._engine = create_async_engine(
            url,
            echo=False,
            future=True,
            # PG-specific: 连接池与单线程 SQLite 不同
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # 自动检测连接掉线
        )
        self._sessionmaker = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False,
        )
        # 把 url 里的密码遮掉再 log
        sanitized = _sanitize_url(url)
        logger.info(f"PostgresMetadataStore 初始化 — url={sanitized}")

    async def init_schema(self) -> None:
        """在 PG 上建表 — 复用同一份 Base.metadata, 不需要单独维护 PG schema.

        SQLAlchemy 2.0 的 DDL 自动适配 dialect, JSON 列在 PG 上自动用 JSONB.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("PostgreSQL schema 已就绪")


def _sanitize_url(url: str) -> str:
    """把 SQLAlchemy URL 里的密码遮成 ***, 用于 log/error 输出."""
    # 简单字符串处理: postgresql+asyncpg://user:pass@host... → user:***@host
    if "@" not in url or ":" not in url:
        return url
    proto, rest = url.split("://", 1) if "://" in url else ("", url)
    if "@" not in rest:
        return url
    cred, host = rest.rsplit("@", 1)
    if ":" in cred:
        user, _ = cred.split(":", 1)
        return f"{proto}://{user}:***@{host}"
    return url
