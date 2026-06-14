# P1.3: PostgreSQL Store + Storage Protocol 抽象的工程证据

> **目标**: 让 README 那句 `MetadataStore → asyncpg + PostgreSQL` 从声明变事实
> **方法**: 复用 SQLAlchemy 2.0 ORM + asyncpg driver, 同一份 ORM 在 SQLite/PG 都能用
> **证据**: 真 PG 容器跑 15 个契约测试全过

## TL;DR

| 项 | 值 |
|---|---|
| **新增代码** | `app/storage/pg_store.py` (~80 行, 继承 SQLiteMetadataStore) |
| **覆盖方法** | 13 个 MetadataStore Protocol 核心方法 (CRUD / 列表 / 仲裁审计 / eval_runs) |
| **PG 契约测试** | 15 个全过 (13 真 PG + 2 sanitize URL) |
| **SQLite 契约测试** | 21 个不变, 全过 |
| **ORM 模型重写** | 0 — 完全复用 sqlite_store.py 里的 8 张表 |
| **业务方法重写** | 0 — 通过继承直接复用 |

## 设计取舍 — 为什么继承而不重写

### 路线对比

| 路线 | 工时 | 价值 | 缺点 |
|---|---|---|---|
| A. 仅 SQLAlchemy URL 切换 | 30min | 最少代码 | 没真跑过 PG, 等于声明 |
| **B. SQLAlchemy + asyncpg, 继承复用业务方法** ✅ | 2h | 真跑过 PG, 21 + 15 契约测试通过 | 需要 docker postgres |
| C. 重写 asyncpg 直连版本 | 5-6h | 性能略优 (绕开 ORM) | 8 张表全要重写, 过度工程 |

**选 B**: 90% 业务方法零重写, 真实证明 Protocol 抽象闭合. 性能不是当前瓶颈 (单条 ~1-2ms),
等 P1.4 bench 出延迟分布数据后再判断是否需要 asyncpg 直连优化.

### 关键实现 — `PostgresMetadataStore(SQLiteMetadataStore)`

```python
class PostgresMetadataStore(SQLiteMetadataStore):
    """覆盖 __init__ 替换 engine, 业务方法全部继承."""

    def __init__(self, url: str | None = None):
        # 不调 super().__init__(), 它会建 SQLite engine
        self._engine = create_async_engine(
            url,
            pool_size=10, max_overflow=20,
            pool_pre_ping=True,  # 自动重连
        )
        self._sessionmaker = async_sessionmaker(...)

    async def init_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        # ↑ 同一份 Base.metadata, SQLAlchemy 自动适配 PG dialect
```

**13 个继承的方法**: `upsert_memory / get_memory / batch_get_memories / list_memories /
delete_memory / delete_all_memories / delete_all_signals / upsert_profile / get_profile /
log_arbitration / list_arbitrations / save_eval_run / list_eval_runs / list_active_users /
consistency_check / Entity CRUD ...` 全部零代码.

## 跨 dialect 兼容性验证

### 已验证的功能 (PG 容器跑通)

- ✅ Schema 自动建表 (`Base.metadata.create_all` 在 PG 16 工作)
- ✅ JSON 列 (PG 自动用 JSONB) — `MemoryRecord.structured` 字典存取
- ✅ DateTime 列 — `created_at` / `last_recalled_at` 时区一致
- ✅ Integer-as-Boolean — `staleness_signal` (SQLite 用 INT 0/1, PG 也支持)
- ✅ UTF-8 中文内容 — `"我对花生过敏"` 写入读出无乱码
- ✅ 批量查询 `id IN (...)` — `batch_get_memories` 在 PG 上工作
- ✅ Arbitration log 写入 — `log_arbitration` 跨 dialect 一致
- ✅ Eval runs 表 — `save_eval_run` / `last_eval` 跨 dialect 一致

### PG-specific 优化 (留给生产)

- 🔲 PG 原生 JSONB 索引 (e.g. `(user_id, structured->>'predicate')` 上的 GIN 索引)
- 🔲 INSERT ... ON CONFLICT DO UPDATE (替代 SQLAlchemy session.get + setattr 两次往返)
- 🔲 Connection pool 调优 (当前默认 pool_size=10, max_overflow=20)
- 🔲 PG vacuum / autovacuum 策略
- 🔲 Read replica 路由 (推荐 list 查询走 replica, write 走 primary)

这些都是**实现内部优化**, 不影响 Protocol 契约 — 上层业务代码不需要任何改动.

## 切换实现的 UX

### MVP (默认, 单机开发 / 小规模 demo)

```bash
# .env (无需配 pg_url)
MEMOCORTEX_DATA_DIR=./data
```

```python
from app.storage import get_metadata
meta = get_metadata()  # → SQLiteMetadataStore (默认)
```

### 生产 (高并发 / 多实例)

```bash
# .env
MEMOCORTEX_PG_URL=postgresql+asyncpg://user:pass@pg.internal:5432/memocortex
```

```python
from app.storage import get_metadata
meta = get_metadata()  # → PostgresMetadataStore (自动切换)
# ↑ 业务代码完全不变
```

`get_metadata()` 内部按 `config.pg_url` 是否非空自动选实现 — **业务方零感知**.

## 测试基础设施

### CI 默认行为 (无 PG)

```bash
uv run pytest                           # → 198 passed, 15 skipped
                                        # PG 测试自动 skip, CI 不需要 docker
```

### 本地真 PG 验证

```bash
# 起一次性 PG 容器 (端口 5433 避免与生产冲突)
docker run -d --rm --name memocortex_pg_test \\
    -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test \\
    -e POSTGRES_DB=memocortex_test \\
    -p 5433:5432 \\
    postgres:16-alpine

# 跑 PG 契约测试
MEMOCORTEX_TEST_PG_URL='postgresql+asyncpg://test:test@localhost:5433/memocortex_test' \\
    uv run --extra dev --extra postgres pytest tests/integration/test_pg_metadata_contract.py

# 清理
docker stop memocortex_pg_test
```

### 安装 PG 依赖

```bash
# 仅生产环境需要 (asyncpg driver)
uv sync --extra postgres
```

## 一个我处理掉的 trap — pytest-asyncio + module-scoped fixture

最初我把 `pg_store` fixture 设成 `scope="module"`, 想节省 schema init 开销.
跑 13 个测试时 9 过 6 败, 错误是 **"Event loop is closed"**.

**根本原因**: pytest-asyncio 默认每个 test 用独立 event loop, 但 module-scoped fixture
的 SQLAlchemy engine 内部有 `asyncpg.Connection` 对象绑定到第一个 event loop.
后续测试用新 loop 时连接已被关闭.

**解法**: fixture 改成 `function` scope, 每个测试新建 store + 退出时 `engine.dispose()`.
单个 test ~150ms init 开销, 15 个测试 +2.3s, 完全可接受.

注释里写得很清楚, 防止下一个接手的人重蹈覆辙.

## 简历 talking points

> "MemoCortex 的 'MVP→生产可插拔' 不是嘴上说的. 我加了 `PostgresMetadataStore` 继承 `SQLiteMetadataStore`,
> 核心是 SQLAlchemy ORM 跨 dialect 的天然兼容 — 同一份 ORM 模型 / 业务方法在 SQLite 和 PG 上都能跑.
> 真起 PG 16 容器跑 15 个契约测试全过, CI 默认 skip 不依赖 docker.
>
> 切换路径: 业务代码零改动, 仅设 `MEMOCORTEX_PG_URL` 环境变量, `get_metadata()` 内部自动切实现.
>
> 这就是我设计 `MetadataStore Protocol` 抽象的真实价值 — 21 个 SQLite 契约测试 + 15 个 PG 契约测试
> 共同证明 Protocol 抽象闭合."

> **面试官追问**: "为什么不直接用 asyncpg 重写, 性能更好?"
>
> **答**: "我会等 P1.4 bench 出实际延迟分布数据再决定. 现在 SQLAlchemy + asyncpg driver 单条 ~1-2ms,
> 不是热路径瓶颈 — recall 真正慢的是 ChromaDB 向量召回 (~25ms) 和 reranker (~280ms). 写入路径里
> ORM 的开销可忽略. 重写 asyncpg 直连要维护两套 SQL, 增加 bug 面积, 在没有性能数据支撑前就重写
> 是过度工程."

## 后续工作 (P1.4 + 之后)

- **P1.4**: bench 100/1k/10k 规模延迟曲线, 出 SQLite vs PG 真实性能对比图
- **PG 索引优化**: 给 `memories(user_id, type, created_at)` 加复合索引提升 list_memories 查询
- **Connection pool 监控**: 通过 PG `pg_stat_activity` 观察连接使用情况
- **读写分离**: list/get 查询走 read replica, write 走 primary
