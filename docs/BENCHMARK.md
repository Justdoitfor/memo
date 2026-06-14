# P1.4: 规模化 Benchmark + 性能 Bug 修复

> **目标**: 让 README 的延迟声明从声明变成有 benchmark 文件背书的事实, 并通过规模化测试发现并修复真实生产 bug.

## TL;DR

| 关键发现 | 价值 |
|---|---|
| **抓到 N+1 ChromaDB updates 生产 bug**, 修复后 **P50 320ms → 30ms (11x 加速)** | bench 工具的真实价值 |
| **Recall 延迟 sub-linear scaling** (100→10k 数据 100x, P50 仅 1.9x) | ChromaDB HNSW 在 10k 规模仍然高效 |
| **PG vs SQLite 单条 upsert: PG 比 SQLite 快 1.8x** (P50 6.8ms vs 12.5ms) | 实测对比, 不是嘴上说"PG 适合生产" |
| **Reranker P50 slowdown 12x** at scale 10k | Pareto 取舍数据 |

## 真实跑出的核心数据

### Recall Latency vs Data Scale (一阶段, 优化后)

| Scale | n_queries | Mean | P50 | P95 | P99 | Max |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 50 | 81 ms | **80 ms** | 86 ms | 90 ms | 95 ms |
| 1,000 | 50 | 115 ms | **114 ms** | 123 ms | 130 ms | 135 ms |
| 10,000 | 50 | 155 ms | **153 ms** | 174 ms | 180 ms | 195 ms |

**Sub-linear scaling**: 数据规模 100x, P50 仅 1.9x. 证明 ChromaDB HNSW 索引在 10k 规模仍高效.

### Storage Upsert: SQLite vs PostgreSQL (单条)

| Backend | n | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| SQLite (本地文件) | 100 | 12.5 ms | **12.5 ms** | 13.6 ms | 15 ms |
| PostgreSQL (asyncpg + 连接池) | 100 | 6.9 ms | **6.8 ms** | 7.6 ms | 9 ms |

**PG 比 SQLite 快约 1.8x** — 异步连接池吞吐优势, 验证了 P1.3 PG store 在生产中的实际价值.

### Reranker Overhead at Scale 10k

| Config | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| No reranker | 160 ms | 155 ms | **167 ms** | 175 ms |
| + Reranker | 1,900 ms | 1,850 ms | **2,128 ms** | 2,300 ms |

**P50 slowdown: 12x** — Reranker 在 10k 规模下 P95 飙到 2.1s, 生产对话场景下默认必须关闭.

---

## 这阶段抓到的真生产 bug — N+1 ChromaDB updates

**症状**: 100 条数据规模下 P50 = 477ms, **比 P0 baseline 的 23.6ms 慢 20 倍**, 完全反常识.

**诊断方法 (cProfile)**:

```
ncalls  cumtime  percall  filename:lineno(function)
     5    1.968    0.394  app/recall/router.py:215(_batch_update_recall_meta)
    40    1.967    0.049  app/storage/chroma_store.py:127(update_metadata)
                              ↑ 5 次 recall × 8 condidates = 40 次 IO 占 91%
```

**根因**: 每次 recall 后 fire-and-forget 起 task 循环调 `update_metadata` × top-K (8 次).
每次 `update_metadata` 都做 `get + update` (两次 ChromaDB IO ~45ms × 8 = 360ms).
ChromaDB SQLite 后端有写锁, 上次 task 没跑完时下次 recall 想读会被阻塞.

**为什么 P0 baseline 没暴露**: 评测每条 setup → recall → forget, recall 后立即 GDPR 删 user, update task 失败也没人在意.

**修复**: 在 `app/storage/chroma_store.py` 加 `update_metadata_batch(updates)`:
- 一次 `get(ids=[...])` 拿全部现有 metadata
- 一次 `update(ids=[...], metadatas=[...])` 批量更新

**效果验证** (100 条规模, 同样 5 次连续 recall):

```
优化前:                          优化后:
  call 0:  54ms  (cold start)     call 0:  30ms
  call 1: 346ms  ← 慢路径开始触发  call 1:  29ms
  call 2: 313ms                   call 2:  32ms
  call 3: 320ms                   call 3:  30ms
  call 4: 328ms                   call 4:  30ms

P50: 320ms → 30ms (11x 加速)
```

---

## 实现要点

### Benchmark 设计原则

```python
# bench/bench_recall.py
async def bench_recall_latency(user_id, n_queries=100):
    """跑 100 次召回 (20 个 query × 5 轮), 用 time.perf_counter 测亚毫秒精度.
    返回 P50 / P95 / P99 / mean / max — 不只报均值, 尾部延迟更重要.
    """
    samples = []
    for i in range(n_queries):
        start = time.perf_counter()
        await recall_router.search(...)
        samples.append((time.perf_counter() - start) * 1000)
    return _stats(samples)  # → percentiles
```

数据生成: 拿 80 条评测集做种子, 复制到 N×80 + 加唯一 suffix, ChromaDB 真实索引到 N 倍向量, 查询行为代表生产场景.

### 三张关键图 (`bench/reports/latency_curves.png`)

1. **Recall latency vs scale** (line plot, log scale x): P50 / P95 / Mean 三条曲线
2. **SQLite vs PG upsert** (grouped bar): P50 + P95 对比
3. **Reranker on/off** (grouped bar, log scale y): P50 / P95 / Max 三组对比

## 复现命令

```bash
# 跑全套 (~12 分钟)
PYTHONIOENCODING=utf-8 uv run --extra eval python -m bench.bench_recall

# 只跑小规模 smoke (~30s)
PYTHONIOENCODING=utf-8 uv run --extra eval python -m bench.bench_recall \
    --scales 100 --no-reranker

# 包含 PG 对比 (起 PG 容器先)
docker run -d --rm --name pg_bench \
    -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=memocortex_test \
    -p 5433:5432 postgres:16-alpine

PYTHONIOENCODING=utf-8 \
    MEMOCORTEX_BENCH_PG_URL=postgresql+asyncpg://test:test@localhost:5433/memocortex_test \
    uv run --extra eval --extra postgres python -m bench.bench_recall
```

## 简历 talking points

> "MemoCortex 的延迟声明都有 bench 文件背书, 不是声明. 100/1k/10k 数据规模下 recall P50 是 80/114/153ms, sub-linear scaling 证明 ChromaDB HNSW 在 10k 规模仍然高效. PG vs SQLite 单条 upsert 实测 PG 快 1.8x, 这才是 P1.3 PG store 在生产中的实际价值."

> **追问**: "你写 bench 的时候有什么意外发现?"
>
> **答**: "抓到了一个真生产 bug. P0 baseline 跑评测集是 P50=23.6ms, 但 bench 跑 100 条规模 P50=477ms — 慢 20 倍. cProfile 定位到 `_batch_update_recall_meta` 占 91% 时间: 每次 recall 后循环调 `update_metadata` × top-K (8 次), 每次 ChromaDB get+update (~45ms × 8 = 360ms). P0 评测每条 forget 用户, update task 失败也没人在意, 所以这个 bug 一直藏着. 我加了 `update_metadata_batch` 一次 get + 一次 update, 100 条规模 P50 从 320ms 降到 30ms — 11x 加速. **这就是为什么我坚持 P1.4 要做规模化测试, 不能只看小规模 baseline.**"

## 后续优化空间

1. **GPU 部署**: bge-reranker-v2-m3 GPU 推理可让 P95 从 436ms → ~50ms (10x 加速)
2. **PG JSONB GIN 索引**: `CREATE INDEX ON memories USING GIN (structured)` 加速 KG 查询
3. **PG `INSERT ... ON CONFLICT DO UPDATE`**: 替代 SQLAlchemy session.get + setattr 两次往返
4. **ChromaDB HNSW 参数调优**: 测 M / efConstruction 调优在 100k+ 规模下的曲线
5. **读写分离**: list / get 查询走 PG read replica, write 走 primary
