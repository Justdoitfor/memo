# MemoCortex

Agent-agnostic 长期记忆中间件 — MCP-Native、5 类分层记忆、4 信号 Hybrid Recall、LLM-as-Arbitrator 冲突消解。

## 核心特性

- **MCP-Native** — 仅通过 MCP (Model Context Protocol) 提供服务，7 个 Tool + 4 个 Resource，原生 async，无 REST API/SDK 依赖
- **5 类分层记忆** — Episodic (事件)、Semantic (事实)、Procedural (流程)、Reflective (画像)、Implicit (隐式偏好) + 内部 Working (短期缓冲)
- **4 信号 Hybrid Recall** — vector_sim (向量相似) + temporal_decay (时间衰减) + keyword_match (BM25 关键词) + importance (有效强度)，加权融合 + 可解释排序
- **LLM-as-Arbitrator 冲突消解** — 语义写入时自动检测冲突，支持 defer (启发式快速)、staleness (软废弃)、arbitrator (LLM 决策) 三种策略
- **Zep 风格时间窗口** — VERSIONED 三元组带 valid_from/valid_until，过期事实降权而非硬删，归档到 Cold Storage 保留可追溯性
- **热记忆 Snapshot 缓存** — 参考 Letta Core Memory，<1ms 延迟读取核心事实 + 画像 + 隐式偏好快照
- **Pattern Miner** — 从 6 种行为信号自动挖掘隐式偏好，生成 Implicit Memory
- **双写一致性** — SQLite 为 source of truth，ChromaDB 缺失时自动补偿，多余时自动清理

## 架构概览

```
┌─────────────────────────────────────────────────────┐
│                    MCP Server                        │
│  (FastMCP 2.14+, streamable-http, async native)     │
│  Tools: remember / recall / recall_workflow /        │
│         get_profile / track_signal / reflect /        │
│         manage_memory                                │
│  Resources: memory://summary|profile|workflows|      │
│             snapshot/{user_id}                        │
├─────────────────────────────────────────────────────┤
│              Memory Orchestrator                      │
│  write → search → get_profile → forget               │
├──────────┬──────────┬──────────┬──────────┬──────────┤
│Episodic  │Semantic  │Procedural│Reflective│Implicit  │
│(事件记忆) │(事实知识) │(流程模板) │(用户画像) │(隐式偏好) │
│          │KG+Vector │steps[]   │Worker聚合 │Miner挖掘 │
├──────────┴──────────┴──────────┴──────────┴──────────┤
│          Hybrid Recall Router (4 信号融合)            │
│  vector_sim × 0.4 + temporal × 0.2                   │
│  + keyword_match × 0.2 + importance × 0.2            │
├─────────────────────────────────────────────────────┤
│                   Storage Layer                       │
│  ChromaDB (向量+BM25) │ SQLite (元数据) │ NetworkX KG │
│  FTS5 (BM25) │ Cold Storage (归档) │ Snapshot Cache   │
├─────────────────────────────────────────────────────┤
│              Reflection Workers (APScheduler)         │
│  distill / merge / decay / pattern_mine /             │
│  graph_flush / consistency_check / archive_expired    │
└─────────────────────────────────────────────────────┘
```

## 5 类记忆体系

| 类型 | 理论根基 | 存储方式 | 写入方式 |
|------|----------|----------|----------|
| **Episodic** | Tulving 1985 事件记忆 | ChromaDB + SQLite | MCP `remember` |
| **Semantic** | Tulving 1985 事实知识 | NetworkX KG + ChromaDB 双索引 | MCP `remember` (自动 LLM 抽取) |
| **Procedural** | Tulving 1985 程序性 | ChromaDB + SQLite | MCP `remember` (带 steps) |
| **Reflective** | 自研 — 显式画像 | SQLite | Worker 自动聚合 / MCP `reflect` |
| **Implicit** | 自研 — 隐式偏好 | ChromaDB + SQLite | Pattern Miner 自动挖掘 |
| Working (内部) | Baddeley 1974 短期缓冲 | SQLite (不进 ChromaDB) | Orchestrator 自动管理 |

## MCP 工具

### remember — 写入记忆

```python
remember(user_id="alice", content="我对花生过敏", memory_type="semantic", importance="high")
```

支持 episodic / semantic / procedural 三种手动写入类型。Reflective 和 Implicit 由后台自动生成，不可手动写。

### recall — 检索记忆

```python
recall(user_id="alice", query="花生过敏", top_k=5, min_confidence=0.55)
```

返回 4 信号融合分数，可解释每条结果的相关性来源。

### recall_workflow — 检索流程模板

```python
recall_workflow(user_id="alice", trigger_context="code review")
```

返回结构化步骤列表，便于 Agent 直接执行。

### get_profile — 获取用户画像

```python
get_profile(user_id="alice", auto_refresh=False)
```

### track_signal — 上报行为信号

```python
track_signal(user_id="alice", signal_type="positive_feedback", context_tags=["python", "debug"])
```

6 种信号类型：regenerate_request / explicit_correction / format_preference / tool_selection / positive_feedback / topic_pivot。

### reflect — 触发模式挖掘

```python
reflect(user_id="alice", window_days=14)
```

分析行为信号，生成 Implicit Memory。

### manage_memory — 记忆管理

```python
manage_memory(user_id="alice", action="list")                # 列表
manage_memory(user_id="alice", action="forget", memory_id="xxx", confirm=True)  # 删除单条
manage_memory(user_id="alice", action="forget", confirm=True)  # GDPR 全量删除
manage_memory(user_id="alice", action="mark_stale", memory_id="xxx")  # 软废弃
manage_memory(user_id="alice", action="arbitrations")         # 冲突审计
```

## MCP Resources

| URI | 说明 | 延迟 |
|-----|------|------|
| `memory://summary/{user_id}` | Semantic 核心事实摘要 | ~10ms |
| `memory://profile/{user_id}` | 用户画像 (Markdown) | ~10ms |
| `memory://workflows/{user_id}` | Procedural 工作流索引 | ~10ms |
| `memory://snapshot/{user_id}` | 热记忆快照 (<500 tokens) | 缓存命中 <1ms |

建议每轮对话开始时读 `memory://snapshot/{user_id}`，需要更多细节时再调 `recall` tool。

## 快速开始

### 1. 安装依赖

```bash
# 需要 Python >= 3.11, < 3.14
pip install uv
uv sync
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑 .env, 至少设置 MEMOCORTEX_LLM_API_KEY
```

核心配置项：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MEMOCORTEX_LLM_API_KEY` | 空 (必填) | OpenAI 兼容 API Key |
| `MEMOCORTEX_LLM_API_BASE` | `https://api.deepseek.com/v1` | API 基础 URL |
| `MEMOCORTEX_LLM_MODEL` | `deepseek-chat` | LLM 模型名 |
| `MEMOCORTEX_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地嵌入模型 |
| `MEMOCORTEX_DATA_DIR` | `./data` | 数据存储目录 |
| `MEMOCORTEX_MCP_PORT` | `8766` | MCP 服务端口 |

### 3. 启动服务

```bash
uv run python -m mcp_server.server
# → http://127.0.0.1:8766/mcp
```

### 4. 接入 Agent 客户端

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "memocortex": {
      "url": "http://127.0.0.1:8766/mcp",
      "transport": "streamable-http"
    }
  }
}
```

**QoderWork / Cursor** — 同样使用 streamable-http transport，配置格式与 Claude Desktop 一致。

### 5. 运行验证 Demo

```bash
# 方式一: 先手动启动 MCP Server, 再运行 Demo
uv run python -m mcp_server.server   # 先启动 Server
uv run python demo/chat_demo.py      # 再运行 Demo (连接到已运行的 Server)

# 方式二: Demo 自动启动 Server (方便一键测试)
uv run python demo/chat_demo.py --auto-start

# 指定用户 ID
uv run python demo/chat_demo.py --user-id alice
```

Demo 通过 FastMCP Client 以 MCP streamable-http 协议连接到真实 MCP Server, 依次调用所有 7 个 Tool 和 4 个 Resource, 并在每步操作后展示具体的记忆变化。所有操作走完整 MCP 协议栈 (网络传输 + Tool/Resource 调用), 不绕过 Server 直接调用 Orchestrator。

详细使用说明见 [docs/USAGE.md](docs/USAGE.md)。

## 冲突消解策略

Semantic Memory 写入时自动检测 (subject, predicate) 冲突，支持三种策略：

| 策略 | 速度 | 说明 |
|------|------|------|
| **defer** (默认) | 快 | 启发式处理：unique → 新覆盖旧 (软废弃)，list → append，versioned → 同时保留 |
| **staleness** | 快 | 直接软废弃旧 triple (effective_strength × 0.2)，不做 LLM 冲突仲裁 |
| **arbitrator** | 慢 | LLM-as-Arbitrator 决策 REPLACE/MERGE/VERSIONED/IGNORE，适合调试 |

字段语义决定冲突时的默认行为：`lives_in` (unique) → REPLACE，`allergic_to` (list) → MERGE，`worked_in` (versioned) → 保留历史。

## 反思 Worker

后台周期任务 (APScheduler)，自动遍历活跃用户：

| Worker | 间隔 | 作用 |
|--------|------|------|
| distill | 3600s | 从 Episodic 提炼 Semantic 事实 |
| merge | 7200s | 合并近似重复的 Semantic 三元组 |
| decay | 3600s | 长期未召回的 importance 指数衰减 |
| pattern_mine | 1800s | 从行为信号挖掘 Implicit 偏好 |
| graph_flush | 60s | NetworkX WAL 定时刷盘 |
| consistency_check | 300s | 比对 SQLite/ChromaDB 双写一致性 |
| archive_expired | 3600s | 归档过期 >30 天的 VERSIONED triple |

## 性能优化要点

- **Embedding 异步化** — HuggingFace encode 在独立线程池执行，不阻塞主协程
- **向量 + BM25 并行召回** — `asyncio.gather` 同时执行 ChromaDB + FTS5 查询
- **BM25 batch 查询** — 补充候选时 `_fetch_records_by_ids` 使用 batch 查询而非 N+1
- **FTS 写入 batching + async** — ChromaDB dual-write 时批量 async 写入 FTS
- **NetworkX WAL** — 写入操作 O(1) append WAL，定时 60s 批量刷盘
- **热记忆 Snapshot 缓存** — 命中 <1ms，未命中 ~10ms (SQLite 直查，不走向量)
- **ChromaDB 元数据 async update** — recall 后 last_recalled_at / recall_count 异步更新，不阻塞返回
- **forget 批量删除** — `DELETE WHERE` 代替全量加载逐条删除

## 项目结构

```
MemoCortex/
  mcp_server/
    server.py              # MCP Server (7 Tools + 4 Resources, async native)
  app/
    config.py              # Pydantic Settings (MEMOCORTEX_ 前缀)
    models.py              # 核心数据模型 (MemoryRecord, Triple, RecallSignals 等)
    core/
      embedder.py          # HuggingFace bge-small-zh-v1.5 (512-dim, async)
      llm_factory.py       # OpenAI 兼容 LLM (structured_invoke + fallback)
      snapshot_cache.py    # 热记忆 Snapshot 缓存 (LRU + TTL)
    storage/
      base.py              # 4 Protocol 接口 (VectorStore, KnowledgeGraph, MetadataStore, ColdStorage)
      chroma_store.py      # ChromaDB + FTS 双写
      sqlite_store.py      # SQLAlchemy 2.0 async (5 ORM 表)
      fts_store.py         # SQLite FTS5 BM25
      nx_graph.py          # NetworkX MultiDiGraph + WAL + 定时刷盘
      fs_cold.py           # 文件系统 Cold Storage
    memories/
      episodic.py          # Episodic Memory
      semantic.py          # Semantic Memory (KG + Vector 双索引, 冲突消解)
      procedural.py        # Procedural Memory (steps 模板)
      reflective.py        # Reflective Memory (画像生成)
      working.py           # Working Memory (短期缓冲, 不进 ChromaDB)
    recall/
      router.py            # Hybrid Recall Router (4 信号融合 + Zep 时间窗口)
      signals.py           # 4 信号计算 + fuse_signals
    arbitrator/
      conflict.py          # LLM-as-Arbitrator 冲突仲裁
    lifecycle/
      decay.py             # importance 指数衰减
      staleness.py         # 软废弃 (effective_strength × 0.2)
    reflection/
      workers.py           # APScheduler 周期任务 (7 个 Worker)
    pattern/
      miner.py             # Pattern Miner (行为信号 → Implicit 偏好)
      signals.py           # 行为信号模型 (6 种 SignalType)
    orchestrator/
      graph.py             # Memory Orchestrator (write/search/forget/get_profile)
    utils/
      metrics.py           # 计量 (timer / counter)
      logger.py            # Loguru 配置
  demo/
    chat_demo.py           # 智能体对话验证 Demo
  docs/
    USAGE.md               # 详细使用说明
```

## License

MIT