# MCP 智能体长期记忆框架架构对比分析

> 分析日期: 2026-06-13
> 分析范围: Mem0、Zep/Graphiti、MemGPT/Letta、LangMem 四大主流框架
> 目标: 深入分析各框架通过 MCP 使用时的架构设计和核心机制

---

## 一、框架概览

| 维度 | Mem0 | Zep/Graphiti | Letta (MemGPT) | LangMem |
|------|------|--------------|----------------|---------|
| **最新版本** | v3 (2026) | graphiti-core (2026) | v1.0+ (2025-2026) | v0.0.30 (2025) |
| **定位** | 通用记忆层中间件 | 时序知识图谱引擎 | OS 式自主记忆 Agent | LangGraph 记忆 SDK |
| **开源协议** | Apache 2.0 | Apache 2.0 / MIT | Apache 2.0 | MIT |
| **GitHub Stars** | ~52,500 | ~8,000 (Graphiti) | ~36,000 | ~2,000 |
| **MCP 支持** | 官方 MCP Server | 官方 MCP Server | 双向 MCP (Client+Server) | 无 MCP 支持 |
| **MCP 传输协议** | stdio | HTTP (SSE) / stdio | HTTP / stdio | N/A |
| **图谱能力** | Neo4j (仅付费版) | Neo4j/FalkorDB/Kuzu/Neptune (开源) | 无图谱 | 无图谱 |
| **多租户** | user_id / agent_id / run_id / app_id | group_id | 独立 Agent 实例 | Namespace 层级隔离 |

---

## 二、MCP 集成架构对比

### 2.1 Mem0 MCP 架构

Mem0 提供两种 MCP Server:

**官方云端版 (`mem0-mcp-server`)**: 通过 stdio 传输，需要 `MEM0_API_KEY` 连接 Mem0 Cloud。记忆存储在云端，适合快速接入但数据不本地化。

**开源本地版 (OpenMemory MCP)**: Docker 部署，100% 本地存储，支持零知识加密。社区还有 `mem0-mcp-selfhosted` 等第三方实现，支持 Qdrant + Ollama 全本地栈。

MCP 工具集 (11 个工具):

```
写入类:  add_memory (LLM 抽取事实 → 去重 → 存储)
查询类:  search_memories, get_memories, get_memory
修改类:  update_memory, delete_memory, delete_all_memories
图谱类:  list_entities, get_entity, delete_entities, search_graph
```

关键设计: Mem0 的 MCP 工具是**无状态的薄代理**，所有智能逻辑（事实抽取、冲突检测、去重）在 SDK 层完成。Agent 只需调用 `add_memory` 传入原始对话文本，框架自动处理一切。

### 2.2 Zep/Graphiti MCP 架构

Graphiti 的 MCP Server 位于仓库 `mcp_server/` 目录，支持 HTTP (SSE) 和 stdio 双传输协议。配置通过 `config.yaml` 或环境变量，默认 LLM 为 `gpt-5.5`。

MCP 工具集 (6 个工具):

```
写入类:  add_episode (自动提取实体+关系 → 图谱更新 → 社区检测)
查询类:  search_facts (实体边/事实), search_nodes (实体节点), get_episodes
删除类:  delete_episode
管理:    clear_graph
```

关键设计: Graphiti 的 `add_episode` 是一个**重量级操作** — 一次调用完成实体抽取、关系抽取、实体消解、图谱更新、社区检测全流程。Agent 只需传入原始文本和可选的时间戳，框架全自动处理。查询分离为 `search_facts`（关系/事实）和 `search_nodes`（实体），支持 `valid_at` 时间过滤实现精确的历史时刻查询。

### 2.3 Letta MCP 架构

Letta 在 MCP 生态中扮演**双向角色**:

**作为 MCP Client**: Letta Agent 可以连接外部 MCP Server（文件系统、数据库、Web API），将 MCP 工具纳入自身的工具调用链。这是 Letta "Agent 是长期运行的有状态服务" 理念的核心支撑。

**作为 MCP Server**: 社区项目 `claude-subconscious` 和 `Letta-MCP-server` 将 Letta 的 REST API 封装为 MCP Server，暴露 Agent 管理、记忆读写、归档搜索等能力。

关键设计: Letta 的 MCP 集成深度远超其他框架 — 它不仅暴露工具，更暴露**有状态的 Agent 实例**作为 MCP 资源。外部 AI 工具通过 MCP 调用 Letta，实际上是让 Letta Agent 作为记忆增强的子代理参与协作。

### 2.4 LangMem MCP 状况

**LangMem 不提供 MCP 支持。** 它是一个纯 Python SDK，设计目标是嵌入 LangGraph Agent 工作流。如果需要 MCP 暴露，必须自行用 FastMCP 等库包装 `manage_memory` 和 `search_memory` 两个核心工具。

---

## 三、记忆模型对比

### 3.1 记忆类型体系

| 框架 | 记忆分类 | 核心特点 |
|------|----------|----------|
| **Mem0** | Semantic / Episodic / Procedural | 统一为"原子事实"模型，每条记忆是一句自然语言陈述 |
| **Graphiti** | Entity (实体) / Episode (事件) / Community (社区) | 以知识图谱为核心，记忆 = 节点 + 边 + 时间窗口 |
| **Letta** | Core Memory / Recall Memory / Archival Memory | OS 内存层级类比，Agent 自主管理各层级内容 |
| **LangMem** | Semantic / Episodic / Procedural | 语义记忆=事实/画像，情景记忆=few-shot 示例，程序性记忆=prompt 优化 |

### 3.2 记忆 Schema

**Mem0**: 扁平化原子事实 + 丰富 metadata

```
Memory {
  id: UUID,
  memory: "User prefers dark mode",   // 自然语言事实
  embedding: [float; 512],
  user_id / agent_id / run_id,        // 4 级作用域
  categories: ["preference"],
  hash: sha256,                       // 去重
  metadata: { ... }                   // 任意 KV
}
```

**Graphiti**: 图结构 + 双时间轴

```
EntityNode { uuid, name, labels, attributes, summary, name_embedding }
EpisodicNode { uuid, source, content, valid_at, entity_edges }
CommunityNode { uuid, community_key, member_uuids, center_node_uuid }
EntityEdge { uuid, name, fact, source_node, target_node, valid_at, invalid_at }
```

**Letta**: 文本块 + 向量归档

```
Core Memory Block { label: "persona"|"human"|..., content: str, ~2KB }
Archival Passage { content, embedding, timestamp, tags }
Conversation Message { role, content, timestamp }
```

**LangMem**: 结构化条目 + Namespace 层级

```
Memory {
  namespace: ("org", "team", "user_id"),  // 层级命名空间
  key: UUID,
  value: str | dict,                      // 自由格式
  embedding: [float; dims],
  metadata: { ... }
}
```

### 3.3 存储后端

| 框架 | 向量库 | 图谱库 | 元数据 | 嵌入模型 |
|------|--------|--------|--------|----------|
| **Mem0** | Qdrant/Chroma/Pinecone/pgvector/Milvus/ES | Neo4j (付费) | SQLite | OpenAI/Ollama/Cohere |
| **Graphiti** | 图数据库内置 | Neo4j/FalkorDB/Kuzu/Neptune | 图数据库 | OpenAI/Anthropic |
| **Letta** | pgvector/Chroma | 无 | PostgreSQL/SQLite | OpenAI/Anthropic/Ollama |
| **LangMem** | pgvector (via BaseStore) | 无 | PostgreSQL/SQLite/InMemory | 可配置 |

---

## 四、核心机制深度对比

### 4.1 记忆写入流程

**Mem0 — 8 阶段 Pipeline**:
```
对话 → LLM 抽取原子事实 → 生成 embedding → 相似记忆搜索
→ LLM 判定 (ADD/UPDATE/DELETE/NONE) → 执行动作 → 图谱更新(可选) → 返回
```
特点: 每个事实独立判定动作，粒度精细。v2 后简化为 ADD-only 引发争议。

**Graphiti — 增量图谱构建**:
```
Episode → 实体抽取 → 关系抽取 → 实体消解 (embedding + BM25 + LLM)
→ 图谱更新 (旧边标 invalid_at) → 社区检测(可选) → 返回
```
特点: 一次 episode 可产生多个实体和关系。时间双轴模型（valid_at + invalid_at）是其最大亮点。

**Letta — Agent 自主决策**:
```
对话 → Agent 处理 → Agent 决定是否调用 core_memory_append/replace
→ 上下文满时 Agent 主动 archival_memory_insert 分页 → 返回
```
特点: 记忆管理决策权完全交给 LLM Agent，没有外部抽取 Pipeline。

**LangMem — 热路径 + 后台双模式**:
```
热路径: Agent 调用 manage_memory(action="create") → BaseStore 写入
后台: ReflectionExecutor → LLM 分析对话 → create/update/delete
```
特点: 灵活的两种形成路径，支持 prompt 优化（程序性记忆）。

### 4.2 冲突消解

| 框架 | 冲突检测方式 | 冲突处理策略 |
|------|-------------|-------------|
| **Mem0** | LLM 判定 (ADD/UPDATE/DELETE/NONE) | UPDATE: 修改现有记忆; DELETE: 删除矛盾记忆; 图谱: 软删除旧关系 |
| **Graphiti** | 同实体对边对比 + LLM 判断 | **时间失效**: 旧边设 invalid_at，不删除，保留完整历史 |
| **Letta** | Agent 自主检测 | Agent 调用 core_memory_replace 主动修正; 无自动化冲突处理 |
| **LangMem** | Memory Manager LLM 检测 | enable_updates=True 时自动更新; enable_deletes=True 时自动删除 |

Graphiti 的时间失效模型最为优雅 — 不做破坏性删除，而是标记时间窗口 `[valid_at, invalid_at]`，支持精确的历史时刻查询。

### 4.3 召回/检索机制

| 框架 | 检索策略 | 评分信号 | 特色 |
|------|---------|---------|------|
| **Mem0** | Semantic + BM25 + Entity Match + Reranker | 余弦相似度 + BM25 + 图谱实体相关性 + Reranker 重排 | Token 预算控制 (~6900 tokens/query) |
| **Graphiti** | Semantic + BM25 + Graph Traversal + HyDE + RRF | 向量相似度 + BM25 + 图遍历 + HyDE 假设文档嵌入 | HyDE: 先生成假设答案再嵌入，弥合查询-文档语义差 |
| **Letta** | Core Memory (全量在上下文) + Archival Semantic Search | 向量相似度 (Archival); Core Memory 固定可见 | Agent 主动决定何时搜索什么 |
| **LangMem** | Semantic Search + Metadata Filter | 向量相似度 + 结构化过滤 | 分页 (limit/offset) + query_model 分离 |

**Graphiti 的 HyDE** 是一个值得借鉴的策略: 不直接嵌入查询文本，而是先让 LLM 生成一段"假设性答案文档"，再嵌入该文档做语义搜索。这能弥合"用户问句"和"存储的事实陈述"之间的语义鸿沟。

### 4.4 实体消解

| 框架 | 实体消解方式 | 别名处理 |
|------|-------------|---------|
| **Mem0** | embedding 余弦相似度 + 阈值合并 (仅付费版) | 自动合并相似实体 |
| **Graphiti** | 3 阶段: embedding + BM25 候选生成 → LLM 语义判断 → 合并/新建 | 属性 + 上下文感知的精细消解 |
| **Letta** | 无实体系统 | N/A |
| **LangMem** | 无实体系统 | N/A |

### 4.5 时间感知

| 框架 | 时间模型 | 历史查询能力 |
|------|---------|-------------|
| **Mem0** | created_at + updated_at | 有限 (按日期过滤) |
| **Graphiti** | **双时间轴**: valid_at + invalid_at + created_at | 精确的历史时刻查询 |
| **Letta** | timestamp (对话时间) | 对话历史搜索 (日期范围) |
| **LangMem** | 无显式时间模型 | 无时间感知 |

---

## 五、MCP 工具设计对比总结

| 工具能力 | Mem0 (11) | Graphiti (6) | Letta (REST→MCP) | MemoCortex (10) |
|----------|-----------|--------------|-------------------|-----------------|
| 写入记忆 | add_memory | add_episode | core_memory_append | remember |
| 语义搜索 | search_memories | search_facts | archival_memory_search | recall |
| 列出记忆 | get_memories | get_episodes | — | manage_memory(list) |
| 获取单条 | get_memory | — | — | — |
| 更新记忆 | update_memory | — | core_memory_replace | — |
| 删除记忆 | delete_memory | delete_episode | — | manage_memory(forget) |
| 画像/Profile | — | — | — | get_profile |
| 图谱查询 | search_graph | search_nodes | — | graph_query |
| 实体管理 | list/get/delete_entities | — | — | list_entities / entity_merge |
| 工作流查询 | — | — | — | recall_workflow |
| 行为信号 | — | — | — | track_signal / reflect |

MemoCortex 的 MCP 工具集在覆盖面上已是最全面的，特别是画像、行为信号、工作流查询是其他框架没有的。可以考虑补充的方向:

1. **Token 预算控制参数**: 在 `recall` 工具增加 `max_tokens` 参数
2. **时间过滤参数**: 在 `recall` 和 `graph_query` 增加 `valid_at` 参数
3. **批量操作**: 参考 Mem0 的 `delete_all_memories`，已有 `manage_memory(forget)` 覆盖

---

## 六、参考来源

- [Mem0 官方博客: State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [Mem0 vs Letta vs Zep vs Cognee 2026 对比](https://mcp.directory/blog/mem0-vs-letta-vs-zep-vs-cognee-2026)
- [Graphiti 官方文档: MCP Server](https://help.getzep.com/graphiti/getting-started/mcp-server)
- [Graphiti 官方文档: 架构概览](https://help.getzep.com/graphiti/getting-started/overview)
- [ThoughtWorks Technology Radar: Graphiti (Trial)](https://www.thoughtworks.com/en-th/radar/platforms/graphiti)
- [Letta 官方文档](https://docs.letta.com/introduction)
- [Letta GitHub: claude-subconscious MCP Server](https://github.com/letta-ai/claude-subconscious)
- [LangMem 官方文档](https://langchain-ai.github.io/langmem/)
- [LangMem GitHub](https://github.com/langchain-ai/langmem)
- [Mem0 GitHub](https://github.com/mem0ai/mem0)
- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [OpenMemory MCP 概览](https://ai-bot.cn/openmemory-mcp/)
