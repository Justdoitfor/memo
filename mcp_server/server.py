"""MemoCortex MCP Server — MCP-Native Long-Term Memory

工具 (动词驱动, 符合 MCP 语义习惯):
  - remember            主动写入记忆
  - recall              检索相关记忆 (4 信号 Hybrid Recall)
  - recall_workflow     检索程序性记忆 (Procedural)
  - get_profile         获取用户画像 (Reflective)
  - track_signal        上报行为信号
  - reflect             触发 Pattern Miner + 记忆反思
  - manage_memory       记忆管理 (list/forget/mark_stale/arbitrations)
  - graph_query         知识图谱查询 (多跳/关系链/社区) — P0
  - list_entities       列出消解后的实体 — P0
  - entity_merge        手动合并实体 — P0

MCP Resources (供 Agent SystemPrompt 注入):
  - memory://summary/{user_id}    用户核心 Semantic 摘要
  - memory://profile/{user_id}    用户画像 JSON
  - memory://workflows/{user_id}  Procedural 索引
  - memory://snapshot/{user_id}   热记忆快照 (Letta Core Memory, 命中 <1ms)
  - memory://entities/{user_id}   实体图谱摘要 — P0

启动:
  uv run python -m mcp_server.server
  → http://127.0.0.1:{config.mcp_port}/mcp

接入 Claude Desktop / QoderWork / Cursor:
  {
    "mcpServers": {
      "memocortex": {
        "url": "http://127.0.0.1:8766/mcp",
        "transport": "streamable-http"
      }
    }
  }

优化说明:
  - FastMCP 2.14+ 原生支持 async def tool/resource, 不再需要 nest_asyncio 桥接
  - 所有 Tool/Resource 均为 async def, 直接 await orchestrator 调用
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from loguru import logger

from app.config import config
from app.models import (
    MemoryType,
    SearchRequest,
    SignalType,
    WriteRequest,
)
from app.orchestrator import orchestrator
from app.core.snapshot_cache import get_snapshot
from app.reflection import start_scheduler, stop_scheduler
from app.storage import get_kg, get_metadata
from app.utils.logger import setup_logger


# ── MCP Server lifespan: 初始化存储 + 启动调度器 ────────────────────────
@asynccontextmanager
async def _mcp_lifespan(server: FastMCP):
    setup_logger()
    logger.info("=" * 60)
    logger.info("MemoCortex MCP Server 启动")
    logger.info(f"数据目录: {config.data_dir.resolve()}")
    logger.info("=" * 60)

    # 初始化存储 schema + 确保目录存在
    config.ensure_dirs()
    await get_metadata().init_schema()

    # 启动反思调度器
    start_scheduler()

    yield

    # 优雅关闭: 持久化图 + 停调度器
    logger.info("正在关闭 MemoCortex MCP Server...")
    try:
        await get_kg().persist()
    except Exception as e:
        logger.warning(f"持久化图失败: {e}")
    stop_scheduler()
    logger.info("MemoCortex MCP Server 已停止")


setup_logger()
mcp = FastMCP("MemoCortex", lifespan=_mcp_lifespan)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     核心 MCP 工具 (async)                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


@mcp.tool()
async def remember(
    user_id: str,
    content: str,
    memory_type: str = "episodic",
    importance: str = "medium",
    context: str | None = None,
    source_type: str | None = None,
    conflict_strategy: str = "defer",
) -> dict[str, Any]:
    """将重要信息存入长期记忆.

    适用场景: 用户提供了个人信息、偏好、工作上下文, 或完成了一项值得记录的任务.
    不需要每次消息时调用, 只在信息具有跨会话价值时使用.

    Args:
        user_id: 用户标识
        content: 要记忆的核心信息 (自然语言)
        memory_type: episodic / semantic / procedural (默认 episodic)
            注: reflective 由 Worker 自动生成, implicit 由 Pattern Miner 挖掘, 不可手动写
        importance: low / medium / high — 影响 decay rate 与召回排序
        context: 可选, 补充上下文场景
        source_type: explicit_statement (默认) / agent_confirmed / inferred / corrected
        conflict_strategy: defer (启发式快速处理, 默认) / staleness (软废弃) / arbitrator (LLM 决策)
    """
    try:
        mtype = MemoryType(memory_type.lower())
        if mtype in (MemoryType.REFLECTIVE, MemoryType.IMPLICIT, MemoryType.WORKING):
            return {
                "error": f"memory_type='{mtype.value}' 不可手动写. "
                         f"Reflective 由 Worker 聚合, Implicit 由 Pattern Miner 挖掘, "
                         f"Working 不对外暴露."
            }
    except ValueError:
        return {"error": f"非法 memory_type: {memory_type}, 支持: episodic/semantic/procedural"}

    imp_map = {"low": 0.3, "medium": 0.5, "high": 0.8}
    req = WriteRequest(
        user_id=user_id,
        content=content if not context else f"{content}\n[context: {context}]",
        type=mtype,
        importance=imp_map.get(importance.lower(), 0.5),
        source_type=source_type,
        conflict_strategy=conflict_strategy,
    )
    res = await orchestrator.write(req)
    return res.model_dump(mode="json")


@mcp.tool()
async def recall(
    user_id: str,
    query: str,
    memory_types: list[str] | None = None,
    top_k: int = 5,
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    """在回答用户问题前, 检索可能相关的历史记忆.

    适用场景: 用户询问之前讨论过的话题、个人偏好、需要保持上下文一致性时.
    返回结果含 4 信号融合分数 (vector / temporal / keyword_match / importance) 可解释.

    Args:
        user_id: 用户标识
        query: 检索关键词或问题
        memory_types: 可选, 限定类型 (episodic/semantic/procedural/reflective/implicit)
        top_k: 默认 5
        min_confidence: vector_sim 阈值, 默认 0.55 (低于则视为无关 → 返回空)
    """
    parsed_types: list[MemoryType] | None = None
    if memory_types:
        try:
            parsed_types = [MemoryType(t.lower()) for t in memory_types]
            for t in parsed_types:
                if t == MemoryType.WORKING:
                    return {"error": "Working 不对外暴露"}
        except ValueError as e:
            return {"error": str(e)}

    req = SearchRequest(
        user_id=user_id,
        query=query,
        types=parsed_types,
        top_k=top_k,
        score_threshold=min_confidence,
    )
    res = await orchestrator.search(
        user_id=req.user_id,
        query=req.query,
        types=req.types,
        top_k=req.top_k,
        session_id=req.session_id,
        score_threshold=req.score_threshold,
    )
    return res.model_dump(mode="json")


@mcp.tool()
async def recall_workflow(
    user_id: str,
    trigger_context: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """检索用户在特定场景下的工作流程和操作规范 (Procedural Memory).

    适用场景: 用户要求执行某类任务时, 先查是否有定制化工作流偏好.
    返回结构化步骤而非自由文本, 便于 Agent 直接执行.

    Args:
        user_id: 用户标识
        trigger_context: 任务场景描述, 如 'code review' / 'writing PR description'
        top_k: 返回最相关的 N 个工作流模板
    """
    req = SearchRequest(
        user_id=user_id,
        query=trigger_context,
        types=[MemoryType.PROCEDURAL],
        top_k=top_k,
        score_threshold=0.45,
    )
    res = await orchestrator.search(
        user_id=req.user_id, query=req.query, types=req.types,
        top_k=req.top_k, score_threshold=req.score_threshold,
    )
    data = res.model_dump(mode="json")
    # 抽出 structured.steps 给 Agent 直接用
    workflows = []
    for r in data.get("results", []):
        s = r["record"].get("structured", {})
        workflows.append({
            "task_pattern": s.get("task_pattern", r["record"]["content"][:60]),
            "steps": s.get("steps", []),
            "memory_id": r["record"]["id"],
            "score": r["signals"]["final_score"],
        })
    data["workflows"] = workflows
    return data


@mcp.tool()
async def get_profile(user_id: str, auto_refresh: bool = False) -> dict[str, Any]:
    """获取用户画像 (Reflective Memory).

    Args:
        user_id: 用户标识
        auto_refresh: True 时若无缓存即时生成
    """
    return await orchestrator.get_profile(user_id, auto_refresh=auto_refresh)


@mcp.tool()
async def track_signal(
    user_id: str,
    signal_type: str,
    context_tags: list[str] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """上报用户行为信号, 供 Pattern Miner 挖掘隐式偏好.

    适用场景:
      - 用户要求重新生成 → signal_type='regenerate_request'
      - 用户明确纠正 → 'explicit_correction'
      - 用户改变格式偏好 → 'format_preference'
      - 用户选择了某 Tool 的结果 → 'tool_selection'
      - 用户表示满意 → 'positive_feedback'
      - 用户转换话题 → 'topic_pivot'

    Args:
        user_id: 用户标识
        signal_type: 6 种之一
        context_tags: 当时场景标签 (如 ['code_review', 'python'])
        session_id: 可选会话 ID
    """
    from app.pattern import track_signal as _track
    try:
        st = SignalType(signal_type.lower())
    except ValueError:
        return {"error": f"非法 signal_type: {signal_type}. 支持: "
                         f"{[s.value for s in SignalType]}"}
    sid = await _track(
        user_id=user_id, signal_type=st,
        context_tags=context_tags or [], session_id=session_id,
    )
    return {"signal_id": sid, "status": "recorded"}


@mcp.tool()
async def reflect(
    user_id: str,
    window_days: int = 14,
) -> dict[str, Any]:
    """分析最近行为信号, 触发 Pattern Miner 生成 Implicit Memory.

    建议在长对话结束时或用户明确要求时调用. 不需要频繁调用 (后台 Worker 每 30 min 自动跑).

    Args:
        user_id: 用户标识
        window_days: 分析最近 N 天的信号 (默认 14)
    """
    from app.pattern import mine_patterns_for_user
    new_records = await mine_patterns_for_user(user_id, window_days=window_days)
    return {
        "user_id": user_id,
        "window_days": window_days,
        "new_implicit_count": len(new_records),
        "new_records": [
            {
                "id": r.id,
                "content": r.content,
                "confidence": r.confidence_score,
                "keywords": r.structured.get("keywords", []),
                "evidence_count": r.structured.get("evidence_count"),
            }
            for r in new_records
        ],
    }


@mcp.tool()
async def manage_memory(
    user_id: str,
    action: str,
    memory_id: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """查看、标记或删除特定记忆 (统一管理入口).

    Args:
        action: list / forget / mark_stale / arbitrations
        memory_id: 操作单条记忆 ID; action=forget 且 memory_id=None 时清空全部
        confirm: forget 时必须为 True
    """
    meta = get_metadata()

    if action == "list":
        items = await meta.list_memories(user_id, limit=50)
        return {
            "user_id": user_id,
            "count": len(items),
            "items": [
                {"id": r.id, "type": r.type.value, "content": r.content[:200],
                 "confidence": r.confidence_score, "staleness": r.staleness_signal,
                 "created_at": r.created_at.isoformat()}
                for r in items
            ],
        }
    if action == "forget":
        if not confirm:
            return {"error": "forget 需要 confirm=True"}
        if memory_id:
            return await orchestrator.forget(user_id=user_id, memory_id=memory_id)
        return await orchestrator.forget(user_id=user_id, all_user_data=True)
    if action == "mark_stale":
        if not memory_id:
            return {"error": "mark_stale 需要 memory_id"}
        rec = await meta.get_memory(memory_id)
        if not rec or rec.user_id != user_id:
            return {"error": "memory_id 不存在或不属于此 user"}
        rec.staleness_signal = True
        await meta.upsert_memory(rec)
        return {"status": "marked_stale", "memory_id": memory_id}
    if action == "arbitrations":
        items = await meta.list_arbitrations(user_id, limit=20)
        return {"user_id": user_id, "count": len(items), "items": items}
    return {"error": f"未知 action: {action}, 支持: list / forget / mark_stale / arbitrations"}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                 P0: 知识图谱 & 实体管理工具                           ║
# ╚══════════════════════════════════════════════════════════════════════╝


@mcp.tool()
async def graph_query(
    user_id: str,
    query_type: str,
    entity: str | None = None,
    max_hops: int = 3,
    predicate_filter: list[str] | None = None,
    relation_chain: list[str] | None = None,
    min_community_size: int = 3,
) -> dict[str, Any]:
    """查询知识图谱中实体之间的关系.

    适用场景:
      - 需要理解用户社交网络或实体关联时 (multi_hop)
      - 查找与某实体通过特定关系连接的其他实体 (related)
      - 发现用户知识图谱中的实体社区/簇 (community)

    Args:
        user_id: 用户标识
        query_type: multi_hop / related / community
            multi_hop: 从实体出发探索可达路径 (e.g. user → works_at → 字节 → has_colleague → Alice)
            related: 查找与某实体通过指定关系连接的实体 (e.g. 找 user 的所有同事)
            community: 发现强连通的实体簇 (label propagation)
        entity: 起始实体名 (multi_hop 和 related 必填)
        max_hops: multi_hop 最大跳数 (默认 3)
        predicate_filter: 仅保留指定谓词的路径 (multi_hop, e.g. ['likes', 'works_at'])
        relation_chain: 关系链过滤 (related, e.g. ['has_colleague', 'has_friend'])
        min_community_size: 社区最小实体数 (community, 默认 3)
    """
    return await orchestrator.graph_query(
        user_id=user_id,
        query_type=query_type,
        entity=entity,
        max_hops=max_hops,
        predicate_filter=predicate_filter,
        relation_chain=relation_chain,
        min_community_size=min_community_size,
    )


@mcp.tool()
async def list_entities(
    user_id: str,
    entity_type: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """列出用户知识图谱中已消解的实体.

    实体是由 Entity Resolution 系统从 Semantic Memory 写入中自动识别的人/地点/组织等.
    每个实体可能有多个别名 (e.g. '小明' / '张小明' → 同一个 Entity).

    Args:
        user_id: 用户标识
        entity_type: 可选, 按类型过滤 (person/location/organization/product/concept/event)
        limit: 返回数量上限 (默认 50)
    """
    entities = await orchestrator.list_entities(
        user_id=user_id, entity_type=entity_type, limit=limit,
    )
    return {
        "user_id": user_id,
        "count": len(entities),
        "entities": entities,
    }


@mcp.tool()
async def entity_merge(
    user_id: str,
    primary_entity_id: str,
    secondary_entity_id: str,
    confirm: bool = False,
) -> dict[str, Any]:
    """手动合并两个实体 (将 secondary 的别名和关系归入 primary).

    适用场景: 当 Entity Resolution 未自动识别出两个名称指向同一实体时, Agent 或用户可手动合并.

    Args:
        user_id: 用户标识
        primary_entity_id: 主实体 ID (合并后保留)
        secondary_entity_id: 副实体 ID (合并后删除)
        confirm: 必须为 True 才执行
    """
    if not confirm:
        return {"error": "entity_merge 需要 confirm=True"}
    return await orchestrator.merge_entities(
        user_id=user_id,
        primary_id=primary_entity_id,
        secondary_id=secondary_entity_id,
    )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     MCP Resources (async)                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


@mcp.resource("memory://summary/{user_id}")
async def resource_summary(user_id: str) -> str:
    """返回用户核心 Semantic 记忆的精简摘要 (< 500 tokens), 供 Agent SystemPrompt 注入."""
    meta = get_metadata()
    semantic_records = await meta.list_memories(
        user_id, memory_type=MemoryType.SEMANTIC.value, limit=20,
    )
    if not semantic_records:
        return f"# {user_id} — 无 semantic 记忆"
    lines = [f"# {user_id} — 核心事实 ({len(semantic_records)} 条)"]
    for r in semantic_records:
        marker = " ⚠STALE" if r.staleness_signal else ""
        lines.append(f"- {r.content}{marker}")
    return "\n".join(lines)


@mcp.resource("memory://profile/{user_id}")
async def resource_profile(user_id: str) -> str:
    """返回结构化用户画像 (Reflective Memory). Markdown 格式."""
    profile_data = await orchestrator.get_profile(user_id, auto_refresh=False)
    p = profile_data.get("profile", {})
    if not p:
        return f"# {user_id} — 无画像 (调 reflect 触发生成)"
    lines = [
        f"# {user_id} 用户画像",
        f"**简介**: {p.get('one_liner', 'N/A')}",
        f"**偏好**: {', '.join(p.get('preferences', []))}",
        f"**禁忌**: {', '.join(p.get('constraints', []))}",
        f"**交互风格**: {p.get('interaction_style', 'N/A')}",
    ]
    return "\n\n".join(lines)


@mcp.resource("memory://workflows/{user_id}")
async def resource_workflows(user_id: str) -> str:
    """返回所有 Procedural Memory 索引."""
    meta = get_metadata()
    workflows = await meta.list_memories(
        user_id, memory_type=MemoryType.PROCEDURAL.value, limit=50,
    )
    if not workflows:
        return f"# {user_id} — 无工作流"
    lines = [f"# {user_id} 工作流索引 ({len(workflows)} 个)"]
    for r in workflows:
        s = r.structured or {}
        lines.append(f"\n## {s.get('task_pattern', r.content[:50])}")
        for i, step in enumerate(s.get("steps", []), 1):
            lines.append(f"  {i}. {step}")
    return "\n".join(lines)


@mcp.resource("memory://snapshot/{user_id}")
async def resource_snapshot(user_id: str) -> str:
    """返回用户热记忆快照 (< 500 tokens), 供 Agent SystemPrompt 注入.

    命中缓存时 < 1ms, 未命中时 ~10ms (SQLite 直查, 不走向量检索).
    包含: 核心事实 + 画像摘要 + 隐式偏好.
    建议每轮对话开始时读此 Resource, 需更多细节时再调 recall tool.
    """
    snap = await get_snapshot(user_id)
    lines = [f"# {user_id} 热记忆快照"]
    facts = snap.get("facts", [])
    if facts:
        lines.append(f"\n## 核心事实 ({len(facts)} 条)")
        for f in facts:
            lines.append(f"- {f}")
    profile = snap.get("profile", {})
    if profile:
        lines.append(f"\n## 画像")
        lines.append(f"- 简介: {profile.get('one_liner', 'N/A')}")
        prefs = profile.get("preferences", [])
        if prefs:
            lines.append(f"- 偏好: {', '.join(prefs)}")
        cons = profile.get("constraints", [])
        if cons:
            lines.append(f"- 禁忌: {', '.join(cons)}")
    preferences = snap.get("preferences", [])
    if preferences:
        lines.append(f"\n## 隐式偏好 ({len(preferences)} 条)")
        for p in preferences:
            lines.append(f"- {p}")
    if not facts and not profile and not preferences:
        lines.append(f"\n(暂无记忆)")
    lines.append(f"\n[快照时间: {snap.get('updated_at', 'N/A')}]")
    return "\n".join(lines)


@mcp.resource("memory://entities/{user_id}")
async def resource_entities(user_id: str) -> str:
    """返回用户实体图谱摘要 — 列出已消解的实体及其别名/关系.

    供 Agent 理解用户知识图谱中的关键人物/地点/组织.
    建议 Agent 在需要引用具体实体时读此 Resource.
    """
    entities = await orchestrator.list_entities(user_id=user_id, limit=50)
    if not entities:
        return f"# {user_id} — 无已识别实体"
    lines = [f"# {user_id} 实体图谱 ({len(entities)} 个实体)"]
    for e in entities:
        aliases_str = f" (别名: {', '.join(e['aliases'][:3])})" if len(e["aliases"]) > 1 else ""
        summary_str = f" — {e['summary']}" if e.get("summary") else ""
        lines.append(f"- **{e['name']}** [{e['entity_type']}]{aliases_str}{summary_str}")
    return "\n".join(lines)


if __name__ == "__main__":
    logger.info(f"MemoCortex MCP Server 启动 → http://127.0.0.1:{config.mcp_port}/mcp")
    logger.info(f"  Tools: remember / recall / recall_workflow / get_profile / "
                f"track_signal / reflect / manage_memory / "
                f"graph_query / list_entities / entity_merge")
    logger.info(f"  Resources: memory://summary|profile|workflows|snapshot|entities/{{user_id}}")
    mcp.run(transport="streamable-http", host="127.0.0.1", port=config.mcp_port, path="/mcp")