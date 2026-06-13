"""Pydantic 数据模型 — 全项目核心数据结构"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         记忆类型与记录                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


class MemoryType(str, Enum):  # noqa: UP042  — 保持 Pydantic v2 兼容
    """5 类长期分层记忆 (对外 API 暴露).

    理论根基:
      - Tulving 1985 long-term memory 三分类 → EPISODIC / SEMANTIC / PROCEDURAL
      - 自研 REFLECTIVE: 显式用户画像 (Worker 周期从 Semantic 聚合)
      - 自研 IMPLICIT: 从行为信号挖掘的隐式偏好 (Pattern Miner 后台生成)
        参考 Honcho 的 dialectic pattern inference 思路

    WORKING (Baddeley 1974 short-term memory) 在内部保留作为 Episodic 缓冲层,
    **不对外 API 暴露** — 短期会话上下文是上游 Agent 框架 (LangGraph state /
    Redis) 的职责, 不是 *长期* 记忆中间件的职责.
    """

    EPISODIC = "episodic"         # 时序事件 ("X 时间发生了 Y")
    SEMANTIC = "semantic"         # 事实知识 (用户偏好/属性, 三元组)
    PROCEDURAL = "procedural"     # 程序性 (任务模板, 解决方法)
    REFLECTIVE = "reflective"     # 元记忆 / 显式用户画像 (Worker 聚合)
    IMPLICIT = "implicit"         # 隐式偏好 (Pattern Miner 从行为信号挖掘)

    # 内部使用, 不对外 API 暴露 — Episodic 路径的可选短期缓冲
    WORKING = "working"


class SourceType(str, Enum):  # noqa: UP042
    """记忆来源类型 — 影响 effective_strength 计算的 source_weight."""

    EXPLICIT_STATEMENT = "explicit_statement"  # 用户亲口说的 → 1.0
    AGENT_CONFIRMED = "agent_confirmed"        # Agent 推断后用户确认 → 0.85
    INFERRED = "inferred"                      # 纯推断, 未经确认 → 0.60
    CORRECTED = "corrected"                    # 用户主动纠正的, 最高权重 → 1.20
    DISTILLED = "distilled"                    # 从 Episodic 提炼成 Semantic → 0.90
    MERGED = "merged"                          # 多条合并 → 0.95


SOURCE_WEIGHTS: dict[str, float] = {
    SourceType.EXPLICIT_STATEMENT.value: 1.00,
    SourceType.AGENT_CONFIRMED.value: 0.85,
    SourceType.INFERRED.value: 0.60,
    SourceType.CORRECTED.value: 1.20,
    SourceType.DISTILLED.value: 0.90,
    SourceType.MERGED.value: 0.95,
}


class MemoryRecord(BaseModel):
    """单条记忆 — 统一数据模型, 5 类记忆都用此结构."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str
    session_id: str | None = None
    type: MemoryType

    # 内容
    content: str = Field(min_length=1, description="可被向量化的自然语言文本")
    structured: dict[str, Any] = Field(
        default_factory=dict,
        description="结构化补充信息 (semantic 的 triple / procedural 的 steps 等)",
    )

    # 重要度 (0-1), 入库时 LLM/启发式打分, reflection 可更新
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    # ── Phase 1: 置信度生命周期 (Staleness Detection) ────────────────────
    confidence_score: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="基础置信度 — 越高越可信, 受 source_type 影响"
    )
    source_type: str = Field(
        default=SourceType.EXPLICIT_STATEMENT.value,
        description="记忆来源, 影响 effective_strength 计算 (见 SOURCE_WEIGHTS)"
    )
    staleness_signal: bool = Field(
        default=False,
        description="是否检测到矛盾 → True 时 effective_strength × 0.2 (软废弃)"
    )
    superseded_by: str | None = Field(
        default=None,
        description="若被新记忆取代, 指向新记忆 ID (双向链接, 可追溯历史)"
    )
    decay_rate: float = Field(
        default=0.01, ge=0.0,
        description="个性化衰减系数 λ, 用于 e^(-λ×active_days) (默认半衰期约 70 天)"
    )

    # 时序
    created_at: datetime = Field(default_factory=datetime.now)
    last_recalled_at: datetime | None = None
    recall_count: int = 0
    ttl_at: datetime | None = None  # 显式过期时间, 主要 working 用

    # 冷热分层
    tier: str = Field(default="hot", description="hot / cold / frozen")
    storage_uri: str | None = None  # cold/frozen 时指向 ColdStorage

    # 额外标签 (业务方自定义)
    tags: list[str] = Field(default_factory=list)

    # 来源 (供审计与冲突仲裁追溯) — 保留以兼容旧调用方
    source: str = Field(default="explicit", description="explicit / distilled / merged / inferred")

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower()
        return v

    def to_chroma_metadata(self) -> dict[str, Any]:
        """转 ChromaDB metadata — 只能存原始类型, 复杂结构 JSON 化."""
        import json

        return {
            "user_id": self.user_id,
            "session_id": self.session_id or "",
            "type": self.type.value,
            "importance": self.importance,
            "confidence_score": self.confidence_score,
            "source_type": self.source_type,
            "staleness_signal": 1 if self.staleness_signal else 0,
            "superseded_by": self.superseded_by or "",
            "decay_rate": self.decay_rate,
            "created_at_iso": self.created_at.isoformat(),
            "created_at_ts": self.created_at.timestamp(),
            "recall_count": self.recall_count,
            "tier": self.tier,
            "source": self.source,
            "structured_json": json.dumps(self.structured, ensure_ascii=False) if self.structured else "",
            "tags_csv": ",".join(self.tags),
        }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         实体模型 (P0)                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝


class Entity(BaseModel):
    """实体节点 — 知识图谱中的人/地点/组织/产品/概念.

    P0 新增: Entity Resolution 系统的基础数据模型.
    每个 Entity 代表一个消解后的唯一实体, aliases 收集了所有引用名称.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str
    name: str = Field(description="规范名称 (e.g. '小明' / '上海' / '字节跳动')")
    aliases: list[str] = Field(
        default_factory=list,
        description="别名/引用名列表 (e.g. ['小明', '张小明', 'Mr. Zhang'])",
    )
    entity_type: str = Field(
        default="person",
        description="实体类型: person / location / organization / product / concept / event",
    )
    summary: str = Field(
        default="",
        description="LLM 生成的实体摘要 (一句话描述)",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ResolvedEntity(BaseModel):
    """Entity Resolution 的结果 — 表示消解后的实体引用."""

    entity_id: str = Field(description="指向已消解的 Entity.id")
    name: str = Field(description="本次输入使用的名称")
    is_new: bool = Field(default=True, description="是否是新发现的实体")
    canonical_name: str = Field(description="实体的规范名称")


class GraphPath(BaseModel):
    """知识图谱中的路径 — 多跳推理结果."""

    nodes: list[str] = Field(description="路径上的实体名列表")
    edges: list[dict[str, Any]] = Field(
        default_factory=list,
        description="路径上的边信息 (predicate, confidence 等)",
    )
    length: int = Field(description="路径长度 (跳数)")


class Community(BaseModel):
    """知识图谱社区 — 强连通实体簇."""

    community_id: str
    entities: list[str] = Field(description="社区中的实体名列表")
    summary: str = Field(default="", description="社区摘要描述")
    size: int = Field(description="社区大小 (实体数)")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         知识图谱 Triple                              ║
# ╚══════════════════════════════════════════════════════════════════════╝


class Triple(BaseModel):
    """RDF 风格三元组 — Semantic Memory 的事实表示."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    subject: str          # 通常是 "user" 或具体实体
    predicate: str        # lives_in / likes / allergic_to / ...
    object: str           # 值
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_memory_id: str | None = None  # 该 triple 由哪条 MemoryRecord 派生
    created_at: datetime = Field(default_factory=datetime.now)
    valid_from: datetime | None = None  # for VERSIONED action
    valid_until: datetime | None = None


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         冲突仲裁                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


class ConflictAction(str, Enum):  # noqa: UP042
    """冲突消解的 4 种动作."""

    REPLACE = "replace"       # 新事实覆盖旧的, 旧的归档
    MERGE = "merge"           # list 字段合并 (allergies / hobbies)
    VERSIONED = "versioned"   # 同时保留, 标记 valid_from/until
    IGNORE = "ignore"         # 新事实可疑, 不写 Semantic, 只记 Episodic


class ArbitrationDecision(BaseModel):
    """LLM-as-Arbitrator 的结构化输出."""

    action: ConflictAction
    reasoning: str = Field(description="为什么做这个决定, 一句话")
    confidence: float = Field(ge=0.0, le=1.0)
    merged_value: str | None = Field(
        default=None,
        description="action=MERGE 时, 合并后的新值 (e.g. '乳糖,花生')",
    )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         召回结果                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝


class RecallSignals(BaseModel):
    """单条召回结果的 4 信号分数 — 用于可解释性."""

    vector_sim: float = 0.0
    temporal_decay: float = 0.0
    keyword_match: float = 0.0  # BM25 关键词匹配分数 (原 graph_proximity, 重命名)
    importance: float = 0.0
    final_score: float = 0.0


class RecallResult(BaseModel):
    """Hybrid Recall 返回的单条结果."""

    record: MemoryRecord
    signals: RecallSignals
    rank: int = 0


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         请求/响应 (API 层用)                          ║
# ╚══════════════════════════════════════════════════════════════════════╝


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                  Phase 2: 隐式行为模式挖掘                            ║
# ╚══════════════════════════════════════════════════════════════════════╝


class SignalType(str, Enum):  # noqa: UP042
    """6 种用户行为信号 — Pattern Miner 聚合后挖掘为 Implicit Memory.

    参考 MemoryMesh §5.3, Honcho dialectic pattern inference.
    """

    REGENERATE_REQUEST = "regenerate_request"    # 用户要求重新生成
    EXPLICIT_CORRECTION = "explicit_correction"  # 用户明确纠正
    FORMAT_PREFERENCE = "format_preference"      # 用户改变了格式要求
    TOOL_SELECTION = "tool_selection"            # 用户选择了哪个 Tool 的结果
    POSITIVE_FEEDBACK = "positive_feedback"      # 用户表示满意
    TOPIC_PIVOT = "topic_pivot"                  # 用户中途转换话题


class BehaviorSignal(BaseModel):
    """单条行为信号 — Agent 在用户产生上述行为时主动通过 MCP track_signal 上报."""

    id: str | None = None
    user_id: str
    session_id: str | None = None
    signal_type: SignalType
    context_tags: list[str] = Field(
        default_factory=list,
        description="当时的上下文标签 (e.g. ['code_review', 'python'])"
    )
    memory_ids_in_context: list[str] = Field(
        default_factory=list,
        description="本次召回用了哪些记忆"
    )
    extra: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class WriteRequest(BaseModel):
    user_id: str
    content: str
    type: MemoryType = MemoryType.EPISODIC  # 默认 episodic, 由 Orchestrator 路由
    session_id: str | None = None
    importance: float | None = None
    tags: list[str] = Field(default_factory=list)
    structured: dict[str, Any] = Field(default_factory=dict)
    # Phase 1: 业务方可显式选冲突消解策略
    conflict_strategy: str = Field(
        default="defer",
        description="defer (启发式快速处理, 默认, 参考 Mem0 v3) / staleness (软废弃) / arbitrator (LLM 决策, 慢)",
    )
    source_type: str | None = Field(
        default=None,
        description="覆盖默认 source_type (explicit_statement). 上游 Agent 可标 corrected / inferred 等",
    )


class WriteResponse(BaseModel):
    memory_id: str
    routed_type: MemoryType
    arbitration: ArbitrationDecision | None = None  # 仅 semantic 写入时可能有


class SearchRequest(BaseModel):
    user_id: str
    query: str
    types: list[MemoryType] | None = None  # None = 全类型
    top_k: int = 8
    session_id: str | None = None
    score_threshold: float | None = None  # None=默认 0.55, 0.0=不过滤 (调试用)


class SearchResponse(BaseModel):
    results: list[RecallResult]
    latency_ms: float
    signals_used: list[str]


class ForgetRequest(BaseModel):
    user_id: str
    memory_id: str | None = None
    query: str | None = None  # 按语义模糊删除 (危险, 需 confirm)
    confirm: bool = False


class ProfileResponse(BaseModel):
    user_id: str
    profile: dict[str, Any]
    updated_at: datetime | None = None
