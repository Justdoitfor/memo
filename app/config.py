"""配置管理 — Pydantic Settings, 全部从环境变量/.env 加载 (MCP-Only)"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """MemoCortex 全局配置.

    所有环境变量统一前缀 MEMOCORTEX_ , 避免与上游 Agent 框架冲突.
    MCP-Only 模式: 仅保留 MCP 服务端口, 不再暴露 REST API.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MEMOCORTEX_",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (OpenAI 兼容) ─────────────────────────────────────────────
    llm_api_key: str = ""
    llm_api_base: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    # ── Embedding (本地 HuggingFace) ──────────────────────────────────
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # ── 存储 ──────────────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("./data"))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def graph_dir(self) -> Path:
        return self.data_dir / "graph"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cold_dir(self) -> Path:
        return self.data_dir / "cold"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "memocortex.db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlite_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.sqlite_path.as_posix()}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlite_sync_url(self) -> str:
        return f"sqlite:///{self.sqlite_path.as_posix()}"

    # ── MCP 服务端口 ──────────────────────────────────────────────────
    mcp_port: int = 8766

    # ── 记忆参数 ──────────────────────────────────────────────────────
    working_capacity: int = 20
    episodic_ttl_days: int = 30
    default_top_k: int = 8

    # ── 召回权重 (4 信号) ─────────────────────────────────────────────
    recall_w_vector: float = 0.40
    recall_w_temporal: float = 0.20
    recall_w_keyword: float = 0.20  # BM25 关键词匹配信号权重 (原 graph_proximity, 重命名)
    recall_w_importance: float = 0.20
    temporal_tau_days: float = 30.0

    # ── Reranker (二阶段重排, 可选) ───────────────────────────────────
    # 默认关闭, 开启后召回路径走两阶段:
    #   1. 一阶段拿 top_k_before_rerank (默认 30) 候选
    #   2. bge-reranker-v2-m3 cross-encoder 重排
    #   3. 加权融合 reranker 分数和一阶段 final_score
    # 性能代价: rerank 30 条 ~ 100-300ms (CPU), 显著高于一阶段 ~25ms.
    enable_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_weight: float = 0.7  # rerank × 0.7 + final_score × 0.3
    top_k_before_rerank: int = 30  # 拿多少条进 reranker (越大越准, 越慢)

    # ── 反思 Worker 间隔 ──────────────────────────────────────────────
    reflect_distill_interval_sec: int = 3600
    reflect_merge_interval_sec: int = 7200
    reflect_decay_interval_sec: int = 3600
    reflect_profile_interval_sec: int = 86400
    pattern_mine_interval_sec: int = 1800  # Pattern Miner: 30 min 扫一次

    # ── 调试 ──────────────────────────────────────────────────────────
    debug: bool = False
    log_level: str = "INFO"

    # ── 中文分词 (BM25 路径) ──────────────────────────────────────────
    # 关闭后 FTS5 仍可工作但中文召回退化为按字切分 (unicode61 风格).
    # 切换 enable_jieba 后需重建 FTS5 索引 (重启服务时若 fts.db 已存在,
    # 旧索引仍按建表时的 tokenizer 工作; 需要 rm data/fts.db 重建).
    enable_jieba: bool = True
    # 用户自定义词典路径 (一行一个词, 可带词频和词性).
    # e.g. /path/to/userdict.txt 内容:
    #   花生过敏 1000 n
    #   字节跳动 1000 nt
    jieba_user_dict_path: Path | None = None

    def ensure_dirs(self) -> None:
        """确保所有运行时目录存在 — lifespan 启动时调用."""
        for d in [self.data_dir, self.chroma_dir, self.graph_dir, self.cold_dir]:
            d.mkdir(parents=True, exist_ok=True)


# 全局单例
config = Settings()
