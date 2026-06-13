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

    # ── 反思 Worker 间隔 ──────────────────────────────────────────────
    reflect_distill_interval_sec: int = 3600
    reflect_merge_interval_sec: int = 7200
    reflect_decay_interval_sec: int = 3600
    reflect_profile_interval_sec: int = 86400
    pattern_mine_interval_sec: int = 1800  # Pattern Miner: 30 min 扫一次

    # ── 调试 ──────────────────────────────────────────────────────────
    debug: bool = False
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """确保所有运行时目录存在 — lifespan 启动时调用."""
        for d in [self.data_dir, self.chroma_dir, self.graph_dir, self.cold_dir]:
            d.mkdir(parents=True, exist_ok=True)


# 全局单例
config = Settings()
