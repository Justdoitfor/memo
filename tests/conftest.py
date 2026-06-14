"""共享 fixtures — 临时 data_dir / Mock LLM / 测试 user.

设计原则:
  - 单测 (tests/unit/) 不依赖任何 storage/LLM, 纯函数测试
  - 集成测试 (tests/integration/) 用 tmp_data_dir fixture 拿独立 ChromaDB/SQLite
  - LLM 默认走 mock (deterministic), 加 @pytest.mark.live_llm 跑真模型
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

import pytest


# ────────────────────────────────────────────────────────────────────────
#  pytest-asyncio 配置
# ────────────────────────────────────────────────────────────────────────


def pytest_configure(config):
    """注册自定义 marker."""
    config.addinivalue_line(
        "markers",
        "live_llm: 跑真实 LLM API (需要 MEMOCORTEX_LLM_API_KEY), CI 默认 skip",
    )
    config.addinivalue_line(
        "markers",
        "integration: 集成测试, 需要临时 data_dir + 真实 ChromaDB/SQLite",
    )
    config.addinivalue_line(
        "markers",
        "slow: 较慢的测试 (> 1s), 默认跑, 可用 -m 'not slow' 跳过",
    )


def pytest_collection_modifyitems(config, items):
    """没有 LLM API key 时自动 skip live_llm 标记的测试.

    优先级: 环境变量 > .env 文件中的 MEMOCORTEX_LLM_API_KEY.
    pytest collection 阶段 Pydantic Settings 还没初始化, 这里手动加载一遍.
    """
    has_key = bool(os.getenv("MEMOCORTEX_LLM_API_KEY"))
    if not has_key:
        # 兜底: 从项目根的 .env 文件读
        try:
            from dotenv import dotenv_values

            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.exists():
                values = dotenv_values(env_path)
                key = values.get("MEMOCORTEX_LLM_API_KEY", "").strip()
                # 排除明显的 placeholder
                if key and not key.startswith("sk-your-"):
                    has_key = True
        except Exception:
            pass

    if has_key:
        return
    skip_live = pytest.mark.skip(reason="MEMOCORTEX_LLM_API_KEY 未设置, 跳过 live_llm 测试")
    for item in items:
        if "live_llm" in item.keywords:
            item.add_marker(skip_live)


# ────────────────────────────────────────────────────────────────────────
#  通用 fixtures
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def test_user_id() -> str:
    """测试用 user_id, 每个测试函数独立, 避免 state 串扰."""
    import uuid

    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """提供独立 data_dir, 测试结束自动清理.

    通过 monkeypatch 改 MEMOCORTEX_DATA_DIR 环境变量, 让 storage layer 写到隔离目录.
    注意: 已 import 的 storage 单例不会自动重新读取环境变量, 集成测试需用
    `_reset_storage_singletons` fixture 显式重置.
    """
    data_dir = tmp_path / "memocortex_test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MEMOCORTEX_DATA_DIR", str(data_dir))
    yield data_dir
    # 清理 (tmp_path 会自动清, 这里仅为显式)
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def fixed_now() -> "datetime":
    """提供固定时间锚点, 让时间相关测试可重复."""
    from datetime import datetime

    return datetime(2026, 6, 14, 12, 0, 0)
