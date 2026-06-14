"""集成测试 conftest — 在任何 app.* import 前把 config.data_dir 重定向到临时目录.

⚠️ 关键约束:
  - app/orchestrator/graph.py 模块尾部有 `orchestrator = MemoryOrchestrator()` eager init,
    会触发 ChromaVectorStore / SQLiteMetadataStore 单例创建 (绑定当时的 config.data_dir).
  - 所以任何 import app.orchestrator 之前, 必须先把 config 重定向, 否则 storage 会写到
    项目真实 ./data 污染开发环境.

设计:
  - 本文件的 module-level 代码在 pytest collection 期更早执行, 比 test 文件 import 更早
  - 创建 session 级 tmpdir, 用 object.__setattr__ 绕过 Pydantic frozen 限制 (Settings 默认非 frozen)
  - 测试 user_id 用 uuid 保证并发安全, 不需要每个测试 reset 数据

测试 user 隔离策略:
  - 每个测试函数获得独立 user_id (test_user_id fixture 已在 root conftest 提供)
  - storage 单例本身不 reset, 所有用户共享同一份 chromadb / sqlite 文件
  - 这正是生产环境的真实形态: 多 user 共存于同一存储后端
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# ────────────────────────────────────────────────────────────────────────
#  Module-level: 立刻把 config 重定向到临时 data_dir
#  必须在任何 app.* import 之前完成 (除了 app.config 本身)
# ────────────────────────────────────────────────────────────────────────

_INTEGRATION_TMPDIR = Path(tempfile.mkdtemp(prefix="memocortex_integration_"))
os.environ["MEMOCORTEX_DATA_DIR"] = str(_INTEGRATION_TMPDIR)

# 重新创建 Settings 实例; 此时 .env 已被读, 但 env 变量已被改写
from app.config import Settings  # noqa: E402

_test_settings = Settings()  # 重新实例化, 会拾取新的 MEMOCORTEX_DATA_DIR
_test_settings.ensure_dirs()

# 把全局 config 单例的 data_dir 替换 (computed_field 会自动重算 chroma_dir 等)
import app.config as _app_config  # noqa: E402

# Pydantic v2 BaseSettings 默认允许 attr 写入
object.__setattr__(_app_config.config, "data_dir", _test_settings.data_dir)

# 此时 app.* 任何 import 都会拿到重定向后的 config


# ────────────────────────────────────────────────────────────────────────
#  Session-scope cleanup
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _cleanup_integration_tmpdir():
    """测试结束清理临时 data_dir."""
    yield
    # 给 ChromaDB / SQLite 一点时间释放文件句柄 (Windows 平台敏感)
    import gc

    gc.collect()
    if _INTEGRATION_TMPDIR.exists():
        shutil.rmtree(_INTEGRATION_TMPDIR, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────
#  Storage 单例就绪 — 第一个使用 storage 的 fixture 触发 schema init
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
async def storage_initialized():
    """初始化 SQLite schema, 让所有集成测试可直接用 get_metadata() / get_vector_store()."""
    from app.storage import get_metadata

    meta = get_metadata()
    await meta.init_schema()
    yield
