# MemoCortex 测试套件

## 跑测试

```bash
# 全部 (含 live LLM, 需要 .env 设 MEMOCORTEX_LLM_API_KEY)
uv run pytest

# 仅单测 (秒级, 无外部依赖)
uv run pytest tests/unit

# 仅集成测试 (跑真 ChromaDB + SQLite, 临时 data_dir)
uv run pytest tests/integration -m "not live_llm"

# 仅 live LLM 测试 (跑真 DeepSeek, 慢, 烧 token)
uv run pytest -m live_llm

# 跑 CI 默认子集 (单测 + 非 LLM 集成测试)
uv run pytest -m "not live_llm"

# 带覆盖率报告
uv run pytest --cov=app --cov-report=term-missing
```

## 目录结构

```
tests/
├── conftest.py                    # 共享 fixtures + live_llm marker auto-skip
├── unit/                          # 纯函数单测, 无外部依赖
│   ├── test_recall_signals.py     # 4 信号 + fuse_signals 边界
│   ├── test_effective_strength.py # Ebbinghaus + 复习 + source 权重
│   ├── test_snapshot_cache.py     # 版本号 + LRU + 并发安全
│   ├── test_arbitrator_fallback.py # 启发式 fallback (无 LLM 路径)
│   ├── test_pattern_miner_grouping.py # 信号→bucket 映射
│   └── test_models.py             # Pydantic 契约 + Chroma metadata 序列化
└── integration/                   # 集成测试, 临时 data_dir 跑真 storage
    ├── conftest.py                # 重定向 config.data_dir 到 tmpdir
    ├── test_storage_protocol.py   # MetadataStore + VectorStore + KG 契约
    ├── test_dual_write_consistency.py # 双写一致性 + 故障注入 + GDPR
    ├── test_orchestrator_e2e.py   # write→recall→forget 全链路
    └── test_arbitrator_live.py    # @live_llm: 真模型仲裁 + Semantic 抽取
```

## Markers

| Marker | 含义 | 默认行为 |
|---|---|---|
| `live_llm` | 调用真 LLM API | 无 `MEMOCORTEX_LLM_API_KEY` 时自动 skip |
| `integration` | 用真实 ChromaDB/SQLite (临时目录) | 默认跑 |
| `slow` | 单个测试 > 1s | 默认跑, 用 `-m "not slow"` 可跳 |

## live_llm 测试机制

`tests/conftest.py` 在 collection 阶段读 `os.getenv()` 和项目根 `.env` 检查 API key:
- 有 key → live_llm 测试正常跑
- 无 key (或仅 `sk-your-...` 占位符) → 自动 skip

CI 默认不传 key, 所有 `@pytest.mark.live_llm` 测试一律 skip, 不烧 token.

## 集成测试隔离策略

`tests/integration/conftest.py` 在任何 `app.*` import **之前** 把 `MEMOCORTEX_DATA_DIR`
重定向到 `tempfile.mkdtemp()` 创建的临时目录, 确保:

- 不污染开发环境的 `./data/`
- 测试之间用独立 user_id (uuid 后缀) 避免数据串扰
- session 结束统一清理

## 覆盖率快照 (Stage 1 + 1.5 完成时)

```
TOTAL  2711  1182  56.40%
```

热路径覆盖率:
- `app/recall/signals.py` 100% (4 信号融合零盲点)
- `app/lifecycle/decay.py` 100% (Ebbinghaus 公式)
- `app/models.py` 99% (Pydantic 契约 + 序列化)
- `app/arbitrator/conflict.py` 88% (含 LLM 主路径)
- `app/storage/sqlite_store.py` 71%
- `app/recall/router.py` 69%
