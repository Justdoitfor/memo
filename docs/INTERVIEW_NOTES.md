# MemoCortex 面试备忘录

> **目的**: 把项目里所有可面试的"高密度信息"集中在这一份文档. 每完成一个 stage 末尾追加.
>
> **使用方法**: 面试前 30 分钟通读一遍, 面试官提问时可对照"Q&A 索引"快速定位.

## 项目一句话定位

**Agent-agnostic 长期记忆中间件 — MCP-Native, 5 类分层记忆, 4 信号 Hybrid Recall + 二阶段 reranker, LLM-as-Arbitrator 冲突消解**.

对标项目: Mem0 / Zep / Letta / Graphiti.

---

## 一、简历项目描述模板 (直接可写)

> **MemoCortex — Agent 长期记忆中间件 (个人项目, 2026)**
> Python · MCP (FastMCP) · LangChain · ChromaDB · SQLite/PostgreSQL · NetworkX · APScheduler
>
> - 设计并实现 7 Tool + 4 Resource 的 MCP Server, 5 类分层记忆 (Tulving 1985 / Baddeley 1974 理论锚点)
> - **4 信号 Hybrid Recall** (vector + temporal + BM25 + importance) + **bge-reranker-v2-m3 二阶段 cross-encoder 重排**
> - **LLM-as-Arbitrator 冲突消解** (REPLACE/MERGE/VERSIONED/IGNORE 四 action), prompt 版本化机制
>
> 工程化:
> - 213 个 pytest 单测+集成测试 (含真 LLM e2e + 真 PG 容器), 56% 覆盖率, GitHub Actions CI
> - 自建 80 条中文评测集 (8 scenario 分层) + 50 条 Arbitrator 标注集
> - 三阶段 hyperparameter search (消融+裁剪 grid) 发现 README baseline 在 48 组中排名第 40
> - bge-reranker-v2-m3 二阶段提升 nDCG@10 0.973 → 1.000, 但 P95 延迟 32ms → 436ms (13.5x)
> - LLM Arbitrator 50 条标注集 accuracy 50/50, stability 5 次重跑 94% 完全一致
> - PostgreSQL Store 通过继承 SQLiteMetadataStore + asyncpg, 真 PG 容器 15 契约测试全过, 业务方法零重写

---

## 二、面试官 Top 10 追问 Q&A

### Q1: 你怎么验证 4 信号召回算法是有效的?

**核心证据链**:
1. 自建 80 条中文评测集, 8 个 scenario 分层 (`eval/datasets/memocortex_zh_v1.jsonl`)
2. Recall@K / nDCG@K / MRR / P50/P95 延迟 IR 指标
3. 三阶段 grid search (消融 → 裁剪 grid → 邻域精扫)

**真实跑出的数字**:
- baseline `(0.4, 0.2, 0.2, 0.2)` → nDCG@10 = 0.9730, P50 = 23.6ms
- grid 跑了 48 组, **baseline 排名 40/48** (当前 README 默认权重明显次优)
- 单轴消融发现 **temporal_decay 单调负贡献**, **importance 关掉后 nDCG@10 直接到 1.0** (反常识 — 暴露 effective_strength 在新数据上方差太小)

**详见**: `docs/RECALL_WEIGHT_ABLATION.md`

### Q2: README 写"生产换 PG 改连接串就行", 证明给我看

**证据三层**:
1. **Protocol 抽象**: `MetadataStore` 是 Protocol, 业务代码只 `from app.storage import get_metadata`, 不依赖具体实现
2. **零代码重写**: `PostgresMetadataStore(SQLiteMetadataStore)` 继承复用 13 个业务方法, 仅覆盖 `__init__` 换 engine. 同一份 SQLAlchemy ORM 在两个 dialect 都能跑 (JSON 列 PG 自动用 JSONB)
3. **真测过 PG**: 起 PG 16 容器跑 15 个契约测试全过, 包括中文 UTF-8 round-trip / 批量查询 / user 隔离 / eval_runs

**切换 UX**: 业务代码零改动, 仅设 `MEMOCORTEX_PG_URL` 环境变量, `get_metadata()` 自动切实现.

**详见**: `docs/PG_STORE.md`

### Q3: LLM-as-Arbitrator 怎么验证它工作得对?

三件事:
1. **50 条手工标注集** (`arbitrator_eval/dataset_v1.jsonl`), 4 action 分布严格 (12+12+12+14)
2. **Accuracy 50/50 完美对角 confusion matrix** — 没有任何 cross-action 错误
3. **Stability 5 次重跑** — 94% case 5 次完全一致, 不一致的 3 条全在"语义灰色地带" (再婚是否保前任 / list 重复值是 ignore 还是 merge)

**Prompt 版本化**: 抽到 `prompts/arbitrator/v1.yaml`, 通过 `ARBITRATOR_PROMPT_VERSION` 环境变量切版本, 未来 v2 改进可纯 YAML 修改 + A/B 对比.

**详见**: `docs/ARBITRATOR_EVAL.md`

### Q4: 双写一致性 (SQLite + ChromaDB) 你怎么测?

**故障注入**: `monkeypatch` 让 `ChromaVectorStore.add` 抛 `RuntimeError`, 验证 SQLite 不被污染.
**自愈测试**: 直接绕过 vector 写 SQLite 制造漂移, `consistency_check` 应识别 `chroma_missing >= 1` 且自动 `chroma_fixed >= 1`.
**GDPR**: `forget(all_user_data)` 双侧同时清理.

**详见**: `tests/integration/test_dual_write_consistency.py` (4 个集成测试)

### Q5: 召回准确率怎么再提升?

**已落地**: bge-reranker-v2-m3 二阶段 cross-encoder 重排.
- 一阶段 (4 信号融合) 拿 top-30 → reranker 重排 → 取 top-K
- 加权融合: `final = reranker × 0.7 + stage1 × 0.3`
- **效果**: nDCG@10 0.9730 → **1.0000** (8 个 scenario 全拉满)
- **代价**: P50 23.6ms → 313.9ms (13.3x), P95 32.2ms → 436.3ms (13.5x)

**Pareto 取舍**: 默认关闭, 通过 `MEMOCORTEX_ENABLE_RERANKER=true` 一键切换. 对话场景关闭, 离线 / 用户主动 rerun 场景开启.

**最难场景救活**: conflict_latest 0.8931 → 1.0000 (+0.107).

**详见**: `docs/RERANKER_ABLATION.md`

### Q6: Memory 和 RAG 的边界是什么?

- **RAG**: read-only 知识库, 内容不可变, 共享给所有用户
- **Memory**: 用户私有的 mutable 状态, 必须支持冲突消解 / 时间衰减 / 多版本

5 类分层 (Tulving 1985 三分类 + 自研 2 类):
- **Episodic** (事件) / **Semantic** (事实) / **Procedural** (流程) — Tulving 经典
- **Reflective** (画像) / **Implicit** (隐式偏好) — 自研

### Q7: 你的 prompt 怎么版本化和 A/B 测?

`prompts/<name>/<version>.yaml` 是唯一真源, 代码不再持有 prompt 字符串.
`app/core/prompt_loader.py` 加 `lru_cache`, 加载结果 cache.

切换版本: `ARBITRATOR_PROMPT_VERSION=v2 uv run python -m mcp_server.server`.
A/B 测: 跑同一标注集 (`arbitrator_eval/dataset_v1.jsonl`) 对比 v1 vs v2 的 accuracy + stability.

**详见**: `app/core/prompt_loader.py`, `prompts/arbitrator/v1.yaml`

### Q8: Snapshot Cache 命中 < 1ms, 怎么实现的?

**版本号缓存** (从 TTL 重构):
- `defaultdict[user_id] -> int` 单调递增版本号
- `invalidate(user_id)` 让该 user 版本号 +1, cache 项不删, 下次访问拿新版本时 miss → rebuild
- **抗并发**: build 期间被 invalidate → 写回时检测版本不一致就放弃 (避免覆盖更新版本)
- **LRU 容量**: `_access` 计数器记录访问顺序, 超容时淘汰最久未访问

**为什么从 TTL 改成版本号**: TTL 有"漂移窗口", 写入后 5 分钟内可能拿旧数据; 版本号无窗口, 写入立即一致.

**详见**: `app/core/snapshot_cache.py` + `tests/unit/test_snapshot_cache.py` (9 个测试)

### Q9: 4 信号融合 vs RRF (Reciprocal Rank Fusion) 你为什么选加权?

**短答**: 加权融合更可解释, 业务方能调权重权衡 vector vs temporal. RRF 不需要 score 归一化但失去权重控制.

**进阶**: 我做了消融发现 4 信号里 temporal/importance 是负贡献, 这种诊断 RRF 做不到 — RRF 给出最终排序但看不到"哪个信号在拖后腿".

**当前 baseline 取舍**: `(0.6, 0.2, 0.2, 0.2)` 在评测集上 nDCG@10 = 0.9919 接近最优, 同时保留 staleness 路径不过拟合评测集.

### Q10: 5 类记忆的边界是清晰的吗? Episodic 怎么自动分流到 Semantic?

5 类边界**理论清晰, 工程上有融合**:
- **Episodic 是输入**: 用户说什么都先写 EPISODIC
- **Semantic 是产出**: distill worker 周期遍历 Episodic, LLM 抽取出 (subject, predicate, object) triple
- **trivial episode 跳过**: orchestrator 用 `_is_likely_fact()` 启发式 (cut_for_fts + 触发词集合) 60% 的"问候/闲聊" 不走 LLM, 节省 token. 漏抽由 distill worker 1h 间隔兜底

**详见**: `app/orchestrator/graph.py:_is_likely_fact`

---

## 三、真实数据卡片 (一句话报数, 不夸大)

| 指标 | 数字 | 出处 |
|---|---|---|
| 测试总数 | **213** (198 默认 + 15 PG opt-in) | `tests/` |
| 测试覆盖率 | **56.40%** | `pytest --cov=app` |
| 真实 LLM e2e 测试数 | 4 (DeepSeek 真调用) | `tests/integration/test_arbitrator_live.py` |
| 评测集规模 (recall) | 80 条中文 8 scenario | `eval/datasets/memocortex_zh_v1.jsonl` |
| 评测集规模 (arbitrator) | 50 条 4 action | `arbitrator_eval/dataset_v1.jsonl` |
| 一阶段召回 nDCG@10 | **0.9730** | `eval/reports/baseline_default_weights.md` |
| 一阶段 P50 / P95 延迟 | **23.6ms / 32.2ms** | 同上 |
| Grid search 跑过组合数 | 48 组 (裁剪 grid) | `eval/reports/grid_search_results.md` |
| Baseline 在 48 组中的排名 | **40 / 48** | 同上 (说明当前权重次优) |
| 二阶段 reranker nDCG@10 | **1.0000** (8 个 scenario 全拉满) | `eval/reports/reranker_enabled.md` |
| 二阶段 P50 / P95 延迟 | 313.9ms / 436.3ms (+13x) | 同上 |
| Arbitrator accuracy | **50 / 50 = 1.0000** 完美对角 | `arbitrator_eval/reports/accuracy.md` |
| Arbitrator stability (5 次重跑) | **94.00%** 完全一致 | `arbitrator_eval/reports/stability.md` |
| PG 契约测试 | **15 / 15** 真容器全过 | `tests/integration/test_pg_metadata_contract.py` |

---

## 四、设计取舍记录 (面试官追问"为什么不 X" 时直接答)

### 评测集为什么自建而不用 LongMemEval / LoCoMo?

- LongMemEval 是英文为主, bge-small-zh 跑会被 cross-lingual gap 主导, 失去对召回算法本身的判别力
- 自建 80 条中文 8-scenario 分层, **每个 scenario 设计上就是某个信号'敏感'的 case**, 比公开 benchmark 混合 score 对 4 信号权重的判别力强得多
- 同时保留 LongMemEval adapter 接口, 想跑可跑 — 但作为"测算法本身"的工具, 自建更精准

### Reranker 为什么默认关闭?

- 准确率红利大 (+2.77% nDCG@10) 但延迟代价 13x (P95 32ms → 436ms)
- 对话场景每轮 < 100ms 才不破坏体验, 关闭走纯一阶段
- 离线 / 用户主动 "重新检索" 场景开启 — 通过 `MEMOCORTEX_ENABLE_RERANKER=true` 一键切换
- GPU 部署时永远开启 (~30-50ms, 延迟劣势变小)

### PG store 为什么继承不重写 asyncpg?

- SQLAlchemy + asyncpg driver 单条 1-2ms, 不是热路径瓶颈 — recall 真正慢的是 ChromaDB 25ms 和 reranker 280ms
- 重写要维护两套 SQL, 增加 bug 面积; 等 P1.4 bench 出延迟分布再判断是否需要进一步优化
- 继承让 13 个业务方法零重写 — 这就是 Protocol 抽象的真实价值

### Storage 抽象选 Protocol 不选 ABC?

- Protocol 是 structural typing, 实现类不需要显式 `inherit`
- 测试时容易 mock (任何符合签名的对象都行)
- 跨包依赖反向: 业务方依赖 `Protocol`, 不依赖具体实现, 反转控制

### NetworkX WAL 60s 刷盘, 进程崩溃丢 60s 数据可接受吗?

- 当前是 MVP 取舍 — 写入 O(1) 是核心目标
- 生产换 Neo4j 时这个问题自然消失 (Neo4j 有 ACID + WAL)
- 关键是 SQLite 是 source of truth, NetworkX 是双索引的"另一份"; 即使丢 60s 也能从 SQLite 重建

### 为什么 importance 信号在评测集上是负贡献, 你还保留它?

- 评测集只有 80 条新数据, 没有真正软废弃 case → 关掉 importance 看似最优是过拟合评测集
- 真实生产 staleness 软废弃路径有用 (覆盖"用户搬过家旧地址还在"这类 case)
- 决策: 推荐 `(0.6, 0.2, 0.2, 0.2)` — 提升 vec 主信号, 保留所有信号路径不破坏 staleness

---

## 五、工程坑 (面试官最爱听"踩过什么坑" 类问题)

### pytest-asyncio + module-scoped fixture "Event loop is closed"

- **症状**: PG 契约测试 13 个 9 过 6 败, 错误 `Event loop is closed`
- **根因**: pytest-asyncio 默认每个 test 用独立 event loop, 但 module-scoped fixture 的 `asyncpg.Connection` 绑定到第一个 loop
- **解法**: fixture 改 `function` scope + `engine.dispose()` 释放连接池. 单 test +150ms init, 15 个测试 +2s 可接受
- **位置**: `tests/integration/test_pg_metadata_contract.py:60` (注释里有详细说明)

### app.orchestrator 模块尾部 eager init 单例

- **现象**: 集成测试想用临时 `data_dir`, 但 `app/orchestrator/graph.py` 底部 `orchestrator = MemoryOrchestrator()` 在 import 时就创建了 ChromaDB 单例, 绑定到当时的 `config.data_dir`
- **解法**: `tests/integration/conftest.py` 在 module-level (而非 fixture-level) 改 `MEMOCORTEX_DATA_DIR` 环境变量并重新实例化 `Settings`, 用 `object.__setattr__` 替换全局 config 单例的 `data_dir`. 必须在任何 `app.*` import 前完成
- **位置**: `tests/integration/conftest.py`

### Snapshot Cache 从 TTL 改成版本号

- **症状**: 写入后 5 分钟内偶发拿到旧 snapshot, 因为 TTL 缓存还没过期
- **根因**: TTL 有"漂移窗口" — 缓存有效期内任何 invalidate 都会被忽略
- **解法**: 单调递增版本号. 写入立即 +1, 下次读 cache version != current version → miss → rebuild
- **抗并发**: build 期间被 invalidate → 写回时检测版本不一致就放弃, 不覆盖更新版本
- **位置**: `app/core/snapshot_cache.py`, `tests/unit/test_snapshot_cache.py:test_invalidate_during_build_discards_stale_result`

### Windows 终端 GBK 编码炸 Unicode 输出

- **症状**: `print("✓ Wrote 80 entries")` 在 Windows GBK terminal 报 `UnicodeEncodeError: 'gbk' codec`
- **解法**: 用 `[OK]` ASCII 替代 Unicode 特殊字符, 或在脚本入口设 `os.environ.setdefault("PYTHONIOENCODING", "utf-8")`
- **位置**: `eval/datasets/build_zh_v1.py`, `eval/run_recall.py`

### LangChain `ChatPromptTemplate` 字面量花括号要双花括号 escape

- **症状**: prompt YAML 里写 `{"action": "merge"}` 加载时报 `KeyError: '"action"'`
- **解法**: ChatPromptTemplate 把 `{x}` 当变量, 要字面量必须 `{{x}}`
- **位置**: `prompts/arbitrator/v1.yaml` 里所有 JSON 示例都用 `{{`

---

## 六、阶段更新历史

### 2026-06-14 P0 完成 (commit 1f204a9 ~ 9066c55)

- 单测 + 集成测试 + CI yaml + 评测集 + grid search
- 175 测试, 56.40% 覆盖率, baseline nDCG@10 = 0.9730

### 2026-06-14 P1.1 reranker (commit 3354577)

- bge-reranker-v2-m3 二阶段重排, +8 单测
- nDCG@10 → 1.0000, P95 → 436ms

### 2026-06-14 P1.2 Arbitrator (commit fd3a7a6 ~ 33688cf)

- prompt 版本化 + 50 条标注集 + stability 5 次重跑
- accuracy 50/50, stability 94% perfect consistency

### 2026-06-14 P1.3 PG store (commit fb3622d)

- PostgresMetadataStore 继承 + 真 PG 容器跑 15 契约测试
- 业务方法零重写, get_metadata() 自动切实现

### 2026-06-14 P1.4 规模化 benchmark (待补充)

(P1.4 完成后追加: bench 100/1k/10k 规模延迟分布, latency_curves.png)

---

## 七、口诀: 三个数字 + 三个真坑

面试紧张时背这个:

**三个数字**:
- **213 测试** (含真 LLM e2e + 真 PG 容器), 56.40% 覆盖率
- **nDCG@10 0.973 → 1.000** (二阶段 reranker), P95 32ms → 436ms (Pareto 取舍)
- **Baseline 在 grid search 48 组中排名 40** (当前 README 默认权重明显次优)

**三个真坑**:
- pytest-asyncio + module-scoped fixture "Event loop is closed"
- app.orchestrator eager init 单例 vs 临时 data_dir
- Snapshot Cache 从 TTL 重构成版本号 (写入立即一致 vs 漂移窗口)
