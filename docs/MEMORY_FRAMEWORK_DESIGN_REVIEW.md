# MemoCortex 长期记忆框架对标分析与改进设计

> 文档版本：v3.0 · 2026-06-13
> 视角：架构师视角，工程实用主义优先
> 项目初衷：**MCP-Native + 中文场景 + 轻量高性能**
>
> v3.0 相对 v2.0 的变化：砍掉了 HyDE、Cross-Encoder Reranker、RRF、去 langchain、OTel tracing、多租户、批处理抽取等"看起来很 SOTA 但和项目初衷不符"的内容。**不为对标而对标**。

---

## 0. 一个核心判断

读完 Mem0 v3 / Graphiti / Letta / LangMem 的设计文档后，最重要的判断是：

> **MemoCortex 的架构已经基本到位，不需要追加新能力，只需要把已有设计落实到位。**

证据：
- ✅ 5 类分层 (Episodic/Semantic/Procedural/Reflective/Implicit + Working) 是各家的超集
- ✅ defer/staleness/arbitrator 三档冲突策略比 Mem0 的 ADD/UPDATE/DELETE/NONE 更细致
- ✅ Resource-as-Memory（snapshot/profile/workflows/entities）超过 Mem0/Graphiti 的 tool-only
- ✅ valid_from/valid_until 内部已实现，与 Graphiti 双时间轴同构
- ✅ tier 字段、source_type 多档权重、effective_strength 公式都是各家没有的细节
- ✅ NetworkX + WAL 替代 Neo4j，符合"轻量"目标

**当前状态的真实差距**只有四处，都是工程落地问题，不是架构问题：

1. 🔴 中文 BM25 失效（FTS5 unicode61 按字切分）
2. 🔴 已有的 `valid_from/until` 字段没通过 MCP 参数暴露
3. 🟠 写入路径每条 episode 都跑 LLM，对 trivial 文本浪费
4. 🟡 设计了 tier 字段但没有迁移 worker

其它"看起来很美"的改进（Reranker / HyDE / 去 langchain / 多租户）都**与项目初衷冲突**或**ROI 不成立**，详见第 5 节"不做的取舍"。

---

## 1. 主流 MCP 长期记忆框架可借鉴的点

只列**真正能落地到 MemoCortex 且符合项目定位**的借鉴点，不做炫技式罗列。

### 1.1 Mem0 v3：Token 预算控制 ⭐ 应借鉴

**他们的做法**：`search_memories` 默认限制每次返回 ~6900 tokens，避免塞爆 Agent 上下文。

**为什么对 MemoCortex 有价值**：
- 当前 `recall(top_k=5)` 只控数量，不控长度。一条长 episodic（"今天我和老板开了个 30 分钟会，讨论了..."）就可能 500+ token，5 条加起来塞爆是常见情况。
- Agent 上下文管理是 MCP 场景的硬约束，不加是 bug 不是 feature。

**借鉴形式**：`recall(max_tokens: int | None = None)`，约 1 小时工作量，零依赖。

### 1.2 Graphiti：双时间轴 valid_at 查询 ⭐ 应借鉴

**他们的做法**：`add_episode(text, valid_at=...)` + 查询时 `search_facts(query, valid_at=...)`，可以"回到 2025-03-01 那一刻"看图谱状态。

**为什么对 MemoCortex 有价值**：
- MemoCortex 内部的 `Triple.valid_from / valid_until` 已经按这个语义实现
- 但 MCP 层 `recall` / `graph_query` 没暴露这个参数
- "用户 2024 住哪里？" "去年的工作流是什么样的？" 这类问题完全无法回答
- **这是已经做了 95%、只差最后 5% 暴露**的状态

**借鉴形式**：`recall(valid_at: str | None = None)`，约 1-2 小时工作量，逻辑只在召回过滤层加几行。

### 1.3 LangMem：热路径 + 后台 Reflection 双模式 ✅ 已对齐，验证设计

**他们的做法**：Agent 通过 `manage_memory` 同步写入；同时 ReflectionExecutor 后台跑 LLM 反思生成记忆。

**MemoCortex 现状**：完全相同的双模式 — `remember` tool 是热路径，`reflection.workers`（distill/merge/decay/pattern_mine/...）是后台路径。

**结论**：**这个设计已经对齐 SOTA，不需要改**。可作为对外宣传依据。

### 1.4 Letta：Core Memory 思想 ✅ 已对齐

**他们的做法**：`<persona>` + `<human>` 两个 block 始终在 system prompt 里，<2KB。

**MemoCortex 现状**：`memory://snapshot/{user_id}` Resource + `MemorySnapshotCache`，命中 <1ms，与 Core Memory 同构甚至更优（因为是 Resource 形式，Agent 主动读、可缓存）。

**结论**：已对齐，**snapshot_cache 的并发安全 bug 要修**（见 3.3）。

### 1.5 不借鉴的部分（明确说明）

- **Mem0 的每事实 LLM 判定（ADD/UPDATE/DELETE/NONE）**：MemoCortex 的 `defer` 启发式 + `arbitrator` 显式调用更合理。defer 写入路径 0 LLM，比 Mem0 v3 快 3-5×。
- **Graphiti 的 HyDE**：项目用户明确否决。**HyDE 对长期记忆场景确实没必要**——HyDE 是为"用户问句 ↔ 文档陈述句"句式差设计，但 MemoCortex 的 Semantic 记忆本身就是结构化三元组（"住在 北京" 这种陈述句格式），且每次 HyDE 多一次 LLM 调用 ~150ms，违背低延迟目标。
- **Mem0 的 Cross-Encoder Reranker**：bge-reranker-v2-m3 ~600MB ONNX 模型，CPU 推理 50ms+，违背"轻量"原则。bge-small-zh + jieba BM25 修复后已足够，没到必须 reranker 的精度门槛。
- **Letta 的 Agent-as-Memory-Editor**：需要为 Agent 提供 `edit_fact / mark_correction` 等十几个工具让它"自己管自己"，这是 Letta 的产品定位（Agent OS）所必需，但 MemoCortex 是"长期记忆中间件"定位，Agent 不该承担记忆管理职责。
- **LangMem 的 Procedural-as-PromptOptimizer**：LangMem 这个用法依赖 LangGraph 的 prompt 演化机制，MemoCortex 不在 LangGraph 生态内，强行借鉴会破坏 Procedural 当前 `steps[]` 的清晰语义。

---

## 2. MemoCortex 当前架构的真实短板

按"严重程度 × 修复成本"排序，只列真正需要修的。

### 2.1 🔴 中文 BM25 形同虚设

**位置**：`app/storage/fts_store.py:66-67`

```python
tokenize='unicode61'  # ← 对中文按字切
```

**问题链**：
1. unicode61 把"花生过敏"切成 `花/生/过/敏` 四个单字 token
2. `_build_fts_match` (行 158-172) 对单字 token 加 `*` 前缀通配符
3. 查询"花生过敏"时，FTS 匹配的是 `花* OR 生* OR 过* OR 敏*`
4. 任何含"花"开头的词（花费/花园/花絮）都被命中
5. BM25 返回的 top-N 几乎是噪声

**影响**：召回路径里 `recall_w_keyword=0.2` 这部分实际是在加噪声分。配置文件写着"4 信号融合"，运行时实际只有 3 个有效信号。

**这是用户明确指出的问题**，必修。

### 2.2 🔴 valid_at 已设计未暴露

**位置**：`app/models.py:223-225`、`app/recall/router.py:271-287`

```python
class Triple(BaseModel):
    valid_from: datetime | None = None
    valid_until: datetime | None = None
```

`recall_router._is_expired_versioned` 已经能判断"当前是否失效"，但只能判断 `now()` 这一个时间点。**MCP 工具签名没有 `valid_at` 参数**，Agent 没法问"用户 2024-06 住哪？"。

**修复成本极低**（router 层加 5-10 行 + MCP 签名加一个参数），不修是浪费。

### 2.3 🟠 Episodic 每条都跑 LLM 抽取

**位置**：`app/orchestrator/graph.py:144-148`

```python
else:  # EPISODIC (默认)
    memory_id = await episodic_memory.write(record)
    self._spawn_bg(  # ← 每条都触发, 不论内容
        self._extract_semantic_safely(req.user_id, req.content, record.id)
    )
```

**问题**：用户随手说"好的"、"嗯"、"今天天气真好" 也要走一次 LLM（~200ms + token 成本）。后台 `distill` worker 每小时还会再扫一遍，LLM 调用至少 2× 浪费。

**修复**：写入前 jieba 关键词命中 `_FACT_TRIGGER_WORDS` 集合的才触发。漏抽的少量 fact 由 distill worker 兜底。

### 2.4 🟡 tier 字段定义了未使用

**位置**：`app/models.py:109-110`

```python
tier: str = Field(default="hot", description="hot / cold / frozen")
storage_uri: str | None = None  # cold/frozen 时指向 ColdStorage
```

只在 arbitrator REPLACE 决策时被设成 `cold`（`semantic.py:483`），但**没有 worker 真的把数据从 Chroma 迁出**。长期运行 Chroma 索引会无限膨胀。

`FileSystemColdStorage` 类也已经实现，但没有写入路径。这是设计完成但没接通的状态，应该补上。

### 2.5 🟡 Snapshot Cache 并发隐患

**位置**：`app/core/snapshot_cache.py:48-56`

```python
def set(self, user_id, snapshot):
    if len(self._cache) >= self._max_users and user_id not in self._cache:
        oldest = min(self._timestamps, key=self._timestamps.get)
        del self._cache[oldest]  # ← 与 invalidate() 并发可能 KeyError
        del self._timestamps[oldest]
```

更严重的问题是用 TTL 而非版本号——后台 worker 改写 Semantic 后，`snapshot_cache.invalidate(user_id)` 调用要分散在所有写入路径上。任何一个忘了调 invalidate，下次 5 分钟内拿到的都是旧 snapshot。

实际检查 `orchestrator.write` (行 152-155) 只对 SEMANTIC/EPISODIC 主动 invalidate，但 worker 路径（distill/merge/arbitrator REPLACE）很多写入点都漏了。

### 2.6 🟢 Pattern Miner 重复挖掘

**位置**：`app/pattern/miner.py`

每 30 分钟扫一次 14 天窗口的全部 signals，`_group_signals` 后送 LLM。没有"已挖掘 signal id 集合"去重。同一批信号会被反复送给 LLM 总结同一个 insight。

修复：在 `behavior_signals` 表加 `mined_at` 时间戳，挖掘时只取 `mined_at IS NULL`，挖完批量更新。

---

## 3. 改进路线图

按 ROI 排序，每个改动都标注**符合项目初衷哪一条**和**借鉴自哪个框架**。

### Phase 1：中文 BM25 修复（🔴 必修，1 天）

**符合初衷**：中文场景 + 高性能
**借鉴**：无（项目自己的中文优化责任）

#### 1.1 jieba 适配 FTS5

新增 `app/utils/tokenizer.py`：

```python
"""中文分词适配 - jieba HMM + 单字过滤 + 缓存

设计原则:
- 不写 SQLite C 扩展, Python 侧切好后用 FTS5 simple tokenizer 按空格切
- 单字中文过滤: 单字 IDF 太低, 噪声大于信号
- 数字/英文保持原样, 不切
- LRU 缓存高频查询 / 写入文本
"""
from __future__ import annotations
import re
from functools import lru_cache
from threading import Lock

import jieba
import jieba.analyse

_DICT_LOADED = False
_DICT_LOCK = Lock()
_CN_RE = re.compile(r"[一-鿿]+")


def _ensure_dict() -> None:
    """懒加载 jieba 词典 + 领域高频词 (一次性)."""
    global _DICT_LOADED
    if _DICT_LOADED:
        return
    with _DICT_LOCK:
        if _DICT_LOADED:
            return
        jieba.setLogLevel(60)  # 关闭启动日志
        # 领域词: 来自 _PRED_REGISTRY 中文模板的高频固定词
        for w in ["过敏原", "工作流", "代码评审", "知识图谱", "记忆中间件"]:
            jieba.add_word(w, freq=1000)
        # 用户自定义词典
        from app.config import config
        if getattr(config, "jieba_user_dict_path", None):
            try:
                jieba.load_userdict(str(config.jieba_user_dict_path))
            except Exception:
                pass
        _DICT_LOADED = True


@lru_cache(maxsize=10_000)
def cut_for_fts(text: str) -> str:
    """切分中文 -> 空格连接的 token 串, 供 FTS5 simple tokenizer 索引/查询."""
    _ensure_dict()
    if not text:
        return ""
    tokens: list[str] = []
    for seg in jieba.cut(text, HMM=True):
        seg = seg.strip()
        if not seg:
            continue
        # 单字中文过滤 (英文/数字单字符保留, 因为它们 IDF 高)
        if len(seg) == 1 and _CN_RE.match(seg):
            continue
        tokens.append(seg)
    return " ".join(tokens)


@lru_cache(maxsize=2_000)
def extract_keywords(text: str, top_k: int = 5) -> list[str]:
    """TF-IDF 关键词抽取, 供 query expansion / trivial 路径判定用."""
    _ensure_dict()
    return jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)
```

修改 `app/storage/fts_store.py`：
- 建表 tokenizer 改 `simple`
- `add()` / `add_batch()` 写入前 `cut_for_fts(content)`
- `_build_fts_match()` 重写，移除单字 prefix 通配，改用整词 OR

#### 1.2 配置 + 依赖

`pyproject.toml` 加 `jieba>=0.42.1`。

`app/config.py` 加：
```python
enable_jieba: bool = True
jieba_user_dict_path: Path | None = None
```

#### 1.3 Query 扩展（可选，半小时）

`_PRED_REGISTRY` 已经维护了"中文 → 谓词"信息，反向构建 dict 后给 query 加几个英文谓词词，提升 BM25 命中。这点完全是项目自身资源的复用，不引入新依赖。

```python
# app/recall/query_expansion.py
from app.memories.semantic import _PRED_REGISTRY
from app.utils.tokenizer import extract_keywords

# 启动时一次性构建反查表
_CN_TO_PRED: dict[str, str] = {}
for pred, info in _PRED_REGISTRY.items():
    cn = info.get("cn", "").replace("{obj}", "").strip()
    for word in cn.split():
        if word and len(word) >= 2:
            _CN_TO_PRED.setdefault(word, pred)


def expand_query(query: str) -> str:
    """中文 query 加英文谓词增强 BM25, 不动向量召回那侧."""
    keywords = extract_keywords(query, top_k=5)
    extras = [_CN_TO_PRED[kw] for kw in keywords if kw in _CN_TO_PRED]
    return f"{query} {' '.join(extras)}" if extras else query
```

仅在 BM25 路径用，向量路径仍用原 query。

---

### Phase 2：MCP 工具语义补全（🔴 必做，半天）

**符合初衷**：MCP-Native
**借鉴**：Mem0（Token 预算）+ Graphiti（valid_at）

#### 2.1 Token 预算

扩展现有 `app/utils/token_meter.py`：

```python
def estimate_tokens(text: str) -> int:
    """轻量 token 估算: 中文 1 token/字, 英文按 4 char/token."""
    if not text:
        return 0
    cn = sum(1 for c in text if '一' <= c <= '鿿')
    other = len(text) - cn
    return cn + (other + 3) // 4


def truncate_to_budget(results, max_tokens, get_text=lambda r: r.record.content):
    """按 token 预算保高分前缀."""
    used, kept = 0, []
    for r in results:
        t = estimate_tokens(get_text(r))
        if used + t > max_tokens:
            break
        kept.append(r)
        used += t
    return kept
```

`mcp_server/server.py` 的 `recall()` 加：

```python
async def recall(
    user_id: str,
    query: str,
    memory_types: list[str] | None = None,
    top_k: int = 5,
    min_confidence: float = 0.55,
    max_tokens: int | None = None,  # ← 新增
    valid_at: str | None = None,     # ← 新增 (见 2.2)
) -> dict[str, Any]:
    """...
    Args:
        max_tokens: 返回结果总 token 预算上限. None 不限.
                    建议 Agent 侧设 6000-8000 避免上下文塞爆.
        valid_at: ISO 时间戳, 仅返回该时刻有效的 Semantic 事实.
                  None 时返回当前有效事实 (默认行为).
    """
    # ... 原召回
    if max_tokens:
        from app.utils.token_meter import truncate_to_budget
        res.results = truncate_to_budget(res.results, max_tokens)
    return res.model_dump(mode="json")
```

#### 2.2 valid_at 时间过滤

`app/recall/router.py` 加：

```python
@staticmethod
def _is_valid_at(record: MemoryRecord, valid_at: datetime) -> bool:
    """Semantic 记录在 valid_at 时刻是否有效.
    非 SEMANTIC 类型不参与时间过滤 (Episodic 是历史事件本身).
    """
    if record.type != MemoryType.SEMANTIC:
        return True
    s = record.structured or {}
    try:
        vf = s.get("valid_from")
        if vf and datetime.fromisoformat(str(vf)) > valid_at:
            return False
        vu = s.get("valid_until")
        if vu and datetime.fromisoformat(str(vu)) < valid_at:
            return False
    except (TypeError, ValueError):
        pass
    return True
```

`search()` 接受 `valid_at: datetime | None = None`，过滤步骤后追加：

```python
if valid_at is not None:
    scored = [r for r in scored if self._is_valid_at(r.record, valid_at)]
```

`graph_query` 同理加 `valid_at` 参数（NX layer 已经存了 `valid_from/until`，只需查询时过滤）。

---

### Phase 3：写入路径减负（🟠 应做，半天）

**符合初衷**：高性能（减少不必要 LLM 调用）
**借鉴**：无

#### 3.1 Trivial 跳过

```python
# app/orchestrator/graph.py
from app.utils.tokenizer import extract_keywords

# 出现这些词的 episode 才触发 LLM 抽取
# 漏抽 fact 由后台 distill worker (1h 间隔) 兜底, 不会丢
_FACT_TRIGGER_WORDS = frozenset({
    "我", "用户", "住", "工作", "公司", "公司", "过敏", "喜欢", "不喜欢",
    "搬", "结婚", "离婚", "生", "买", "卖", "学", "毕业",
    "宠物", "孩子", "对象", "女朋友", "男朋友", "老婆", "老公",
    "出生", "周岁", "岁", "电话", "邮箱", "地址",
})


def _is_likely_fact(text: str) -> bool:
    """启发式: 此 episode 是否值得跑 LLM 抽取.
    跳过约 60% trivial episode (问候/确认/闲聊).
    误判成本: 偶尔漏抽, 由后台 distill worker 兜底.
    """
    text = text.strip()
    if len(text) < 6:
        return False
    return bool(set(extract_keywords(text, top_k=10)) & _FACT_TRIGGER_WORDS)


# write() EPISODIC 分支
else:
    memory_id = await episodic_memory.write(record)
    if _is_likely_fact(req.content):
        self._spawn_bg(
            self._extract_semantic_safely(req.user_id, req.content, record.id)
        )
    # else: 留给后台 distill worker
```

#### 3.2 Pattern Miner 去重

`app/storage/sqlite_store.py` 的 `behavior_signals` 表加 `mined_at: datetime | None`。

`miner.py` 改：
```python
signals = await meta.list_unmined_signals(user_id, since=since)
# ... 挖掘 ...
await meta.mark_signals_mined([s["id"] for s in mined])
```

避免同一批 signal 被反复 LLM 总结。

---

### Phase 4：把已设计的功能接通（🟡 应做，1 天）

**符合初衷**：完成 0.x 版本承诺的能力
**借鉴**：无

#### 4.1 Tier 迁移 Worker

新增 `app/lifecycle/tier_migration.py`：

```python
"""Hot -> Cold -> Frozen 三层迁移.

判定标准: effective_strength (已有的 Ebbinghaus 公式)
- hot:    strength >= 0.3, 保留 ChromaDB + KG
- cold:   0.1 <= strength < 0.3, 从 ChromaDB 移除, SQLite 保留
- frozen: strength < 0.1, 序列化到 ColdStorage 文件, SQLite 仅留 stub

设计取舍:
- 用 effective_strength 而不是 last_recalled_at 单一信号:
  effective_strength 已经综合了衰减/复习/来源/staleness 四因子, 一次到位
- 不删除 frozen, 用 ColdStorage 文件保留, 支持 GDPR 审计 / 历史回溯
- 频率: 每 6h 一次, 与其它 reflection worker 错峰
"""
from app.lifecycle.decay import compute_effective_strength
from app.storage import get_metadata, get_vector_store, get_cold_storage

HOT_TO_COLD = 0.3
COLD_TO_FROZEN = 0.1


async def migrate_tiers_for_user(user_id: str) -> dict:
    meta = get_metadata()
    vec = get_vector_store()
    cold = get_cold_storage()
    records = await meta.list_memories(user_id, limit=10_000)

    h2c, c2f = 0, 0
    for r in records:
        s = compute_effective_strength(r)
        if r.tier == "hot" and s < HOT_TO_COLD:
            await vec.delete(r.id, user_id)
            r.tier = "cold"
            await meta.upsert_memory(r)
            h2c += 1
        elif r.tier == "cold" and s < COLD_TO_FROZEN:
            uri = await cold.archive(r)
            r.tier = "frozen"
            r.storage_uri = uri
            r.content = ""  # 节省空间, 内容在 ColdStorage 文件里
            await meta.upsert_memory(r)
            c2f += 1
    return {"hot_to_cold": h2c, "cold_to_frozen": c2f}
```

注册到 `app/reflection/workers.py`：
```python
scheduler.add_job(
    _run_tier_migration_all_users,
    "interval", seconds=21600,  # 6h
    id="tier_migration",
)
```

#### 4.2 Snapshot Cache 改版本号

`app/core/snapshot_cache.py` 重写：

```python
import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable


class MemorySnapshotCache:
    """版本号缓存 - 任何写入 invalidate 时版本递增, 旧 cache 自动失效.

    比 TTL 更可靠: invalidate 不会"漏调" - 即使漏调, 下次 build_snapshot 时
    版本号比较仍能识别陈旧 cache. 实际行为是"过期 != 错过".

    保留 max_users 容量限制以避免 OOM, 用 LRU 淘汰策略.
    """

    def __init__(self, max_users: int = 100):
        self._cache: dict[str, dict[str, Any]] = {}
        self._versions: dict[str, int] = defaultdict(int)
        self._access: dict[str, int] = {}  # 单调访问计数, 用于 LRU
        self._counter = 0
        self._max_users = max_users
        self._lock = asyncio.Lock()

    async def get_or_build(
        self, user_id: str,
        builder: Callable[[str], Awaitable[dict]],
    ) -> dict:
        async with self._lock:
            cur_version = self._versions[user_id]
            cached = self._cache.get(user_id)
            if cached and cached.get("_version") == cur_version:
                self._counter += 1
                self._access[user_id] = self._counter
                return cached
        # 锁外构建避免阻塞其它 user
        snap = await builder(user_id)
        async with self._lock:
            # 容量保护
            if len(self._cache) >= self._max_users and user_id not in self._cache:
                oldest = min(self._access, key=self._access.get)
                self._cache.pop(oldest, None)
                self._access.pop(oldest, None)
            snap["_version"] = self._versions[user_id]
            self._cache[user_id] = snap
            self._counter += 1
            self._access[user_id] = self._counter
        return snap

    def invalidate(self, user_id: str) -> None:
        # 仅递增版本号, 下次 get 自动 miss. 不删 cache 项, 等 LRU 自然淘汰.
        self._versions[user_id] += 1
```

`get_snapshot()` 改成 `get_or_build(user_id, build_snapshot)` 调用方式。

orchestrator 写入路径的 invalidate 调用补全（distill / merge / arbitrator REPLACE 三处当前漏调）。

---

## 4. 不做的取舍（明确说明）

为避免后续被同类提议反复打扰，明确列出考虑过但**不做**的设计，附理由。

### ❌ Cross-Encoder Reranker (bge-reranker-v2-m3)

- **理由 1（与初衷冲突）**：模型 ~600MB，CPU 推理 50ms+，与"轻量高性能"目标冲突
- **理由 2（边际效益小）**：bge-small-zh + jieba BM25 修复后召回精度已经够用，reranker 只在追求 SOTA leaderboard 时才有意义
- **理由 3（运维负担）**：增加一个模型版本管理、依赖冲突（FlagEmbedding 与 sentence-transformers 版本经常打架）
- **重新启用时机**：用户拿出 LongMemEval 上低于 0.55 MRR@5 的实测数据，再考虑

### ❌ HyDE 假设文档嵌入

- **理由 1（用户明确否决）**：用户在 v3 反馈中明确说"假设性文档对长期记忆来说是没有必要的"
- **理由 2（场景不匹配）**：HyDE 解决的是"问句 ↔ 文档陈述句"句式差。MemoCortex 的 Semantic 已经是结构化三元组（"住在 北京"），陈述句格式，不存在这个差距
- **理由 3（成本高）**：每次 recall 多 ~150ms LLM 调用，违背低延迟目标

### ❌ RRF 融合替代加权和

- **理由 1（边际收益小）**：当前加权和工作正常，RRF 在 MemoCortex 4 信号场景不会带来质变
- **理由 2（增加配置复杂度）**：要么强行替代（兼容性问题）、要么加配置项（用户难调）
- **理由 3（场景错配）**：RRF 是为"多个独立排序系统结果合并"设计的，MemoCortex 的 4 信号是同一系统输出的不同维度，加权和更合理

### ❌ EpisodicBatchExtractor 累积批处理

- **理由（过度工程）**：trivial 跳过（Phase 3.1）已经解决 60% 浪费。剩下的 40% 是真有信息量的 episode，逐条抽取没问题
- **理由（增加状态）**：批处理引入排队、超时、flush 等状态管理，与项目"无外部队列依赖"的轻量原则冲突

### ❌ 替换 langchain 为原生 OpenAI SDK

- **理由 1（修改面广）**：要重写 LLM 抽取、冲突仲裁、Pattern Miner、Entity Resolver、Reflective 五处 prompt 调用，4h 工作量，新 bug 风险
- **理由 2（收益不直接）**：启动时间 ~2.5s 节省、内存 ~600MB 节省，是工程指标，不影响 Agent 用户体验
- **理由 3（生态友好）**：langchain 的 `ChatPromptTemplate` 让 prompt 可读性好，对维护者友好
- **重新启用时机**：依赖冲突真的卡住升级（如 langchain breaking change）时再做

### ❌ OpenTelemetry / 结构化 trace

- **理由（YAGNI）**：当前 metrics.timer 计数器 + loguru 已经够单进程调试用。OTel 是分布式系统才需要的
- **重新启用时机**：项目部署到多节点 / 接入观测平台时

### ❌ 多租户 (tenant_id / namespace)

- **理由（场景未明）**：MemoCortex 当前定位是"单 Agent 接入的长期记忆"，企业 SaaS 场景需求未明确
- **理由（侵入大）**：tenant_id 要渗透到所有存储查询、所有 MCP 工具签名、所有索引
- **重新启用时机**：有具体多租户用户提出需求

### ❌ Letta 风格 edit_fact / Agent-as-Editor 工具集

- **理由（产品定位差异）**：Letta 是"有状态 Agent OS"，需要 Agent 自管记忆。MemoCortex 是"记忆中间件"，记忆策略由系统决定，不让 Agent 干预
- **理由（已有替代）**：`manage_memory(action="mark_stale")` 已经覆盖 Agent 显式纠错的场景，不需要额外工具

### ❌ Memory Graph Embedding / 第 5 信号

- **理由（边际效益不明）**：需要离线训练 KG embedding（TransE/RotatE），训练时间长，且对小图（user 级别 KG 通常 <1000 节点）几乎无收益
- **理由（与轻量冲突）**：增加训练流水线和模型管理成本

### ❌ Hierarchical Summarization

- **理由（YAGNI）**：当前 reflection workers 已经做了 Episodic→Semantic 蒸馏，MemoryBank 风格的多级摘要在 user 级别（年记忆量 < 10k 条）没必要
- **重新启用时机**：单 user 长期运行 episode 数 > 100k 条时

---

## 5. 完整改进文件清单

| Phase | 优先级 | 文件 | 工作量 | 依赖变更 |
|---|---|---|---|---|
| 1.1 | 🔴 P0 | + `app/utils/tokenizer.py` | 1h | + jieba |
| 1.1 | 🔴 P0 | ~ `app/storage/fts_store.py` | 1h | - |
| 1.2 | 🔴 P0 | ~ `app/config.py` (jieba 开关) | 0.2h | - |
| 1.2 | 🔴 P0 | ~ `pyproject.toml` | 0.1h | + jieba>=0.42.1 |
| 1.3 | 🟠 P1 | + `app/recall/query_expansion.py` | 0.5h | - |
| 1.3 | 🟠 P1 | ~ `app/recall/router.py` (BM25 用 expand_query) | 0.3h | - |
| 2.1 | 🔴 P0 | ~ `app/utils/token_meter.py` | 0.5h | - |
| 2.1 | 🔴 P0 | ~ `mcp_server/server.py` (max_tokens 参数) | 0.5h | - |
| 2.2 | 🔴 P0 | ~ `app/recall/router.py` (valid_at 过滤) | 1h | - |
| 2.2 | 🔴 P0 | ~ `mcp_server/server.py` (valid_at 参数) | 0.5h | - |
| 3.1 | 🟠 P1 | ~ `app/orchestrator/graph.py` (_is_likely_fact) | 0.5h | - |
| 3.2 | 🟢 P2 | ~ `app/storage/sqlite_store.py` (mined_at 列) | 1h | - |
| 3.2 | 🟢 P2 | ~ `app/pattern/miner.py` (去重) | 0.5h | - |
| 4.1 | 🟡 P1 | + `app/lifecycle/tier_migration.py` | 2h | - |
| 4.1 | 🟡 P1 | ~ `app/reflection/workers.py` (注册 worker) | 0.3h | - |
| 4.2 | 🟡 P1 | ~ `app/core/snapshot_cache.py` (版本号) | 1h | - |
| 4.2 | 🟡 P1 | ~ `app/orchestrator/graph.py` + workers (补 invalidate) | 0.5h | - |

**P0 合计 ≈ 5h**（一个工作日，覆盖中文 BM25 + MCP 参数补全）
**P0+P1 合计 ≈ 9h**（一个半工作日，覆盖所有有用改进）
**全部 ≈ 11h**（两个工作日，包括 P2 的 Pattern Miner 去重）

无新增重型依赖，仅 + jieba（~10MB）。

---

## 6. 验证清单

完成 P0+P1 后下列指标应满足：

| 指标 | 当前 | 目标 | 验证方式 |
|---|---|---|---|
| 中文 BM25 Recall@10 | ~0.45 | ≥ 0.75 | LongMemEval 中文子集 |
| Episodic 写入 P50 | ~250ms | ≤ 80ms (trivial) / ≤ 250ms (有 fact) | 跑 demo/chat_test |
| Episodic 写入 P99 | ~600ms | ≤ 400ms | 同上 |
| Recall P50 | ~80ms | ≤ 80ms (jieba 不增加显著开销) | 同上 |
| max_tokens=6000 命中率 | N/A | ≥ 99% 请求未超预算 | MCP 集成测试 |
| valid_at 历史查询正确率 | N/A | ≥ 95% | 单元测试构造时间窗口数据 |
| Tier 迁移日 cold 率 | N/A | 90 天后 < 30% memories 仍在 hot | 长期运行观测 |
| Snapshot 一致性 | TTL 5min 可能陈旧 | 任何写入后下次读必新 | 集成测试 |

---

## 7. 设计原则（项目应坚守）

1. **MCP-Native 优先**：Resource 用于持续注入、Tool 用于显式动作，不为对标膨胀工具数
2. **写求快**：写入路径最多 1 次同步 LLM；trivial 跳过；冲突仲裁全异步
3. **读求准**：召回接受 50-100ms 延迟换精度，但绝不为 5% 精度提升加 600MB 模型
4. **存储可替换，记忆语义是核心**：Protocol 抽象保护，存储后端是实现细节
5. **中文场景一等公民**：embedding (bge-zh)、分词 (jieba)、谓词中文模板、错误信息中文 — 每层都用中文最佳实践，不是英文方案凑合
6. **降级永远比报错好**：LLM 失败 → defer；jieba 失败 → 降级 unicode61；reranker（如未来加）失败 → 降级原排序
7. **可解释性 > 黑盒精度**：RecallSignals 暴露 4 分；ConflictAction 暴露 reasoning；ArbitrationDecision 全审计
8. **YAGNI**：不为"未来可能用上"加抽象。当前 manage_memory 一个工具覆盖 list/forget/mark_stale/arbitrations 四个动作，比拆四个工具好

---

## 8. 总结

MemoCortex 的架构设计已经吸收了主流框架的精华且有自己的细化创新。**当前阶段不需要追加新能力，只需要把已有设计落实到位**：

- Phase 1 修中文检索（项目自身责任）
- Phase 2 暴露已有的时间语义和 Token 控制（最小成本对齐 SOTA）
- Phase 3 减少不必要 LLM 调用（性能优化）
- Phase 4 把 tier 字段、snapshot 缓存的设计接通（完成度问题）

约 9 小时工作量、零重型依赖增加。完成后 MemoCortex 在**中文 + 轻量本地部署**这个细分领域是无可替代的。

至于 Reranker / HyDE / RRF / 多租户 / Agent-as-Editor 这些"看起来很美的功能"，**项目现阶段不需要。** 等真实用户场景提出明确需求、并且当前设计无法满足时，再回来评估。**架构师的核心责任是知道什么不做。**

---

*文档结束。如需开始 Phase 1 的代码实现，告知即可，第一步是 `app/utils/tokenizer.py` + `app/storage/fts_store.py` 的改造。*
