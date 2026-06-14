# MemoCortex 评测集 (eval datasets)

## v1 — 中文长期记忆评测集 (80 条)

**文件**: `memocortex_zh_v1.jsonl`
**生成器**: `build_zh_v1.py` — 数据集是 deterministic 的, 改生成器跑一遍就能复现.

### 为什么自建

| 公开数据集 | 不用的原因 |
|---|---|
| LongMemEval | 英文为主, bge-small-zh-v1.5 在 cross-lingual 场景下分数会被 embedding gap 主导, 失去对召回算法本身的判别力 |
| LoCoMo | 同上 + 对话格式重, 需要先把每条对话喂进系统; 跑通成本远高于自建 |
| Mem0 用例 | 只能跑全套, 不能针对 4 信号的具体行为做消融 |

简单说: **我的目标是用评测集做 grid search, 而不是刷 leaderboard**.
为了让"权重 (0.4, 0.2, 0.2, 0.2)"这一选择有数据支撑, 数据集必须能区分:

- 单纯关键词匹配的 case (`exact_recall`)
- 必须靠向量泛化的 case (`paraphrase`)
- 必须靠时间衰减的 case (`temporal_window` / `episodic_temporal`)
- 必须靠 importance 信号的 case (`conflict_latest` 用 staleness)
- 召回容易翻车的反例 (`negation`)

公开 benchmark 的混合 score 对 4 信号权重的判别力远不如这种分场景设计的数据集.

### 80 条分布

| Scenario | 条数 | 主要考察的召回信号 |
|---|---:|---|
| `exact_recall` | 20 | 向量 + BM25 都应高分 (基线) |
| `paraphrase` | 15 | 向量泛化, BM25 弱 |
| `temporal_window` | 10 | valid_from / valid_until 语义 |
| `conflict_latest` | 10 | staleness_signal × 0.2 应让旧版降权 |
| `negation` | 5 | 向量是否区分 "喜欢" / "不喜欢" |
| `episodic_temporal` | 10 | 时间衰减 + episodic 路径 |
| `procedural` | 5 | 任务模板召回 |
| `mixed_types` | 5 | 跨记忆类型, 应同时命中 sem + epi |

### 单条样例结构

```json
{
  "id": "zh-001",
  "scenario": "exact_recall",
  "user_suffix": "001",
  "setup": [
    {
      "mid": "target",
      "type": "semantic",
      "content": "我对花生过敏",
      "structured": {
        "subject": "user",
        "predicate": "allergic_to",
        "object": "花生"
      },
      "created_days_ago": 5,
      "importance": 0.8
    },
    { "mid": "001_d0", "type": "semantic", "content": "...", ... }
  ],
  "query": "花生过敏",
  "expected_mids": ["target"],
  "top_k": 5,
  "score_threshold": 0.0,
  "notes": "..."
}
```

- `mid`: dataset-local 标识符. Runner 会在 setup 时把 mid 映射到真实 uuid memory_id, 对比 expected 时用映射后的 id.
- `created_days_ago`: 写入时手动 backfill `created_at`, 让时间衰减信号能起作用.
- `score_threshold`: 大多数评测条目用 0.0 关闭过滤, 拿全部候选; 少量场景用默认阈值测端到端.
- 每条数据集独立 `user_suffix` → `eval_user_<suffix>` 形式的 user_id, 避免数据串扰.

### 干扰项 (distractors)

每条 setup 有 3-5 条不相关的"用户事实"作为干扰. 没有干扰时 K=5 一律 trivially 全召回, 测不出分辨率.

### 重新生成

```bash
PYTHONIOENCODING=utf-8 uv run python eval/datasets/build_zh_v1.py
```

输出会覆盖 `memocortex_zh_v1.jsonl`. 80 条 ID 唯一性 / 总数 / 字段完整性由 `tests/integration/test_eval_dataset.py` 校验.

### 后续兼容

`runner.py` 会兼容 LongMemEval / LoCoMo 适配器接口 (Stage 3 实现), 让此评测集和公开数据集能在同一个 metrics 框架下出分.
