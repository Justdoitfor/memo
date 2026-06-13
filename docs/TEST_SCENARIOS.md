# MemoCortex 对话测试脚本

> 通过 `demo/chat_agent.py` 的对话界面验证全部能力。
> 11 个 MCP Tool + 5 个 Resource + Phase 1-4 改造的全部新能力。
>
> 启动方式:
> ```bash
> # 终端 1
> uv run python -m mcp_server.server
>
> # 终端 2 (推荐每个剧本用独立 user_id, 避免污染)
> uv run python demo/chat_agent.py --user-id test_basic
> ```

---

## 准备工作

### 验证点查看方式

| 验证点 | 怎么看 |
|---|---|
| Agent 调了哪个工具、参数是什么 | 终端会打印 `[Tool] toolname({...})` |
| 工具实际返回 | 看 MCP Server 终端的日志 |
| 数据落库情况 | 退出对话后 demo 会自动打印 Snapshot + 记忆统计 |
| 记忆条数 / staleness / 仲裁记录 | 让 Agent 调 `manage_memory` |

### 推荐流程
1. 先跑剧本 0 做冒烟测试（5 分钟覆盖核心路径）
2. 再按需挑剧本 1-10 验证特定能力
3. 剧本 11 是 Resource 专项

---

## 剧本 0 · 冒烟测试 (5 分钟，先跑这个)

**目标**: 一次对话覆盖最高频路径（remember + recall + snapshot resource）。

**user_id**: `smoke_test`

```
You: 你好, 我叫小明, 今年 30 岁, 在字节跳动工作, 对花生过敏
```
**期望**:
- Agent 应识别多个 fact, 调 `remember(memory_type="semantic")`
- 可能拆成 1-3 次 remember 调用 (每个 fact 一次, 或批量)
- 终端能看到 `[Tool] remember({"content": "...", "memory_type": "semantic", "importance": "high"})`

```
You: 你还记得我对什么过敏吗
```
**期望**:
- Agent 调 `[Tool] recall({"query": "过敏"})`
- 命中 "对花生过敏"
- 回复包含 "花生"

退出后看打印 (输入 `quit`):
- `Snapshot 记忆快照` 应有 4-5 条核心事实
- 记忆统计有 `semantic` 类型若干条

**通过标准**: Agent 能从 fact 中抽取并存为 semantic; recall 能用中文查找命中。

---

## 剧本 1 · 中文 BM25 召回 (验证 jieba 改造)

**目标**: 验证 Phase 1.1 的 jieba 切词在长 query 下仍能命中正确事实。

**user_id**: `chinese_bm25`

```
You: 我对花生和乳糖都过敏, 不能吃含奶制品的东西
You: 我家有只叫小白的猫, 还有只叫旺财的狗
You: 我下个月要去上海出差三天
You: 我手机是 iPhone 16 Pro Max, 笔记本是 MacBook Pro
You: 我在上一家公司花费了不少时间做后端架构
```

```
You: 你还记得我对什么过敏吗
```
**关键验证点**:
- 旧实现 (按字切): query "过敏" 会通配命中 "花费了不少时间" (因为 "花" 字也命中)
- 新实现 (jieba 切): 应**只命中**真正含"花生 / 乳糖 / 过敏"的记忆, 不命中"花费"

```
You: 我的宠物都有什么
```
**期望**: recall 命中 "猫"/"狗"/"小白"/"旺财", 不会命中无关 fact。

---

## 剧本 2 · 冲突仲裁三档 (覆盖 ConflictAction)

**目标**: 验证 SemanticMemory 的字段语义判定 (unique / list / versioned)。

**user_id**: `conflict_test`

### 2A. Unique 字段 (lives_in) → REPLACE

```
You: 我现在住在北京海淀
You: 等等, 我搞错了, 我刚搬家了, 现在住在上海浦东
```
**期望**:
- 第一条 → 写入 `(user, lives_in, 北京海淀)`
- 第二条 → defer 策略检测到 unique 字段冲突, 旧记忆 staleness=True, 新记忆 source_type="corrected"

```
You: 我现在住哪里
```
**期望**: Agent 回复"上海浦东", 不应混淆为"北京"。

```
You: 看看我的所有 semantic 记忆
```
**期望**: Agent 调 `manage_memory(action="list")`, 旧"住在北京"应标记 `staleness=True`。

### 2B. List 字段 (allergic_to) → MERGE

```
You: 我对花生过敏
You: 我还对乳糖过敏
You: 还有, 我对芒果也过敏
```
**期望**: 三条都保留 (不冲突), 因为 allergic_to 是 list 字段。

### 2C. Versioned 字段 (worked_in) → 双保留

```
You: 我之前在腾讯工作过两年
You: 后来跳槽到字节做基础架构
You: 现在在阿里
```
**期望**: 三条 worked_in 关系都保留, 各自带 `valid_from / valid_until`。

---

## 剧本 3 · 历史时刻查询 ⭐ (Phase 2.2 新能力 valid_at)

**目标**: 验证 `recall(valid_at=...)` 能回到指定时刻。

**user_id**: `historical_query`

**前置**: 跑完剧本 2C, 数据已有 worked_in 时态链。

```
You: 我 2024 年在哪工作?
```
**关键期望**:
- Agent 应调 `[Tool] recall({"query": "...", "valid_at": "2024-XX-XXT00:00:00"})`
- 注意终端打印的 `valid_at` 参数是否出现 ⭐
- 返回 2024 当时有效的工作经历

```
You: 我现在在哪工作?
```
**期望**: Agent 调 recall **不带** valid_at, 返回当前工作。

**通过标准**: Agent 能识别"以前 / 去年 / 2024" 触发 valid_at 参数。

---

## 剧本 4 · Token 预算控制 ⭐ (Phase 2.1 新能力 max_tokens)

**目标**: 验证 `recall(max_tokens=...)` 能截断长返回。

**user_id**: `token_budget`

写入大量长 episode:
```
You: 今天和老板开了个 30 分钟会议, 主要讨论了 Q3 OKR 的拆解, 包括团队的 KPI 目标、资源分配方案、关键里程碑设置, 以及和上下游团队的协作机制等等
You: 昨天晚上看了一部纪录片, 关于深海生物的, 介绍了 5 种鲜为人知的发光生物, 它们的生存环境和捕食方式都很奇特
You: 上周三去看了医生, 医生说我血压有点偏高, 建议每天监测, 减少咖啡因摄入, 增加运动, 半个月后复查
You: 周末去爬了山, 走了大约 12 公里, 海拔上升 800 米, 用了 4 个小时, 中途遇到一个老朋友
You: 朋友推荐了几本书, 包括《深度工作》《原子习惯》《心流》《刻意练习》, 都是讲个人成长的
```

```
You: 简短地告诉我最近发生的事, 控制在 200 字以内不要超
```
**期望**:
- Agent 应识别"控制长度"指令, 调 `[Tool] recall({"max_tokens": ~600, ...})`
- 终端打印的 args 含 `max_tokens` ⭐
- MCP Server 日志会有 `recall token 预算截断: 5 -> 2 (budget=600)` 这样的 DEBUG 行

**通过标准**: Agent 在用户明确长度约束时使用 max_tokens 参数。

---

## 剧本 5 · Trivial 跳过 (Phase 3.1 新能力)

**目标**: 验证 `_is_likely_fact` 启发式: 闲聊不触发 LLM 抽取, 含 fact 才抽取。

**user_id**: `trivial_test`

打开 MCP Server 端的 DEBUG 日志，跑这段：

```
You: 你好
You: 嗯
You: 好的
You: 收到
You: 今天天气真好
You: 哈哈
You: ok
```
**期望**: 这 7 条几乎都不应触发 semantic 抽取。
**验证**: MCP Server 日志应只有 episodic 写入, 没有 LLM extract 调用; metrics 计数器 `orchestrator.write.trivial_skipped` 累计 +7。

```
You: 我对花生过敏
You: 我搬家到了上海
```
**期望**: 这两条触发 LLM 抽取。
**验证**: MCP Server 日志看到 `Semantic extraction returned ...`; semantic 记忆 +2。

**通过标准**: Trivial 文本只写 episodic, 不调 LLM。

---

## 剧本 6 · 知识图谱多跳查询 (graph_query / list_entities)

**目标**: 验证实体消解 + 图谱查询。

**user_id**: `graph_test`

写入关系网:
```
You: 我女朋友叫小雪, 她在腾讯做产品经理
You: 小雪的同事叫 Alice, 我们三个一起出去玩过
You: 小雪的好朋友叫小红, 在阿里做设计
You: 我和小雪都喜欢日料
```

```
You: 列出我认识的所有人
```
**期望**:
- Agent 应调 `[Tool] list_entities({"entity_type": "person"})`
- 返回 4 个实体: 小雪 / Alice / 小红 (+ user 自己)

```
You: 小雪和谁有关系?
```
**期望**:
- Agent 应调 `[Tool] graph_query({"query_type": "related", "entity": "小雪"})`
- 返回 Alice (同事) / 小红 (朋友) / 用户 (女朋友)

```
You: 有没有同时认识小雪和我的人?
```
**期望**:
- 触发 `[Tool] graph_query({"query_type": "multi_hop", "entity": "小雪", "max_hops": 2})`

**通过标准**: list_entities 能列出消解后的实体; graph_query 能做多跳。

---

## 剧本 7 · 实体合并 (entity_merge)

**目标**: 当 Entity Resolution 没自动识别时, 用户主动合并。

**user_id**: `entity_merge_test`

```
You: 我同事 Alice 是个很厉害的工程师
You: Alice Wang 上周给了我一些技术建议
```
**期望**: Entity Resolver 可能把 "Alice" 和 "Alice Wang" 识别为同一人 (LLM 判断), 也可能没识别。

```
You: 列出我认识的所有人
```
**情况 1**: 只有一个 Alice (已自动消解) → 跳过此剧本。
**情况 2**: 有 Alice 和 Alice Wang 两个实体 → 进入下一步。

```
You: Alice 和 Alice Wang 是同一个人, 帮我合并下
```
**期望**:
- Agent 应调 `[Tool] entity_merge({"primary_entity_id": "...", "secondary_entity_id": "...", "confirm": true})`
- 合并后再 list_entities 应只剩一个 Alice

---

## 剧本 8 · 行为信号 + Implicit 挖掘 (track_signal + reflect)

**目标**: 验证 Pattern Miner 能从重复信号中挖掘隐式偏好。

**user_id**: `pattern_test`

模拟用户多次纠正 + 偏好:
```
You: 帮我写个 Python 爬虫的代码示例
[Agent 给出代码]
You: 不要这种风格, 我喜欢用 type hint 而且函数式风格
[Agent 重写]
You: 还要再简洁一点

You: 帮我写个数据分析脚本
[Agent 给出代码]
You: 我说过我喜欢 type hint, 帮我加上
[Agent 修改]
You: 用 polars 不要用 pandas

You: 帮我写个 Web 接口
[Agent 给出代码]
You: 还是要 type hint 啊
You: 而且不要 emoji
```

每次纠正时, 期望 Agent 调:
```
[Tool] track_signal({"signal_type": "explicit_correction", "context_tags": ["python", "code_style"]})
```

最后:
```
You: 总结一下我的代码偏好
```
**期望**:
- Agent 调 `[Tool] reflect({"window_days": 14})`
- 返回新挖掘的 implicit 偏好, e.g. "用户偏好 Python 用 type hint, 函数式风格, 简洁, 不要 emoji"

**通过标准**: 重复 3+ 次的同类 signal 能挖出 Implicit Memory。

---

## 剧本 9 · 流程模板 (Procedural / recall_workflow)

**目标**: 验证 procedural memory 的写入和按场景检索。

**user_id**: `procedural_test`

```
You: 帮我记一下我的 code review 流程: 1) 先看 commit message 是否清晰 2) 检查测试覆盖率 3) 跑本地静态分析 4) 看核心逻辑变更 5) 验证 backward compat 6) 写中肯的评论
```
**期望**: Agent 调 `[Tool] remember({"memory_type": "procedural", "content": "...", "importance": "high"})`, structured 含 steps 数组。

```
You: 我现在要做 code review, 应该怎么做
```
**期望**:
- Agent 调 `[Tool] recall_workflow({"trigger_context": "code review"})`
- 返回上面 6 步流程, structured.steps 是结构化列表

---

## 剧本 10 · 用户画像 (get_profile + Reflective)

**目标**: 验证 Reflective Memory 聚合机制。

**user_id**: `profile_test` (复用剧本 1+2 写过的多 fact 用户)

```
You: 帮我看下我的画像
```
**期望**:
- Agent 调 `[Tool] get_profile({"auto_refresh": true})`
- 返回结构化 profile (one_liner / preferences / constraints / interaction_style)

```
You: 我还有什么没告诉你的?
```
**期望**:
- Agent 可能读 `memory://snapshot/{user_id}` Resource (在 system prompt 阶段已注入)
- 提示用户补充画像缺失项

---

## 剧本 11 · 记忆管理 (manage_memory)

**目标**: list / mark_stale / forget / arbitrations 四个 action。

**user_id**: 复用 `conflict_test` 已有冲突数据。

```
You: 列出我所有的记忆
```
→ `manage_memory({"action": "list"})`

```
You: 给我看看冲突仲裁记录
```
→ `manage_memory({"action": "arbitrations"})`
**期望**: 返回 conflict_test 剧本里 lives_in 北京→上海这次冲突的审计记录。

```
You: 把"住在北京"那条标记为过时
```
→ Agent 先 list 拿到 memory_id, 再 `manage_memory({"action": "mark_stale", "memory_id": "..."})`

```
You: 删除"住在北京"那条记忆
```
→ `manage_memory({"action": "forget", "memory_id": "...", "confirm": true})`
**期望**: Agent 在 forget 前应先确认 (因为是不可逆), 然后传 confirm=true。

---

## 剧本 12 · MCP Resource 直接验证

**目标**: 5 个 Resource 端点都能正确读取 (Agent 在 system prompt 启动时已读 snapshot)。

**user_id**: 复用任意已有数据的 user。

这个剧本不靠对话, 直接用 fastmcp client 验证 (添到 demo 里临时测试):

```python
# 临时脚本
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8766/mcp") as c:
        for uri in [
            "memory://snapshot/{uid}",
            "memory://summary/{uid}",
            "memory://profile/{uid}",
            "memory://workflows/{uid}",
            "memory://entities/{uid}",
        ]:
            r = await c.read_resource(uri.format(uid="conflict_test"))
            print(f"--- {uri} ---")
            print(r[0].text[:500] if r else "EMPTY")
```

**通过标准**: 5 个 Resource 都返回非空内容 (除非该 user 该类记忆为空)。

---

## 性能 / 长期运行验证 (可选)

### 验证 Snapshot Cache 版本号

跑一段对话写入 N 条 fact, 然后**立即**问 Agent "我有什么记忆", 应能拿到刚写入的 (不是 5 分钟前的)。

```
You: 我刚买了辆特斯拉 Model Y
You: (立即) 我刚才说我买了什么车?
```
**期望**: Agent 调 recall 或读 snapshot 都能拿到"特斯拉 Model Y" (Phase 4.2 修的: 写入 → 立即一致)。

### 验证 Decay → 真删 Chroma

需要长期运行 (60+ 天) 才能看到, 不在对话测试范围。可手动改 `pattern_mine_interval_sec` / `reflect_decay_interval_sec` 加速 + 改 importance 衰减常数。

---

## 测试通过标准 (Definition of Done)

| 剧本 | 必须通过 | 加分项 |
|---|---|---|
| 0 (冒烟) | remember + recall 端到端通 | snapshot resource 能读到 |
| 1 (中文 BM25) | "过敏" 不命中"花费" | 长 query 仍命中正确 fact |
| 2 (冲突仲裁) | unique/list/versioned 三种行为正确 | arbitration_logs 有记录 |
| 3 (valid_at) | Agent 自动用 valid_at 参数 | 历史时刻精确返回 |
| 4 (max_tokens) | Agent 在长度约束下用 max_tokens | DEBUG 日志有截断行 |
| 5 (trivial) | 闲聊不触发 LLM | 含 fact 触发 LLM |
| 6 (graph_query) | multi_hop / related 各调一次 | community detect 也能调 |
| 7 (entity_merge) | 合并后实体数减少 | (条件性, 需自动消解未触发) |
| 8 (Pattern Miner) | reflect 后 implicit 记忆 +1 | 含正确的偏好描述 |
| 9 (procedural) | steps 结构化保留 | trigger_context 模糊匹配命中 |
| 10 (profile) | profile 字段非空 | one_liner 准确 |
| 11 (manage_memory) | 4 个 action 都通 | mark_stale 后召回降权 |
| 12 (Resources) | 5 个端点非空 | snapshot 缓存命中 (第二次 <1ms) |

---

## 一些常见问题 (FAQ)

**Q: Agent 不调 valid_at, 直接用 query "我去年住哪"?**
A: 这是 LLM 的判断问题, 不是项目 bug。可以在 system prompt 里强化提示, 或换更强的 LLM。当前 prompt 已经加了相关引导。

**Q: trivial 测试时所有"好的""嗯"也被 remember 调用了?**
A: 检查 Agent 是否走了 episodic 路径 (memory_type 默认 episodic 也会写). 真正要验证的是 episodic 写入后**没有触发 semantic 抽取**, 看 MCP Server 日志而非 tool 调用次数。

**Q: 实体合并 entity_merge 没法测?**
A: Entity Resolver 现在 LLM 判断比较强, 大概率自动合并。可以故意输入完全不同形式 (如 "Alice"/"王爱丽丝") 试试是否被识别为不同实体。

**Q: 需要清空数据重测?**
A: 退出 demo 后 `rm -rf data/`, 重启 server 即可。或者每个剧本用不同 user_id (推荐, 不需重启)。
