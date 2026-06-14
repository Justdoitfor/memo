# P2: 可观测性 (Observability) — trace_id + Prometheus + LLM 成本计量

> **背景**: P0/P1 让项目"声明都被数据证明"; P2 让"线上问题都能被快速定位"。
> 三件事: trace_id 串联调用链 / Prometheus 标准 metrics endpoint / LLM 调用的成本可观测.

## TL;DR

| 子任务 | 产出 | 关键能力 |
|---|---|---|
| P2.1 trace_id + JSON 日志 | `app/utils/trace_context.py`, 升级 `logger.py` | 线上 grep 一个 trace_id 拉出整条调用链 |
| P2.2 Prometheus /metrics | `app/utils/prometheus.py`, server `/metrics` endpoint | Grafana 直接接, 标准 PromQL 可用 |
| P2.3 LLM 成本计量 | `app/utils/cost_meter.py`, 升级 `llm_factory.py` | 每次 LLM 调用记 input/output tokens + 模型 + 耗时 + USD |

---

## P2.1: trace_id 注入

### 设计要点

**`contextvars.ContextVar`** 是 Python asyncio 的标准方案 — 比 `threading.local` 在 async 场景下正确, 每个 `await` 跨越正确传播.

```python
@traced
async def search(...):
    # 装饰器自动开 trace_context, 函数内所有 logger 调用都带上 trace_id
    logger.info("...")  # → [tid:abc123def456] 14:32:01 | INFO | ...
```

### 嵌套行为 (关键细节)

```python
with trace_context("outer-tid"):
    logger.info("...")        # → [tid:outer-tid]
    with trace_context("inner-tid"):
        logger.info("...")    # → [tid:inner-tid]
    logger.info("...")        # → [tid:outer-tid] (恢复)
```

但 `traced` 装饰器**复用现有 trace_id 而不嵌套**:

```python
@traced
async def inner():
    return get_trace_id()

with trace_context("outer-tid"):
    inner_tid = await inner()
    # inner_tid == "outer-tid", 不是新生成的!
```

这让 worker / scheduler 触发的链路可以一直串到底, 不会因为内层 traced 把外层 trace_id 覆盖.

### 输出格式

**text 模式 (开发):**
```
[tid:e50cc1284cf6] 15:51:02 | INFO     | app.recall.router:search:62 | ...
```

**json 模式 (生产, MEMOCORTEX_LOG_FORMAT=json):**
```json
{"timestamp": "2026-06-14T15:51:25+08:00", "level": "INFO",
 "trace_id": "demo-trace-001", "module": "app.recall.router",
 "function": "search", "line": 62, "message": "..."}
```

ELK / Loki / Grafana 直接吃, 不用预处理.

---

## P2.2: Prometheus /metrics endpoint

### 设计要点

不引入 `prometheus_client` 依赖, 手写 exposition format. 业务代码继续用现有 `metrics.incr() / metrics.timer()` API, 零改动.

| 业务调用 | Prometheus 输出 |
|---|---|
| `metrics.incr("recall.invocations")` | `memocortex_recall_invocations_total 1234` |
| `metrics.observe("recall.latency", 25.6)` | `memocortex_recall_latency_milliseconds{quantile="0.5"} 23.6` |
| `metrics.set_gauge("active_users", 42)` | `memocortex_active_users 42` |
| `metrics.add_cost("llm.deepseek", 0.0042)` | `memocortex_llm_deepseek_usd_total 0.0042` |

### 集成 FastMCP 的关键技巧

FastMCP 没有 middleware 接口, 但 `mcp.http_app()` 返回 Starlette app:

```python
app = mcp.http_app(path="/mcp", transport="streamable-http")
app.add_route("/metrics", _metrics_endpoint, methods=["GET"])
app.add_route("/health", _health_endpoint, methods=["GET"])
uvicorn.run(app, ...)
```

启动后:
- `http://127.0.0.1:8766/mcp` — FastMCP 协议 endpoint
- `http://127.0.0.1:8766/metrics` — Prometheus 抓取
- `http://127.0.0.1:8766/health` — k8s liveness/readiness probe

### Grafana / Prometheus 集成示例

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'memocortex'
    scrape_interval: 30s
    static_configs:
      - targets: ['memocortex.internal:8766']
```

PromQL 查询示例:
- 每分钟 recall 调用数: `rate(memocortex_recall_invocations_total[1m])`
- recall P95 延迟: `memocortex_recall_total_latency_milliseconds{quantile="0.95"}`
- LLM 总成本 (周): `increase(memocortex_llm_total_usd_total[7d])`

---

## P2.3: LLM 调用成本计量

### 设计要点

每次 LLM 调用自动记 input/output tokens + 模型 + 耗时 + USD 成本, **业务代码加一个 `purpose` 标签即可**:

```python
# Before (P2.3 之前)
result = await llm_factory.structured_invoke(prompt, schema, vars)

# After
result = await llm_factory.structured_invoke(
    prompt, schema, vars,
    purpose="arbitrator",  # ← 唯一改动
)
```

### 5 个调用点全部 tagged

```
arbitrator (冲突仲裁)        → llm.calls.arbitrator
semantic_extractor (事实抽取) → llm.calls.semantic_extractor
pattern_miner (隐式偏好挖掘)  → llm.calls.pattern_miner
reflective_profile (画像生成) → llm.calls.reflective_profile
entity_resolver (实体消解)    → llm.calls.entity_resolver
```

### 自动指标

每次调用产生 7 个 metrics:

```
llm.calls.{model}                      # counter, 调用次数
llm.calls.{purpose}                    # counter, 按用途分组
llm.errors.{model}                     # counter, 失败次数
llm.tokens.input.{model}               # counter, 输入 token 累计
llm.tokens.output.{model}              # counter, 输出 token 累计
llm.tokens.total.{model}               # counter, 总 token 累计
llm.duration.{model}                   # histogram (summary), 耗时分布
llm.{model}.usd                        # cost, 累积 USD
llm.total.usd                          # cost, 全局累积
```

### 价格表

主流模型的 per-1M-tokens 价格 (USD), 写在 `_MODEL_PRICES`:

| 模型 | Input | Output |
|---|---:|---:|
| deepseek-chat | $0.14 | $0.28 |
| deepseek-reasoner | $0.14 | $2.19 |
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| claude-3-5-sonnet | $3.00 | $15.00 |
| qwen-turbo | $0.05 | $0.20 |

**未知模型走 _DEFAULT** ($0.50 in / $1.50 out per 1M), 兜底估算.

### 真实 demo 输出

跑一次 "中国的首都是哪个城市?" 调用后的 Prometheus 输出:

```
memocortex_llm_calls_demo_test_total 1
memocortex_llm_tokens_input_deepseek_v4_flash_total 40
memocortex_llm_tokens_output_deepseek_v4_flash_total 25
memocortex_llm_duration_deepseek_v4_flash_milliseconds{quantile="0.5"} 1489.72
memocortex_llm_total_usd_total 0.000058
```

直接接 Grafana 就能看 "今天 token 烧了多少钱".

---

## 测试覆盖

| 模块 | 单测数 | 覆盖 |
|---|---:|---|
| `tests/unit/test_trace_context.py` | 13 | async 跨 await / 10 个并发 task 隔离 / 嵌套恢复 / `traced` 装饰器复用 |
| `tests/unit/test_prometheus.py` | 15 | 名称归一化 / 百分位 / counter+summary+gauge+cost 格式 / 命名合规 |
| `tests/unit/test_cost_meter.py` | 17 | 价格表查询 / 模型名归一化 / record_call / 上下文管理器 / extract_usage |

合计 **45 个新单测**, 全套 **243 测试** 全过.

## 简历 talking points

> **面试官**: "线上 recall 慢了你怎么定位?"
>
> **答**: "我加了 contextvars-based trace_id, 装饰器自动给 orchestrator 入口开 trace 上下文.
> 所有 logger 调用通过 loguru patcher 自动注入 trace_id, 跨 async/await 也正确传播.
> 生产用 JSON 格式输出 (`MEMOCORTEX_LOG_FORMAT=json`), 直接 docker logs → ELK / Loki.
> 出问题时用 `grep tid:abc123` 拉出整条调用链, vector 召回 / BM25 / 算分 / metadata update 全部串起来.
> 同时通过 /metrics endpoint 暴露 Prometheus 指标, Grafana 上 P95 延迟趋势一目了然."

> **面试官**: "你们 LLM 一天烧多少 token?"
>
> **答**: "我接了成本计量层 — `cost_meter.record_call()` 在每次 `structured_invoke` 末尾自动调.
> 通过 LangChain BaseCallbackHandler 拿 token_usage, 兼容 with_structured_output 路径.
> 5 个调用点分别打了 purpose 标签 (arbitrator / semantic_extractor / pattern_miner / ...),
> 让成本可以按用途聚合. 未知模型走默认兜底价, 已知模型 (DeepSeek/GPT/Claude/Qwen) 用真实价格表估算.
> 直接 PromQL 查 `increase(memocortex_llm_total_usd_total[7d])` 就是周成本."

## 后续工作 (留给真正生产部署)

- **OpenTelemetry trace 导出** — 跨服务追踪, 替代当前 trace_id (适合微服务架构)
- **Grafana dashboard JSON** — `infra/grafana/dashboard.json` 一键导入
- **告警规则** — `infra/prometheus/alerts.yml` 定义 P95 latency / token 烧得太快等告警
- **PII scan + prompt injection sanitize** — P2.4, 安全护栏 (留给下一个 session)
