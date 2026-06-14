"""Request-scoped trace context — async-safe trace_id 传播.

设计原则:
  - 用 contextvars.ContextVar 在 async 上下文传播 trace_id
  - 每次进入 MCP tool 调用时 generate_trace_id() 生成新 ID
  - 所有 logger 调用通过 loguru.bind 自动带上 trace_id
  - 业务代码 0 改动, 装饰器 + lifespan 完成所有注入

为什么不用 threading.local:
  - asyncio 单线程跑多个 task, threading.local 会在 task 间串
  - contextvars 是 PEP 567 标准, async 安全, await 跨越正确传播

为什么不在 MCP server 入口手动注入:
  - FastMCP 的 tool 是 async 装饰器, 我们自己也写一层 trace_context_tool 装饰器
  - 业务方法仍可手动 with_trace_context() 进入子 context (e.g. 后台 worker)

面试 talking point:
  - 线上 recall 慢了, 直接 grep trace_id 拉出整条调用链 (vector 召回 / BM25 / 算分 / metadata update)
  - Worker / scheduler 跑后台任务用 worker_trace_id 区分, 不和用户请求混
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

# trace_id 默认空, 表示当前不在请求上下文 (后台 worker 也可以显式 bind)
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """读当前上下文的 trace_id; 不在 trace 上下文时返回空字符串."""
    return _trace_id_var.get()


def set_trace_id(trace_id: str) -> None:
    """直接设置 trace_id (高级用法). 一般用 with trace_context() 上下文管理器."""
    _trace_id_var.set(trace_id)


def generate_trace_id() -> str:
    """生成新 trace_id (UUID4 前 12 位, 短而唯一)."""
    return uuid4().hex[:12]


@contextmanager
def trace_context(trace_id: str | None = None):
    """上下文管理器 — 进入时设 trace_id, 退出时恢复原值.

    用法:
        with trace_context() as tid:
            # 这里所有 logger 调用都带 tid
            logger.info("...")

        with trace_context("explicit-123"):
            # 显式指定 trace_id
            ...

    与 asyncio 正交: contextvars 在 async / await 跨越时自动复制上下文,
    子 task 看到父 task 的 trace_id (除非手动 trace_context() 改).
    """
    tid = trace_id or generate_trace_id()
    token = _trace_id_var.set(tid)
    try:
        yield tid
    finally:
        _trace_id_var.reset(token)


def traced(func):
    """装饰器: 自动给 async 函数加 trace_context.

    用法:
        @mcp.tool()
        @traced
        async def remember(...):
            # 函数内所有 logger 调用都带 trace_id
            ...

    每次调用都生成新的 trace_id; 业务方可以从 logger 输出里 grep 出
    "这次调用涉及哪些子操作".

    嵌套调用时, 内层会复用外层 trace_id (不生成新的) — 让 worker / scheduler 触发的链路
    可以一直串到底.
    """
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # 已在 trace 上下文里: 复用 trace_id, 不生成新的
        if get_trace_id():
            return await func(*args, **kwargs)
        # 没有 trace 上下文: 开新的
        with trace_context():
            return await func(*args, **kwargs)
    return wrapper
