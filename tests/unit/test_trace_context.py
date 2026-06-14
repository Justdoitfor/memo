"""trace_context 单元测试 — 验证 async 安全 / 嵌套行为 / 装饰器."""
from __future__ import annotations

import asyncio

import pytest

from app.utils.trace_context import (
    generate_trace_id,
    get_trace_id,
    set_trace_id,
    trace_context,
    traced,
)


class TestGenerateTraceId:
    """同步纯函数测试 — 不需要 asyncio."""

    def test_returns_12_chars(self):
        tid = generate_trace_id()
        assert len(tid) == 12

    def test_returns_hex(self):
        tid = generate_trace_id()
        # 应都是 hex 字符
        int(tid, 16)  # 不报错就是合法 hex

    def test_unique(self):
        tids = {generate_trace_id() for _ in range(100)}
        assert len(tids) == 100  # 100 次调用全部不重复


class TestTraceContext:
    def test_default_empty_outside_context(self):
        # 不在 trace_context 里, 应返回空字符串
        assert get_trace_id() == ""

    def test_context_sets_and_clears(self):
        assert get_trace_id() == ""
        with trace_context() as tid:
            assert get_trace_id() == tid
            assert tid != ""
        # 退出后清空
        assert get_trace_id() == ""

    def test_explicit_trace_id(self):
        with trace_context("custom-tid-123") as tid:
            assert tid == "custom-tid-123"
            assert get_trace_id() == "custom-tid-123"

    def test_nested_context_restores_outer(self):
        """嵌套 trace_context: 内层退出后恢复外层."""
        with trace_context("outer") as outer:
            assert get_trace_id() == "outer"
            with trace_context("inner") as inner:
                assert get_trace_id() == "inner"
                assert inner == "inner"
            # 内层退出, 应恢复外层
            assert get_trace_id() == "outer"

    def test_set_trace_id_direct(self):
        set_trace_id("manual-123")
        assert get_trace_id() == "manual-123"
        # 清掉避免影响其他测试
        set_trace_id("")


class TestAsyncBehavior:
    """asyncio + contextvars: trace_id 在 await 跨越时正确传播."""

    @pytest.mark.asyncio
    async def test_propagates_across_await(self):
        with trace_context("test-async") as tid:
            assert get_trace_id() == tid

            async def inner():
                # 子协程应继承父 task 的 trace_id
                return get_trace_id()

            inner_tid = await inner()
            assert inner_tid == "test-async"

    @pytest.mark.asyncio
    async def test_concurrent_tasks_isolated(self):
        """两个并发 task 应有独立的 trace_id, 不串扰."""
        async def task_with_tid(tid: str, sleep_ms: int) -> str:
            with trace_context(tid):
                await asyncio.sleep(sleep_ms / 1000)
                return get_trace_id()

        # 同时跑 10 个 task, 每个有自己的 trace_id
        tasks = [task_with_tid(f"task-{i}", 5) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # 每个都应该返回自己的 tid
        assert results == [f"task-{i}" for i in range(10)]


class TestTracedDecorator:
    @pytest.mark.asyncio
    async def test_traced_assigns_new_id_when_no_context(self):
        result_tid = None

        @traced
        async def my_func():
            nonlocal result_tid
            result_tid = get_trace_id()

        await my_func()
        assert result_tid != ""
        assert len(result_tid) == 12

    @pytest.mark.asyncio
    async def test_traced_reuses_existing_id(self):
        """嵌套调用: 内层 traced 应复用外层 trace_id, 不生成新的."""
        inner_tid = None

        @traced
        async def inner_func():
            nonlocal inner_tid
            inner_tid = get_trace_id()

        with trace_context("outer-fixed-id"):
            await inner_func()

        assert inner_tid == "outer-fixed-id"

    @pytest.mark.asyncio
    async def test_traced_clears_after_call(self):
        """退出 traced 函数后, 应清空 trace_id."""
        @traced
        async def my_func():
            return get_trace_id()

        tid_inside = await my_func()
        # 退出后没有 context, 返回空
        assert get_trace_id() == ""
        assert tid_inside != ""
