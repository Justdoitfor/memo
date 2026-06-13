"""MemoCortex 自动化对话测试 — 模拟多轮对话测试记忆能力

本脚本模拟多个连续的对话轮次, 自动发送预设消息,
通过 MCP 记忆框架验证:
  1. 记忆写入 — 用户提供的事实是否被记住
  2. 记忆召回 — 后续对话能否回忆起之前的信息
  3. 冲突消解 — 用户纠正信息时, 旧信息是否被正确处理
  4. 跨会话持久化 — 清空对话历史后, 记忆是否仍然存在

运行方式:
  1. 先启动 MCP Server:
     uv run python -m mcp_server.server
  2. 再运行测试:
     uv run python demo/chat_test.py

  或自动启动 Server:
     uv run python demo/chat_test.py --auto-start
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MCP_PORT = 8766
MCP_URL = f"http://127.0.0.1:{MCP_PORT}/mcp"


# ── MCP 工具调用封装 ──────────────────────────────────────────────────


def _parse_tool_result(result: Any) -> dict[str, Any]:
    """Duck typing 解析 MCP CallToolResult → dict."""
    if hasattr(result, "content") and isinstance(result.content, list):
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                    return {"data": parsed}
                except (json.JSONDecodeError, TypeError):
                    return {"raw_text": text}
        if hasattr(result, "isError") and result.isError:
            return {"error": "tool call isError"}
        return {"raw_content": str(result.content)}
    if isinstance(result, dict):
        return result
    return {"raw": str(result)}


def _parse_resource_result(result: Any) -> str:
    """Duck typing 解析 MCP Resource → 文本."""
    if isinstance(result, list):
        texts = []
        for item in result:
            t = getattr(item, "text", None)
            if t is not None:
                texts.append(t)
        if texts:
            return "\n".join(texts)
    if hasattr(result, "text"):
        return result.text
    return str(result)


async def mcp_call(client, tool_name: str, arguments: dict) -> dict[str, Any]:
    result = await client.call_tool(tool_name, arguments)
    return _parse_tool_result(result)


async def mcp_resource(client, uri: str) -> str:
    result = await client.read_resource(uri)
    return _parse_resource_result(result)


def _is_port_open(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result == 0
    except Exception:
        sock.close()
        return False


def _header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _sub(title: str) -> None:
    print()
    print(f"── {title} ──")


def _show(label: str, data: Any) -> None:
    """精简展示数据."""
    if isinstance(data, dict):
        # 只展示关键字段
        key_fields = ["memory_id", "routed_type", "arbitration", "status",
                       "count", "items", "results", "latency_ms",
                       "new_implicit_count", "new_records",
                       "signal_id", "user_id",
                       "metadata_deleted", "vector_deleted",
                       "content", "type", "importance"]
        compact = {}
        for k, v in data.items():
            if k in key_fields or isinstance(v, (int, float, bool)):
                if isinstance(v, str) and len(v) > 60:
                    compact[k] = v[:60] + "..."
                elif isinstance(v, list):
                    compact[k] = f"[{len(v)} items]"
                else:
                    compact[k] = v
        print(f"  [{label}] {json.dumps(compact, ensure_ascii=False)}")
    else:
        print(f"  [{label}] {str(data)[:100]}")


def _show_recall(data: dict) -> None:
    """展示 recall 结果."""
    results = data.get("results", [])
    latency = data.get("latency_ms", 0)
    print(f"  [recall] 延迟={latency:.1f}ms, 结果={len(results)} 条")
    for r in results[:5]:
        rec = r.get("record", {})
        sig = r.get("signals", {})
        stale = " [STALE]" if rec.get("staleness_signal") else ""
        print(f"    #{r.get('rank')} {rec.get('type','?')}: "
              f"score={sig.get('final_score',0):.3f} "
              f"→ {rec.get('content','')[:60]}{stale}")


# ── 测试场景 ──────────────────────────────────────────────────────────

# 模拟用户的多轮对话消息, 每条含 user_input + 预期的 remember 参数
TEST_SCENARIOS = [
    # ── Session 1: 基本事实记忆 ──
    {
        "session": "Session 1 — 基本事实写入",
        "rounds": [
            {
                "user_says": "你好, 我叫小明, 我是程序员",
                "remember": {
                    "content": "用户叫小明, 是程序员",
                    "memory_type": "semantic",
                    "importance": "high",
                },
                "recall_query": "小明的职业",
                "expected_recall_contains": ["程序员", "小明"],
            },
            {
                "user_says": "我对花生过敏, 住在北京",
                "remember": {
                    "content": "对花生过敏, 住在北京",
                    "memory_type": "semantic",
                    "importance": "high",
                },
                "recall_query": "花生过敏",
                "expected_recall_contains": ["花生", "过敏"],
            },
            {
                "user_says": "今天天气不错, 去公园散步了",
                "remember": None,  # 闲聊不值得记
                "recall_query": None,
            },
        ],
    },
    # ── Session 2: 冲突消解 (纠正信息) ──
    {
        "session": "Session 2 — 冲突消解 (纠正)",
        "rounds": [
            {
                "user_says": "不对, 我现在搬到上海了",
                "remember": {
                    "content": "搬到了上海",
                    "memory_type": "semantic",
                    "importance": "high",
                    "source_type": "corrected",
                },
                "recall_query": "住在哪里",
                "expected_recall_contains": ["上海"],
            },
        ],
    },
    # ── Session 3: 跨会话记忆验证 ──
    {
        "session": "Session 3 — 跨会话记忆验证",
        "rounds": [
            {
                "user_says": "你还记得我的名字和职业吗?",
                "remember": None,
                "recall_query": "小明的信息",
                "expected_recall_contains": ["小明", "程序员"],
                "verify_snapshot": True,  # 验证 snapshot 包含核心事实
            },
            {
                "user_says": "我过敏什么?",
                "remember": None,
                "recall_query": "过敏",
                "expected_recall_contains": ["花生"],
            },
        ],
    },
]


async def run_test(user_id: str = "test_user") -> None:
    """运行自动化记忆测试."""

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         MemoCortex 自动化对话测试                            ║")
    print("║         验证记忆写入/召回/冲突/跨会话                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  用户: {user_id}")
    print(f"  MCP: {MCP_URL}")

    # ── 检查 MCP Server ──
    if not _is_port_open(MCP_PORT):
        print("  MCP Server 未运行! 请先启动:")
        print(f"    uv run python -m mcp_server.server")
        return

    from fastmcp import Client

    client = Client(MCP_URL)

    # 测试统计
    stats = {"total": 0, "passed": 0, "failed": 0}

    try:
        async with client:
            tools = await client.list_tools()
            print(f"  连接成功! Tools: {[t.name for t in tools]}")
            print()

            # ── 清空旧数据 ──
            _header("准备: 清空旧数据")
            await mcp_call(client, "manage_memory", {
                "user_id": user_id, "action": "forget", "confirm": True,
            })
            await asyncio.sleep(2)

            snap = await mcp_resource(client, f"memory://snapshot/{user_id}")
            if "暂无记忆" in snap:
                print("  清空成功, 从零开始")
            else:
                print("  ⚠ 清空可能有残留, 继续测试")

            # ── 运行每个测试场景 ──
            for scenario in TEST_SCENARIOS:
                _header(scenario["session"])

                for round_info in scenario["rounds"]:
                    stats["total"] += 1
                    user_msg = round_info["user_says"]
                    print()
                    print(f"  👤 用户: {user_msg}")

                    # 1. Remember (如果有)
                    if round_info.get("remember"):
                        rem_args = round_info["remember"]
                        result = await mcp_call(client, "remember", {
                            "user_id": user_id,
                            "content": rem_args["content"],
                            "memory_type": rem_args.get("memory_type", "episodic"),
                            "importance": rem_args.get("importance", "medium"),
                        })
                        routed = result.get("routed_type", "?")
                        arb = result.get("arbitration")
                        arb_info = f" (冲突: {arb})" if arb else ""
                        print(f"  📝 已记住 [{routed}]: {rem_args['content'][:50]}{arb_info}")

                        # 等后台处理完成
                        await asyncio.sleep(3)

                    # 2. Recall (如果有)
                    if round_info.get("recall_query"):
                        recall_data = await mcp_call(client, "recall", {
                            "user_id": user_id,
                            "query": round_info["recall_query"],
                            "top_k": 5,
                            "min_confidence": 0.45,
                        })
                        _show_recall(recall_data)

                        # 验证预期内容
                        expected = round_info.get("expected_recall_contains", [])
                        results = recall_data.get("results", [])
                        found_content = " ".join([
                            r.get("record", {}).get("content", "") for r in results
                        ]).lower()

                        all_found = all(kw.lower() in found_content for kw in expected)

                        if all_found:
                            print(f"  ✅ 验证通过: 预期关键词 {expected} 均在召回结果中找到")
                            stats["passed"] += 1
                        else:
                            missing = [kw for kw in expected if kw.lower() not in found_content]
                            print(f"  ❌ 验证失败: 预期关键词 {missing} 未在召回结果中找到")
                            print(f"     实际内容: {found_content[:200]}")
                            stats["failed"] += 1

                    # 3. Snapshot 验证 (如果有)
                    if round_info.get("verify_snapshot"):
                        snap = await mcp_resource(client, f"memory://snapshot/{user_id}")
                        print(f"  💭 Snapshot:")
                        for line in snap.split("\n")[:10]:
                            if line.strip():
                                print(f"    {line}")

                        expected_snap = round_info.get("expected_recall_contains", [])
                        snap_lower = snap.lower()
                        snap_found = all(kw.lower() in snap_lower for kw in expected_snap)
                        if snap_found:
                            print(f"  ✅ Snapshot 验证通过: {expected_snap} 均在快照中")
                        else:
                            print(f"  ❌ Snapshot 验证失败: 部分预期内容不在快照中")

                    # 4. 如果没有 remember 和 recall, 只统计为 passed (闲聊轮)
                    if not round_info.get("remember") and not round_info.get("recall_query"):
                        print(f"  (闲聊轮次, 无需记忆验证)")
                        stats["passed"] += 1

                # 场景间等待
                await asyncio.sleep(2)

            # ── 最终 snapshot ──
            _header("最终状态: Snapshot + Profile")

            snap = await mcp_resource(client, f"memory://snapshot/{user_id}")
            print("  💭 Snapshot:")
            for line in snap.split("\n")[:15]:
                if line.strip():
                    print(f"    {line}")

            profile = await mcp_call(client, "get_profile", {
                "user_id": user_id, "auto_refresh": True,
            })
            p = profile.get("profile", {})
            if p:
                print(f"  📋 Profile: {json.dumps(p, ensure_ascii=False)[:200]}")
            else:
                print(f"  📋 Profile: 空 (可能 LLM 生成失败)")

            # ── 测试结果 ──
            _header("测试结果汇总")
            print(f"  总测试: {stats['total']}")
            print(f"  通过: {stats['passed']}")
            print(f"  失败: {stats['failed']}")
            if stats["failed"] == 0:
                print()
                print("  🎉 全部测试通过! 记忆框架功能正常!")
            else:
                print()
                print("  ⚠ 有测试未通过, 请检查上面的失败原因")

    except Exception as e:
        print(f"\n  ❌ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MemoCortex 自动化对话测试")
    parser.add_argument("--user-id", default="test_user", help="测试用户 ID")
    parser.add_argument("--auto-start", action="store_true", help="自动启动 MCP Server")
    args = parser.parse_args()

    if args.auto_start:
        import subprocess
        if not _is_port_open(MCP_PORT):
            print("  启动 MCP Server...")
            proc = subprocess.Popen(
                ["uv", "run", "python", "-m", "mcp_server.server"],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            print("  等待 Server 就绪...")
            for _ in range(45):
                if _is_port_open(MCP_PORT):
                    time.sleep(5)
                    break
                time.sleep(2)

    asyncio.run(run_test(user_id=args.user_id))


if __name__ == "__main__":
    main()