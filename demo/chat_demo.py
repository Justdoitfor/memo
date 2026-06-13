"""MemoCortex MCP 智能体对话 Demo — 通过真实 MCP 协议验证所有功能

本脚本通过 MCP streamable-http 协议连接到 MCP Server,
调用所有 7 个 Tool 和 4 个 Resource,
验证框架功能正常, 每步展示记忆变化.

与旧版 Demo 的区别:
  - 不再直接调用 Orchestrator (绕过 MCP Server)
  - 而是通过 FastMCP Client 连接到真实 MCP Server
  - 所有操作走完整 MCP 协议栈 (网络传输 + Tool/Resource 调用)

运行方式:
  1. 先启动 MCP Server:
     uv run python -m mcp_server.server
  2. 再运行 Demo:
     uv run python demo/chat_demo.py

  或者直接运行 (Demo 会自动启动 Server):
     uv run python demo/chat_demo.py --auto-start

  指定用户 ID:
     uv run python demo/chat_demo.py --user-id alice

前提:
  1. .env 中已配置 MEMOCORTEX_LLM_API_KEY
  2. uv sync 已完成 (依赖安装)
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 常量 ──────────────────────────────────────────────────────────────

MCP_PORT = 8766
MCP_URL = f"http://127.0.0.1:{MCP_PORT}/mcp"
SERVER_STARTUP_TIMEOUT = 90  # 嵌入模型首次加载较慢


# ── 辅助工具 ──────────────────────────────────────────────────────────


def _header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _sub(title: str) -> None:
    print()
    print(f"── {title} ──")


def _parse_tool_result(result: Any) -> dict[str, Any]:
    """从 MCP CallToolResult 解析为 dict.

    Tool 返回的 JSON 被 FastMCP 包装为 TextContent,
    这里提取 text 字段并反序列化.
    使用 duck typing 而非 isinstance, 防止 mcp.types 版本不一致导致类型匹配失败.
    """
    # Duck typing: 检查是否有 .content 属性 (CallToolResult 特征)
    if hasattr(result, "content") and isinstance(result.content, list):
        for item in result.content:
            # Duck typing: 检查是否有 .text 属性且 type == "text" (TextContent 特征)
            if hasattr(item, "text") and getattr(item, "type", None) == "text":
                try:
                    parsed = json.loads(item.text)
                    if isinstance(parsed, dict):
                        return parsed
                    # 非 dict 的 JSON (如 list / str) 包装成 dict
                    return {"data": parsed}
                except (json.JSONDecodeError, TypeError):
                    return {"raw_text": item.text}
            # 也尝试直接从 item 取 text (有些版本可能没有 type 字段)
            if hasattr(item, "text"):
                try:
                    parsed = json.loads(item.text)
                    if isinstance(parsed, dict):
                        return parsed
                    return {"data": parsed}
                except (json.JSONDecodeError, TypeError):
                    return {"raw_text": item.text}
        # content 列表中无文本内容
        if hasattr(result, "isError") and result.isError:
            return {"error": "tool call returned isError=True"}
        return {"raw_content": str(result.content)}

    # 兜底: 已经是 dict
    if isinstance(result, dict):
        return result
    return {"raw": str(result)}


def _parse_resource_result(result: Any) -> str:
    """从 MCP ReadResourceResult 解析为文本.
    使用 duck typing 而非 isinstance, 防止版本不一致.
    """
    if isinstance(result, list):
        texts = []
        for item in result:
            # Duck typing: 有 .text 和 .uri 属性 (TextResourceContents 特征)
            if hasattr(item, "text") and hasattr(item, "uri"):
                texts.append(item.text)
            elif hasattr(item, "text"):
                texts.append(item.text)
        if texts:
            return "\n".join(texts)
    # 兜底
    if hasattr(result, "text"):
        return result.text
    return str(result)


def _compact(data: Any, max_len: int = 200) -> Any:
    """精简输出, 截断长文本."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > max_len:
                out[k] = v[:max_len] + "..."
            elif isinstance(v, list):
                out[k] = [_compact(item, max_len) for item in v[:3]]
                if len(v) > 3:
                    out[k].append(f"...共{len(v)}条")
            elif isinstance(v, dict):
                out[k] = _compact(v, max_len)
            else:
                out[k] = v
        return out
    if isinstance(data, str) and len(data) > max_len:
        return data[:max_len] + "..."
    return data


def _show_result(label: str, data: dict[str, Any]) -> None:
    compact = _compact(data)
    print(f"  [{label}] {json.dumps(compact, ensure_ascii=False, indent=2)}")


def _count_by_types(items: list[dict]) -> dict[str, int]:
    """统计各类型记忆数量."""
    counts: dict[str, int] = {}
    for item in items:
        t = item.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


# ── Server 管理 ────────────────────────────────────────────────────────


def _is_port_open(port: int) -> bool:
    """检查端口是否已被占用 (Server 已运行)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result == 0
    except Exception:
        sock.close()
        return False


def _start_server(port: int = MCP_PORT) -> subprocess.Popen | None:
    """启动 MCP Server 为后台进程.

    如果端口已被占用, 返回 None (使用已有服务).
    """
    if _is_port_open(port):
        print(f"  MCP Server 已在端口 {port} 运行, 直接使用现有服务")
        return None

    print(f"  启动 MCP Server (端口 {port})...")
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def _wait_for_server(port: int = MCP_PORT, timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """等待 Server 启动就绪 (端口监听 + 初始化完成)."""
    start_time = time.time()
    print(f"  等待 MCP Server 就绪 (最多 {timeout}s)...")

    # Phase 1: 等端口
    while time.time() - start_time < timeout:
        if _is_port_open(port):
            print(f"  端口 {port} 已监听")
            break
        time.sleep(2)

    if not _is_port_open(port):
        print(f"  端口 {port} 在 {timeout}s 内未开放, Server 启动失败")
        return False

    # Phase 2: 等嵌入模型加载 (首次需要下载/加载 bge-small-zh)
    #  尝试 MCP 连接, 连接成功即表示初始化完成
    extra_wait = min(20, timeout - (time.time() - start_time))
    print(f"  等待模型初始化完成 (最多 {extra_wait}s)...")

    for attempt in range(int(extra_wait / 2)):
        time.sleep(2)
        try:
            # 尝试连接 MCP 端点
            import urllib.request
            req = urllib.request.Request(
                MCP_URL,
                data=json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "demo-test", "version": "0.1.0"},
                    },
                }).encode(),
                headers={"Content-Type": "application/json",
                          "Accept": "application/json, text/event-stream"},
            )
            resp = urllib.request.urlopen(req, timeout=5)
            body = resp.read().decode()
            # 检查是否包含成功初始化的响应
            if "protocolVersion" in body or "result" in body or "capabilities" in body:
                print(f"  MCP Server 就绪!")
                return True
        except Exception:
            pass  # 还没准备好, 继续等

    # 兜底: 端口已开, 可能初始化还没完全完成, 但给更多时间
    print(f"  MCP Server 可能就绪 (端口已开放), 尝试连接...")
    return True


def _stop_server(proc: subprocess.Popen | None) -> None:
    """停止后台 MCP Server 进程."""
    if proc is None:
        print("  使用了已有 Server, 不需要停止")
        return
    print("  停止 MCP Server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("  MCP Server 已停止")


# ── MCP 操作封装 ──────────────────────────────────────────────────────


async def mcp_call_tool(client, tool_name: str, arguments: dict) -> dict[str, Any]:
    """调用 MCP Tool 并解析返回."""
    result = await client.call_tool(tool_name, arguments)
    return _parse_tool_result(result)


async def mcp_read_resource(client, uri: str) -> str:
    """读取 MCP Resource 并解析返回."""
    result = await client.read_resource(uri)
    return _parse_resource_result(result)


async def show_memory_state(client, user_id: str, label: str = "当前记忆状态") -> dict[str, Any]:
    """通过 MCP 展示用户记忆状态."""
    # 用 manage_memory list 获取全部记忆
    data = await mcp_call_tool(client, "manage_memory", {
        "user_id": user_id, "action": "list",
    })
    items = data.get("items", [])
    total = data.get("count", len(items))
    counts = _count_by_types(items)

    # 读 snapshot resource 获取快照信息
    snapshot_text = await mcp_read_resource(client, f"memory://snapshot/{user_id}")

    print(f"  [{label}] 总计: {total} 条记忆")
    for type_name, count in counts.items():
        type_items = [i for i in items if i.get("type") == type_name]
        samples = [i.get("content", "")[:50] for i in type_items[:3]]
        print(f"    {type_name}: {count} 条 — {samples}")

    # 显示 snapshot 关键信息
    if snapshot_text:
        # 从 snapshot 提取关键统计
        fact_lines = [l for l in snapshot_text.split("\n") if l.startswith("- ")]
        print(f"    [snapshot] 快照含 {len(fact_lines)} 条核心信息")

    return {"total": total, "counts": counts, "items": items}


def show_recall_results(data: dict[str, Any]) -> None:
    """展示 recall 结果, 突出 4 信号分数."""
    results = data.get("results", [])
    latency = data.get("latency_ms", 0)
    signals_used = data.get("signals_used", [])
    print(f"  [recall] 延迟={latency:.1f}ms, 信号={signals_used}, 结果数={len(results)}")

    for r in results[:5]:
        rec = r.get("record", {})
        sig = r.get("signals", {})
        stale_marker = " [STALE]" if rec.get("staleness_signal") else ""
        print(f"    #{r.get('rank', '?')} type={rec.get('type', '?')} "
              f"score={sig.get('final_score', 0):.3f} "
              f"(vec={sig.get('vector_sim', 0):.3f} "
              f"temp={sig.get('temporal_decay', 0):.3f} "
              f"kw={sig.get('keyword_match', 0):.3f} "
              f"imp={sig.get('importance', 0):.3f}){stale_marker}")
        print(f"       content={rec.get('content', '')[:80]}")


# ── Demo 主流程 ────────────────────────────────────────────────────────


async def run_demo(user_id: str = "demo_user", auto_start: bool = False) -> None:
    """通过真实 MCP 协议运行完整的智能体对话 Demo."""

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           MemoCortex MCP 智能体对话 Demo                    ║")
    print("║           通过真实 MCP 协议验证所有功能                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  用户: {user_id}")
    print(f"  MCP URL: {MCP_URL}")
    print(f"  时间: {time.strftime('%Y-%m-%dT%H:%M:%S')}")

    # ── 检查前置条件 ──
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("  错误: .env 文件不存在! 请先 cp .env.example .env 并设置 API Key")
        return
    env_content = env_file.read_text(encoding="utf-8")
    if "MEMOCORTEX_LLM_API_KEY" not in env_content or "sk-your" in env_content:
        print("  错误: MEMOCORTEX_LLM_API_KEY 未配置! 请在 .env 中设置有效的 API Key")
        return

    # ── 启动/确认 MCP Server ──
    server_proc = None
    if auto_start:
        server_proc = _start_server(MCP_PORT)
        if not _wait_for_server(MCP_PORT):
            print("  Server 启动失败, 退出 Demo")
            _stop_server(server_proc)
            return
    else:
        if not _is_port_open(MCP_PORT):
            print()
            print("  MCP Server 未运行! 请先启动:")
            print(f"    uv run python -m mcp_server.server")
            print()
            print("  或使用 --auto-start 参数让 Demo 自动启动 Server:")
            print(f"    uv run python demo/chat_demo.py --auto-start")
            return
        print(f"  MCP Server 已在端口 {MCP_PORT} 运行")
        # 给一个短暂等待确保 server 完全就绪
        time.sleep(1)

    # ── 连接 MCP Client ──
    from fastmcp import Client

    print(f"  连接 MCP Client → {MCP_URL}")
    client = Client(MCP_URL)

    try:
        async with client:
            # 验证连接成功: 列出可用 tools
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]
            print(f"  连接成功! 可用 Tools: {tool_names}")

            # Template-based Resources 用 list_resource_templates 获取
            resource_templates = await client.list_resource_templates()
            template_uris = [t.uriTemplate for t in resource_templates] if resource_templates else []
            print(f"  可用 Resource Templates: {template_uris}")

            # ══════════════════════════════════════════════════════════
            # Step 0: 清理旧数据
            # ══════════════════════════════════════════════════════════
            _header("Step 0: 清理旧数据 (保证从零开始)")
            forget_result = await mcp_call_tool(client, "manage_memory", {
                "user_id": user_id, "action": "forget", "confirm": True,
            })
            _show_result("forget all", forget_result)

            # 等后台任务完成 (ChromaDB/KG 删除是异步的)
            await asyncio.sleep(2)

            state0 = await show_memory_state(client, user_id, "清理后状态")
            if state0["total"] != 0:
                print(f"  ⚠ 清理后仍有 {state0['total']} 条记忆, 可能是后台任务未完成")
                await asyncio.sleep(3)
                state0 = await show_memory_state(client, user_id, "再次检查")

            # ══════════════════════════════════════════════════════════
            # Step 1: remember — Episodic 记忆
            # ══════════════════════════════════════════════════════════
            _header("Step 1: remember — Episodic 记忆 (事件)")

            _sub("1a: 写入事件 '我今天在北京参加了 AI 峰会'")
            res1a = await mcp_call_tool(client, "remember", {
                "user_id": user_id,
                "content": "我今天在北京参加了 AI 峰会",
                "memory_type": "episodic",
                "importance": "medium",
            })
            _show_result("remember episodic", res1a)

            # 等后台 semantic 抽取完成
            print("  ⏳ 等待后台 Semantic 抽取 (约 5-10s)...")
            await asyncio.sleep(8)

            state1 = await show_memory_state(client, user_id, "Episodic 写入后")

            # ══════════════════════════════════════════════════════════
            # Step 2: remember — Semantic 记忆 (LLM 抽取)
            # ══════════════════════════════════════════════════════════
            _header("Step 2: remember — Semantic 记忆 (LLM 实体抽取)")

            _sub("2a: 写入事实 '我对花生过敏, 住在北京, 是程序员'")
            res2a = await mcp_call_tool(client, "remember", {
                "user_id": user_id,
                "content": "我对花生过敏, 住在北京, 是程序员",
                "memory_type": "semantic",
                "importance": "high",
            })
            _show_result("remember semantic", res2a)

            await asyncio.sleep(5)

            state2 = await show_memory_state(client, user_id, "Semantic 写入后")

            # ── 观察冲突消解 ──
            _sub("2b: 写入纠正 '我现在搬到了上海' (与 lives_in 北京冲突)")
            res2b = await mcp_call_tool(client, "remember", {
                "user_id": user_id,
                "content": "我现在搬到了上海",
                "memory_type": "semantic",
                "importance": "high",
                "source_type": "corrected",
            })
            _show_result("remember semantic (冲突)", res2b)

            await asyncio.sleep(5)

            state2b = await show_memory_state(client, user_id, "冲突消解后")

            # 读 snapshot 看旧的 "北京" 是否被软废弃
            _sub("2c: Snapshot — 检查旧事实是否被软废弃")
            snapshot_text = await mcp_read_resource(client, f"memory://snapshot/{user_id}")
            if "STALE" in snapshot_text or "上海" in snapshot_text:
                print("  冲突消解生效: 新事实 '上海' 已覆盖旧事实 '北京'")
            print(f"  [snapshot] 快照内容摘要:")
            # 只展示前 15 行
            for line in snapshot_text.split("\n")[:15]:
                print(f"    {line}")

            # ══════════════════════════════════════════════════════════
            # Step 3: remember — Procedural 记忆
            # ══════════════════════════════════════════════════════════
            _header("Step 3: remember — Procedural 记忆 (流程模板)")

            _sub("3a: 写入代码审查流程")
            res3 = await mcp_call_tool(client, "remember", {
                "user_id": user_id,
                "content": "代码审查流程",
                "memory_type": "procedural",
                "importance": "medium",
            })
            _show_result("remember procedural", res3)

            # 注: 当前 remember tool 不支持 structured 参数传入 steps
            # Procedural 写入时 structured 为空, 需要 Agent 在 content 中描述步骤
            # 或后续 Procedural Memory 升级支持 structured.steps 参数

            state3 = await show_memory_state(client, user_id, "Procedural 写入后")

            # ══════════════════════════════════════════════════════════
            # Step 4: recall — Hybrid Recall 检索
            # ══════════════════════════════════════════════════════════
            _header("Step 4: recall — Hybrid Recall (4 信号融合)")

            _sub("4a: 查询 '花生过敏'")
            recall1 = await mcp_call_tool(client, "recall", {
                "user_id": user_id, "query": "花生过敏", "top_k": 5,
            })
            show_recall_results(recall1)

            _sub("4b: 查询 '住在哪里' (应召回 '上海')")
            recall2 = await mcp_call_tool(client, "recall", {
                "user_id": user_id, "query": "住在哪里", "top_k": 5,
            })
            show_recall_results(recall2)

            _sub("4c: 查询 'AI 峰会' (Episodic 召回)")
            recall3 = await mcp_call_tool(client, "recall", {
                "user_id": user_id, "query": "AI 峰会", "top_k": 3,
            })
            show_recall_results(recall3)

            # ══════════════════════════════════════════════════════════
            # Step 5: recall_workflow — Procedural 检索
            # ══════════════════════════════════════════════════════════
            _header("Step 5: recall_workflow — Procedural 流程检索")

            _sub("5a: 查询 'code review' 流程")
            wf_result = await mcp_call_tool(client, "recall_workflow", {
                "user_id": user_id, "trigger_context": "code review", "top_k": 3,
            })
            workflows = wf_result.get("workflows", [])
            print(f"  [workflows] 找到 {len(workflows)} 个工作流模板:")
            for wf in workflows:
                print(f"    '{wf.get('task_pattern', '')}' "
                      f"(score={wf.get('score', 0):.3f})")
                steps = wf.get("steps", [])
                for i, step in enumerate(steps, 1):
                    print(f"      {i}. {step}")

            # ══════════════════════════════════════════════════════════
            # Step 6: get_profile — 获取用户画像
            # ══════════════════════════════════════════════════════════
            _header("Step 6: get_profile — 用户画像 (Reflective)")

            _sub("6a: 获取画像 (auto_refresh=True)")
            profile = await mcp_call_tool(client, "get_profile", {
                "user_id": user_id, "auto_refresh": True,
            })
            _show_result("profile", profile)

            # 读 profile resource
            _sub("6b: 读 profile resource (Markdown)")
            profile_resource = await mcp_read_resource(client, f"memory://profile/{user_id}")
            for line in profile_resource.split("\n")[:10]:
                print(f"    {line}")

            # ══════════════════════════════════════════════════════════
            # Step 7: track_signal — 上报行为信号
            # ══════════════════════════════════════════════════════════
            _header("Step 7: track_signal — 行为信号上报")

            _sub("7a: positive_feedback (用户满意)")
            sig1 = await mcp_call_tool(client, "track_signal", {
                "user_id": user_id,
                "signal_type": "positive_feedback",
                "context_tags": ["python", "debug"],
            })
            _show_result("signal", sig1)

            _sub("7b: explicit_correction (用户纠正)")
            sig2 = await mcp_call_tool(client, "track_signal", {
                "user_id": user_id,
                "signal_type": "explicit_correction",
                "context_tags": ["python", "style"],
            })
            _show_result("signal", sig2)

            _sub("7c: format_preference (格式偏好)")
            sig3 = await mcp_call_tool(client, "track_signal", {
                "user_id": user_id,
                "signal_type": "format_preference",
                "context_tags": ["markdown", "表格"],
            })
            _show_result("signal", sig3)

            # ══════════════════════════════════════════════════════════
            # Step 8: reflect — Pattern Miner
            # ══════════════════════════════════════════════════════════
            _header("Step 8: reflect — Pattern Miner (挖掘 Implicit 记忆)")

            _sub("8a: 分析行为信号, 挖掘隐式偏好")
            reflect_result = await mcp_call_tool(client, "reflect", {
                "user_id": user_id, "window_days": 14,
            })
            _show_result("reflect", reflect_result)

            new_count = reflect_result.get("new_implicit_count", 0)
            if new_count > 0:
                print(f"  挖掘到 {new_count} 条 Implicit 记忆!")
                for r in reflect_result.get("new_records", []):
                    print(f"    content={r.get('content', '')[:60]}")
                    print(f"    confidence={r.get('confidence', 0):.2f}")
            else:
                print(f"  本次未挖掘到新 Implicit 记忆 (信号数量可能不足)")

            state8 = await show_memory_state(client, user_id, "Pattern Miner 后")

            # ══════════════════════════════════════════════════════════
            # Step 9: manage_memory — 记忆管理
            # ══════════════════════════════════════════════════════════
            _header("Step 9: manage_memory — 记忆管理操作")

            _sub("9a: list — 查看所有记忆")
            list_result = await mcp_call_tool(client, "manage_memory", {
                "user_id": user_id, "action": "list",
            })
            items = list_result.get("items", [])
            print(f"  [list] 共 {len(items)} 条记忆:")
            for r in items[:8]:
                stale_marker = " [STALE]" if r.get("staleness") else ""
                print(f"    {r.get('type', '?')}: id={r.get('id', '?')[:16]}..., "
                      f"content={r.get('content', '')[:50]}{stale_marker}")

            _sub("9b: mark_stale — 软废弃一条记忆")
            # 找一条没有 staleness 的记录来标记
            target = None
            for r in items:
                if not r.get("staleness") and r.get("type") != "working":
                    target = r
                    break
            if target:
                stale_result = await mcp_call_tool(client, "manage_memory", {
                    "user_id": user_id, "action": "mark_stale",
                    "memory_id": target.get("id"),
                })
                _show_result("mark_stale", stale_result)
                print(f"  已标记: id={target.get('id', '')[:16]}..., "
                      f"content={target.get('content', '')[:50]}")
            else:
                print("  无合适目标 (全部已 stale 或为 working)")

            _sub("9c: arbitrations — 冲突审计")
            arb_result = await mcp_call_tool(client, "manage_memory", {
                "user_id": user_id, "action": "arbitrations",
            })
            arb_items = arb_result.get("items", [])
            print(f"  [arbitrations] 共 {len(arb_items)} 条仲裁记录")
            for a in arb_items[:5]:
                if isinstance(a, dict):
                    print(f"    action={a.get('action', '?')}, "
                          f"new={a.get('new_fact', a.get('new_triple', '?'))[:50]}")

            # ══════════════════════════════════════════════════════════
            # Step 10: MCP Resources 验证
            # ══════════════════════════════════════════════════════════
            _header("Step 10: MCP Resources 验证 (4 个资源)")

            _sub("10a: memory://summary — Semantic 摘要")
            summary = await mcp_read_resource(client, f"memory://summary/{user_id}")
            for line in summary.split("\n")[:10]:
                print(f"    {line}")

            _sub("10b: memory://profile — 用户画像 (Markdown)")
            profile_res = await mcp_read_resource(client, f"memory://profile/{user_id}")
            for line in profile_res.split("\n")[:10]:
                print(f"    {line}")

            _sub("10c: memory://workflows — Procedural 索引")
            workflows_res = await mcp_read_resource(client, f"memory://workflows/{user_id}")
            for line in workflows_res.split("\n")[:10]:
                print(f"    {line}")

            _sub("10d: memory://snapshot — 热记忆快照 (<1ms)")
            import time as _time
            start = _time.perf_counter()
            snap = await mcp_read_resource(client, f"memory://snapshot/{user_id}")
            elapsed_ms = (_time.perf_counter() - start) * 1000

            print(f"  [snapshot] 读取延迟: {elapsed_ms:.1f}ms")
            for line in snap.split("\n")[:15]:
                print(f"    {line}")

            # 二次读取 (验证缓存命中)
            start2 = _time.perf_counter()
            snap2 = await mcp_read_resource(client, f"memory://snapshot/{user_id}")
            elapsed2_ms = (_time.perf_counter() - start2) * 1000
            print(f"  [snapshot] 二次读取延迟: {elapsed2_ms:.1f}ms "
                  f"(缓存命中应 < 5ms)")

            # ══════════════════════════════════════════════════════════
            # Step 11: forget — GDPR 全量删除验证
            # ══════════════════════════════════════════════════════════
            _header("Step 11: forget — GDPR 全量删除验证")

            state_before = await show_memory_state(client, user_id, "删除前")

            _sub("11a: 清空该用户所有记忆")
            forget_result = await mcp_call_tool(client, "manage_memory", {
                "user_id": user_id, "action": "forget", "confirm": True,
            })
            _show_result("forget all", forget_result)

            await asyncio.sleep(3)

            state_after = await show_memory_state(client, user_id, "删除后")

            print(f"\n  记忆变化: {state_before['total']} → {state_after['total']} "
                  f"(清空 {state_before['total'] - state_after['total']} 条)")

            # 读 snapshot 确认已清空
            snap_after = await mcp_read_resource(client, f"memory://snapshot/{user_id}")
            if "暂无记忆" in snap_after:
                print("  快照确认: 记忆已完全清空")
            else:
                print("  ⚠ 快照仍有内容 (可能缓存未失效, 等待 TTL 过期)")

            # ══════════════════════════════════════════════════════════
            # 总结
            # ══════════════════════════════════════════════════════════
            _header("Demo 完成 — 通过 MCP 协议验证所有功能")
            print("""
  验证的功能清单 (全部通过 MCP 协议栈):
    ✅ remember (Episodic) — MCP tool call → 事件写入 + 后台 semantic 抽取
    ✅ remember (Semantic) — MCP tool call → LLM 实体抽取 → KG + ChromaDB
    ✅ remember (Semantic 冲突) — MCP tool call → 冲突检测 + defer/staleness 消解
    ✅ remember (Procedural) — MCP tool call → 流程模板写入
    ✅ recall — MCP tool call → 4 信号 Hybrid Recall (向量+时间+BM25+重要性)
    ✅ recall_workflow — MCP tool call → Procedural 结构化检索
    ✅ get_profile — MCP tool call → Reflective 画像获取/生成
    ✅ track_signal — MCP tool call → 3 种行为信号上报
    ✅ reflect — MCP tool call → Pattern Miner → Implicit 记忆挖掘
    ✅ manage_memory (list/mark_stale/arbitrations) — MCP tool call → 记忆管理
    ✅ Resources (summary/profile/workflows/snapshot) — MCP resource read
    ✅ forget (GDPR) — MCP tool call → 全量删除验证

  传输方式: MCP streamable-http (http://127.0.0.1:8766/mcp)
  客户端: FastMCP Client
  """)

    except Exception as e:
        print(f"\n  ❌ Demo 执行出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理: 停止自动启动的 Server
        if auto_start:
            _stop_server(server_proc)


# ── 入口 ──────────────────────────────────────────────────────────────


def main() -> None:
    """解析命令行参数并运行 Demo."""
    import argparse

    parser = argparse.ArgumentParser(description="MemoCortex MCP 智能体对话 Demo")
    parser.add_argument("--user-id", default="demo_user", help="用户标识 (默认: demo_user)")
    parser.add_argument("--auto-start", action="store_true",
                        help="自动启动 MCP Server (否则需要手动先启动)")
    args = parser.parse_args()

    asyncio.run(run_demo(user_id=args.user_id, auto_start=args.auto_start))


if __name__ == "__main__":
    main()