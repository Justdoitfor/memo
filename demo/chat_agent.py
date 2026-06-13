"""MemoCortex LangGraph 对话型 Demo — AI Agent + MCP 记忆框架

基于 LangGraph create_agent 构建 ReAct Agent,
通过 langchain-mcp-adapters 将 MCP 记忆工具转为 LangChain Tool,
Agent 在对话中自动决定何时调用记忆工具.

核心设计:
  - Agent System Prompt 引导记忆决策 (何时 remember / recall)
  - MCP Tools 作为 LangChain StructuredTool 供 Agent 自主调用
  - Snapshot Resource 作为 System Prompt 初始注入 (提供核心上下文)
  - 每轮对话: Agent 先 recall -> 回复用户 -> 主动 remember 有价值的信息

运行方式:
  1. 先启动 MCP Server:
     uv run python -m mcp_server.server
  2. 再运行 Demo:
     uv run python demo/chat_agent.py

  或指定用户 ID:
     uv run python demo/chat_agent.py --user-id alice

  自动启动 Server:
     uv run python demo/chat_agent.py --auto-start

依赖:
  - langgraph >= 1.2
  - langchain-mcp-adapters >= 0.3
  - MCP Server 运行在 http://127.0.0.1:8766/mcp
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ── Windows 终端 UTF-8 兼容 ──
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MCP_PORT = 8766
MCP_URL = f"http://127.0.0.1:{MCP_PORT}/mcp"


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


def _safe_print(text: str, max_len: int = 500) -> None:
    """安全打印: 截断过长内容, 替换不可编码字符."""
    if len(text) > max_len:
        text = text[:max_len] + f"...(共 {len(text)} 字)"
    print(text)


# ── MCP -> LangChain Tool 转换 ────────────────────────────────────────


async def load_memory_tools() -> list[Any]:
    """通过 langchain-mcp-adapters 将 MCP 记忆工具转为 LangChain Tool."""
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_mcp_adapters.sessions import StreamableHttpConnection

    connection = StreamableHttpConnection(
        transport="streamable_http",
        url=MCP_URL,
    )

    tools = await load_mcp_tools(session=None, connection=connection)
    return tools


# ── Agent System Prompt ────────────────────────────────────────────────


AGENT_SYSTEM_PROMPT = """你是一个拥有长期记忆能力的 AI 助手。你可以通过记忆工具记住用户的重要信息, 并在后续对话中回忆这些信息。

## 记忆工具使用指南

你有以下记忆工具可用:

### remember — 写入记忆
当用户提供具有跨会话价值的信息时, 主动调用:
- 个人偏好、事实、属性 → memory_type="semantic", importance="high"
- 事件经历 → memory_type="episodic", importance="medium"
- 工作流程/操作步骤 → memory_type="procedural", importance="medium"
- 用户纠正之前的信息 → source_type="corrected"

**不要记住**: 闲聊 (你好/谢谢/天气)、临时性内容、无跨会话价值的信息

### recall — 检索记忆
当用户提到之前讨论过的话题, 或你需要了解用户背景时调用:
- 用户问 "你还记得..." → 立即 recall
- 用户提到个人相关话题 → recall 获取背景
- 你不确定某个信息时 → recall 查证

### recall_workflow — 检索流程
当用户要执行某类任务时, 先查是否有定制化工作流。

### get_profile — 获取用户画像
当你需要全面了解用户时调用 (首次对话时建议调用一次)。

### track_signal — 上报行为信号
当观察到用户行为模式时上报:
- 用户纠正你 → signal_type="explicit_correction"
- 用户满意 → signal_type="positive_feedback"
- 用户要求换格式 → signal_type="format_preference"
- 用户要求重新生成 → signal_type="regenerate_request"

### reflect — 触发记忆反思
在长对话结束时调用, 让系统从行为信号中挖掘隐式偏好。

### manage_memory — 记忆管理
- action="list": 查看所有记忆
- action="mark_stale": 标记某条记忆过时 (不再准确)
- action="forget": 删除记忆 (需 confirm=True)
- action="arbitrations": 查看冲突仲裁记录

## 对话策略

1. 首次对话: 先调用 get_profile 了解用户, 如无画像则在对话中自然积累
2. 每次用户提到重要信息: 判断是否值得记住, 主动调用 remember
3. 回答问题时: 如果涉及用户个人偏好/历史, 先 recall 再回答
4. 自然地展示你记得的信息: "我记得你说过对花生过敏..."
5. 用户纠正信息: 用 remember + source_type="corrected" 记住纠正后的版本
"""


# ── 交互式对话主循环 ──────────────────────────────────────────────────


async def run_chat(user_id: str = "demo_user") -> None:
    """启动 LangGraph ReAct Agent 进行交互式对话."""

    print()
    print("  MemoCortex x LangGraph 对话型 Demo")
    print("  ReAct Agent + MCP 记忆工具")
    print(f"  用户: {user_id}")
    print(f"  MCP: {MCP_URL}")

    # ── 检查 MCP Server ──
    if not _is_port_open(MCP_PORT):
        print()
        print("  MCP Server 未运行! 请先启动:")
        print("    uv run python -m mcp_server.server")
        print()
        print("  或使用 --auto-start:")
        print("    uv run python demo/chat_agent.py --auto-start")
        return

    # ── 加载 MCP 工具 ──
    _header("初始化: 加载 MCP 记忆工具")
    print("  通过 langchain-mcp-adapters 连接 MCP Server...")
    tools = await load_memory_tools()
    tool_names = [t.name for t in tools]
    print(f"  加载成功! 工具: {tool_names}")

    # ── 创建 LLM ──
    from app.config import config
    from langchain_openai import ChatOpenAI

    print(f"  LLM: {config.llm_model} @ {config.llm_api_base}")
    llm = ChatOpenAI(
        model=config.llm_model,
        base_url=config.llm_api_base,
        api_key=config.llm_api_key,
        temperature=0.7,
    )

    # ── 构建 System Prompt (注入用户 ID 和已有记忆) ──
    system_prompt = AGENT_SYSTEM_PROMPT + f"\n\n当前对话用户 ID: {user_id}\n在调用所有记忆工具时, 使用 user_id=\"{user_id}\"."

    # ── 读取已有记忆快照, 注入到 prompt ──
    try:
        from fastmcp import Client
        client = Client(MCP_URL)
        async with client:
            snap_result = await client.read_resource(f"memory://snapshot/{user_id}")
            # Duck typing 解析
            snap_text = ""
            if isinstance(snap_result, list):
                for item in snap_result:
                    t = getattr(item, "text", None)
                    if t:
                        snap_text += t
            elif hasattr(snap_result, "text"):
                snap_text = snap_result.text
            else:
                snap_text = str(snap_result)

            if snap_text and "暂无记忆" not in snap_text:
                # 提取核心事实
                fact_lines = [l for l in snap_text.split("\n") if l.startswith("- ")]
                if fact_lines:
                    system_prompt += "\n\n## 用户已有记忆摘要\n" + "\n".join(fact_lines[:10])
                    print(f"  已有记忆: {len(fact_lines)} 条核心事实")
            else:
                print("  当前无记忆, 从零开始")
    except Exception as e:
        print(f"  读取快照失败 (忽略): {e}")

    # ── 构建 LangGraph Agent ──
    # 优先使用新 API (langchain.agents.create_agent), 回退到旧 API
    try:
        from langchain.agents import create_agent
        print("  构建 LangGraph Agent (新 API: create_agent)...")
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        print("  构建 LangGraph ReAct Agent (旧 API: create_react_agent)...")
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
        )
    print("  Agent 构建完成!")

    # ── 交互式对话 ──
    _header("开始对话")
    print("  输入消息与 AI 对话 (输入 quit 退出)")
    print("  Agent 会自动决定何时调用记忆工具")
    print()

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  退出对话")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("  退出对话")
            break

        # 调用 Agent
        conversation_history.append({"role": "user", "content": user_input})

        try:
            # LangGraph agent.ainvoke 需要 messages 格式
            result = await agent.ainvoke(
                {"messages": conversation_history},
            )

            # 提取结果中的所有 messages
            all_messages = result.get("messages", conversation_history)

            # ── 提取工具调用和最终 AI 回复 ──
            # Agent 流程: AIMessage(tool_calls) -> ToolMessage(results) -> AIMessage(final reply)
            # 我们要展示: 1) 中间的工具调用  2) 最终的 AI 文本回复
            tool_calls_seen = []
            final_ai_text = ""

            for msg in all_messages[len(conversation_history):]:
                if hasattr(msg, "type"):
                    if msg.type == "ai":
                        # 收集该 AI 消息中的 tool calls
                        tc = getattr(msg, "tool_calls", None) or []
                        if tc:
                            tool_calls_seen.extend(tc)
                        # 如果有文本内容且没有 tool calls, 这是最终回复
                        if msg.content and not tc:
                            final_ai_text = msg.content
                        elif msg.content and tc:
                            # 有些模型在 tool_calls 时也带文本 (并行思考)
                            if not final_ai_text:
                                final_ai_text = msg.content
                    # tool 类型消息 (工具执行结果) 我们不直接展示

            # ── 展示 ──
            if tool_calls_seen:
                print()
                for tc in tool_calls_seen:
                    tool_name = tc.get("name", "?")
                    args = tc.get("args", {})
                    if isinstance(args, dict):
                        key_args = {}
                        for k in ("user_id", "content", "query", "memory_type",
                                   "importance", "action", "signal_type", "source_type"):
                            if k in args and args[k]:
                                val = args[k]
                                if isinstance(val, str) and len(val) > 50:
                                    key_args[k] = val[:50] + "..."
                                else:
                                    key_args[k] = val
                        print(f"  [Tool] {tool_name}({json.dumps(key_args, ensure_ascii=False)})")
                    else:
                        print(f"  [Tool] {tool_name}")

            if final_ai_text:
                _safe_print(f"  AI: {final_ai_text}")
            elif tool_calls_seen and not final_ai_text:
                # 仅 tool calls, 最终回复在之后的 AIMessage 中 (可能被截断)
                # 尝试从最后一条消息获取
                for msg in reversed(all_messages):
                    if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                        _safe_print(f"  AI: {msg.content}")
                        final_ai_text = msg.content
                        break

            # 更新对话历史
            conversation_history = all_messages

        except Exception as e:
            print(f"\n  [Error] Agent 执行出错: {type(e).__name__}: {e}")

        print()

    # ── 对话结束, 展示记忆摘要 ──
    _header("对话结束 -- 记忆摘要")
    try:
        client = Client(MCP_URL)
        async with client:
            snap_result = await client.read_resource(f"memory://snapshot/{user_id}")
            snap_text = ""
            if isinstance(snap_result, list):
                for item in snap_result:
                    t = getattr(item, "text", None)
                    if t:
                        snap_text += t
            elif hasattr(snap_result, "text"):
                snap_text = snap_result.text

            print("  [Snapshot] 记忆快照:")
            for line in snap_text.split("\n")[:15]:
                if line.strip():
                    print(f"    {line}")

            # 记忆列表
            list_data = _parse_tool_result(
                await client.call_tool("manage_memory", {
                    "user_id": user_id, "action": "list",
                })
            )
            items = list_data.get("items", [])
            counts = {}
            for item in items:
                t = item.get("type", "unknown")
                counts[t] = counts.get(t, 0) + 1
            print(f"\n  [Stats] 记忆统计: 共 {len(items)} 条")
            for t, c in counts.items():
                print(f"    {t}: {c} 条")

            # Profile
            profile_data = _parse_tool_result(
                await client.call_tool("get_profile", {
                    "user_id": user_id, "auto_refresh": True,
                })
            )
            p = profile_data.get("profile", {})
            if p:
                _safe_print(f"\n  [Profile] 用户画像: {json.dumps(p, ensure_ascii=False)}", max_len=300)
    except Exception as e:
        print(f"  读取记忆摘要失败: {e}")


def _parse_tool_result(result: Any) -> dict[str, Any]:
    """Duck typing 解析 MCP CallToolResult."""
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
            return {"error": "isError"}
        return {"raw_content": str(result.content)}
    if isinstance(result, dict):
        return result
    return {"raw": str(result)}


# ── 入口 ──────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MemoCortex LangGraph 对话型 Demo")
    parser.add_argument("--user-id", default="demo_user", help="用户标识")
    parser.add_argument("--auto-start", action="store_true", help="自动启动 MCP Server")
    args = parser.parse_args()

    if args.auto_start and not _is_port_open(MCP_PORT):
        print("  启动 MCP Server...")
        proc = subprocess.Popen(
            ["uv", "run", "python", "-m", "mcp_server.server"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        print("  等待 Server 就绪 (最多 90s)...")
        for i in range(45):
            if _is_port_open(MCP_PORT):
                # 端口开了, 等模型加载
                print("  端口已开放, 等模型初始化...")
                time.sleep(10)
                break
            time.sleep(2)
        if not _is_port_open(MCP_PORT):
            print("  Server 启动超时!")
            return

    asyncio.run(run_chat(user_id=args.user_id))


if __name__ == "__main__":
    main()
