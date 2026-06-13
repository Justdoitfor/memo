"""MemoCortex 交互式问答 Demo — AI 对话 + MCP 记忆框架

真正的对话型 Agent Demo:
  - 用户在终端输入消息
  - AI 通过 LLM 生成回复
  - 同时通过 MCP 记忆框架自动:
    * 每轮对话前 recall 相关历史记忆
    * 每轮对话后 remember 有价值的信息
    * 读取 snapshot 获取核心上下文
  - 跨会话测试: 记忆持久化, 新对话仍能回忆旧信息

运行方式:
  1. 先启动 MCP Server:
     uv run python -m mcp_server.server
  2. 再运行 Demo:
     uv run python demo/chat_demo.py --interactive

  指定用户 ID (用于记忆隔离):
     uv run python demo/chat_demo.py --interactive --user-id alice

  Demo 内部使用与 .env 相同的 LLM (DeepSeek/OpenAI 兼容),
  与 MCP Server 共享同一套存储, 记忆真正持久化到 ChromaDB/SQLite/KG.
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
    """Duck typing 解析 MCP Resource 结果 → 文本."""
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
    """调用 MCP Tool."""
    result = await client.call_tool(tool_name, arguments)
    return _parse_tool_result(result)


async def mcp_resource(client, uri: str) -> str:
    """读取 MCP Resource."""
    result = await client.read_resource(uri)
    return _parse_resource_result(result)


# ── AI Agent 核心 ──────────────────────────────────────────────────────


SYSTEM_PROMPT = """你是一个友好的 AI 助手, 拥有长期记忆能力。你可以通过记忆框架记住用户的重要信息。

行为准则:
1. 每次对话时, 你会收到之前的相关记忆摘要, 请自然地利用这些信息
2. 如果用户提供了值得跨会话记住的信息 (偏好、事实、工作流程等), 你应该主动记住
3. 如果信息只是闲聊 (如 "今天天气不错"), 不需要记住
4. 回答时要自然地引用你记得的信息, 不要说 "根据我的记忆..."
5. 如果用户纠正了之前的信息, 要记住纠正后的版本

你的回答应该:
- 自然、友好、有帮助
- 适当展示你记得的信息 (如 "我记得你说过对花生过敏, 所以..." )
- 简洁为主, 不啰嗦
"""


async def decide_what_to_remember(
    llm_client: Any, user_id: str, user_message: str, ai_reply: str
) -> list[dict[str, Any]]:
    """让 LLM 判断本轮对话中有哪些值得记住的信息, 返回 remember 参数列表.

    这模拟了真实 Agent 的 "记忆决策" 能力:
    Agent 不是什么都记, 而是判断哪些信息有跨会话价值.
    """
    prompt = f"""分析以下对话, 判断是否有值得长期记住的信息.

用户说: {user_message}
AI回复: {ai_reply}

规则:
- 只记有跨会话价值的信息: 个人偏好、事实、工作流程、重要事件
- 不记闲聊: "你好"、"天气不错"、"谢谢" 等
- 冲突信息 (如用户纠正之前的说法) 用 source_type="corrected"
- 事实性信息用 memory_type="semantic"
- 事件性信息用 memory_type="episodic"
- 流程性信息用 memory_type="procedural"

如果没有值得记住的信息, 返回空列表 [].

返回 JSON 列表, 每个元素格式:
[
  {{
    "content": "要记住的核心信息 (自然语言)",
    "memory_type": "semantic / episodic / procedural",
    "importance": "low / medium / high",
    "source_type": "explicit_statement / corrected / inferred",
    "reason": "为什么值得记住"
  }}
]

只返回 JSON, 不要其他文字."""

    try:
        result = await llm_client.ainvoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        # 提取 JSON (可能被 markdown code block 包裹)
        text = text.strip()
        if text.startswith("```"):
            # 去掉 code block 标记
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        text = text.strip()
        if text.startswith("["):
            items = json.loads(text)
            if isinstance(items, list):
                return items
        # 尝试从文本中找到 JSON 列表
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            items = json.loads(match.group())
            if isinstance(items, list):
                return items
        return []
    except Exception as e:
        print(f"  [记忆决策] LLM 判断失败: {e}")
        return []


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


# ── 交互式对话主循环 ──────────────────────────────────────────────────


async def run_interactive(user_id: str = "demo_user") -> None:
    """交互式对话 Demo — 用户输入 → AI 回复 + 自动记忆."""

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         MemoCortex 交互式问答 Demo                          ║")
    print("║         AI 对话 + MCP 记忆框架                               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  用户 ID: {user_id}")
    print(f"  MCP URL: {MCP_URL}")
    print()

    # ── 检查 MCP Server ──
    if not _is_port_open(MCP_PORT):
        print("  MCP Server 未运行! 请先启动:")
        print(f"    uv run python -m mcp_server.server")
        return

    # ── 连接 MCP Client ──
    from fastmcp import Client

    print("  连接 MCP 记忆服务...")
    client = Client(MCP_URL)

    # ── 创建 LLM (用于 AI 对话) ──
    # 使用与 MCP Server 相同的 LLM 配置
    from app.config import config
    from app.core.llm_factory import LLMFactory

    print(f"  初始化 LLM: {config.llm_model} @ {config.llm_api_base}")
    chat_llm = LLMFactory.create_chat_model(temperature=0.7, streaming=False)

    try:
        async with client:
            # 验证 MCP 连接
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]
            print(f"  MCP 连接成功! Tools: {tool_names}")
            templates = await client.list_resource_templates()
            template_uris = [t.uriTemplate for t in templates] if templates else []
            print(f"  Resources: {template_uris}")
            print()

            # 读取已有记忆, 告诉用户
            snapshot = await mcp_resource(client, f"memory://snapshot/{user_id}")
            if "暂无记忆" in snapshot:
                print("  💭 当前无记忆, 从零开始对话")
            else:
                print("  💭 已有记忆:")
                for line in snapshot.split("\n")[:8]:
                    if line.strip() and not line.startswith("# MemoCortex"):
                        print(f"    {line}")
            print()

            # ── 对话历史 (本轮会话) ──
            conversation_history: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

            # ── 主对话循环 ──
            print("  ── 开始对话 (输入 'quit' 退出, 'status' 查看记忆状态) ──")
            print()

            while True:
                # 读取用户输入
                try:
                    user_input = input("👤 你: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n  退出对话")
                    break

                if not user_input:
                    continue

                if user_input.lower() == "quit":
                    print("  退出对话")
                    break

                if user_input.lower() == "status":
                    # 查看记忆状态
                    snapshot = await mcp_resource(client, f"memory://snapshot/{user_id}")
                    print("  💭 当前记忆快照:")
                    for line in snapshot.split("\n"):
                        if line.strip():
                            print(f"    {line}")
                    print()

                    # 也列出所有记忆
                    list_data = await mcp_call(client, "manage_memory", {
                        "user_id": user_id, "action": "list",
                    })
                    items = list_data.get("items", [])
                    print(f"  记忆列表 ({len(items)} 条):")
                    for item in items[:10]:
                        stale = " [STALE]" if item.get("staleness") else ""
                        print(f"    {item.get('type', '?')}: {item.get('content', '')[:60]}{stale}")
                    print()
                    continue

                if user_input.lower() == "clear":
                    # 清空记忆
                    print("  清空所有记忆...")
                    await mcp_call(client, "manage_memory", {
                        "user_id": user_id, "action": "forget", "confirm": True,
                    })
                    await asyncio.sleep(2)
                    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
                    print("  记忆已清空, 对话历史重置")
                    print()
                    continue

                # ── Step 1: Recall 相关记忆 ──
                recall_data = await mcp_call(client, "recall", {
                    "user_id": user_id,
                    "query": user_input,
                    "top_k": 5,
                    "min_confidence": 0.45,
                })

                # 构建记忆上下文注入
                memory_context = ""
                results = recall_data.get("results", [])
                if results:
                    memory_lines = []
                    for r in results[:3]:
                        rec = r.get("record", {})
                        sig = r.get("signals", {})
                        stale = " (已过时)" if rec.get("staleness_signal") else ""
                        memory_lines.append(
                            f"- [{rec.get('type', '?')}] {rec.get('content', '')} "
                            f"(相关度 {sig.get('final_score', 0):.2f}){stale}"
                        )
                    memory_context = "\n\n你记得关于此用户的以下信息:\n" + "\n".join(memory_lines)

                # ── Step 2: 读 snapshot (核心事实) ──
                snap = await mcp_resource(client, f"memory://snapshot/{user_id}")
                snap_facts = ""
                if "暂无记忆" not in snap:
                    fact_lines = [l for l in snap.split("\n") if l.startswith("- ")]
                    if fact_lines:
                        snap_facts = "\n\n用户核心事实:\n" + "\n".join(fact_lines[:5])

                # ── Step 3: 组装 prompt + 调用 LLM ──
                # 把记忆注入到当前用户消息中 (作为 context)
                enhanced_message = user_input
                if memory_context or snap_facts:
                    enhanced_message = user_input + memory_context + snap_facts

                conversation_history.append({"role": "user", "content": enhanced_message})

                try:
                    from langchain_core.messages import HumanMessage, SystemMessage

                    messages = [
                        SystemMessage(content=SYSTEM_PROMPT),
                    ]
                    # 只保留最近 6 条对话 (避免 token 过长)
                    recent = conversation_history[-6:]
                    for msg in recent:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            messages.append(HumanMessage(content=msg["content"]))

                    # 最后一条必须是当前用户消息
                    if messages[-1].content != enhanced_message:
                        messages.append(HumanMessage(content=enhanced_message))

                    ai_response = await chat_llm.ainvoke(messages)
                    ai_text = ai_response.content if hasattr(ai_response, "content") else str(ai_response)

                except Exception as e:
                    ai_text = f"(LLM 调用失败: {e})"

                conversation_history.append({"role": "assistant", "content": ai_text})

                # ── Step 4: 记忆决策 — 判断是否有值得记住的信息 ──
                remember_items = await decide_what_to_remember(
                    None, user_id, user_input, ai_text  # llm_client unused, we parse directly
                )

                # 手动做简单判断 (不依赖额外 LLM call, 减少延迟)
                # 如果用户消息中包含事实性信息, 自动记住
                remember_items = _simple_remember_heuristic(user_input, ai_text)

                for item in remember_items:
                    try:
                        result = await mcp_call(client, "remember", {
                            "user_id": user_id,
                            "content": item["content"],
                            "memory_type": item.get("memory_type", "episodic"),
                            "importance": item.get("importance", "medium"),
                            "source_type": item.get("source_type"),
                        })
                        mem_id = result.get("memory_id", "?")
                        routed = result.get("routed_type", "?")
                        arb = result.get("arbitration")
                        arb_info = f" (冲突: {arb})" if arb else ""
                        print(f"  📝 已记住 [{routed}]: {item['content'][:50]} → id={mem_id[:16]}{arb_info}")
                    except Exception as e:
                        print(f"  ⚠ 记忆写入失败: {e}")

                # ── Step 5: 显示 AI 回复 ──
                print(f"  🤖 AI: {ai_text}")
                print()

    except Exception as e:
        print(f"\n  ❌ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def _simple_remember_heuristic(user_message: str, ai_reply: str) -> list[dict[str, Any]]:
    """简单启发式判断: 是否有值得记住的信息.

    真实 Agent 会用 LLM 做这个决策, 但为了减少延迟和 API 调用,
    Demo 中先用启发式快速判断, 用户也可以手动输入 '/remember xxx' 强制记忆.
    """
    items = []
    msg_lower = user_message.lower()

    # 常见事实性模式
    fact_patterns = [
        ("我喜欢", "likes"),
        ("我不喜欢", "dislikes"),
        ("我爱", "likes"),
        ("我讨厌", "dislikes"),
        ("我住", "lives_in"),
        ("我在", "works_at"),
        ("我工作", "occupation"),
        ("我是", "identity"),
        ("我叫", "name"),
        ("我过敏", "allergic_to"),
        ("我对", "allergic_to"),
        ("我会", "skill"),
        ("我不会", "limitation"),
        ("我有", "has"),
        ("我家", "family"),
        ("我的", "attribute"),
        ("我女朋友", "girlfriend"),
        ("我男朋友", "boyfriend"),
        ("我老婆", "spouse"),
        ("我老公", "spouse"),
        ("我搬", "location_change"),
        ("我换", "change"),
        ("我之前", "past_experience"),
        ("我以前", "past_experience"),
        ("我打算", "plan"),
        ("我计划", "plan"),
    ]

    for pattern, category in fact_patterns:
        if pattern in msg_lower:
            items.append({
                "content": user_message,
                "memory_type": "semantic" if category in ("lives_in", "occupation", "allergic_to", "identity", "name", "attribute", "spouse", "girlfriend", "boyfriend", "location_change", "change", "likes", "dislikes", "has", "family") else "episodic",
                "importance": "high" if category in ("allergic_to", "lives_in", "identity", "location_change") else "medium",
                "source_type": "explicit_statement",
                "reason": f"包含{category}类事实信息",
            })
            break  # 只取第一个匹配

    # 纠正类: "不对/不是/应该是/其实"
    correction_patterns = ["不对", "不是", "应该是", "其实", "纠正", "错了"]
    for pattern in correction_patterns:
        if pattern in msg_lower:
            items.append({
                "content": user_message,
                "memory_type": "semantic",
                "importance": "high",
                "source_type": "corrected",
                "reason": "用户纠正之前的信息",
            })
            break

    # 工作流程类: "流程/步骤/怎么做/规范"
    workflow_patterns = ["流程", "步骤", "怎么做", "规范", "审查", "review", "部署", "上线"]
    for pattern in workflow_patterns:
        if pattern in msg_lower:
            items.append({
                "content": user_message,
                "memory_type": "procedural",
                "importance": "medium",
                "source_type": "explicit_statement",
                "reason": "包含工作流程信息",
            })
            break

    return items


# ── 入口 ──────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MemoCortex 交互式问答 Demo")
    parser.add_argument("--interactive", action="store_true", help="交互式对话模式")
    parser.add_argument("--user-id", default="demo_user", help="用户标识")
    parser.add_argument("--auto-start", action="store_true", help="自动启动 MCP Server")
    args = parser.parse_args()

    if args.auto_start:
        # 自动启动 MCP Server
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
                    time.sleep(5)  # 等模型加载
                    break
                time.sleep(2)

    if args.interactive:
        asyncio.run(run_interactive(user_id=args.user_id))
    else:
        # 默认模式: 运行自动化验证 Demo (保留旧版行为)
        print("  使用 --interactive 启动交互式对话模式")
        print("  使用 --auto-start 自动启动 MCP Server")
        print()
        print("  示例:")
        print("    uv run python demo/chat_demo.py --interactive --user-id alice")
        print("    uv run python demo/chat_demo.py --interactive --auto-start")


if __name__ == "__main__":
    import subprocess
    main()