"""Reflective Memory — 元记忆 / 用户画像

特征:
  - 不是直接事实, 而是"对所有事实的总结"
  - JSON Blob 存 SQLite (per-user)
  - 由后台 Reflection Worker 周期性生成与刷新
  - Agent 召回时直接注入 SystemPrompt, 作为高度浓缩的"人物简介"
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.core.llm_factory import llm_factory
from app.memories.semantic import semantic_memory
from app.storage import get_metadata
from app.utils.metrics import metrics


class _Profile(BaseModel):
    """LLM 生成的 Reflective Profile."""

    one_liner: str = Field(description="一句话用户画像 (50 字内)")
    facts: dict[str, list[str]] = Field(
        default_factory=dict,
        description="核心事实 KV, 每个键对应值的列表 (e.g. {'居住地': ['上海'], '过敏原': ['花生', '芝麻']})",
    )
    preferences: list[str] = Field(default_factory=list, description="偏好/喜好 (最多 8 条)")
    constraints: list[str] = Field(default_factory=list, description="限制/禁忌 (e.g. 过敏原/不喜欢的事物)")
    interaction_style: str = Field(
        default="",
        description="互动风格建议 (e.g. '偏好简洁直接回答, 不要长篇大论')",
    )


_REFLECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是一个用户画像分析师. 根据用户的事实列表, 生成简洁的用户画像.\n"
                "\n"
                "原则:\n"
                "- one_liner 必须 50 字内\n"
                "- facts 只放核心事实 (居住地/职业/年龄等), 不要堆砌\n"
                "- preferences 最多 8 条, 按重要性排序\n"
                "- constraints 列出明确的禁忌 (过敏/讨厌)\n"
                "- interaction_style 根据用户表达习惯推断 (不必硬给, 不确定就留空)\n"
                "- 严禁编造未在事实列表中出现的信息\n"
                "\n"
                "返回 JSON: {{\"one_liner\": \"...\", \"facts\": {{\"key\": [\"val1\", ...]}}, "
                "\"preferences\": [...], \"constraints\": [...], \"interaction_style\": \"...\"}}"
            ),
        ),
        ("human", "用户事实:\n{facts_text}"),
    ]
)


class ReflectiveMemory:
    """元记忆 / 用户画像 — 由后台 worker 周期性刷新."""

    def __init__(self) -> None:
        self._meta = get_metadata()
        logger.info("ReflectiveMemory 初始化")

    async def get(self, user_id: str) -> dict[str, Any] | None:
        """直接读 — 给 Agent 注入 SystemPrompt 用."""
        return await self._meta.get_profile(user_id)

    async def refresh(self, user_id: str) -> dict[str, Any]:
        """重新生成用户画像 (从 Semantic Memory 聚合 + LLM 提炼)."""
        semantic_data = await semantic_memory.export_for_profile(user_id)
        facts = semantic_data.get("facts", {})

        if not facts:
            empty_profile: dict[str, Any] = {
                "one_liner": "新用户, 暂无已知事实",
                "facts": {},
                "preferences": [],
                "constraints": [],
                "interaction_style": "",
            }
            await self._meta.upsert_profile(user_id, empty_profile)
            return empty_profile

        facts_text = "\n".join(
            f"- {pred}: {', '.join(vals)}" for pred, vals in facts.items()
        )

        try:
            with metrics.timer("reflective.refresh.latency"):
                result = await llm_factory.structured_invoke(
                    _REFLECT_PROMPT, _Profile, {"facts_text": facts_text},
                    temperature=0.2, purpose="reflective_profile",
                )
            if result is None:
                raise RuntimeError("structured_invoke 返回 None")
            profile = result.model_dump()
        except Exception as e:
            logger.warning(f"Reflective refresh LLM 失败, 降级: {e}")
            profile = {
                "one_liner": f"已知 {len(facts)} 类事实",
                "facts": facts,
                "preferences": [],
                "constraints": [],
                "interaction_style": "",
            }

        await self._meta.upsert_profile(user_id, profile)
        metrics.incr("reflective.refresh.count")
        logger.info(f"Reflective profile 已刷新: user={user_id}")
        return profile


reflective_memory = ReflectiveMemory()
