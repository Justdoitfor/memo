"""LLM 工厂 — OpenAI 兼容协议, 支持 DeepSeek / OpenAI / DashScope / 等价厂商

设计借鉴 SuperBizAgent: 单一 create_chat_model() 入口, 通过 .env 切厂商不改代码.
"""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel

from app.config import config

_S = TypeVar("_S", bound=BaseModel)


class LLMFactory:
    """LLM 工厂 — 通过 OpenAI 兼容模式接入任意厂商.

    切换厂商只改 .env:
      DeepSeek:  MEMOCORTEX_LLM_API_BASE=https://api.deepseek.com/v1
      OpenAI:    MEMOCORTEX_LLM_API_BASE=https://api.openai.com/v1
      DashScope: MEMOCORTEX_LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
    """

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.0,
        streaming: bool = False,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> ChatOpenAI:
        """创建 ChatOpenAI 实例.

        Args:
            model: 模型名, None 用 config.llm_model
            temperature: 默认 0 (仲裁/抽取需要可复现)
            streaming: 是否启用流式
            timeout: 单次调用超时秒数
            max_retries: 失败重试次数 (LangChain 内置指数退避)
        """
        resolved_model = model or config.llm_model
        if not config.llm_api_key:
            logger.warning(
                "MEMOCORTEX_LLM_API_KEY 未配置, LLM 调用会失败. "
                "Eval / Arbitrator 等需要 LLM 的功能将不可用."
            )

        return ChatOpenAI(
            model=resolved_model,
            temperature=temperature,
            streaming=streaming,
            base_url=config.llm_api_base,
            api_key=config.llm_api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    @staticmethod
    async def structured_invoke(
        prompt: ChatPromptTemplate,
        schema: type[_S],
        input_vars: dict[str, Any],
        temperature: float = 0.0,
    ) -> _S | None:
        """统一的结构化输出入口 — 自动 fallback 兼容 thinking 模型.

        DeepSeek-v4-pro / deepseek-reasoner 等 thinking 模型不支持 tool_choice,
        必须用 json_mode; 普通模型 function_calling 体验更稳. 此 helper 按顺序尝试.

        Args:
            prompt: ChatPromptTemplate
            schema: Pydantic 输出模型
            input_vars: prompt 的 invoke 变量
            temperature: 默认 0 (结构化输出需要可复现)

        Returns:
            schema 实例, 或全部尝试失败时 None
        """
        llm = LLMFactory.create_chat_model(temperature=temperature, streaming=False)

        # 1. function_calling — 速度快, 但不支持 thinking 模型
        try:
            chain = prompt | llm.with_structured_output(schema, method="function_calling")
            result = await chain.ainvoke(input_vars)
            if isinstance(result, dict):
                return schema(**result)
            return result
        except Exception as e:
            logger.debug(f"structured_invoke function_calling 失败, 尝试 json_mode: {e}")

        # 2. json_mode fallback — thinking 模型走这条
        try:
            chain = prompt | llm.with_structured_output(schema, method="json_mode")
            result = await chain.ainvoke(input_vars)
            if isinstance(result, dict):
                return schema(**result)
            return result
        except Exception as e:
            logger.warning(f"structured_invoke json_mode 也失败: {e}")
            return None


llm_factory = LLMFactory()
