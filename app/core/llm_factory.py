"""LLM 工厂 — OpenAI 兼容协议, 支持 DeepSeek / OpenAI / DashScope / 等价厂商

设计借鉴 SuperBizAgent: 单一 create_chat_model() 入口, 通过 .env 切厂商不改代码.

P2.3 升级: 集成 cost_meter — 每次 LLM 调用自动记 input/output tokens + 模型 + 耗时 + USD 成本.
通过 LangChain BaseCallbackHandler 拿 token_usage (兼容 with_structured_output 路径).
"""

from __future__ import annotations

import time
from typing import Any, TypeVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel

from app.config import config
from app.utils.cost_meter import record_call

_S = TypeVar("_S", bound=BaseModel)


class _UsageCallback(BaseCallbackHandler):
    """LangChain callback 拿 LLM 调用的 token usage.

    on_llm_end 在每次 LLM 调用结束时触发, response.llm_output 含 token_usage.
    """

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # response.llm_output 在 ChatOpenAI 路径下含 token_usage
        usage = (response.llm_output or {}).get("token_usage", {})
        if usage:
            self.input_tokens = int(
                usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            )
            self.output_tokens = int(
                usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
            )
        else:
            # generations 里的 message 也可能带 usage_metadata
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is not None:
                        meta = getattr(msg, "usage_metadata", None)
                        if meta:
                            self.input_tokens = int(meta.get("input_tokens", 0))
                            self.output_tokens = int(meta.get("output_tokens", 0))
                            break


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
        purpose: str = "unknown",
    ) -> _S | None:
        """统一的结构化输出入口 — 自动 fallback 兼容 thinking 模型.

        DeepSeek-v4-pro / deepseek-reasoner 等 thinking 模型不支持 tool_choice,
        必须用 json_mode; 普通模型 function_calling 体验更稳. 此 helper 按顺序尝试.

        P2.3: 自动记 token usage + 调用耗时 + USD 成本到 metrics, 不需要业务方关心.

        Args:
            prompt: ChatPromptTemplate
            schema: Pydantic 输出模型
            input_vars: prompt 的 invoke 变量
            temperature: 默认 0 (结构化输出需要可复现)
            purpose: 调用用途标签 ('arbitrator' / 'extractor' / 'pattern_miner' / ...),
                用于成本分组分析. 默认 'unknown'.

        Returns:
            schema 实例, 或全部尝试失败时 None
        """
        llm = LLMFactory.create_chat_model(temperature=temperature, streaming=False)
        usage_cb = _UsageCallback()
        callbacks = [usage_cb]
        result: _S | None = None
        error: str | None = None
        start = time.perf_counter()

        # 1. function_calling — 速度快, 但不支持 thinking 模型
        try:
            chain = prompt | llm.with_structured_output(schema, method="function_calling")
            result = await chain.ainvoke(input_vars, config={"callbacks": callbacks})
            if isinstance(result, dict):
                result = schema(**result)
        except Exception as e:
            error = str(e)
            logger.debug(f"structured_invoke function_calling 失败, 尝试 json_mode: {e}")

            # 2. json_mode fallback — thinking 模型走这条
            try:
                chain = prompt | llm.with_structured_output(schema, method="json_mode")
                result = await chain.ainvoke(input_vars, config={"callbacks": callbacks})
                if isinstance(result, dict):
                    result = schema(**result)
                error = None  # fallback 成功, 清掉错误
            except Exception as e2:
                error = str(e2)
                logger.warning(f"structured_invoke json_mode 也失败: {e2}")
                result = None

        duration_ms = (time.perf_counter() - start) * 1000

        # 计入 metrics + 写结构化日志
        record_call(
            model=config.llm_model,
            input_tokens=usage_cb.input_tokens,
            output_tokens=usage_cb.output_tokens,
            duration_ms=duration_ms,
            purpose=purpose,
            error=error,
        )
        return result


llm_factory = LLMFactory()
