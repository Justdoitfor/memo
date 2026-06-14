"""Prompt 版本化加载工具.

用法:
    from app.core.prompt_loader import load_prompt
    prompt = load_prompt("arbitrator", version="v1")

设计原则:
  - prompts/<name>/<version>.yaml 是唯一真源, 代码不再持有 prompt 字符串
  - YAML 结构: {system: "...", human: "...", meta: {description, created_at, ...}}
  - 加载结果 cache, 同一 (name, version) 不重复读盘
  - 双花括号 escape: YAML 里写 `{{...}}` 表示 ChatPromptTemplate 的字面量花括号

为什么要版本化:
  - A/B 测 v2 prompt 不用改代码 (env 切版本即可)
  - 可追溯: arbitration_logs 里记录用的哪个 prompt 版本, 复现历史决策
  - 协作: 改 prompt 不需要懂 Python, YAML 改完测试即可

未来扩展点:
  - 加 prompts/<name>/CHANGELOG.md 记录每个版本的设计动机
  - load_prompt 支持 fallback: v3 找不到 → v2 → v1
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

# 定位项目根 prompts/ 目录: app/core/prompt_loader.py 的 parent.parent.parent
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@lru_cache(maxsize=64)
def load_prompt(name: str, version: str = "v1") -> ChatPromptTemplate:
    """加载指定 prompt 模板.

    Args:
        name: prompt 名 (e.g. "arbitrator")
        version: 版本号 (e.g. "v1", "v2"); 文件路径 prompts/<name>/<version>.yaml

    Returns:
        ChatPromptTemplate, 已 from_messages 构造完毕

    Raises:
        FileNotFoundError: 该 (name, version) 不存在
        ValueError: YAML 缺 system 或 human 字段
    """
    yaml_path = _PROMPTS_DIR / name / f"{version}.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"Prompt 不存在: {yaml_path} "
            f"(请在 prompts/{name}/{version}.yaml 创建)"
        )
    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"{yaml_path} 内容必须是 dict, 实际 {type(spec)}")
    if "system" not in spec or "human" not in spec:
        raise ValueError(
            f"{yaml_path} 缺 system 或 human 字段 (有: {list(spec.keys())})"
        )
    return ChatPromptTemplate.from_messages(
        [
            ("system", spec["system"]),
            ("human", spec["human"]),
        ]
    )


def get_prompt_meta(name: str, version: str = "v1") -> dict:
    """读取 prompt 的 meta 信息 (description / created_at / 版本说明).

    用于审计日志记录"这次仲裁用的是哪个版本的 prompt".
    """
    yaml_path = _PROMPTS_DIR / name / f"{version}.yaml"
    if not yaml_path.is_file():
        return {}
    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return spec.get("meta", {})
