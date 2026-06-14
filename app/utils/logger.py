"""Loguru 日志配置 — 控制台 + 按天滚动文件 + trace_id 注入 + JSON 格式可选.

P2.1 升级:
  - trace_id 自动注入: loguru patcher 从 trace_context.get_trace_id() 取值
  - JSON 格式可选 (config.log_format='json'): 生产环境直接 stdout → docker logs → ELK/Loki

格式说明:
  text 模式 (默认):
    [tid:abc123] 14:32:01 | INFO     | app.recall.router:search:62 | HybridRecall 权重: ...
                ↑ trace_id 8 位前缀, 没在 trace 上下文时显示 [tid:--------]

  json 模式:
    {"timestamp": "2026-06-14T14:32:01.123Z", "level": "INFO", "trace_id": "abc123",
     "module": "app.recall.router", "function": "search", "line": 62,
     "message": "HybridRecall 权重: ..."}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

from app.config import config
from app.utils.trace_context import get_trace_id


def _trace_patcher(record):
    """loguru patcher: 把当前 trace_id 注入到每条日志的 extra 字段."""
    record["extra"]["trace_id"] = get_trace_id() or "--------"


def _json_sink(message):
    """自定义 JSON sink — 给生产 ELK / Loki / Grafana 用.

    时间戳用 ISO 8601 with timezone, 级别大写, 含完整 caller 信息.
    """
    record = message.record
    payload = {
        "timestamp": record["time"].astimezone().isoformat(),
        "level": record["level"].name,
        "trace_id": record["extra"].get("trace_id", "--------"),
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    if record["exception"] is not None:
        payload["exception"] = str(record["exception"])
    # 业务自定义 extra (除了 trace_id) 也带上
    for k, v in record["extra"].items():
        if k == "trace_id":
            continue
        try:
            json.dumps(v)
            payload[k] = v
        except (TypeError, ValueError):
            payload[k] = str(v)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")


def setup_logger() -> None:
    """初始化 loguru — 移除默认 handler, 加控制台 + 文件双输出, 自动注入 trace_id."""
    logger.remove()
    # 注册 patcher, 每条日志生成时自动跑一次, 注入 trace_id
    logger.configure(patcher=_trace_patcher)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    if config.log_format == "json":
        # stdout 用 JSON
        logger.add(
            _json_sink,
            level=config.log_level,
        )
        # 文件也写 JSON Lines (jsonl), 方便 grep / jq
        logger.add(
            log_dir / "memocortex_{time:YYYY-MM-DD}.jsonl",
            level=config.log_level,
            rotation="00:00",
            retention="14 days",
            format=lambda record: json.dumps({
                "timestamp": record["time"].astimezone().isoformat(),
                "level": record["level"].name,
                "trace_id": record["extra"].get("trace_id", "--------"),
                "module": record["name"],
                "function": record["function"],
                "line": record["line"],
                "message": record["message"],
            }, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        # text 模式 (默认, 开发友好)
        logger.add(
            sys.stdout,
            level=config.log_level,
            format=(
                "<dim>[tid:{extra[trace_id]}]</dim> "
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}:{function}:{line}</cyan> | "
                "{message}"
            ),
            colorize=True,
        )
        logger.add(
            log_dir / "memocortex_{time:YYYY-MM-DD}.log",
            level=config.log_level,
            rotation="00:00",
            retention="14 days",
            format=(
                "[tid:{extra[trace_id]}] "
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            encoding="utf-8",
        )

    logger.info(
        f"日志已初始化, level={config.log_level}, format={config.log_format}, "
        f"debug={config.debug}"
    )
