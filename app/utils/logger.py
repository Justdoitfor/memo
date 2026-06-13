"""Loguru 日志配置 — 控制台 + 按天滚动文件"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from app.config import config


def setup_logger() -> None:
    """初始化 loguru — 移除默认 handler, 加控制台 + 文件双输出."""
    logger.remove()

    logger.add(
        sys.stdout,
        level=config.log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{function}:{line}</cyan> | "
            "{message}"
        ),
        colorize=True,
    )

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "memocortex_{time:YYYY-MM-DD}.log",
        level=config.log_level,
        rotation="00:00",
        retention="14 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        encoding="utf-8",
    )

    logger.info(f"日志已初始化, level={config.log_level}, debug={config.debug}")
