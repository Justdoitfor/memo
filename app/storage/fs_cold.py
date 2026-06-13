"""本地文件系统实现 ColdStorage Protocol — MVP 用, 生产换 S3."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import aiofiles
from loguru import logger

from app.config import config

_URI_PREFIX = "fs://cold/"


class FileSystemColdStorage:
    """ColdStorage 的本地目录实现.

    storage_uri 格式: fs://cold/{relpath}
      e.g. fs://cold/2026/06/abc123.json
    """

    def __init__(self) -> None:
        config.ensure_dirs()
        self._root = config.cold_dir
        logger.info(f"FileSystemColdStorage 初始化 — root={self._root}")

    def _resolve(self, storage_uri: str) -> Path:
        if not storage_uri.startswith(_URI_PREFIX):
            raise ValueError(f"非法 storage_uri: {storage_uri}")
        rel = storage_uri[len(_URI_PREFIX):]
        return (self._root / rel).resolve()

    async def archive(self, key: str, content: str | bytes) -> str:
        # key 仅作 hint, 内部生成唯一文件名避免冲突
        safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)[:40]
        rel = f"{safe_key}_{uuid4().hex[:10]}.bin"
        path = self._root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        async with aiofiles.open(path, mode=mode, encoding=None if mode == "wb" else "utf-8") as f:
            await f.write(content)
        uri = _URI_PREFIX + rel
        logger.debug(f"cold archive: {uri}")
        return uri

    async def restore(self, storage_uri: str) -> bytes:
        path = self._resolve(storage_uri)
        if not path.exists():
            raise FileNotFoundError(storage_uri)
        async with aiofiles.open(path, mode="rb") as f:
            return await f.read()

    async def delete(self, storage_uri: str) -> bool:
        try:
            path = self._resolve(storage_uri)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"cold delete 失败 {storage_uri}: {e}")
            return False
