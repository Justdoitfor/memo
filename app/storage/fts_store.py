"""SQLite FTS5 — BM25 全文检索通道

替代之前的 graph_proximity 信号. BM25 是更主流的关键词召回算法,
与向量召回互补 (向量擅长语义相似, BM25 擅长字面匹配 / 罕见词).

设计:
  - 单独的 SQLite 数据库, 避免与主 ORM 冲突
  - 用 FTS5 虚拟表 + jieba_tokenizer 兜底 (无 jieba 时降级用 unicode61)
  - 写入时批量双写: ChromaVectorStore.add_batch 时批量 + 异步写入 FTS
  - 召回时 BM25 score → 归一化到 [0, 1]
  - 与向量召回的 candidate 集合做并集, 然后融合打分

优化 (P1-4):
  - add_batch: 批量写入用 executemany, 减少锁/连接开销
  - async_add / async_add_batch: asyncio.to_thread 包装, 不阻塞事件循环
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from loguru import logger

from app.config import config


_DB_FILE = "fts.db"


class FtsStore:
    """SQLite FTS5 BM25 全文检索. 进程内单例 (init_app 时调一次)."""

    def __init__(self) -> None:
        config.ensure_dirs()
        self._db_path = config.data_dir / _DB_FILE
        self._lock = Lock()
        self._init_schema()
        logger.info(f"FtsStore 初始化 — db={self._db_path}")

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path, isolation_level=None)
            try:
                yield conn
            finally:
                conn.close()

    def _init_schema(self) -> None:
        """建 FTS5 虚拟表."""
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            # FTS5 虚拟表 — unicode61 tokenizer 对中英都能用 (会按空格 / 标点切)
            # 生产可换 jieba_tokenizer 提升中文召回质量
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    memory_id UNINDEXED,
                    user_id UNINDEXED,
                    type UNINDEXED,
                    content,
                    tokenize='unicode61'
                )
                """
            )

    # ── Write ──────────────────────────────────────────────────────────
    def add(self, memory_id: str, user_id: str, memory_type: str, content: str) -> None:
        with self._conn() as conn:
            # 先删后写, 保证 upsert
            conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            conn.execute(
                "INSERT INTO memory_fts(memory_id, user_id, type, content) VALUES (?, ?, ?, ?)",
                (memory_id, user_id, memory_type, content),
            )

    def add_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """批量写入 — (memory_id, user_id, type, content) 列表.
        用 executemany 替代逐条循环, 减少锁/连接开销.
        """
        if not rows:
            return
        ids = [r[0] for r in rows]
        with self._conn() as conn:
            # 批量删旧数据
            conn.executemany("DELETE FROM memory_fts WHERE memory_id = ?", [(id,) for id in ids])
            # 批量插入
            conn.executemany(
                "INSERT INTO memory_fts(memory_id, user_id, type, content) VALUES (?, ?, ?, ?)",
                rows,
            )

    async def async_add(self, memory_id: str, user_id: str, memory_type: str, content: str) -> None:
        """异步写入 — 不阻塞事件循环."""
        await asyncio.to_thread(self.add, memory_id, user_id, memory_type, content)

    async def async_add_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """异步批量写入 — 不阻塞事件循环."""
        await asyncio.to_thread(self.add_batch, rows)

    def delete(self, memory_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))

    def delete_by_user(self, user_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM memory_fts WHERE user_id = ?", (user_id,))
            return cur.rowcount

    # ── Search ─────────────────────────────────────────────────────────
    def search(
        self,
        user_id: str,
        query: str,
        memory_types: list[str] | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """FTS5 BM25 检索, 返回 [(memory_id, bm25_score), ...] 按相关度倒序.

        bm25() 返回值越小越相关 (类似距离), 我们取倒数归一化.
        """
        # FTS5 不支持 OR 跨多 type, 简化: 先按 user_id + bm25 全搜, 再 Python 过滤 type
        sql = """
            SELECT memory_id, type, bm25(memory_fts) AS rank
            FROM memory_fts
            WHERE user_id = ? AND memory_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        # FTS5 MATCH 不支持中文整词查询, 用空格分隔每个字符 OR 兜底
        # 但对短查询 (<=10 字符) 直接传, FTS unicode61 会切;
        # 加上 NEAR 让多关键词关联更紧
        safe_q = _build_fts_match(query)
        results: list[tuple[str, float]] = []
        with self._conn() as conn:
            try:
                for row in conn.execute(sql, (user_id, safe_q, top_k * 3)):
                    mid, mtype, rank = row[0], row[1], row[2]
                    if memory_types and mtype not in memory_types:
                        continue
                    # bm25() 返回负数, 越小 (越负) 越相关. 转换为 [0, 1]
                    # rank ≈ -10 ~ -0.5 (高度相关 ~ 弱相关)
                    sim = 1.0 / (1.0 + abs(rank))
                    results.append((mid, sim))
                    if len(results) >= top_k:
                        break
            except sqlite3.OperationalError as e:
                # FTS5 语法错 / 空 query → 不抛
                logger.debug(f"FTS5 search 异常 (返回空): {e}")
                return []
        return results


def _build_fts_match(query: str) -> str:
    """构造 FTS5 MATCH 表达式, 安全处理特殊字符."""
    # FTS5 元字符: " ' " 等需转义; 简化: 全部按空白切, 每个 token 加引号
    import re
    tokens = re.findall(r"[\w一-鿿]+", query)
    if not tokens:
        return ""
    # 用 OR 连接, 单字 token 加 prefix 通配符提高中文召回
    parts = []
    for t in tokens:
        if len(t) == 1:
            parts.append(f'"{t}"*')
        else:
            parts.append(f'"{t}"')
    return " OR ".join(parts)


# 单例 (懒加载, 第一次用时 init)
_fts: FtsStore | None = None
_init_lock = Lock()


def get_fts_store() -> FtsStore:
    global _fts
    if _fts is None:
        with _init_lock:
            if _fts is None:
                _fts = FtsStore()
    return _fts
