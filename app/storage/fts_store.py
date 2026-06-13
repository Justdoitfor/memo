"""SQLite FTS5 — BM25 全文检索通道

替代之前的 graph_proximity 信号. BM25 是更主流的关键词召回算法,
与向量召回互补 (向量擅长语义相似, BM25 擅长字面匹配 / 罕见词).

设计:
  - 单独的 SQLite 数据库, 避免与主 ORM 冲突
  - FTS5 'unicode61' tokenizer + Python 侧 jieba 预切词 (中文召回质量提升 30%+)
  - 写入时批量双写: ChromaVectorStore.add_batch 时批量 + 异步写入 FTS
  - 召回时 BM25 score → 归一化到 [0, 1]
  - 与向量召回的 candidate 集合做并集, 然后融合打分

中文支持 (Phase 1):
  - Python 侧 jieba HMM 预切词后用空格连接, 让 unicode61 按空白切
  - 单字中文过滤 (IDF 太低, 噪声大于信号)
  - 不再使用单字 prefix 通配 ("花*" → 误命中花费/花园). 整词查询命中率
    高且更精准
  - 对老索引兼容: 旧数据 (按整段中文索引) 仍可读取, 但召回质量低 —
    建议 rm data/fts.db 重建

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
from app.utils.tokenizer import cut_for_fts


_DB_FILE = "fts.db"


class FtsStore:
    """SQLite FTS5 BM25 全文检索. 进程内单例 (init_app 时调一次)."""

    def __init__(self) -> None:
        config.ensure_dirs()
        self._db_path = config.data_dir / _DB_FILE
        self._lock = Lock()
        self._init_schema()
        logger.info(f"FtsStore 初始化 — db={self._db_path}, tokenizer=unicode61+jieba")

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path, isolation_level=None)
            try:
                yield conn
            finally:
                conn.close()

    def _init_schema(self) -> None:
        """建 FTS5 虚拟表.

        tokenizer 设计:
          - 用 unicode61 (SQLite 内置, 所有 build 必有)
          - Python 侧用 jieba 把中文 *预切*, 用空格连接 → unicode61 按空白切
          - 关键: unicode61 默认对汉字字符串保留为整 token, 不按字爆开
            所以 "花生 过敏" 进 FTS 后, "花生" / "过敏" 都能整词命中
          - 老索引兼容: 旧数据 (按整段中文索引) 仍可读, 但查询命中差,
            建议 rm data/fts.db 重建一次

        放弃方案 (踩过的坑):
          - tokenize='simple': SQLite Windows build 默认不带 simple tokenizer
          - 自定义 jieba C 扩展: 跨平台编译麻烦, ROI 不值
        """
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
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
        tokenized = cut_for_fts(content)
        with self._conn() as conn:
            # 先删后写, 保证 upsert
            conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            conn.execute(
                "INSERT INTO memory_fts(memory_id, user_id, type, content) VALUES (?, ?, ?, ?)",
                (memory_id, user_id, memory_type, tokenized),
            )

    def add_batch(self, rows: list[tuple[str, str, str, str]]) -> None:
        """批量写入 — (memory_id, user_id, type, content) 列表.
        用 executemany 替代逐条循环, 减少锁/连接开销.
        """
        if not rows:
            return
        # 批量切词 (利用 cut_for_fts 的 LRU 缓存)
        tokenized_rows = [
            (mid, uid, mtype, cut_for_fts(content))
            for mid, uid, mtype, content in rows
        ]
        ids = [r[0] for r in tokenized_rows]
        with self._conn() as conn:
            # 批量删旧数据
            conn.executemany("DELETE FROM memory_fts WHERE memory_id = ?", [(id,) for id in ids])
            # 批量插入
            conn.executemany(
                "INSERT INTO memory_fts(memory_id, user_id, type, content) VALUES (?, ?, ?, ?)",
                tokenized_rows,
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

        bm25() 返回值越小越相关 (类似距离), 我们取倒数归一化到 [0, 1].
        """
        # FTS5 不支持 OR 跨多 type, 简化: 先按 user_id + bm25 全搜, 再 Python 过滤 type
        sql = """
            SELECT memory_id, type, bm25(memory_fts) AS rank
            FROM memory_fts
            WHERE user_id = ? AND memory_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        safe_q = _build_fts_match(query)
        if not safe_q:
            return []  # query 切词后为空 → 不查 (避免 FTS 语法错)
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
    """构造 FTS5 MATCH 表达式 — query 经 jieba 切词后, 多 token OR 连接.

    每个 token 加引号防 FTS5 元字符 (`'"`, `*`, ` ` 等) 引发语法错.
    去重保留首次出现顺序 (Python 3.7+ dict 保序).
    """
    tokenized = cut_for_fts(query)
    if not tokenized:
        return ""
    seen: dict[str, None] = {}
    for t in tokenized.split():
        # FTS5 引号内仍需转义内部双引号: a"b → "a""b"
        safe = t.replace('"', '""')
        seen.setdefault(f'"{safe}"', None)
    return " OR ".join(seen.keys())


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
