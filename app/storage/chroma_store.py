"""ChromaDB 内嵌实现 VectorStore Protocol

per-user 隔离通过 collection metadata filter 实现 (where 子句).
embeddings 用 langchain HuggingFaceEmbeddings.

生产替换为 Milvus: 改 collection 概念为 partition, embedding 同源即可.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from app.config import config
from app.core.embedder import async_embed_batch, async_embed_text
from app.models import MemoryRecord, MemoryType
from app.utils.metrics import metrics

# 单一 collection, per-user/per-type 通过 metadata filter 隔离
# 选择"扁平 collection + where 过滤"而非"per-user collection":
#   - 写入时不用提前创建 collection
#   - 适合 MVP, Chroma 单集合 100K 级文档无压力
#   - 生产换 Milvus 时建议 per-tenant collection / partition
_COLLECTION_NAME = "memocortex"


def _build_where(user_id: str, memory_types: list[str] | None = None) -> dict[str, Any]:
    """构造 Chroma where 过滤. Chroma 要求多 key 必须用 $and 显式组合."""
    if not memory_types:
        return {"user_id": user_id}
    if len(memory_types) == 1:
        # 多 key 必须 $and 包起来 (Chroma 0.5+ 强制)
        return {"$and": [{"user_id": user_id}, {"type": memory_types[0]}]}
    return {"$and": [{"user_id": user_id}, {"type": {"$in": memory_types}}]}


class ChromaVectorStore:
    """VectorStore 的 ChromaDB 实现 (内嵌持久化)."""

    def __init__(self) -> None:
        config.ensure_dirs()
        self._client = chromadb.PersistentClient(
            path=str(config.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        # Chroma 自带 embedding 但我们用自己的 embedder 保持外部一致
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaVectorStore 初始化 — dir={config.chroma_dir}, "
            f"count={self._collection.count()}"
        )

    # ── Write ──────────────────────────────────────────────────────────
    async def add(self, record: MemoryRecord) -> None:
        await self.add_batch([record])

    async def add_batch(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        with metrics.timer("chroma.add_batch.latency"):
            vectors = await async_embed_batch([r.content for r in records])
        self._collection.add(
            ids=[r.id for r in records],
            embeddings=vectors,  # type: ignore[arg-type]
            documents=[r.content for r in records],
            metadatas=[r.to_chroma_metadata() for r in records],
        )
        # Phase 3: 异步批量写入 FTS5 (BM25 召回通道) — 不阻塞主流程
        try:
            from app.storage.fts_store import get_fts_store
            fts = get_fts_store()
            rows = [(r.id, r.user_id, r.type.value, r.content) for r in records]
            await fts.async_add_batch(rows)
        except Exception as e:
            logger.warning(f"FTS 异步批量写入失败 (不影响主流程): {e}")
        metrics.incr("chroma.writes", len(records))

    # ── Search ─────────────────────────────────────────────────────────
    async def search(
        self,
        user_id: str,
        query: str,
        memory_types: list[str] | None = None,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        if self._collection.count() == 0:
            return []
        with metrics.timer("chroma.search.latency"):
            query_vec = await async_embed_text(query)
            where = _build_where(user_id, memory_types)
            res = self._collection.query(
                query_embeddings=[query_vec],  # type: ignore[arg-type]
                n_results=top_k,
                where=where,
            )

        if not res or not res.get("ids") or not res["ids"][0]:
            return []

        ids = res["ids"][0]
        distances = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
        documents = res["documents"][0] if res.get("documents") else [""] * len(ids)
        metadatas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(ids)

        out: list[tuple[MemoryRecord, float]] = []
        for mid, dist, doc, meta in zip(ids, distances, documents, metadatas, strict=False):
            # cosine distance ∈ [0, 2], similarity = 1 - dist/2 → [0, 1]
            similarity = max(0.0, min(1.0, 1.0 - dist / 2.0))
            if similarity < score_threshold:
                continue
            record = self._reconstruct_record(mid, doc, meta)
            out.append((record, similarity))

        metrics.incr("chroma.searches")
        return out

    # ── Update / Delete ────────────────────────────────────────────────
    async def update_metadata(
        self, memory_id: str, user_id: str, metadata_patch: dict[str, Any]
    ) -> bool:
        # Chroma 不支持 partial update, 必须先 get 再 update
        try:
            res = self._collection.get(ids=[memory_id])
            if not res["ids"]:
                return False
            existing_meta = res["metadatas"][0] if res.get("metadatas") else {}
            if existing_meta.get("user_id") != user_id:
                logger.warning(f"update_metadata: user_id 不匹配 {memory_id}")
                return False
            new_meta = {**existing_meta, **metadata_patch}
            self._collection.update(ids=[memory_id], metadatas=[new_meta])
            return True
        except Exception as e:
            logger.error(f"update_metadata 失败: {e}")
            return False

    async def update_metadata_batch(
        self,
        updates: list[tuple[str, dict[str, Any]]],
        user_id: str | None = None,
    ) -> int:
        """批量更新 metadata — 一次 get + 一次 update, N 倍快于循环调 update_metadata.

        recall 路径用此 API: 每次 recall 后批量更新 top-K 的 last_recalled_at / recall_count.
        循环调 update_metadata 会触发 N 次 ChromaDB IO (每次 ~45ms), 在 100+ 条数据规模下成为瓶颈
        (P95 延迟 477ms 大部分被这步占, 见 docs/BENCHMARK.md).

        Args:
            updates: [(memory_id, patch_dict), ...]
            user_id: 可选, 校验所有 memory 都属于该 user (防止跨 user metadata 泄漏)

        Returns:
            实际更新成功的条数 (skip 不存在或 user_id 不匹配的)
        """
        if not updates:
            return 0
        ids = [mid for mid, _ in updates]
        try:
            # 一次 get 拿全部现有 metadata
            res = self._collection.get(ids=ids)
            existing_ids = res.get("ids") or []
            existing_metas = res.get("metadatas") or []
            if not existing_ids:
                return 0

            # 构建 id → existing_meta 映射
            id_to_meta = dict(zip(existing_ids, existing_metas, strict=False))

            # 合并 patch, 跳过 user_id 不匹配
            final_ids = []
            final_metas = []
            for mid, patch in updates:
                existing = id_to_meta.get(mid)
                if existing is None:
                    continue
                if user_id is not None and existing.get("user_id") != user_id:
                    logger.warning(f"update_metadata_batch: user_id 不匹配 {mid}")
                    continue
                final_ids.append(mid)
                final_metas.append({**existing, **patch})

            if final_ids:
                self._collection.update(ids=final_ids, metadatas=final_metas)
            return len(final_ids)
        except Exception as e:
            logger.error(f"update_metadata_batch 失败 ({len(ids)} ids): {e}")
            return 0

    async def delete(self, memory_id: str, user_id: str) -> bool:
        try:
            self._collection.delete(ids=[memory_id], where={"user_id": user_id})
            # Phase 3: 异步删除 FTS
            try:
                from app.storage.fts_store import get_fts_store
                await asyncio.to_thread(get_fts_store().delete, memory_id)
            except Exception:
                pass
            metrics.incr("chroma.deletes")
            return True
        except Exception as e:
            logger.error(f"delete 失败: {e}")
            return False

    async def delete_by_user(self, user_id: str) -> int:
        count_before = self._collection.count()
        self._collection.delete(where={"user_id": user_id})
        deleted = count_before - self._collection.count()
        # Phase 3: 异步删除 FTS
        try:
            from app.storage.fts_store import get_fts_store
            await asyncio.to_thread(get_fts_store().delete_by_user, user_id)
        except Exception:
            pass
        logger.info(f"GDPR delete: user={user_id}, deleted={deleted}")
        return deleted

    async def count(self, user_id: str, memory_type: str | None = None) -> int:
        where = _build_where(user_id, [memory_type] if memory_type else None)
        try:
            # Chroma 没有直接 count(where), 这里用 get 全量再 len, 性能可接受
            res_all = self._collection.get(where=where, include=[])
            return len(res_all.get("ids", []))
        except Exception:
            return 0

    # ── Helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _reconstruct_record(
        mid: str, document: str, meta: dict[str, Any]
    ) -> MemoryRecord:
        """从 Chroma 检索结果还原 MemoryRecord."""
        import json

        structured: dict[str, Any] = {}
        if meta.get("structured_json"):
            try:
                structured = json.loads(meta["structured_json"])
            except (TypeError, ValueError):
                pass

        created_at: datetime
        if meta.get("created_at_iso"):
            try:
                created_at = datetime.fromisoformat(meta["created_at_iso"])
            except (TypeError, ValueError):
                created_at = datetime.now()
        else:
            created_at = datetime.now()

        return MemoryRecord(
            id=mid,
            user_id=str(meta.get("user_id", "")),
            session_id=str(meta.get("session_id") or "") or None,
            type=MemoryType(meta.get("type", "episodic")),
            content=document,
            structured=structured,
            importance=float(meta.get("importance", 0.5)),
            confidence_score=float(meta.get("confidence_score", 0.7)),
            source_type=str(meta.get("source_type", "explicit_statement")),
            staleness_signal=bool(int(meta.get("staleness_signal", 0))),
            superseded_by=str(meta.get("superseded_by") or "") or None,
            decay_rate=float(meta.get("decay_rate", 0.01)),
            created_at=created_at,
            recall_count=int(meta.get("recall_count", 0)),
            tier=str(meta.get("tier", "hot")),
            source=str(meta.get("source", "explicit")),
            tags=[t for t in str(meta.get("tags_csv", "")).split(",") if t],
        )
