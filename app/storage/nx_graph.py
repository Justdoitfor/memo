"""NetworkX 实现 KnowledgeGraph Protocol

per-user 一个 MultiDiGraph 实例, JSON 持久化到 data/graph/{user_id}.json.

优化 (P2-1): WAL + dirty 标记 + 定时刷盘
  - add_triple / delete_triple 只更新内存图, 标 dirty, append WAL (O(1))
  - flush() 将 dirty 用户的图一次性序列化到磁盘, 清空 WAL
  - 启动时 replay WAL 恢复上次崩溃未刷盘的写操作
  - persist() (shutdown) 先 flush 再保存

生产替换为 Neo4j: 改 add/find/neighbors 用 Cypher, 接口不变.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

import networkx as nx
from loguru import logger

from app.config import config
from app.models import Entity, GraphPath, Triple
from app.utils.metrics import metrics


def _safe_user_dir(user_id: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in user_id)


class NetworkXGraph:
    """KnowledgeGraph 的 NetworkX MVP 实现.

    内存图: nx.MultiDiGraph
      - 节点 = 实体名 (str)
      - 边 = (subject, object, key=triple_id, attrs={predicate, confidence, ...})

    持久化: data/graph/{user_id}.json + wal.jsonl
      - 写操作: 只更新内存图 + 标 dirty + append WAL (O(1))
      - flush: dirty 用户图序列化到磁盘, 清 WAL (定时或 shutdown 触发)
      - 启动时: replay WAL 恢复未刷盘数据
    """

    def __init__(self, root_dir: Path | None = None) -> None:
        """root_dir 可注入便于测试; 默认走 config.graph_dir."""
        if root_dir is None:
            config.ensure_dirs()
            self._root = config.graph_dir
        else:
            self._root = root_dir
            self._root.mkdir(parents=True, exist_ok=True)
        self._graphs: dict[str, nx.MultiDiGraph] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = Lock()
        self._dirty_users: set[str] = set()
        self._wal_path = self._root / "wal.jsonl"
        # 启动时 replay WAL 恢复未刷盘数据
        self._replay_wal()
        logger.info(f"NetworkXGraph 初始化 — dir={self._root}")

    # ── WAL (Write-Ahead Log) ──────────────────────────────────────────

    def _replay_wal(self) -> None:
        """启动时从 WAL 文件恢复上次崩溃未刷盘的写操作."""
        if not self._wal_path.exists():
            return
        replayed = 0
        try:
            with open(self._wal_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        user_id = entry.get("user_id", "")
                        op = entry.get("op", "")
                        if op == "add":
                            g = self._get_graph_sync(user_id)
                            triple_data = entry.get("triple", {})
                            g.add_edge(
                                triple_data.get("subject", ""),
                                triple_data.get("object", ""),
                                key=triple_data.get("id", ""),
                                predicate=triple_data.get("predicate", ""),
                                confidence=triple_data.get("confidence", 1.0),
                                source_memory_id=triple_data.get("source_memory_id", ""),
                                created_at=triple_data.get("created_at", ""),
                                valid_from=triple_data.get("valid_from", ""),
                                valid_until=triple_data.get("valid_until", ""),
                                object=triple_data.get("object", ""),
                                subject=triple_data.get("subject", ""),
                            )
                            self._dirty_users.add(user_id)
                            replayed += 1
                        elif op == "delete":
                            g = self._get_graph_sync(user_id)
                            triple_id = entry.get("triple_id", "")
                            for u, v, key in list(g.edges(keys=True)):
                                if key == triple_id:
                                    g.remove_edge(u, v, key)
                                    break
                            self._dirty_users.add(user_id)
                            replayed += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"WAL replay 跳过异常行: {e}")
        except Exception as e:
            logger.warning(f"WAL replay 失败: {e}")
        if replayed > 0:
            logger.info(f"WAL replay 完成: 恢复 {replayed} 条写操作")
            # replay 完成后立即刷盘, 避免 WAL 持续累积
            self._flush_sync()

    def _append_wal(self, user_id: str, op: str, data: dict[str, Any]) -> None:
        """append 一条 WAL 记录 — O(1) 操作, 进程崩溃时用于恢复."""
        entry = {"user_id": user_id, "op": op, **data}
        try:
            with open(self._wal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"WAL append 失败 (数据已在内存图中): {e}")

    # ── Persistence ────────────────────────────────────────────────────
    def _user_file(self, user_id: str) -> Path:
        return self._root / f"{_safe_user_dir(user_id)}.json"

    def _get_graph_sync(self, user_id: str) -> nx.MultiDiGraph:
        """同步版懒加载 — 供 WAL replay 使用."""
        if user_id in self._graphs:
            return self._graphs[user_id]
        g: nx.MultiDiGraph = nx.MultiDiGraph()
        path = self._user_file(user_id)
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                g = nx.node_link_graph(data, multigraph=True, directed=True, edges="edges")
            except Exception as e:
                logger.warning(f"加载图失败 ({path}): {e}, 用空图")
        self._graphs[user_id] = g
        return g

    def _get_graph(self, user_id: str) -> nx.MultiDiGraph:
        """懒加载 + 缓存. 线程安全."""
        if user_id in self._graphs:
            return self._graphs[user_id]
        with self._global_lock:
            if user_id in self._graphs:
                return self._graphs[user_id]
            return self._get_graph_sync(user_id)

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        with self._global_lock:
            if user_id not in self._locks:
                self._locks[user_id] = asyncio.Lock()
            return self._locks[user_id]

    # ── Flush ──────────────────────────────────────────────────────────

    def _flush_sync(self) -> int:
        """同步刷盘 — 将所有 dirty 用户的图写入磁盘, 清空 WAL."""
        flushed = 0
        for user_id in list(self._dirty_users):
            g = self._get_graph(user_id)
            path = self._user_file(user_id)
            try:
                data = nx.node_link_data(g, edges="edges")
                tmp = path.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                os.replace(tmp, path)
                flushed += 1
            except Exception as e:
                logger.error(f"flush 保存图失败 ({path}): {e}")
        if flushed > 0 or self._wal_path.exists():
            # 清空 WAL (写入空文件而非删除, 避免并发读写冲突)
            try:
                with open(self._wal_path, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception as e:
                logger.warning(f"清 WAL 失败: {e}")
        self._dirty_users.clear()
        if flushed > 0:
            logger.info(f"NetworkX flush: {flushed} 个用户图已刷盘")
        return flushed

    async def flush(self) -> int:
        """异步刷盘 — 供定时任务和 shutdown 调用."""
        return await asyncio.to_thread(self._flush_sync)

    # ── Write ──────────────────────────────────────────────────────────
    async def add_triple(self, user_id: str, triple: Triple) -> None:
        async with self._get_lock(user_id):
            g = self._get_graph(user_id)
            attrs = {
                "predicate": triple.predicate,
                "confidence": triple.confidence,
                "source_memory_id": triple.source_memory_id or "",
                "created_at": triple.created_at.isoformat(),
                "valid_from": triple.valid_from.isoformat() if triple.valid_from else "",
                "valid_until": triple.valid_until.isoformat() if triple.valid_until else "",
                "object": triple.object,
                "subject": triple.subject,
            }
            g.add_edge(triple.subject, triple.object, key=triple.id, **attrs)
            metrics.incr("graph.triples_added")
            # 只标 dirty + append WAL, 不立即刷盘
            self._dirty_users.add(user_id)
            self._append_wal(user_id, "add", {
                "triple": {
                    "id": triple.id,
                    "subject": triple.subject,
                    "predicate": triple.predicate,
                    "object": triple.object,
                    "confidence": triple.confidence,
                    "source_memory_id": triple.source_memory_id or "",
                    "created_at": triple.created_at.isoformat(),
                    "valid_from": triple.valid_from.isoformat() if triple.valid_from else "",
                    "valid_until": triple.valid_until.isoformat() if triple.valid_until else "",
                },
            })

    # ── Query ──────────────────────────────────────────────────────────
    async def find_triples(
        self,
        user_id: str,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> list[Triple]:
        g = self._get_graph(user_id)
        out: list[Triple] = []
        for u, v, key, data in g.edges(keys=True, data=True):
            if subject is not None and u != subject:
                continue
            if obj is not None and v != obj:
                continue
            if predicate is not None and data.get("predicate") != predicate:
                continue
            out.append(self._edge_to_triple(u, v, key, data))
        return out

    async def delete_triple(self, user_id: str, triple_id: str) -> bool:
        async with self._get_lock(user_id):
            g = self._get_graph(user_id)
            found_edge = None
            for u, v, key in g.edges(keys=True):
                if key == triple_id:
                    found_edge = (u, v, key)
                    break
            if not found_edge:
                return False
            g.remove_edge(*found_edge)
            # 只标 dirty + append WAL, 不立即刷盘
            self._dirty_users.add(user_id)
            self._append_wal(user_id, "delete", {"triple_id": triple_id})
            return True

    async def neighbors(
        self, user_id: str, entity: str, max_hops: int = 2
    ) -> set[str]:
        g = self._get_graph(user_id)
        if entity not in g:
            return set()
        # 无向 BFS, 距离 ≤ max_hops
        undirected = g.to_undirected(as_view=True)
        result: set[str] = set()
        try:
            lengths = nx.single_source_shortest_path_length(undirected, entity, cutoff=max_hops)
            for node, dist in lengths.items():
                if 0 < dist <= max_hops:
                    result.add(str(node))
        except Exception as e:
            logger.warning(f"neighbors BFS 失败: {e}")
        return result

    # ── P0: 增强查询 ──────────────────────────────────────────────────

    async def multi_hop_query(
        self,
        user_id: str,
        start_entity: str,
        max_hops: int = 3,
        predicate_filter: list[str] | None = None,
    ) -> list[GraphPath]:
        """多跳路径查询 — BFS 发现从 start_entity 出发的可达路径."""
        g = self._get_graph(user_id)
        if start_entity not in g:
            return []

        paths: list[GraphPath] = []
        # 用 BFS 找所有 ≤ max_hops 的简单路径
        undirected = g.to_undirected(as_view=True)
        try:
            # single_source_shortest_path 给出所有可达节点及其路径
            all_paths = nx.single_source_shortest_path(undirected, start_entity, cutoff=max_hops)
            for target, path_nodes in all_paths.items():
                if len(path_nodes) <= 1:
                    continue
                # 构造边的详细信息
                edges_info: list[dict[str, Any]] = []
                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i + 1]
                    # 在有向图中找对应的边 (可能多条)
                    if g.has_edge(u, v):
                        for key, data in g.get_edge_data(u, v).items():
                            pred = data.get("predicate", "")
                            if predicate_filter and pred not in predicate_filter:
                                continue
                            edges_info.append({
                                "predicate": pred,
                                "confidence": data.get("confidence", 1.0),
                                "direction": f"{u} → {v}",
                            })
                    elif g.has_edge(v, u):
                        for key, data in g.get_edge_data(v, u).items():
                            pred = data.get("predicate", "")
                            if predicate_filter and pred not in predicate_filter:
                                continue
                            edges_info.append({
                                "predicate": pred,
                                "confidence": data.get("confidence", 1.0),
                                "direction": f"{v} → {u}",
                            })

                paths.append(GraphPath(
                    nodes=[str(n) for n in path_nodes],
                    edges=edges_info,
                    length=len(path_nodes) - 1,
                ))
        except Exception as e:
            logger.warning(f"multi_hop_query BFS 失败: {e}")

        return paths

    async def find_related_entities(
        self,
        user_id: str,
        entity: str,
        relation_chain: list[str] | None = None,
    ) -> list[str]:
        """关系链查询 — 按指定 predicate 链发现相关实体."""
        g = self._get_graph(user_id)
        if entity not in g:
            return []

        result_entities: set[str] = set()

        # 直接邻居 (1-hop)
        for u, v, key, data in g.edges(keys=True, data=True):
            pred = data.get("predicate", "")
            if relation_chain and pred not in relation_chain:
                continue
            if u == entity:
                result_entities.add(v)
            elif v == entity:
                result_entities.add(u)

        return list(result_entities)

    async def community_detect(
        self, user_id: str, min_size: int = 3
    ) -> list[dict[str, Any]]:
        """社区检测 — 用 label propagation 算法识别强连通实体簇."""
        g = self._get_graph(user_id)
        if g.number_of_nodes() < min_size:
            return []

        try:
            undirected = g.to_undirected(as_view=True)
            # Label propagation community detection
            communities = nx.algorithms.community.label_propagation_communities(undirected)

            result: list[dict[str, Any]] = []
            for idx, community in enumerate(communities):
                entities = [str(n) for n in community]
                if len(entities) < min_size:
                    continue
                result.append({
                    "community_id": f"comm_{idx}",
                    "entities": entities,
                    "size": len(entities),
                    "summary": f"包含 {len(entities)} 个相关实体的社区",
                })
            return result
        except Exception as e:
            logger.warning(f"community_detect 失败: {e}")
            return []

    async def delete_by_user(self, user_id: str) -> int:
        async with self._get_lock(user_id):
            g = self._get_graph(user_id)
            count = g.number_of_edges()
            self._graphs.pop(user_id, None)
            self._dirty_users.discard(user_id)
            path = self._user_file(user_id)
            if path.exists():
                path.unlink()
            logger.info(f"GDPR delete graph: user={user_id}, edges={count}")
            return count

    async def persist(self) -> None:
        """全量保存所有用户图 (lifespan shutdown 调用).
        先 flush dirty + WAL, 再保存.
        """
        await self.flush()

    # ── Helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _edge_to_triple(u: str, v: str, key: str, data: dict[str, Any]) -> Triple:
        from datetime import datetime

        def _parse_ts(s: str | None) -> datetime | None:
            if not s:
                return None
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                return None

        return Triple(
            id=key,
            subject=str(u),
            predicate=str(data.get("predicate", "")),
            object=str(v),
            confidence=float(data.get("confidence", 1.0)),
            source_memory_id=str(data.get("source_memory_id") or "") or None,
            created_at=_parse_ts(data.get("created_at")) or datetime.now(),
            valid_from=_parse_ts(data.get("valid_from")),
            valid_until=_parse_ts(data.get("valid_until")),
        )