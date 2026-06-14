"""Eval Runner — 加载 jsonl → setup → query → 算分.

执行流程 (per dataset entry):
  1. 解析 entry, 用 user_suffix 拼出独立 user_id (避免数据串扰)
  2. 把 setup 列表里的每条 memory 写入 storage:
     - SEMANTIC: 直接走 vector + meta + KG 三方写, 跳过 LLM 抽取
     - EPISODIC: episodic_memory.write
     - PROCEDURAL: procedural_memory.write
     - 同时按 created_days_ago 回填 created_at, 让时间衰减信号生效
     - staleness_signal=True 时手动设置, 模拟 arbitrator 软废弃后状态
  3. 调用 orchestrator.search(user_id, query) 拿 top_k 结果
  4. 把 retrieved id 列表通过 mid_to_uuid 映射回 dataset 的 mid 空间
  5. per_query_metrics 算分
  6. 跑完后 forget(all_user_data=True) 清理

返回: per-query 指标列表 + 聚合 (按 scenario / overall) + latency 分布

权重支持: 通过 weights=(...) 参数显式覆盖 4 信号融合权重, 让 grid_search 复用.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from app.models import MemoryRecord, MemoryType, Triple
from eval.metrics import mean, per_query_metrics, percentile


@dataclass
class RunnerConfig:
    dataset_path: Path
    weights: tuple[float, float, float, float] | None = None  # 覆盖 config 默认权重
    score_threshold: float | None = 0.0  # 默认 0.0 拿全部候选
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    forget_after: bool = True  # 评测完清理 user 数据
    limit: int | None = None  # 仅跑前 N 条 (debug)
    suite_name: str = "memocortex_zh_v1"


@dataclass
class QueryResult:
    entry_id: str
    scenario: str
    user_id: str
    query: str
    expected_mids: list[str]
    retrieved_mids: list[str]   # 已映射回 dataset 的 mid 空间
    retrieved_uuids: list[str]
    metrics: dict[str, float]
    latency_ms: float
    error: str | None = None


@dataclass
class RunReport:
    suite: str
    n_total: int
    n_failed: int
    overall: dict[str, float]
    by_scenario: dict[str, dict[str, float]]
    latency: dict[str, float]
    weights: tuple[float, float, float, float] | None
    timestamp: str
    results: list[QueryResult] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────
#  Setup helpers — 把 dataset entry 写入 storage
# ────────────────────────────────────────────────────────────────────────


async def _setup_entry(
    entry: dict, user_id: str,
) -> dict[str, str]:
    """把 entry["setup"] 全部写入 storage.

    返回 mid → real memory_id 的映射, 评分时把 retrieved 的 uuid 映射回 mid.
    """
    from app.storage import get_kg, get_metadata, get_vector_store

    vec = get_vector_store()
    meta = get_metadata()
    kg = get_kg()
    mid_to_uuid: dict[str, str] = {}
    now = datetime.now()

    for item in entry["setup"]:
        mid = item["mid"]
        days_ago = item.get("created_days_ago", 0)
        created_at = now - timedelta(days=days_ago)

        if item["type"] == "semantic":
            # 直接绕过 LLM 抽取, 用 dataset 提供的 structured 写
            structured = dict(item.get("structured", {}))
            record = MemoryRecord(
                id=uuid4().hex,
                user_id=user_id,
                type=MemoryType.SEMANTIC,
                content=item["content"],
                structured=structured,
                importance=item.get("importance", 0.7),
                staleness_signal=item.get("staleness_signal", False),
                created_at=created_at,
            )
            # 双写 vector + meta
            await vec.add(record)
            await meta.upsert_memory(record)
            # 写 KG triple (供 entity 加权信号生效)
            subj = structured.get("subject")
            pred = structured.get("predicate")
            obj = structured.get("object")
            if subj and pred and obj:
                triple = Triple(
                    subject=subj, predicate=pred, object=obj,
                    confidence=item.get("confidence", 0.85),
                    source_memory_id=record.id,
                    valid_from=_parse_dt(structured.get("valid_from")),
                    valid_until=_parse_dt(structured.get("valid_until")),
                )
                await kg.add_triple(user_id, triple)
            mid_to_uuid[mid] = record.id

        elif item["type"] == "episodic":
            record = MemoryRecord(
                id=uuid4().hex,
                user_id=user_id,
                type=MemoryType.EPISODIC,
                content=item["content"],
                importance=item.get("importance", 0.5),
                created_at=created_at,
            )
            await vec.add(record)
            await meta.upsert_memory(record)
            mid_to_uuid[mid] = record.id

        elif item["type"] == "procedural":
            record = MemoryRecord(
                id=uuid4().hex,
                user_id=user_id,
                type=MemoryType.PROCEDURAL,
                content=item["content"],
                structured=item.get("structured", {}),
                importance=item.get("importance", 0.6),
                created_at=created_at,
            )
            await vec.add(record)
            await meta.upsert_memory(record)
            mid_to_uuid[mid] = record.id

        else:
            logger.warning(f"未知 setup type: {item['type']}, 跳过 {mid}")

    return mid_to_uuid


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


# ────────────────────────────────────────────────────────────────────────
#  Per-entry 评测
# ────────────────────────────────────────────────────────────────────────


async def _run_entry(entry: dict, runner_cfg: RunnerConfig) -> QueryResult:
    """执行单条 dataset entry: setup → query → score → 清理."""
    from app.orchestrator.graph import orchestrator
    from app.recall.router import recall_router

    user_id = f"eval_{runner_cfg.suite_name}_{entry['user_suffix']}_{uuid4().hex[:6]}"
    err: str | None = None
    retrieved_uuids: list[str] = []
    retrieved_mids: list[str] = []
    latency_ms = 0.0
    metrics_dict: dict[str, float] = {}

    try:
        mid_to_uuid = await _setup_entry(entry, user_id)
        uuid_to_mid = {v: k for k, v in mid_to_uuid.items()}

        # Pattern 1: 走 router 直接传 weights (grid search 用); orchestrator 不开此口子.
        # 不走 working memory 注入路径 (eval 不用 session_id).
        start = time.perf_counter()
        results = await recall_router.search(
            user_id=user_id,
            query=entry["query"],
            top_k=entry.get("top_k", 8),
            weights=runner_cfg.weights,
            score_threshold=(
                entry.get("score_threshold")
                if entry.get("score_threshold") is not None
                else runner_cfg.score_threshold
            ),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_uuids = [r.record.id for r in results]
        # 仅保留我们写入的 (在 mid_to_uuid 范围内的); 跨 user 不会出现, 但兜底
        retrieved_mids = [uuid_to_mid.get(uid, "<unknown>") for uid in retrieved_uuids]

        # 算分: 在 mid 空间对比, 不需要管真实 uuid
        metrics_dict = per_query_metrics(
            retrieved_mids,
            entry["expected_mids"],
            k_values=runner_cfg.k_values,
        )
    except Exception as e:
        logger.exception(f"Eval entry {entry['id']} 失败")
        err = str(e)
    finally:
        if runner_cfg.forget_after:
            try:
                await orchestrator.forget(user_id=user_id, all_user_data=True)
            except Exception as ce:
                logger.warning(f"清理 user {user_id} 失败: {ce}")

    return QueryResult(
        entry_id=entry["id"],
        scenario=entry["scenario"],
        user_id=user_id,
        query=entry["query"],
        expected_mids=list(entry["expected_mids"]),
        retrieved_mids=retrieved_mids,
        retrieved_uuids=retrieved_uuids,
        metrics=metrics_dict,
        latency_ms=latency_ms,
        error=err,
    )


# ────────────────────────────────────────────────────────────────────────
#  Suite-level run
# ────────────────────────────────────────────────────────────────────────


def _aggregate(results: list[QueryResult], k_values: tuple[int, ...]) -> RunReport:
    """聚合 per-query 结果到 overall + by_scenario + latency."""
    metric_names = []
    for k in k_values:
        metric_names += [f"recall@{k}", f"precision@{k}", f"hit@{k}", f"ndcg@{k}"]
    metric_names += ["mrr"]

    successful = [r for r in results if r.error is None]

    overall = {
        m: round(mean(r.metrics.get(m, 0.0) for r in successful), 4)
        for m in metric_names
    }

    by_scenario: dict[str, dict[str, float]] = {}
    scenarios = sorted({r.scenario for r in successful})
    for sc in scenarios:
        rs = [r for r in successful if r.scenario == sc]
        by_scenario[sc] = {
            "n": len(rs),
            **{
                m: round(mean(r.metrics.get(m, 0.0) for r in rs), 4)
                for m in metric_names
            },
        }

    lats = [r.latency_ms for r in successful]
    latency = {
        "p50_ms": round(percentile(lats, 50), 2),
        "p95_ms": round(percentile(lats, 95), 2),
        "mean_ms": round(mean(lats), 2),
        "max_ms": round(max(lats) if lats else 0, 2),
    }

    return RunReport(
        suite="",  # 由调用方填
        n_total=len(results),
        n_failed=len(results) - len(successful),
        overall=overall,
        by_scenario=by_scenario,
        latency=latency,
        weights=None,
        timestamp=datetime.now().isoformat(),
        results=results,
    )


async def run_suite(cfg: RunnerConfig) -> RunReport:
    """运行整个评测套件. 串行执行避免 storage 单例并发踩踏."""
    entries = _load_dataset(cfg.dataset_path)
    if cfg.limit:
        entries = entries[: cfg.limit]
    logger.info(f"开始评测 {len(entries)} 条 entries (weights={cfg.weights})")

    # 确保 SQLite schema 已建 (运行入口幂等)
    from app.storage import get_metadata
    await get_metadata().init_schema()

    results: list[QueryResult] = []
    for i, entry in enumerate(entries, 1):
        r = await _run_entry(entry, cfg)
        results.append(r)
        if i % 10 == 0 or i == len(entries):
            logger.info(f"  进度 {i}/{len(entries)}")

    report = _aggregate(results, cfg.k_values)
    report.suite = cfg.suite_name
    report.weights = cfg.weights

    # 写入 SQLite eval_runs 表 (跨版本回归对比的基础)
    try:
        meta = get_metadata()
        await meta.save_eval_run(
            suite=cfg.suite_name,
            score=report.overall.get("ndcg@10", 0.0),
            details={
                "overall": report.overall,
                "by_scenario": report.by_scenario,
                "latency": report.latency,
                "weights": list(report.weights) if report.weights else None,
                "n_total": report.n_total,
                "n_failed": report.n_failed,
                "timestamp": report.timestamp,
            },
        )
    except Exception as e:
        logger.warning(f"写入 eval_runs 失败: {e}")

    return report


def _load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset 不存在: {path}; 先跑 eval/datasets/build_zh_v1.py")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ────────────────────────────────────────────────────────────────────────
#  Markdown report renderer
# ────────────────────────────────────────────────────────────────────────


def render_markdown(report: RunReport) -> str:
    lines = []
    lines.append(f"# Eval Report — {report.suite}")
    lines.append("")
    lines.append(f"- Timestamp: `{report.timestamp}`")
    if report.weights:
        lines.append(f"- Weights (vec, temp, kw, imp): `{report.weights}`")
    else:
        lines.append("- Weights: config 默认")
    lines.append(f"- Total entries: {report.n_total}")
    lines.append(f"- Failed: {report.n_failed}")
    lines.append("")

    # Overall
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k in ("recall@1", "recall@3", "recall@5", "recall@10",
              "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
              "hit@1", "hit@3", "hit@5",
              "mrr"):
        if k in report.overall:
            lines.append(f"| {k} | {report.overall[k]:.4f} |")
    lines.append("")

    # Latency
    lines.append("## Latency")
    lines.append("")
    lines.append("| Stat | ms |")
    lines.append("|---|---:|")
    for k, v in report.latency.items():
        lines.append(f"| {k} | {v:.2f} |")
    lines.append("")

    # By scenario
    lines.append("## By Scenario")
    lines.append("")
    cols = ["recall@5", "ndcg@5", "ndcg@10", "mrr", "hit@1"]
    lines.append("| Scenario | n | " + " | ".join(cols) + " |")
    lines.append("|---|---:|" + "|".join("---:" for _ in cols) + "|")
    for sc, data in sorted(report.by_scenario.items()):
        row = [sc, str(data["n"])]
        for c in cols:
            row.append(f"{data.get(c, 0.0):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines)
