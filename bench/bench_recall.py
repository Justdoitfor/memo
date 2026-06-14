"""规模化 benchmark — 测召回延迟随数据规模变化 + storage 后端对比.

设计:
  1. 数据生成: 拿 80 条评测集做种子, 复制到 N 倍 (每条 id 加 suffix)
     这样 ChromaDB 真实索引到 N×80 条向量, 查询行为代表生产场景
  2. 三个规模: 100 / 1000 / 10000 条
  3. 三个测量:
     a. recall_latency_by_scale: 一阶段召回延迟 vs 数据规模
     b. reranker_overhead: 同样规模下 reranker on/off 延迟对比
     c. storage_upsert_throughput: SQLite vs PG (PG 需要 docker) 单条 upsert 延迟
  4. 出图: bench/reports/latency_curves.png (matplotlib)

跑测试:
  PYTHONIOENCODING=utf-8 uv run python -m bench.bench_recall

  # 仅跑小规模 (~30s, smoke)
  PYTHONIOENCODING=utf-8 uv run python -m bench.bench_recall --scales 100

  # 包含 PG 对比 (需要本地起 PG: docker run -d --rm --name memocortex_pg_bench ...)
  MEMOCORTEX_BENCH_PG_URL=postgresql+asyncpg://test:test@localhost:5433/memocortex_test \\
      PYTHONIOENCODING=utf-8 uv run python -m bench.bench_recall

测量原则:
  - 每个数据规模独立 user_id (隔离, 避免上一规模残留影响)
  - 召回查询用预设 20 个 query, 跑 5 轮 = 100 次采样
  - 用 time.perf_counter (不是 time.time) 测亚毫秒精度
  - 出 P50 / P95 / mean / max, 别只报均值 (尾部延迟更重要)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from uuid import uuid4

from loguru import logger

from app.config import config
from app.models import MemoryRecord, MemoryType


REPORTS_DIR = Path(__file__).parent / "reports"
DATASET_PATH = (
    Path(__file__).parent.parent / "eval" / "datasets" / "memocortex_zh_v1.jsonl"
)


# ────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────


def _percentile(values: list[float], p: float) -> float:
    """简单百分位 (0-100) — 复用 eval.metrics.percentile 逻辑避免循环依赖."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    if p <= 0:
        return sorted_v[0]
    if p >= 100:
        return sorted_v[-1]
    import math

    k = (len(sorted_v) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f)


def _stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
    return {
        "n": len(samples),
        "mean": round(mean(samples), 3),
        "p50": round(_percentile(samples, 50), 3),
        "p95": round(_percentile(samples, 95), 3),
        "p99": round(_percentile(samples, 99), 3),
        "max": round(max(samples), 3),
        "min": round(min(samples), 3),
    }


def _load_seed_contents() -> list[str]:
    """从评测集 setup 字段读取所有 SEMANTIC + EPISODIC content 作为种子."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"评测集 {DATASET_PATH} 不存在; 先跑 eval/datasets/build_zh_v1.py"
        )
    contents = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        for item in entry.get("setup", []):
            if item.get("type") in ("semantic", "episodic"):
                contents.append(item["content"])
    # 去重
    return sorted(set(contents))


# 20 个预设查询, 覆盖不同 scenario
_QUERIES = [
    "花生过敏",
    "在哪家公司工作",
    "用什么手机",
    "对什么过敏",
    "最近的工作经历",
    "喜欢的食物",
    "用户的车",
    "家乡在哪",
    "女朋友是谁",
    "学历背景",
    "目前的住址",
    "周末做了什么",
    "上次去哪出差",
    "孩子的情况",
    "用什么编辑器",
    "宠物名字",
    "爱好是什么",
    "不喜欢的东西",
    "技术栈",
    "运动习惯",
]


# ────────────────────────────────────────────────────────────────────────
#  Setup: 把数据写入到 N 条规模
# ────────────────────────────────────────────────────────────────────────


async def setup_data(user_id: str, n_target: int, seed_contents: list[str]) -> int:
    """把 seed_contents 复制 + 加唯一 suffix 写入到 user_id, 直到达到 n_target 条.

    返回实际写入条数.
    """
    from app.storage import get_metadata, get_vector_store

    vec = get_vector_store()
    meta = get_metadata()

    written = 0
    batch_size = 50
    while written < n_target:
        remaining = n_target - written
        batch_count = min(batch_size, remaining)
        records = []
        for _ in range(batch_count):
            seed = seed_contents[written % len(seed_contents)]
            # 加 uuid 前缀确保 content 有差异 (ChromaDB 会拒绝完全一样的向量)
            content = f"{seed} (#{written})"
            record = MemoryRecord(
                user_id=user_id,
                type=MemoryType.SEMANTIC if written % 2 == 0 else MemoryType.EPISODIC,
                content=content,
                importance=0.5,
            )
            records.append(record)
            written += 1
        # batch 写
        await vec.add_batch(records)
        for r in records:
            await meta.upsert_memory(r)
        if written % 500 == 0 or written == n_target:
            logger.info(f"  setup 进度 {written}/{n_target}")
    return written


async def teardown_user(user_id: str) -> None:
    """清理 bench user 数据, 避免残留影响下一规模."""
    from app.storage import get_kg, get_metadata, get_vector_store

    try:
        await get_vector_store().delete_by_user(user_id)
    except Exception as e:
        logger.warning(f"清理 vector 失败: {e}")
    try:
        await get_metadata().delete_all_memories(user_id)
    except Exception as e:
        logger.warning(f"清理 meta 失败: {e}")
    try:
        await get_kg().delete_by_user(user_id)
    except Exception as e:
        logger.warning(f"清理 KG 失败: {e}")


# ────────────────────────────────────────────────────────────────────────
#  Bench A: recall 延迟 vs 数据规模
# ────────────────────────────────────────────────────────────────────────


async def bench_recall_latency(
    user_id: str, n_queries: int = 100,
) -> dict[str, float]:
    """跑 n_queries 次召回, 返回延迟统计."""
    from app.recall.router import recall_router

    samples = []
    for i in range(n_queries):
        query = _QUERIES[i % len(_QUERIES)]
        start = time.perf_counter()
        await recall_router.search(
            user_id=user_id,
            query=query,
            top_k=8,
            score_threshold=0.0,
        )
        samples.append((time.perf_counter() - start) * 1000)
    return _stats(samples)


# ────────────────────────────────────────────────────────────────────────
#  Bench B: SQLite vs PG 单条 upsert 延迟
# ────────────────────────────────────────────────────────────────────────


async def bench_storage_upsert(
    store, user_id: str, n_samples: int = 100,
) -> dict[str, float]:
    """跑 n_samples 次单条 upsert, 返回延迟统计."""
    from app.storage import get_metadata as _gm

    samples = []
    for i in range(n_samples):
        rec = MemoryRecord(
            user_id=user_id,
            type=MemoryType.SEMANTIC,
            content=f"bench-upsert-{i}-{uuid4().hex[:6]}",
            importance=0.5,
        )
        start = time.perf_counter()
        await store.upsert_memory(rec)
        samples.append((time.perf_counter() - start) * 1000)
    return _stats(samples)


async def bench_sqlite_vs_pg(n_samples: int = 100) -> dict:
    """对比 SQLite vs PG 单条 upsert 延迟. PG 需要 MEMOCORTEX_BENCH_PG_URL 环境变量."""
    from app.storage.sqlite_store import SQLiteMetadataStore

    out = {}

    # SQLite
    sqlite_store = SQLiteMetadataStore()
    await sqlite_store.init_schema()
    sqlite_user = f"bench_sqlite_{uuid4().hex[:6]}"
    out["sqlite"] = await bench_storage_upsert(sqlite_store, sqlite_user, n_samples)
    await sqlite_store.delete_all_memories(sqlite_user)

    # PG (可选)
    pg_url = os.getenv("MEMOCORTEX_BENCH_PG_URL", "")
    if pg_url:
        try:
            from app.storage.pg_store import PostgresMetadataStore

            pg_store = PostgresMetadataStore(url=pg_url)
            await pg_store.init_schema()
            pg_user = f"bench_pg_{uuid4().hex[:6]}"
            out["postgres"] = await bench_storage_upsert(pg_store, pg_user, n_samples)
            await pg_store.delete_all_memories(pg_user)
            await pg_store._engine.dispose()
        except Exception as e:
            logger.warning(f"PG bench 失败: {e}")
            out["postgres_error"] = str(e)
    else:
        out["postgres"] = None
        out["postgres_skip_reason"] = "MEMOCORTEX_BENCH_PG_URL 未设置"

    return out


# ────────────────────────────────────────────────────────────────────────
#  Bench C: Reranker overhead 跨规模
# ────────────────────────────────────────────────────────────────────────


async def bench_reranker_overhead(
    user_id: str, n_queries: int = 30,
) -> dict[str, dict[str, float]]:
    """同样规模下对比开/关 reranker 的延迟. 因 reranker 慢, n_queries 减到 30."""
    # 先关 reranker 跑一轮
    config.enable_reranker = False
    no_rerank = await bench_recall_latency(user_id, n_queries=n_queries)

    # 再开 reranker 跑一轮 (会触发模型加载, 第一次跑较慢)
    config.enable_reranker = True
    try:
        # warm-up: 加载模型
        from app.recall.reranker import get_reranker
        get_reranker()
        with_rerank = await bench_recall_latency(user_id, n_queries=n_queries)
    finally:
        config.enable_reranker = False  # 还原

    return {
        "no_reranker": no_rerank,
        "with_reranker": with_rerank,
        "overhead_p50_ms": round(with_rerank["p50"] - no_rerank["p50"], 2),
        "overhead_p95_ms": round(with_rerank["p95"] - no_rerank["p95"], 2),
        "slowdown_p50_x": round(with_rerank["p50"] / max(no_rerank["p50"], 1e-3), 2),
    }


# ────────────────────────────────────────────────────────────────────────
#  画图
# ────────────────────────────────────────────────────────────────────────


def plot_latency_curves(report: dict, output_path: Path) -> None:
    """画 3 张子图:
      1. Recall latency vs scale (P50 / P95)
      2. Storage upsert SQLite vs PG (柱状)
      3. Reranker on/off (P50 P95 max 三组柱状)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── 1. Recall latency vs scale ──
    scales = sorted(report["recall_latency"].keys(), key=int)
    p50s = [report["recall_latency"][s]["p50"] for s in scales]
    p95s = [report["recall_latency"][s]["p95"] for s in scales]
    means = [report["recall_latency"][s]["mean"] for s in scales]
    ax = axes[0]
    ax.plot(scales, p50s, marker="o", label="P50", linewidth=2)
    ax.plot(scales, p95s, marker="s", label="P95", linewidth=2)
    ax.plot(scales, means, marker="^", label="Mean", linewidth=2, alpha=0.7)
    ax.set_xlabel("Number of memories (per user)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Recall latency vs data scale\n(stage 1 only, 100 queries each)")
    ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # ── 2. Storage upsert SQLite vs PG ──
    ax = axes[1]
    storage_data = report.get("storage_upsert", {})
    backends = []
    p50_vals = []
    p95_vals = []
    if storage_data.get("sqlite"):
        backends.append("SQLite\n(local file)")
        p50_vals.append(storage_data["sqlite"]["p50"])
        p95_vals.append(storage_data["sqlite"]["p95"])
    if storage_data.get("postgres"):
        backends.append("PostgreSQL\n(asyncpg)")
        p50_vals.append(storage_data["postgres"]["p50"])
        p95_vals.append(storage_data["postgres"]["p95"])
    if backends:
        x = list(range(len(backends)))
        width = 0.35
        ax.bar([i - width / 2 for i in x], p50_vals, width, label="P50",
               color="#1f77b4")
        ax.bar([i + width / 2 for i in x], p95_vals, width, label="P95",
               color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(backends)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Single upsert: SQLite vs PostgreSQL\n(100 samples each)")
        ax.legend()
        for i, v in enumerate(p50_vals):
            ax.text(i - width / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        for i, v in enumerate(p95_vals):
            ax.text(i + width / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No storage data", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # ── 3. Reranker on/off ──
    ax = axes[2]
    rerank = report.get("reranker_overhead", {})
    if rerank.get("no_reranker") and rerank.get("with_reranker"):
        labels = ["P50", "P95", "Max"]
        no = [rerank["no_reranker"]["p50"], rerank["no_reranker"]["p95"], rerank["no_reranker"]["max"]]
        wi = [rerank["with_reranker"]["p50"], rerank["with_reranker"]["p95"], rerank["with_reranker"]["max"]]
        x = list(range(len(labels)))
        width = 0.35
        ax.bar([i - width / 2 for i in x], no, width, label="No reranker",
               color="#2ca02c")
        ax.bar([i + width / 2 for i in x], wi, width, label="+ Reranker (bge-v2-m3)",
               color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Latency (ms)")
        slowdown = rerank.get("slowdown_p50_x", 0)
        ax.set_title(f"Reranker overhead\n(P50 slowdown {slowdown}x)")
        ax.legend()
        ax.set_yscale("log")
        for i, v in enumerate(no):
            ax.text(i - width / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        for i, v in enumerate(wi):
            ax.text(i + width / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No reranker data", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info(f"图表已保存: {output_path}")


def render_markdown(report: dict) -> str:
    lines = [
        "# MemoCortex Benchmark Report",
        "",
        f"- Timestamp: `{report['timestamp']}`",
        f"- Platform: CPU-only (no GPU), bge-small-zh-v1.5 + bge-reranker-v2-m3",
        "",
        "## 1. Recall Latency vs Data Scale (一阶段)",
        "",
        "| Scale | n_queries | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scale in sorted(report["recall_latency"].keys(), key=int):
        s = report["recall_latency"][scale]
        lines.append(
            f"| {scale} | {s['n']} | {s['mean']:.2f} | {s['p50']:.2f} | "
            f"{s['p95']:.2f} | {s['p99']:.2f} | {s['max']:.2f} |"
        )

    lines.extend([
        "",
        "## 2. Storage Upsert: SQLite vs PostgreSQL",
        "",
        "| Backend | n | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for backend in ("sqlite", "postgres"):
        s = report.get("storage_upsert", {}).get(backend)
        if s:
            lines.append(
                f"| {backend} | {s['n']} | {s['mean']:.2f} | {s['p50']:.2f} | "
                f"{s['p95']:.2f} | {s['max']:.2f} |"
            )
        else:
            skip = report.get("storage_upsert", {}).get(f"{backend}_skip_reason", "n/a")
            lines.append(f"| {backend} | — skipped: {skip} |")

    rerank = report.get("reranker_overhead")
    if rerank and rerank.get("no_reranker") and rerank.get("with_reranker"):
        lines.extend([
            "",
            "## 3. Reranker Overhead",
            "",
            "| Config | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) |",
            "|---|---:|---:|---:|---:|",
        ])
        for label, key in [("No reranker", "no_reranker"), ("+ Reranker", "with_reranker")]:
            r = rerank[key]
            lines.append(
                f"| {label} | {r['mean']:.2f} | {r['p50']:.2f} | "
                f"{r['p95']:.2f} | {r['max']:.2f} |"
            )
        lines.append(
            f"\n**P50 slowdown**: {rerank.get('slowdown_p50_x')}x  | "
            f"**P95 overhead**: +{rerank.get('overhead_p95_ms')}ms"
        )

    lines.append("")
    lines.append("## 图表")
    lines.append("")
    lines.append("`bench/reports/latency_curves.png`")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────


async def main_async(args):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_contents = _load_seed_contents()
    logger.info(f"加载 {len(seed_contents)} 条种子内容")

    report = {
        "timestamp": datetime.now().isoformat(),
        "platform": "CPU only",
        "recall_latency": {},
        "storage_upsert": {},
        "reranker_overhead": {},
    }

    # ── Bench A: recall latency vs scale ──
    scales = args.scales
    for scale in scales:
        user_id = f"bench_scale_{scale}_{uuid4().hex[:6]}"
        logger.info(f"=== Scale {scale}: setup ===")
        t0 = time.perf_counter()
        actual = await setup_data(user_id, scale, seed_contents)
        setup_time = time.perf_counter() - t0
        logger.info(f"  setup 完成 {actual} 条, 耗时 {setup_time:.1f}s")

        logger.info(f"=== Scale {scale}: recall {args.n_queries} 次 ===")
        stats = await bench_recall_latency(user_id, n_queries=args.n_queries)
        report["recall_latency"][str(scale)] = stats
        logger.info(
            f"  scale={scale}: P50={stats['p50']:.2f}ms P95={stats['p95']:.2f}ms"
        )

        # 仅最大规模留下数据给 bench C 用 (节省总时间)
        if scale != scales[-1]:
            await teardown_user(user_id)
        else:
            largest_user_id = user_id

    # ── Bench B: SQLite vs PG upsert ──
    logger.info("=== Bench B: storage upsert ===")
    report["storage_upsert"] = await bench_sqlite_vs_pg(n_samples=args.n_storage_samples)
    s = report["storage_upsert"]
    if s.get("sqlite"):
        logger.info(f"  SQLite: P50={s['sqlite']['p50']:.2f}ms P95={s['sqlite']['p95']:.2f}ms")
    if s.get("postgres"):
        logger.info(f"  PG:     P50={s['postgres']['p50']:.2f}ms P95={s['postgres']['p95']:.2f}ms")
    else:
        logger.info(f"  PG: skipped ({s.get('postgres_skip_reason', 'unknown')})")

    # ── Bench C: Reranker overhead (在最大规模上) ──
    if args.with_reranker:
        logger.info(f"=== Bench C: reranker overhead at scale={scales[-1]} ===")
        report["reranker_overhead"] = await bench_reranker_overhead(
            largest_user_id, n_queries=args.n_reranker_queries,
        )
        r = report["reranker_overhead"]
        logger.info(
            f"  Off: P95={r['no_reranker']['p95']:.2f}ms; "
            f"On: P95={r['with_reranker']['p95']:.2f}ms; "
            f"Slowdown P50={r.get('slowdown_p50_x')}x"
        )

    # 清理最后一个 user
    await teardown_user(largest_user_id)

    # 写报告
    md = render_markdown(report)
    md_path = REPORTS_DIR / "bench_report.md"
    md_path.write_text(md, encoding="utf-8")

    json_path = REPORTS_DIR / "bench_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 画图
    try:
        plot_latency_curves(report, REPORTS_DIR / "latency_curves.png")
    except Exception as e:
        logger.warning(f"画图失败 (跳过): {e}")

    print()
    print("=== Bench 完成 ===")
    print(f"  Recall latency:")
    for scale in sorted(report["recall_latency"].keys(), key=int):
        s = report["recall_latency"][scale]
        print(f"    scale={scale:>5}: P50={s['p50']:>6.2f}ms  P95={s['p95']:>6.2f}ms")
    print()
    print(f"  报告: {md_path}")
    print(f"  图表: {REPORTS_DIR / 'latency_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description="MemoCortex 规模化 benchmark")
    parser.add_argument(
        "--scales", type=int, nargs="+", default=[100, 1000, 10000],
        help="数据规模列表 (default: 100 1000 10000)",
    )
    parser.add_argument(
        "--n-queries", type=int, default=100,
        help="每个规模跑多少次召回 (default 100)",
    )
    parser.add_argument(
        "--n-storage-samples", type=int, default=100,
        help="storage upsert 跑多少次 (default 100)",
    )
    parser.add_argument(
        "--n-reranker-queries", type=int, default=30,
        help="reranker on/off 各跑多少次 (default 30, 因 reranker 慢)",
    )
    parser.add_argument(
        "--no-reranker", dest="with_reranker", action="store_false",
        default=True,
        help="跳过 reranker bench (节省时间)",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
