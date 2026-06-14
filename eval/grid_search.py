"""Grid Search — 4 信号融合权重的消融 + 粗筛 + 精扫.

执行 3 阶段, 跑完出 Pareto 报告 + matplotlib 散点图.

阶段 A — 单轴消融 (Single-axis ablation, 20 组):
  固定其他三维 = baseline (0.2), 单独 sweep 一维 ∈ {0.0, 0.2, 0.4, 0.6, 0.8}.
  目的: 看每个信号的边际贡献, 给"权重定多少"提供 ground-truth 依据.

阶段 B — 粗筛 grid (Coarse grid, 约 80 组):
  stride=0.2, 4 维 ∈ {0.0, 0.2, 0.4, 0.6, 0.8}, 共 5⁴ = 625 组.
  剪枝: 保留权重和 ∈ [0.6, 1.4] 且非全零的组合, ≈ 80 组.
  目的: 粗略找到 nDCG@10 排前 10 的组合, 缩小搜索空间.

阶段 C — 精扫 (Fine grid, 约 50 组):
  在阶段 B Top-10 周围 ±0.05 stride 精扫, 找最优.

输出:
  eval/reports/ablation_single_axis.md
  eval/reports/grid_search_results.md           # 完整 Pareto 表
  eval/reports/grid_search_pareto.png            # nDCG@10 vs P95 latency 散点
  eval/reports/grid_search_artifacts.json        # 原始 (weights, metrics) 列表

用法:
  PYTHONIOENCODING=utf-8 uv run python -m eval.grid_search
  PYTHONIOENCODING=utf-8 uv run python -m eval.grid_search --skip-fine  # 跳精扫
  PYTHONIOENCODING=utf-8 uv run python -m eval.grid_search --quick       # 仅消融
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from eval.run_recall import DEFAULT_DATASET, DEFAULT_REPORTS_DIR
from eval.runner import RunReport, RunnerConfig, run_suite


# ────────────────────────────────────────────────────────────────────────
#  配置
# ────────────────────────────────────────────────────────────────────────


# Baseline: README 当前的 4 信号权重
BASELINE: tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20)

# 单轴消融
ABLATION_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8)

# 粗筛 grid — 默认范围 (4^4 = 256, 剪枝后 ~80)
COARSE_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8)
# Targeted grid — 基于消融结论裁剪后的搜索空间, 跑约 6-8 min
# 设计依据 (见 docs/RECALL_WEIGHT_ABLATION.md):
#   - vec 主信号, 最优在 0.6 附近 → 扫 0.4-0.7 (4 个值)
#   - temp 单调下降, 关掉甚至更好 → 扫 0.0-0.2 (3 个值)
#   - kw 中性 → 扫 0.0/0.2 即可 (2 个值)
#   - imp 在评测集场景下负贡献 → 扫 0.0/0.2 即可 (2 个值)
# 4 × 3 × 2 × 2 = 48 组前剪枝
TARGETED_VEC_VALUES = (0.4, 0.5, 0.6, 0.7)
TARGETED_TEMP_VALUES = (0.0, 0.1, 0.2)
TARGETED_KW_VALUES = (0.0, 0.2)
TARGETED_IMP_VALUES = (0.0, 0.2)

COARSE_SUM_RANGE = (0.4, 1.4)

# 精扫 stride
FINE_STRIDE = 0.05

# 主排序指标
PRIMARY_METRIC = "ndcg@10"


# ────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────


def _w(vec: float, temp: float, kw: float, imp: float) -> tuple[float, float, float, float]:
    return (round(vec, 4), round(temp, 4), round(kw, 4), round(imp, 4))


def _row_from_report(report: RunReport, weights: tuple) -> dict[str, Any]:
    """把 RunReport 摊平成一行可写表格的 dict."""
    o = report.overall
    return {
        "weights": list(weights),
        "ndcg@10": o.get("ndcg@10", 0.0),
        "ndcg@5": o.get("ndcg@5", 0.0),
        "recall@5": o.get("recall@5", 0.0),
        "mrr": o.get("mrr", 0.0),
        "hit@1": o.get("hit@1", 0.0),
        "p50_ms": report.latency.get("p50_ms", 0.0),
        "p95_ms": report.latency.get("p95_ms", 0.0),
        "by_scenario": {
            sc: data.get(PRIMARY_METRIC, 0.0)
            for sc, data in report.by_scenario.items()
        },
    }


async def _eval_one(
    weights: tuple[float, float, float, float], cfg_template: dict,
) -> dict[str, Any]:
    """跑一组权重, 返回结果行."""
    cfg = RunnerConfig(
        dataset_path=cfg_template["dataset_path"],
        weights=weights,
        score_threshold=cfg_template["score_threshold"],
        suite_name=cfg_template["suite_name"],
        forget_after=True,
        limit=cfg_template.get("limit"),
    )
    start = time.perf_counter()
    report = await run_suite(cfg)
    wall = time.perf_counter() - start
    row = _row_from_report(report, weights)
    row["wall_sec"] = round(wall, 2)
    return row


# ────────────────────────────────────────────────────────────────────────
#  阶段 A: 单轴消融
# ────────────────────────────────────────────────────────────────────────


async def run_ablation(cfg_template: dict) -> list[dict[str, Any]]:
    """4 维 × 5 个值 = 20 组. 固定其他三维 = baseline 0.2."""
    rows = []
    axis_names = ("vec", "temp", "kw", "imp")
    for axis_idx, axis_name in enumerate(axis_names):
        for v in ABLATION_VALUES:
            weights = list(BASELINE)
            weights[axis_idx] = v
            row = await _eval_one(tuple(weights), cfg_template)
            row["axis"] = axis_name
            row["axis_value"] = v
            rows.append(row)
            logger.info(
                f"[Ablation] axis={axis_name}={v} → "
                f"ndcg@10={row['ndcg@10']:.4f} mrr={row['mrr']:.4f}"
            )
    return rows


# ────────────────────────────────────────────────────────────────────────
#  阶段 B: 粗筛
# ────────────────────────────────────────────────────────────────────────


def _enumerate_coarse(targeted: bool = True) -> list[tuple[float, float, float, float]]:
    """4 维笛卡尔积, 剪枝: 总和 ∈ [0.4, 1.4] 且非全零.

    targeted=True 时使用基于消融结论裁剪的搜索空间 (~80 组), 否则用完整 5^4 范围.
    """
    if targeted:
        product = itertools.product(
            TARGETED_VEC_VALUES, TARGETED_TEMP_VALUES,
            TARGETED_KW_VALUES, TARGETED_IMP_VALUES,
        )
    else:
        product = itertools.product(COARSE_VALUES, repeat=4)

    out = []
    for w in product:
        s = sum(w)
        if s == 0:
            continue
        if not (COARSE_SUM_RANGE[0] <= s <= COARSE_SUM_RANGE[1]):
            continue
        out.append(tuple(round(x, 4) for x in w))
    return out


async def run_coarse(cfg_template: dict, targeted: bool = True) -> list[dict[str, Any]]:
    weights_list = _enumerate_coarse(targeted=targeted)
    logger.info(f"[Coarse] {'targeted' if targeted else 'full'} 共 {len(weights_list)} 组待跑")
    rows = []
    for i, w in enumerate(weights_list, 1):
        row = await _eval_one(w, cfg_template)
        rows.append(row)
        if i % 10 == 0 or i == len(weights_list):
            top = max(rows, key=lambda r: r[PRIMARY_METRIC])
            logger.info(
                f"[Coarse] {i}/{len(weights_list)} "
                f"current best={top[PRIMARY_METRIC]:.4f} @ {top['weights']}"
            )
    return rows


# ────────────────────────────────────────────────────────────────────────
#  阶段 C: 精扫
# ────────────────────────────────────────────────────────────────────────


def _fine_neighborhood(
    center: tuple[float, ...], stride: float = FINE_STRIDE,
) -> list[tuple[float, ...]]:
    """生成 center 周围 ±stride 的所有 81 组合 (3×3×3×3 - center)."""
    deltas = (-stride, 0, stride)
    out = []
    for d in itertools.product(deltas, repeat=4):
        w = tuple(round(max(0.0, min(1.0, c + dd)), 4) for c, dd in zip(center, d))
        if sum(w) == 0:
            continue
        out.append(w)
    return list(set(out))  # 去重


async def run_fine(
    cfg_template: dict, top_centers: list[tuple], top_n: int = 10,
) -> list[dict[str, Any]]:
    """围绕 top-N 中心点做 ±0.05 精扫. 去重后约 50-150 组."""
    candidates: set[tuple[float, ...]] = set()
    for c in top_centers[:top_n]:
        for w in _fine_neighborhood(c):
            candidates.add(w)
    weights_list = sorted(candidates)
    logger.info(f"[Fine] 围绕 top-{top_n} 共 {len(weights_list)} 组待跑")

    rows = []
    for i, w in enumerate(weights_list, 1):
        row = await _eval_one(w, cfg_template)
        rows.append(row)
        if i % 20 == 0 or i == len(weights_list):
            top = max(rows, key=lambda r: r[PRIMARY_METRIC])
            logger.info(
                f"[Fine] {i}/{len(weights_list)} best={top[PRIMARY_METRIC]:.4f} @ {top['weights']}"
            )
    return rows


# ────────────────────────────────────────────────────────────────────────
#  报告渲染
# ────────────────────────────────────────────────────────────────────────


def render_ablation_md(ablation_rows: list[dict[str, Any]], baseline_row: dict) -> str:
    lines = []
    lines.append("# 4 信号权重 — 单轴消融报告")
    lines.append("")
    lines.append(f"- Dataset: `memocortex_zh_v1.jsonl` (80 条)")
    lines.append(f"- Baseline: `{tuple(baseline_row['weights'])}` → "
                 f"nDCG@10={baseline_row['ndcg@10']:.4f}, "
                 f"MRR={baseline_row['mrr']:.4f}")
    lines.append("")
    lines.append("## 实验设计")
    lines.append("")
    lines.append(
        "对每个信号 (vec / temp / kw / imp), 固定其他三维 = baseline (0.2), "
        "单独 sweep 该维 ∈ {0.0, 0.2, 0.4, 0.6, 0.8}, 测每个信号的边际贡献."
    )
    lines.append("")

    axes = sorted({r["axis"] for r in ablation_rows})
    for axis in axes:
        rows = [r for r in ablation_rows if r["axis"] == axis]
        rows.sort(key=lambda r: r["axis_value"])
        lines.append(f"### Axis: `{axis}`")
        lines.append("")
        lines.append("| 权重值 | weights | nDCG@10 | nDCG@5 | MRR | hit@1 | P95 ms |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r['axis_value']} | {tuple(r['weights'])} | "
                f"{r['ndcg@10']:.4f} | {r['ndcg@5']:.4f} | {r['mrr']:.4f} | "
                f"{r['hit@1']:.4f} | {r['p95_ms']:.1f} |"
            )
        lines.append("")
        # 最优值
        best = max(rows, key=lambda r: r["ndcg@10"])
        lines.append(f"**该轴最优**: `{axis}={best['axis_value']}` → nDCG@10={best['ndcg@10']:.4f}")
        lines.append("")

    return "\n".join(lines)


def render_grid_md(
    coarse_rows: list[dict[str, Any]],
    fine_rows: list[dict[str, Any]],
    baseline_row: dict,
    top_n: int = 15,
) -> str:
    lines = []
    lines.append("# 4 信号权重 Grid Search 报告")
    lines.append("")
    lines.append(f"- Dataset: `memocortex_zh_v1.jsonl` (80 条)")
    lines.append(f"- 主排序指标: `nDCG@10`")
    lines.append(f"- 粗筛: stride 0.2 + 总和 ∈ [0.6, 1.4] = {len(coarse_rows)} 组")
    lines.append(f"- 精扫: 粗筛 top-10 邻域 ±0.05 = {len(fine_rows)} 组")
    lines.append("")
    lines.append(f"**Baseline**: `{tuple(baseline_row['weights'])}` → "
                 f"nDCG@10={baseline_row['ndcg@10']:.4f}, MRR={baseline_row['mrr']:.4f}")
    lines.append("")

    # 合并去重 (按权重 tuple)
    all_rows = {tuple(r["weights"]): r for r in coarse_rows}
    for r in fine_rows:
        all_rows[tuple(r["weights"])] = r
    sorted_rows = sorted(all_rows.values(), key=lambda r: r["ndcg@10"], reverse=True)

    # Top N 表
    lines.append(f"## Top {top_n} 组合 (按 nDCG@10 排序)")
    lines.append("")
    lines.append(
        "| Rank | weights (vec, temp, kw, imp) | nDCG@10 | nDCG@5 | MRR | hit@1 | conflict_latest | episodic_temporal | P95 ms |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(sorted_rows[:top_n], 1):
        sc = r.get("by_scenario", {})
        is_baseline = tuple(r["weights"]) == tuple(baseline_row["weights"])
        marker = " ⭐ baseline" if is_baseline else ""
        lines.append(
            f"| {i}{marker} | {tuple(r['weights'])} | "
            f"{r['ndcg@10']:.4f} | {r['ndcg@5']:.4f} | "
            f"{r['mrr']:.4f} | {r['hit@1']:.4f} | "
            f"{sc.get('conflict_latest', 0):.4f} | "
            f"{sc.get('episodic_temporal', 0):.4f} | "
            f"{r['p95_ms']:.1f} |"
        )
    lines.append("")

    # Baseline 排名
    base_rank = next(
        (i for i, r in enumerate(sorted_rows, 1)
         if tuple(r["weights"]) == tuple(baseline_row["weights"])),
        None,
    )
    if base_rank is not None:
        lines.append(f"**Baseline `{tuple(baseline_row['weights'])}` 排名: {base_rank} / {len(sorted_rows)}**")
        lines.append("")

    # 最优 vs baseline 的提升
    best = sorted_rows[0]
    delta_ndcg = best["ndcg@10"] - baseline_row["ndcg@10"]
    delta_mrr = best["mrr"] - baseline_row["mrr"]
    lines.append("## 最优组合 vs Baseline")
    lines.append("")
    lines.append(f"- 最优权重: `{tuple(best['weights'])}`")
    lines.append(f"- nDCG@10 Δ = {delta_ndcg:+.4f} ({delta_ndcg/baseline_row['ndcg@10']*100:+.2f}%)")
    lines.append(f"- MRR Δ = {delta_mrr:+.4f}")
    lines.append("")

    # 决策记录
    lines.append("## 决策记录")
    lines.append("")
    lines.append(
        "选定权重时除了看 `nDCG@10`, 还需考虑:\n"
        "1. **鲁棒性**: 在 P95 latency 上是否有显著回归\n"
        "2. **场景均衡**: 在 8 个场景上是否都不掉, 不能因为优化 conflict_latest 而牺牲 paraphrase\n"
        "3. **可解释性**: 选择信号语义对得上的权重 (e.g. vector 占比应是大头)\n"
    )
    lines.append("")
    lines.append("Pareto 散点图: `eval/reports/grid_search_pareto.png`")
    lines.append("")

    return "\n".join(lines)


def plot_pareto(
    coarse_rows: list[dict[str, Any]],
    fine_rows: list[dict[str, Any]],
    baseline_row: dict,
    output_path: Path,
) -> None:
    """画 nDCG@10 vs P95 latency 散点图.

    点的颜色: coarse=灰, fine=蓝. baseline 高亮红色 X.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # 粗筛
    cx = [r["p95_ms"] for r in coarse_rows]
    cy = [r["ndcg@10"] for r in coarse_rows]
    ax.scatter(cx, cy, c="#cccccc", s=22, alpha=0.6, label=f"Coarse grid (n={len(coarse_rows)})")

    # 精扫
    fx = [r["p95_ms"] for r in fine_rows]
    fy = [r["ndcg@10"] for r in fine_rows]
    ax.scatter(fx, fy, c="#1f77b4", s=30, alpha=0.7, label=f"Fine grid (n={len(fine_rows)})")

    # Baseline
    ax.scatter(
        [baseline_row["p95_ms"]], [baseline_row["ndcg@10"]],
        c="#d62728", s=180, marker="X",
        label=f"Baseline {tuple(baseline_row['weights'])}",
        edgecolors="black", linewidths=1.2, zorder=10,
    )

    # 最优点
    all_rows = list(coarse_rows) + list(fine_rows)
    best = max(all_rows, key=lambda r: r["ndcg@10"])
    ax.scatter(
        [best["p95_ms"]], [best["ndcg@10"]],
        c="#2ca02c", s=220, marker="*",
        label=f"Best {tuple(best['weights'])} → {best['ndcg@10']:.4f}",
        edgecolors="black", linewidths=1.2, zorder=11,
    )

    ax.set_xlabel("P95 Latency (ms)", fontsize=12)
    ax.set_ylabel("nDCG@10", fontsize=12)
    ax.set_title("4-Signal Weights Grid Search — nDCG@10 vs P95 Latency Pareto",
                 fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info(f"散点图已保存: {output_path}")


# ────────────────────────────────────────────────────────────────────────
#  主入口
# ────────────────────────────────────────────────────────────────────────


async def main_async(args):
    cfg_template = {
        "dataset_path": args.dataset,
        "score_threshold": 0.0,
        "suite_name": "memocortex_zh_v1_grid",
        "limit": args.limit,
    }

    # Baseline 跑一次, 后续比较的基准
    logger.info(f"=== Phase 0: Baseline {BASELINE} ===")
    baseline_row = await _eval_one(BASELINE, cfg_template)
    baseline_row["weights"] = list(BASELINE)
    logger.info(
        f"Baseline: ndcg@10={baseline_row['ndcg@10']:.4f}, "
        f"mrr={baseline_row['mrr']:.4f}, p95={baseline_row['p95_ms']:.1f}ms"
    )

    # Phase A: 单轴消融
    ablation_rows: list[dict] = []
    if args.skip_ablation:
        logger.info("[Ablation] --skip-ablation 跳过 Phase A; 尝试从已有 artifact 读取")
        prev_artifact = DEFAULT_REPORTS_DIR / "grid_search_artifacts.json"
        if prev_artifact.exists():
            try:
                ablation_rows = json.loads(
                    prev_artifact.read_text(encoding="utf-8")
                ).get("ablation", [])
                logger.info(f"[Ablation] 从 {prev_artifact.name} 加载 {len(ablation_rows)} 行")
            except Exception as e:
                logger.warning(f"加载 artifact 失败: {e}")
    else:
        logger.info("=== Phase A: Single-axis ablation (20 组) ===")
        t0 = time.perf_counter()
        ablation_rows = await run_ablation(cfg_template)
        logger.info(f"[Ablation] 完成, 耗时 {time.perf_counter() - t0:.1f}s")

    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if ablation_rows:
        abl_md = render_ablation_md(ablation_rows, baseline_row)
        (DEFAULT_REPORTS_DIR / "ablation_single_axis.md").write_text(abl_md, encoding="utf-8")

    if args.quick:
        logger.info("--quick 模式: 仅消融, 跳过 grid search")
        # 写 artifact
        (DEFAULT_REPORTS_DIR / "grid_search_artifacts.json").write_text(
            json.dumps({
                "baseline": baseline_row,
                "ablation": ablation_rows,
                "coarse": [], "fine": [],
                "timestamp": datetime.now().isoformat(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[OK] Ablation report: {DEFAULT_REPORTS_DIR / 'ablation_single_axis.md'}")
        return

    # Phase B: 粗筛
    logger.info("=== Phase B: Coarse grid (targeted) ===")
    t0 = time.perf_counter()
    coarse_rows = await run_coarse(cfg_template, targeted=not args.full_grid)
    logger.info(f"[Coarse] 完成, 耗时 {time.perf_counter() - t0:.1f}s")

    # Phase C: 精扫
    fine_rows: list[dict] = []
    if not args.skip_fine:
        coarse_sorted = sorted(coarse_rows, key=lambda r: r["ndcg@10"], reverse=True)
        top_centers = [tuple(r["weights"]) for r in coarse_sorted[:10]]
        logger.info(f"=== Phase C: Fine grid (围绕 top-10 邻域) ===")
        t0 = time.perf_counter()
        fine_rows = await run_fine(cfg_template, top_centers, top_n=10)
        logger.info(f"[Fine] 完成, 耗时 {time.perf_counter() - t0:.1f}s")

    # 渲染报告
    grid_md = render_grid_md(coarse_rows, fine_rows, baseline_row)
    (DEFAULT_REPORTS_DIR / "grid_search_results.md").write_text(grid_md, encoding="utf-8")

    # 散点图
    try:
        plot_pareto(
            coarse_rows, fine_rows, baseline_row,
            DEFAULT_REPORTS_DIR / "grid_search_pareto.png",
        )
    except Exception as e:
        logger.warning(f"散点图生成失败 (跳过): {e}")

    # JSON artifact
    artifacts = {
        "baseline": baseline_row,
        "ablation": ablation_rows,
        "coarse": coarse_rows,
        "fine": fine_rows,
        "timestamp": datetime.now().isoformat(),
    }
    (DEFAULT_REPORTS_DIR / "grid_search_artifacts.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Top 10 摘要打印到 stdout
    all_rows = list(coarse_rows) + list(fine_rows)
    sorted_rows = sorted(all_rows, key=lambda r: r["ndcg@10"], reverse=True)
    print()
    print("=== Grid Search 完成 ===")
    print(f"  Baseline: {tuple(baseline_row['weights'])} → ndcg@10={baseline_row['ndcg@10']:.4f}")
    print()
    print("  Top 5:")
    for i, r in enumerate(sorted_rows[:5], 1):
        marker = " ⭐ baseline" if tuple(r["weights"]) == BASELINE else ""
        print(f"    {i}. {tuple(r['weights'])} → ndcg@10={r['ndcg@10']:.4f} mrr={r['mrr']:.4f}{marker}")
    print()
    print(f"  报告: {DEFAULT_REPORTS_DIR / 'grid_search_results.md'}")
    print(f"  散点: {DEFAULT_REPORTS_DIR / 'grid_search_pareto.png'}")


def main():
    parser = argparse.ArgumentParser(description="MemoCortex 4 信号权重 Grid Search")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None,
                        help="每组只跑 dataset 的前 N 条 (debug, 加速)")
    parser.add_argument("--quick", action="store_true",
                        help="仅跑单轴消融, 跳过完整 grid")
    parser.add_argument("--skip-fine", action="store_true",
                        help="跳过精扫阶段, 仅粗筛")
    parser.add_argument("--full-grid", action="store_true",
                        help="跑完整 5^4 grid 而非裁剪过的 targeted grid")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="跳过 Phase A (从上次 artifact 读取)")
    args = parser.parse_args()

    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
