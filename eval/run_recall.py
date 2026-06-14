"""eval CLI 入口 — 跑一次评测, 输出 markdown 报告.

用法:
    # 跑全套, 用 config 默认权重
    uv run python -m eval.run_recall

    # 仅跑前 5 条 (快速 smoke)
    uv run python -m eval.run_recall --limit 5

    # 显式权重 (grid search 复用)
    uv run python -m eval.run_recall --weights 0.45,0.15,0.25,0.15

    # 指定数据集 / 输出路径
    uv run python -m eval.run_recall \\
        --dataset eval/datasets/memocortex_zh_v1.jsonl \\
        --report eval/reports/run_$(date +%Y%m%d_%H%M).md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

# 必须在任何 app.* import 前执行 (确保 ChromaDB / SQLite 写到正确目录)
# 此处只设置 PYTHONIOENCODING 让 stdout 能打印中文
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from eval.runner import RunnerConfig, render_markdown, run_suite


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "datasets" / "memocortex_zh_v1.jsonl"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"


def _parse_weights(s: str) -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--weights 需要 4 个逗号分隔的浮点数, 实际 {len(parts)}: {s}"
        )
    return tuple(parts)  # type: ignore[return-value]


def main():
    parser = argparse.ArgumentParser(description="MemoCortex Recall Eval Runner")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help=f"评测集 jsonl 路径 (默认 {DEFAULT_DATASET.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--weights", type=_parse_weights, default=None,
        help="覆盖 4 信号融合权重, 格式 'vec,temp,kw,imp', e.g. '0.4,0.2,0.2,0.2'",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="仅跑前 N 条 (debug)",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.0,
        help="final_score 阈值 (默认 0.0 拿全部候选)",
    )
    parser.add_argument(
        "--suite", default="memocortex_zh_v1",
        help="suite 名称 (写入 eval_runs 表用)",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="markdown 报告输出路径 (默认 eval/reports/<suite>_<timestamp>.md)",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="JSON 报告输出路径 (供 grid search 消费)",
    )
    parser.add_argument(
        "--no-forget", action="store_true",
        help="评测后不清理 user 数据 (debug)",
    )
    args = parser.parse_args()

    cfg = RunnerConfig(
        dataset_path=args.dataset,
        weights=args.weights,
        score_threshold=args.score_threshold,
        limit=args.limit,
        forget_after=not args.no_forget,
        suite_name=args.suite,
    )

    report = asyncio.run(run_suite(cfg))

    # 渲染并写盘
    md = render_markdown(report)
    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = args.report or (
        DEFAULT_REPORTS_DIR
        / f"{cfg.suite_name}_{report.timestamp.replace(':', '').replace('.', '_')}.md"
    )
    report_path.write_text(md, encoding="utf-8")

    # JSON artifact (grid search 消费)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "suite": report.suite,
            "weights": list(report.weights) if report.weights else None,
            "n_total": report.n_total,
            "n_failed": report.n_failed,
            "overall": report.overall,
            "by_scenario": report.by_scenario,
            "latency": report.latency,
            "timestamp": report.timestamp,
        }
        args.json_out.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 打印关键指标到 stdout
    print()
    print(f"=== Eval Report: {report.suite} ===")
    print(f"  weights: {report.weights or 'config-default'}")
    print(f"  n_total: {report.n_total}, n_failed: {report.n_failed}")
    print()
    print("  Overall:")
    for k in ("recall@5", "recall@10", "ndcg@5", "ndcg@10", "mrr", "hit@1"):
        if k in report.overall:
            print(f"    {k:12s} = {report.overall[k]:.4f}")
    print()
    print(f"  Latency: p50={report.latency['p50_ms']:.1f}ms "
          f"p95={report.latency['p95_ms']:.1f}ms "
          f"max={report.latency['max_ms']:.1f}ms")
    print()
    print(f"  Report: {report_path}")
    print()


if __name__ == "__main__":
    main()
