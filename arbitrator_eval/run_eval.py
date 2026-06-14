"""LLM Arbitrator 评测 runner — 跑 50 条标注集 + stability 重复测试.

两个产出物:
  1. accuracy: 单次跑 50 条, 算 LLM action vs expected_action 的 confusion matrix + 准确率
  2. stability: 同一条 case 跑 N 次 (默认 5), 算 action 一致率 (5 次中最多次的 action 占比)

设计原则:
  - 复用 ConflictArbitrator.arbitrate (走真实代码路径), 不绕过 prompt 加载机制
  - 每次调用前手动构造 Triple 列表喂入, 不依赖 storage (可在任意环境跑)
  - confusion matrix 输出 markdown, 便于贴 PR
  - stability 主要看 action 稳定性 (reasoning 文本变异性是次要指标)

CLI 用法:
  PYTHONIOENCODING=utf-8 uv run python -m arbitrator_eval.run_eval --mode accuracy
  PYTHONIOENCODING=utf-8 uv run python -m arbitrator_eval.run_eval --mode stability --runs 5
  PYTHONIOENCODING=utf-8 uv run python -m arbitrator_eval.run_eval --mode both --runs 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from app.models import Triple
from app.utils.metrics import metrics

DATASET_PATH = Path(__file__).parent / "dataset_v1.jsonl"
REPORTS_DIR = Path(__file__).parent / "reports"


def _load_dataset() -> list[dict]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset 不存在: {DATASET_PATH}; 先跑 build_dataset_v1.py"
        )
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _entry_to_triples(entry: dict) -> tuple[Triple, list[Triple]]:
    """把 dataset entry 转换为 (new_triple, existing_triples)."""
    now = datetime.now()
    new_triple = Triple(
        subject=entry["subject"],
        predicate=entry["predicate"],
        object=entry["new"]["object"],
        confidence=entry["new"]["confidence"],
    )
    existing = []
    for ex in entry["existing"]:
        days_ago = ex.get("days_ago", 30)
        existing.append(
            Triple(
                subject=entry["subject"],
                predicate=entry["predicate"],
                object=ex["object"],
                confidence=ex["confidence"],
                created_at=now - timedelta(days=days_ago),
            )
        )
    return new_triple, existing


# ────────────────────────────────────────────────────────────────────────
#  Accuracy: 单次跑 → LLM action vs expected, confusion matrix
# ────────────────────────────────────────────────────────────────────────


async def _arbitrate_one(entry: dict, eval_user_id: str) -> dict[str, Any]:
    """对单条 entry 跑一次 arbitrator. 返回 action / reasoning / confidence."""
    # 局部 import 避免 module-level 加载 (eval 不开 arbitrator 也能 import)
    from app.arbitrator.conflict import conflict_arbitrator

    new_triple, existing = _entry_to_triples(entry)
    start = time.perf_counter()
    decision = await conflict_arbitrator.arbitrate(
        user_id=eval_user_id,
        new_triple=new_triple,
        existing_triples=existing,
        field_semantics=entry["field_semantics"],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "action": decision.action.value,
        "reasoning": decision.reasoning,
        "confidence": decision.confidence,
        "merged_value": decision.merged_value,
        "latency_ms": round(latency_ms, 2),
    }


async def run_accuracy(entries: list[dict]) -> dict[str, Any]:
    """跑全部 entries 一次, 算准确率 + confusion matrix."""
    logger.info(f"[Accuracy] 开始评测 {len(entries)} 条")
    eval_user_id = f"arb_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results = []
    for i, entry in enumerate(entries, 1):
        try:
            output = await _arbitrate_one(entry, eval_user_id)
            results.append({
                "id": entry["id"],
                "category": entry["category"],
                "expected": entry["expected_action"],
                "actual": output["action"],
                "correct": output["action"] == entry["expected_action"],
                "reasoning": output["reasoning"],
                "confidence": output["confidence"],
                "merged_value": output["merged_value"],
                "latency_ms": output["latency_ms"],
            })
            if i % 10 == 0 or i == len(entries):
                acc = sum(1 for r in results if r["correct"]) / len(results)
                logger.info(f"  进度 {i}/{len(entries)}, 累计 acc={acc:.3f}")
        except Exception as e:
            logger.exception(f"[Accuracy] entry {entry['id']} 失败")
            results.append({
                "id": entry["id"],
                "category": entry["category"],
                "expected": entry["expected_action"],
                "actual": "<ERROR>",
                "correct": False,
                "error": str(e),
            })

    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / len(results) if results else 0.0

    # Confusion matrix: rows=expected, cols=actual
    actions = ("replace", "merge", "versioned", "ignore")
    confusion: dict[str, dict[str, int]] = {a: {b: 0 for b in actions} for a in actions}
    for r in results:
        exp = r["expected"]
        act = r["actual"]
        if exp in confusion and act in confusion.get(exp, {}):
            confusion[exp][act] += 1

    # Per-category 准确率
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0}
    )
    for r in results:
        cat = r["category"].rsplit("_", 1)[0] if "_" in r["category"] else r["category"]
        # 先用 expected_action 当大类
        bucket = r["expected"]
        by_category[bucket]["n"] += 1
        if r["correct"]:
            by_category[bucket]["correct"] += 1

    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    return {
        "n_total": len(results),
        "n_correct": n_correct,
        "accuracy": round(accuracy, 4),
        "confusion": confusion,
        "by_expected_action": dict(by_category),
        "results": results,
        "latency_avg_ms": round(avg_latency, 2),
        "latency_max_ms": round(max_latency, 2),
        "timestamp": datetime.now().isoformat(),
    }


# ────────────────────────────────────────────────────────────────────────
#  Stability: 同 entry 跑 N 次, 看 action 稳定性
# ────────────────────────────────────────────────────────────────────────


async def run_stability(entries: list[dict], n_runs: int = 5) -> dict[str, Any]:
    """每条 entry 重复跑 n_runs 次, 算 action 一致率.

    关键指标:
      - mode_rate: 同条 case 中, 出现次数最多的 action 占比 (5 次里最多 5 次都同 action → 1.0)
      - perfect_consistency_rate: 多少条 case 能跑出 5 次完全一致
      - per_category 不一致率
    """
    logger.info(f"[Stability] 开始 {len(entries)} 条 × {n_runs} 次 = {len(entries) * n_runs} 次调用")
    eval_user_id = f"arb_stab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    per_entry_actions: dict[str, list[str]] = defaultdict(list)

    for i, entry in enumerate(entries, 1):
        for run in range(n_runs):
            try:
                output = await _arbitrate_one(entry, eval_user_id)
                per_entry_actions[entry["id"]].append(output["action"])
            except Exception as e:
                logger.warning(f"  {entry['id']} run {run} 失败: {e}")
                per_entry_actions[entry["id"]].append("<ERROR>")
        if i % 5 == 0 or i == len(entries):
            logger.info(f"  进度 {i}/{len(entries)}")

    # 统计每条 entry 的 mode_rate 和 perfect_consistency
    stats = []
    for entry in entries:
        actions = per_entry_actions[entry["id"]]
        counter = Counter(actions)
        if not counter:
            continue
        mode_action, mode_count = counter.most_common(1)[0]
        mode_rate = mode_count / len(actions)
        stats.append({
            "id": entry["id"],
            "category": entry["category"],
            "expected": entry["expected_action"],
            "runs": actions,
            "distinct_actions": len(set(actions)),
            "mode_action": mode_action,
            "mode_rate": round(mode_rate, 4),
            "perfectly_consistent": mode_rate == 1.0,
            "matches_expected": mode_action == entry["expected_action"],
        })

    avg_mode_rate = sum(s["mode_rate"] for s in stats) / len(stats) if stats else 0.0
    perfect_pct = (
        sum(1 for s in stats if s["perfectly_consistent"]) / len(stats) if stats else 0.0
    )
    expected_match_pct = (
        sum(1 for s in stats if s["matches_expected"]) / len(stats) if stats else 0.0
    )

    # 不稳定 case (mode_rate < 1.0)
    unstable = [s for s in stats if not s["perfectly_consistent"]]

    return {
        "n_entries": len(entries),
        "n_runs_per_entry": n_runs,
        "total_calls": len(entries) * n_runs,
        "avg_mode_rate": round(avg_mode_rate, 4),
        "perfect_consistency_pct": round(perfect_pct, 4),
        "majority_match_expected_pct": round(expected_match_pct, 4),
        "unstable_cases": unstable,
        "per_entry_stats": stats,
        "timestamp": datetime.now().isoformat(),
    }


# ────────────────────────────────────────────────────────────────────────
#  Markdown rendering
# ────────────────────────────────────────────────────────────────────────


def render_accuracy_md(report: dict[str, Any]) -> str:
    lines = [
        "# Arbitrator Accuracy Report",
        "",
        f"- Timestamp: `{report['timestamp']}`",
        f"- Dataset: `arbitrator_eval/dataset_v1.jsonl` ({report['n_total']} 条)",
        f"- **Accuracy: {report['accuracy']:.4f} ({report['n_correct']}/{report['n_total']})**",
        f"- Latency: avg {report['latency_avg_ms']:.0f}ms, max {report['latency_max_ms']:.0f}ms",
        "",
        "## Confusion Matrix",
        "",
        "rows = expected, cols = actual (LLM 输出)",
        "",
        "| expected \\ actual | replace | merge | versioned | ignore |",
        "|---|---:|---:|---:|---:|",
    ]
    for exp in ("replace", "merge", "versioned", "ignore"):
        row = report["confusion"].get(exp, {})
        lines.append(
            f"| **{exp}** | {row.get('replace', 0)} | {row.get('merge', 0)} | "
            f"{row.get('versioned', 0)} | {row.get('ignore', 0)} |"
        )
    lines.extend(["", "## Per Expected-Action Accuracy", ""])
    lines.append("| Expected Action | n | correct | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for action in ("replace", "merge", "versioned", "ignore"):
        cat = report["by_expected_action"].get(action, {"n": 0, "correct": 0})
        n = cat["n"]
        c = cat["correct"]
        acc = c / n if n else 0
        lines.append(f"| {action} | {n} | {c} | {acc:.4f} |")

    # 错误案例
    errors = [r for r in report["results"] if not r.get("correct")]
    if errors:
        lines.extend(["", "## Mismatched Cases", "", "| ID | Category | Expected | Actual | Reasoning |", "|---|---|---|---|---|"])
        for r in errors[:20]:
            reason = (r.get("reasoning") or "").replace("|", "\\|").replace("\n", " ")[:80]
            lines.append(
                f"| {r['id']} | {r['category']} | {r['expected']} | {r['actual']} | {reason} |"
            )

    return "\n".join(lines)


def render_stability_md(report: dict[str, Any]) -> str:
    lines = [
        "# Arbitrator Stability Report",
        "",
        f"- Timestamp: `{report['timestamp']}`",
        f"- Setup: {report['n_entries']} 条 × {report['n_runs_per_entry']} 次 = "
        f"{report['total_calls']} 次 LLM 调用",
        "",
        "## 关键指标",
        "",
        f"- **Average mode rate (每条 case 多数派 action 占比的均值)**: "
        f"`{report['avg_mode_rate']:.4f}`",
        f"- **Perfect consistency (多次跑 action 完全一致的 case 比例)**: "
        f"`{report['perfect_consistency_pct']:.2%}`",
        f"- **Majority action matches expected (多数派 action 与 ground truth 吻合)**: "
        f"`{report['majority_match_expected_pct']:.2%}`",
        "",
    ]
    if report["unstable_cases"]:
        lines.extend([
            "## Unstable Cases",
            "",
            f"共 {len(report['unstable_cases'])} 条 (mode_rate < 1.0):",
            "",
            "| ID | Category | Expected | Runs | Mode | Mode Rate |",
            "|---|---|---|---|---|---:|",
        ])
        for s in report["unstable_cases"]:
            runs_str = " / ".join(s["runs"])
            lines.append(
                f"| {s['id']} | {s['category']} | {s['expected']} | "
                f"{runs_str} | {s['mode_action']} | {s['mode_rate']:.4f} |"
            )
    else:
        lines.append("✅ 所有 case 多次跑 action 完全一致 — 模型输出稳定.")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────
#  Main CLI
# ────────────────────────────────────────────────────────────────────────


async def main_async(args):
    entries = _load_dataset()
    if args.limit:
        entries = entries[: args.limit]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode in ("accuracy", "both"):
        acc_report = await run_accuracy(entries)
        acc_md = render_accuracy_md(acc_report)
        (REPORTS_DIR / "accuracy.md").write_text(acc_md, encoding="utf-8")
        (REPORTS_DIR / "accuracy.json").write_text(
            json.dumps(acc_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print()
        print(f"=== Accuracy: {acc_report['accuracy']:.4f} "
              f"({acc_report['n_correct']}/{acc_report['n_total']}) ===")
        print(f"  报告: {REPORTS_DIR / 'accuracy.md'}")

    if args.mode in ("stability", "both"):
        stab_report = await run_stability(entries, n_runs=args.runs)
        stab_md = render_stability_md(stab_report)
        (REPORTS_DIR / "stability.md").write_text(stab_md, encoding="utf-8")
        (REPORTS_DIR / "stability.json").write_text(
            json.dumps(stab_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print()
        print(f"=== Stability ===")
        print(f"  avg_mode_rate: {stab_report['avg_mode_rate']:.4f}")
        print(f"  perfect_consistency: {stab_report['perfect_consistency_pct']:.2%}")
        print(f"  majority matches expected: {stab_report['majority_match_expected_pct']:.2%}")
        print(f"  报告: {REPORTS_DIR / 'stability.md'}")


def main():
    parser = argparse.ArgumentParser(description="LLM Arbitrator Eval Runner")
    parser.add_argument("--mode", choices=("accuracy", "stability", "both"),
                        default="both",
                        help="跑 accuracy / stability / both")
    parser.add_argument("--runs", type=int, default=5,
                        help="stability 模式下每条 entry 重复多少次 (default 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅跑前 N 条 (debug)")
    args = parser.parse_args()

    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
