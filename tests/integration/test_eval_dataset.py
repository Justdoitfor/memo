"""评测集本身的健康检查 — 防止以后生成器改坏数据.

跑测试: uv run pytest tests/integration/test_eval_dataset.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

DATASET = Path(__file__).resolve().parents[2] / "eval" / "datasets" / "memocortex_zh_v1.jsonl"


@pytest.fixture(scope="module")
def entries():
    if not DATASET.exists():
        pytest.skip(f"Dataset 未生成: {DATASET}; 运行 build_zh_v1.py 后再跑此测试")
    return [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines()]


def test_total_count(entries):
    assert len(entries) == 80


def test_ids_unique(entries):
    ids = [e["id"] for e in entries]
    assert len(set(ids)) == len(ids), "存在重复 id"


def test_id_format(entries):
    for e in entries:
        assert e["id"].startswith("zh-")
        assert len(e["id"]) == 6  # zh-NNN


def test_user_suffix_unique(entries):
    """user_suffix 唯一保证测试间数据隔离."""
    suffixes = [e["user_suffix"] for e in entries]
    assert len(set(suffixes)) == len(suffixes)


def test_scenario_distribution(entries):
    """按设计分布断言, 防止生成器漂."""
    from collections import Counter

    counts = Counter(e["scenario"] for e in entries)
    expected = {
        "exact_recall": 20,
        "paraphrase": 15,
        "temporal_window": 10,
        "conflict_latest": 10,
        "episodic_temporal": 10,
        "procedural": 5,
        "negation": 5,
        "mixed_types": 5,
    }
    assert dict(counts) == expected, f"分布偏离: {counts}"


def test_required_fields(entries):
    required = {"id", "scenario", "user_suffix", "setup", "query", "expected_mids", "top_k"}
    for e in entries:
        missing = required - e.keys()
        assert not missing, f"{e['id']} 缺字段: {missing}"


def test_setup_target_referenced_in_expected(entries):
    """每条 expected_mids 必须真实存在于 setup 中, 否则永远召不到."""
    for e in entries:
        setup_mids = {item["mid"] for item in e["setup"]}
        for emid in e["expected_mids"]:
            assert emid in setup_mids, f"{e['id']} expected={emid} 不在 setup 中"


def test_setup_has_distractors(entries):
    """除 procedural / mixed 外, setup 必须有干扰项 (>= 3 条) 防止 trivially 全召回."""
    for e in entries:
        if e["scenario"] in ("procedural", "mixed_types"):
            continue
        n_setup = len(e["setup"])
        n_target = len(e["expected_mids"])
        # setup 中 target 之外的都是干扰
        n_distract = n_setup - n_target
        assert n_distract >= 3, (
            f"{e['id']} 干扰项不足 ({n_distract}), 评测无判别力"
        )


def test_setup_item_types_valid(entries):
    """setup 项 type 必须是合法 MemoryType 字符串."""
    valid_types = {"episodic", "semantic", "procedural", "reflective", "implicit", "working"}
    for e in entries:
        for item in e["setup"]:
            assert item["type"] in valid_types, f"{e['id']} 非法 type: {item['type']}"


def test_semantic_items_have_structured(entries):
    """SEMANTIC 类型必须有 structured.subject + predicate + object."""
    for e in entries:
        for item in e["setup"]:
            if item["type"] == "semantic":
                s = item.get("structured", {})
                assert "subject" in s and "predicate" in s and "object" in s, (
                    f"{e['id']} mid={item['mid']} structured 字段不全"
                )


def test_temporal_window_has_valid_until(entries):
    """temporal_window 场景必须有过期事实 (valid_until 字段) 才能测时间过滤."""
    for e in entries:
        if e["scenario"] != "temporal_window":
            continue
        has_expired = any(
            item.get("structured", {}).get("valid_until")
            for item in e["setup"]
        )
        assert has_expired, f"{e['id']} temporal_window 必须有 valid_until 字段"


def test_conflict_latest_has_staleness(entries):
    """conflict_latest 场景必须有 staleness_signal=True 的旧事实."""
    for e in entries:
        if e["scenario"] != "conflict_latest":
            continue
        has_stale = any(item.get("staleness_signal") for item in e["setup"])
        assert has_stale, f"{e['id']} conflict_latest 必须有 staleness_signal"


def test_top_k_reasonable(entries):
    for e in entries:
        assert 1 <= e["top_k"] <= 20
