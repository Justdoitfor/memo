"""arbitrator_eval/dataset_v1.jsonl 健康检查 + run_eval 内部纯函数测试."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "arbitrator_eval"
    / "dataset_v1.jsonl"
)


@pytest.fixture(scope="module")
def entries():
    if not DATASET_PATH.is_file():
        pytest.skip(
            f"Arbitrator dataset 未生成: {DATASET_PATH}; "
            f"先跑 build_dataset_v1.py"
        )
    return [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()]


def test_total_count(entries):
    assert len(entries) == 50


def test_ids_unique(entries):
    ids = [e["id"] for e in entries]
    assert len(set(ids)) == len(ids)


def test_action_distribution(entries):
    """action 分布按设计精确."""
    from collections import Counter

    counts = Counter(e["expected_action"] for e in entries)
    assert dict(counts) == {
        "replace": 12,
        "merge": 12,
        "versioned": 12,
        "ignore": 14,
    }


def test_required_fields(entries):
    required = {
        "id", "category", "subject", "predicate", "field_semantics",
        "existing", "new", "expected_action", "rationale",
    }
    for e in entries:
        missing = required - e.keys()
        assert not missing, f"{e['id']} 缺字段: {missing}"


def test_existing_non_empty(entries):
    """每条 case 至少有 1 条 existing fact 才能构成冲突."""
    for e in entries:
        assert len(e["existing"]) >= 1, f"{e['id']} existing 为空"


def test_new_has_object_and_confidence(entries):
    for e in entries:
        n = e["new"]
        assert "object" in n and "confidence" in n
        assert 0 <= n["confidence"] <= 1


def test_field_semantics_valid(entries):
    valid = {"unique", "list", "versioned"}
    for e in entries:
        assert e["field_semantics"] in valid

    # list category 必须 field_semantics=list
    list_cases = [e for e in entries if e["expected_action"] == "merge"]
    for e in list_cases:
        assert e["field_semantics"] == "list", (
            f"{e['id']}: merge 期望 → field_semantics 应为 list"
        )


def test_versioned_action_predicate_makes_sense(entries):
    """versioned action 的 predicate 应该是时态相关的."""
    versioned_cases = [e for e in entries if e["expected_action"] == "versioned"]
    versioned_predicate_keywords = (
        "_in", "lived", "worked_at", "studied", "planned", "will_", "past", "ex_"
    )
    for e in versioned_cases:
        pred = e["predicate"]
        assert any(kw in pred for kw in versioned_predicate_keywords), (
            f"{e['id']}: versioned 期望但 predicate '{pred}' 看起来非时态字段"
        )


def test_action_strings_lowercase(entries):
    """expected_action 必须小写, 与 ConflictAction enum 对齐."""
    for e in entries:
        assert e["expected_action"] == e["expected_action"].lower()
