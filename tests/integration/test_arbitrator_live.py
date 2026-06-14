"""真 LLM 集成测试 — 验证 Arbitrator / Semantic 抽取的端到端正确性.

⚠️ 默认 skip — 需要 MEMOCORTEX_LLM_API_KEY 才会跑.
本地跑命令:
    uv run pytest tests/integration/test_arbitrator_live.py -m live_llm --no-cov

CI 默认跳过, 避免烧 API quota / 不稳定.

测试设计:
  - 不依赖 LLM 给出特定 reasoning 字符串 (LLM 输出有变异性)
  - 只断言 action 类型在合理范围内 (e.g. unique 字段冲突 → REPLACE 或 IGNORE, 不会 MERGE)
  - 不依赖具体 confidence 分数 (LLM 给的分有抖动)
"""
from __future__ import annotations

import uuid

import pytest

from app.models import ConflictAction, Triple

pytestmark = [pytest.mark.live_llm, pytest.mark.integration, pytest.mark.asyncio]


def _t(obj: str, predicate: str = "lives_in", confidence: float = 0.85) -> Triple:
    return Triple(
        subject="user", predicate=predicate, object=obj, confidence=confidence
    )


# ────────────────────────────────────────────────────────────────────────
#  Arbitrator: LLM 决策 4 种 action
# ────────────────────────────────────────────────────────────────────────


async def test_unique_field_conflict_replaces_or_ignores(storage_initialized):
    """unique 字段 (lives_in) 冲突 → LLM 应给 REPLACE 或 IGNORE, 不可能 MERGE."""
    from app.arbitrator.conflict import conflict_arbitrator

    user_id = "arb_live_" + uuid.uuid4().hex[:6]
    new = _t("上海", confidence=0.9)
    existing = [_t("北京", confidence=0.85)]

    decision = await conflict_arbitrator.arbitrate(
        user_id=user_id,
        new_triple=new,
        existing_triples=existing,
        field_semantics="unique",
    )

    # unique 字段不应 MERGE (语义错误)
    assert decision.action != ConflictAction.MERGE
    # action 必须是合法 enum
    assert decision.action in {
        ConflictAction.REPLACE,
        ConflictAction.IGNORE,
        ConflictAction.VERSIONED,
    }
    # 推理非空, confidence 合法
    assert decision.reasoning
    assert 0.0 <= decision.confidence <= 1.0


async def test_list_field_conflict_merges(storage_initialized):
    """list 字段 (allergic_to) 冲突 → LLM 应给 MERGE 并返回 merged_value."""
    from app.arbitrator.conflict import conflict_arbitrator

    user_id = "arb_live_" + uuid.uuid4().hex[:6]
    new = _t("芝麻", predicate="allergic_to", confidence=0.9)
    existing = [
        _t("花生", predicate="allergic_to", confidence=0.9),
        _t("乳糖", predicate="allergic_to", confidence=0.85),
    ]

    decision = await conflict_arbitrator.arbitrate(
        user_id=user_id,
        new_triple=new,
        existing_triples=existing,
        field_semantics="list",
    )

    assert decision.action == ConflictAction.MERGE
    assert decision.merged_value is not None
    # 合并后必须包含全部 3 个值
    merged_set = set(decision.merged_value.split(","))
    assert {"花生", "乳糖", "芝麻"} <= merged_set


async def test_arbitration_writes_audit_log(storage_initialized):
    """Arbitrator 决策必须写 arbitration_logs (审计可追溯, README 承诺)."""
    from app.arbitrator.conflict import conflict_arbitrator
    from app.storage import get_metadata

    user_id = "arb_audit_" + uuid.uuid4().hex[:6]
    meta = get_metadata()

    new = _t("上海", confidence=0.9)
    existing = [_t("北京", confidence=0.85)]

    await conflict_arbitrator.arbitrate(
        user_id=user_id, new_triple=new, existing_triples=existing,
        field_semantics="unique",
    )

    logs = await meta.list_arbitrations(user_id, limit=5)
    assert len(logs) >= 1
    last = logs[-1] if logs[0]["action"] != "ignore" else logs[0]
    # 审计必须包含: subject / predicate / new_value / action / reasoning
    found = next(
        (l for l in logs if l["new_value"] == "上海" and l["predicate"] == "lives_in"),
        None,
    )
    assert found is not None, f"未找到本次仲裁的日志, logs={logs}"
    assert found["action"] in {"replace", "ignore", "versioned"}
    assert found["reasoning"]


# ────────────────────────────────────────────────────────────────────────
#  Semantic 抽取 + 写入 e2e (LLM 路径)
# ────────────────────────────────────────────────────────────────────────


async def test_semantic_extraction_and_kg_write(storage_initialized):
    """SEMANTIC 写入 → LLM 抽取 triples → KG 应包含对应事实."""
    from app.memories.semantic import semantic_memory
    from app.storage import get_kg

    user_id = "sem_e2e_" + uuid.uuid4().hex[:6]
    await semantic_memory.write_from_text(
        user_id=user_id,
        text="我对花生过敏, 而且我家有只叫小白的猫",
    )

    kg = get_kg()
    triples = await kg.find_triples(user_id, subject="user")

    # LLM 应至少抽到 2 个事实
    predicates = [t.predicate for t in triples]
    objects = [t.object for t in triples]
    # 要么用我们模板里的 allergic_to, 要么 LLM 起的别名 (但应包含"花生" / "小白" 实体)
    has_allergy = "allergic_to" in predicates and "花生" in objects
    has_pet = ("has_pet" in predicates) and ("小白" in objects)

    assert has_allergy, f"应抽到 allergic_to 花生, 实际 triples={[(t.predicate, t.object) for t in triples]}"
    assert has_pet, f"应抽到 has_pet 小白, 实际 triples={[(t.predicate, t.object) for t in triples]}"
