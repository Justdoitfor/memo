"""LLM Arbitrator 评测集 v1 生成器.

50 条人工标注的冲突 case, 每条带 expected_action ground truth.
分布: replace 12, merge 12, versioned 12, ignore 14 (含若干 edge case).

输出: arbitrator_eval/dataset_v1.jsonl

标注规范 (面试官追问点):
  - 每条 case 由项目作者基于真实用户场景设计 + 人工拍 expected_action
  - 边界 case 故意保留 (低置信度新事实 / 时态模糊) 测 LLM 是否过度自信
  - field_semantics 提示给定时遵循"list→merge / unique→replace/ignore" 默认
    LLM 仍可基于内容推翻默认 (e.g. unique 字段但新值很可疑应 IGNORE)
  - rationale 字段是给人读的, LLM 看不到, 仅供人工校验时参考
"""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).parent / "dataset_v1.jsonl"


def _entry(
    eid: str,
    category: str,
    subject: str,
    predicate: str,
    field_semantics: str,
    existing: list[dict],
    new_object: str,
    new_confidence: float,
    expected_action: str,
    rationale: str,
) -> dict:
    return {
        "id": eid,
        "category": category,
        "subject": subject,
        "predicate": predicate,
        "field_semantics": field_semantics,
        "existing": existing,
        "new": {"object": new_object, "confidence": new_confidence},
        "expected_action": expected_action,
        "rationale": rationale,
    }


def _ex(obj: str, conf: float = 0.85, days_ago: int = 30) -> dict:
    return {"object": obj, "confidence": conf, "days_ago": days_ago}


# ────────────────────────────────────────────────────────────────────────
#  REPLACE — 12 条 (unique 字段, 新值置信度高 / 明确变更)
# ────────────────────────────────────────────────────────────────────────


def build_replace() -> list[dict]:
    return [
        _entry("arb-001", "replace_clear_move", "user", "lives_in", "unique",
               [_ex("北京", 0.85, 365)], "上海", 0.95, "replace",
               "用户主动告知搬家, 单值字段且新置信度高"),
        _entry("arb-002", "replace_job_change", "user", "works_at", "unique",
               [_ex("腾讯", 0.85, 730)], "字节跳动", 0.95, "replace",
               "用户跳槽, 雇主单值字段"),
        _entry("arb-003", "replace_age_increment", "user", "age", "unique",
               [_ex("28", 0.95, 400)], "29", 0.95, "replace",
               "年龄递增覆盖旧值"),
        _entry("arb-004", "replace_phone", "user", "owns_phone", "unique",
               [_ex("iPhone 13", 0.9, 730)], "iPhone 16 Pro", 0.95, "replace",
               "换手机, 单值"),
        _entry("arb-005", "replace_car", "user", "owns_car", "unique",
               [_ex("本田思域", 0.85, 1000)], "特斯拉 Model Y", 0.95, "replace",
               "换车, 单值字段"),
        _entry("arb-006", "replace_marital", "user", "married_to", "unique",
               [_ex("张三", 0.9, 730)], "李四", 0.92, "replace",
               "再婚配偶变更"),
        _entry("arb-007", "replace_weight", "user", "weight_kg", "unique",
               [_ex("85", 0.9, 365)], "72", 0.95, "replace",
               "减肥后体重更新"),
        _entry("arb-008", "replace_email", "user", "email", "unique",
               [_ex("alice@old.com", 0.95, 365)], "alice@new.com", 0.98, "replace",
               "邮箱变更"),
        _entry("arb-009", "replace_occupation", "user", "occupation", "unique",
               [_ex("工程师", 0.9, 1000)], "产品经理", 0.92, "replace",
               "职业转型"),
        _entry("arb-010", "replace_height_correction", "user", "height_cm", "unique",
               [_ex("175", 0.7, 200)], "178", 0.95, "replace",
               "测量误差校正, 新值置信度更高"),
        _entry("arb-011", "replace_blood_type_correction", "user", "blood_type", "unique",
               [_ex("A", 0.7, 365)], "O", 0.99, "replace",
               "化验报告校正"),
        _entry("arb-012", "replace_favorite_color", "user", "favorite_color", "unique",
               [_ex("红色", 0.6, 1000)], "深蓝色", 0.85, "replace",
               "偏好更新, 单值"),
    ]


# ────────────────────────────────────────────────────────────────────────
#  MERGE — 12 条 (list 字段, 应合并多值)
# ────────────────────────────────────────────────────────────────────────


def build_merge() -> list[dict]:
    return [
        _entry("arb-013", "merge_allergy_disjoint", "user", "allergic_to", "list",
               [_ex("花生", 0.95, 100)], "芝麻", 0.95, "merge",
               "list 字段, 不重叠应保留全部过敏原"),
        _entry("arb-014", "merge_allergy_three", "user", "allergic_to", "list",
               [_ex("花生", 0.95, 200), _ex("乳糖", 0.9, 365)], "芒果", 0.9, "merge",
               "list 字段, 三个独立过敏原"),
        _entry("arb-015", "merge_likes_overlap", "user", "likes", "list",
               [_ex("跑步", 0.85, 100), _ex("游泳", 0.85, 100)], "跑步", 0.85, "merge",
               "list 字段, 新旧重叠应去重"),
        _entry("arb-016", "merge_pets", "user", "has_pet", "list",
               [_ex("小白(猫)", 0.95, 365)], "汪汪(狗)", 0.95, "merge",
               "list 字段, 新增宠物"),
        _entry("arb-017", "merge_languages", "user", "speaks_language", "list",
               [_ex("中文", 0.99, 1000), _ex("英语", 0.95, 1000)], "日语", 0.85, "merge",
               "list 字段, 语言能力扩展"),
        _entry("arb-018", "merge_hobbies", "user", "hobby", "list",
               [_ex("摄影", 0.9, 500)], "登山", 0.9, "merge",
               "list 字段, 新增爱好"),
        _entry("arb-019", "merge_visited_cities", "user", "visited", "list",
               [_ex("东京", 0.95, 365), _ex("巴黎", 0.95, 730)], "纽约", 0.95, "merge",
               "list 字段, 旅行足迹累积"),
        _entry("arb-020", "merge_dislikes", "user", "dislikes", "list",
               [_ex("芒果", 0.85, 200)], "香菜", 0.9, "merge",
               "list 字段, 不喜欢的食物列表"),
        _entry("arb-021", "merge_children", "user", "has_child", "list",
               [_ex("大宝", 0.99, 1000)], "二宝", 0.99, "merge",
               "list 字段, 二胎"),
        _entry("arb-022", "merge_skills", "user", "skill", "list",
               [_ex("Python", 0.95, 365), _ex("Go", 0.9, 200)], "Rust", 0.85, "merge",
               "list 字段, 技能栈累积"),
        _entry("arb-023", "merge_overlap_three", "user", "allergic_to", "list",
               [_ex("花生", 0.95, 100), _ex("芝麻", 0.9, 200)], "花生", 0.95, "merge",
               "新值已在旧值列表 → 合并去重不增"),
        _entry("arb-024", "merge_favorite_food_multi", "user", "favorite_food", "list",
               [_ex("川菜", 0.85, 365)], "粤菜", 0.85, "merge",
               "list 字段, 喜爱菜系扩展"),
    ]


# ────────────────────────────────────────────────────────────────────────
#  VERSIONED — 12 条 (时态明确的过去/未来事实)
# ────────────────────────────────────────────────────────────────────────


def build_versioned() -> list[dict]:
    return [
        _entry("arb-025", "versioned_worked_in_past", "user", "worked_in", "versioned",
               [_ex("北京", 0.9, 730)], "上海", 0.95, "versioned",
               "过去工作经历, 应保留历史版本"),
        _entry("arb-026", "versioned_lived_in", "user", "lived_in", "versioned",
               [_ex("广州", 0.9, 1500), _ex("深圳", 0.9, 800)], "杭州", 0.92, "versioned",
               "时态字段, 历史居住地"),
        _entry("arb-027", "versioned_studied_at", "user", "studied_at", "versioned",
               [_ex("清华大学", 0.99, 2000)], "麻省理工", 0.95, "versioned",
               "教育经历多段, 应保留时间链"),
        _entry("arb-028", "versioned_planned_move", "user", "planned_move_to", "versioned",
               [_ex("成都", 0.85, 30)], "重庆", 0.9, "versioned",
               "未来计划变更但旧计划值得追溯"),
        _entry("arb-029", "versioned_will_visit", "user", "will_visit", "versioned",
               [_ex("日本", 0.85, 60)], "韩国", 0.85, "versioned",
               "未来出行计划, 多段保留"),
        _entry("arb-030", "versioned_history_replace_keyword", "user", "worked_at", "versioned",
               [_ex("阿里", 0.95, 1000)], "字节", 0.95, "versioned",
               "worked_at 是 versioned 字段, 不是 unique 的 works_at"),
        _entry("arb-031", "versioned_planned_event", "user", "planned_event", "versioned",
               [_ex("结婚", 0.95, 90)], "买房", 0.9, "versioned",
               "重大计划事件历史链"),
        _entry("arb-032", "versioned_owned_car_past", "user", "owned_car_past", "versioned",
               [_ex("大众朗逸", 0.85, 1500)], "本田思域", 0.85, "versioned",
               "历史拥有过的车, 时态字段"),
        _entry("arb-033", "versioned_relationship_history", "user", "ex_partner", "versioned",
               [_ex("张三", 0.85, 1000)], "李四", 0.85, "versioned",
               "历史伴侣, 时态字段"),
        _entry("arb-034", "versioned_lived_in_period", "user", "lived_in", "versioned",
               [_ex("北京 (2018-2020)", 0.9, 1500)], "上海 (2020-2022)", 0.9, "versioned",
               "明确时间段的居住地"),
        _entry("arb-035", "versioned_education_phases", "user", "studied_at", "versioned",
               [_ex("北大附中", 0.95, 3000), _ex("清华大学", 0.99, 2000)], "斯坦福研究生", 0.95, "versioned",
               "学历分阶段, 全保留"),
        _entry("arb-036", "versioned_will_work_at", "user", "will_work_at", "versioned",
               [_ex("微软", 0.8, 60)], "Google", 0.85, "versioned",
               "Offer 比较, 未来工作选择"),
    ]


# ────────────────────────────────────────────────────────────────────────
#  IGNORE — 14 条 (新事实可疑 / 低置信度 / 模糊冲突)
# ────────────────────────────────────────────────────────────────────────


def build_ignore() -> list[dict]:
    return [
        _entry("arb-037", "ignore_low_confidence", "user", "lives_in", "unique",
               [_ex("北京", 0.95, 365)], "上海", 0.3, "ignore",
               "新值置信度 (0.3) 远低于旧值, 应忽略"),
        _entry("arb-038", "ignore_age_decrease", "user", "age", "unique",
               [_ex("30", 0.99, 100)], "25", 0.7, "ignore",
               "年龄不可能减少, 矛盾事实"),
        _entry("arb-039", "ignore_low_conf_blood_type", "user", "blood_type", "unique",
               [_ex("O", 0.99, 1000)], "AB", 0.4, "ignore",
               "血型生物学固定, 低置信度新值不可信"),
        _entry("arb-040", "ignore_marital_unclear", "user", "married_to", "unique",
               [_ex("张三", 0.95, 365)], "未知", 0.3, "ignore",
               "新值'未知', 表述含糊"),
        _entry("arb-041", "ignore_email_typo", "user", "email", "unique",
               [_ex("alice@example.com", 0.99, 365)], "alic@example.com", 0.4, "ignore",
               "新邮箱看似拼写错误, 低置信度"),
        _entry("arb-042", "ignore_height_implausible", "user", "height_cm", "unique",
               [_ex("178", 0.95, 100)], "150", 0.4, "ignore",
               "成人身高不会突变, 低置信度新值"),
        _entry("arb-043", "ignore_weight_extreme", "user", "weight_kg", "unique",
               [_ex("75", 0.95, 30)], "30", 0.3, "ignore",
               "30 公斤对成人不合理"),
        _entry("arb-044", "ignore_phone_low_conf", "user", "owns_phone", "unique",
               [_ex("iPhone 16", 0.95, 30)], "可能是华为", 0.3, "ignore",
               "新值含'可能', 低置信度"),
        _entry("arb-045", "ignore_hometown_change", "user", "hometown", "unique",
               [_ex("成都", 0.99, 1000)], "重庆", 0.5, "ignore",
               "家乡是出生地, 通常不变, 中等置信度新值不足以覆盖"),
        _entry("arb-046", "ignore_gender_low_conf", "user", "gender", "unique",
               [_ex("男", 0.99, 1000)], "女", 0.3, "ignore",
               "性别低置信度变更应忽略"),
        _entry("arb-047", "ignore_education_downgrade", "user", "educational_background",
               "unique",
               [_ex("清华大学计算机系", 0.99, 1000)], "中专", 0.3, "ignore",
               "学历降级是异常事件, 低置信度需要忽略"),
        _entry("arb-048", "ignore_inconsistent_subject", "user", "favorite_color", "unique",
               [_ex("深蓝色", 0.85, 365)], "随便", 0.3, "ignore",
               "'随便' 不是有效偏好值"),
        _entry("arb-049", "ignore_partial_confidence", "user", "favorite_food", "unique",
               [_ex("川菜", 0.85, 200)], "好像是粤菜", 0.4, "ignore",
               "含'好像是'不确定语气, 低置信度"),
        _entry("arb-050", "ignore_blank_object", "user", "occupation", "unique",
               [_ex("工程师", 0.9, 365)], "暂未确定", 0.3, "ignore",
               "'暂未确定' 不是有效职业值"),
    ]


def build_all() -> list[dict]:
    entries = (
        build_replace()
        + build_merge()
        + build_versioned()
        + build_ignore()
    )
    # 校验
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "重复 id"
    assert len(entries) == 50, f"应为 50 条, 实际 {len(entries)}"
    actions = [e["expected_action"] for e in entries]
    from collections import Counter
    counts = Counter(actions)
    assert counts == {"replace": 12, "merge": 12, "versioned": 12, "ignore": 14}, (
        f"action 分布偏离: {counts}"
    )
    return entries


def main():
    entries = build_all()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    from collections import Counter

    by_cat = Counter(e["category"].rsplit("_", 1)[0].split("_")[0] for e in entries)
    print(f"[OK] Wrote {len(entries)} entries to {OUTPUT}")
    by_action = Counter(e["expected_action"] for e in entries)
    for action, n in sorted(by_action.items()):
        print(f"  expected={action:10s} {n:>3d}")


if __name__ == "__main__":
    main()
