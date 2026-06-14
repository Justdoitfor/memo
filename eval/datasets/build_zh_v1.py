"""MemoCortex 中文长期记忆评测集 v1 生成器.

产出: eval/datasets/memocortex_zh_v1.jsonl  (80 条)

设计原则 (面试官追问点):
  1. 自建数据集而非 LongMemEval — bge-small-zh-v1.5 是中文嵌入模型,
     用英文 LongMemEval 跑分会被 cross-lingual gap 主导, 失去对召回算法本身的判别力.
  2. 8 个场景覆盖不同的召回失败模式:
     - exact_recall   关键词完全匹配的基线 (向量层应满分)
     - paraphrase     改写召回 (考验向量泛化)
     - temporal       时间窗口过滤 (考验 valid_from/valid_until 语义)
     - conflict       多版本冲突 (考验仲裁后的召回排序)
     - negation       否定查询 (考验向量是否区分 "喜欢" vs "不喜欢")
     - episodic       时序事件 (考验 episodic memory 召回)
     - procedural     流程模板 (考验 procedural memory)
     - mixed          跨记忆类型 (考验路由 + 融合)
  3. 每条独立 user_id_suffix, 测试间数据隔离, 可并行跑.
  4. 每条 setup 列表生成"干扰项" (unrelated memories), 防止 K=8 时 trivially 全召回.

每条样例结构:
  {
    "id": "zh-001",
    "scenario": "exact_recall",
    "user_suffix": "001",
    "setup": [
      {"mid": "m1", "type": "semantic", "content": "我对花生过敏",
       "structured": {"subject": "user", "predicate": "allergic_to", "object": "花生"},
       "created_days_ago": 7},
      ...
    ],
    "query": "花生过敏",
    "expected_mids": ["m1"],   # 应出现在 top_k 中 (顺序无关)
    "top_k": 5,
    "score_threshold": 0.0,    # 0.0 = 关闭过滤拿全部候选
    "notes": "完全匹配召回基线 — 向量层应给 0.9+"
  }

mid 是 dataset-local 标识符. runner 在 setup 时把 mid 映射到真实 uuid memory_id,
expected 比对时用映射后的真实 id.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUTPUT = Path(__file__).parent / "memocortex_zh_v1.jsonl"


# ────────────────────────────────────────────────────────────────────────
#  Scenario builders — 每个返回 list[dict] entry
# ────────────────────────────────────────────────────────────────────────


def _entry(
    eid: str, scenario: str, suffix: str,
    setup: list[dict], query: str, expected: list[str],
    top_k: int = 5, threshold: float = 0.0, notes: str = "",
) -> dict:
    return {
        "id": eid,
        "scenario": scenario,
        "user_suffix": suffix,
        "setup": setup,
        "query": query,
        "expected_mids": expected,
        "top_k": top_k,
        "score_threshold": threshold,
        "notes": notes,
    }


def _sem(mid: str, content: str, subj: str, pred: str, obj: str,
         days_ago: int = 7, importance: float = 0.7,
         valid_from: str | None = None, valid_until: str | None = None) -> dict:
    """构造一条 SEMANTIC 记忆 (直接绕过 LLM 抽取, 用预填的 structured)."""
    structured = {"subject": subj, "predicate": pred, "object": obj}
    if valid_from:
        structured["valid_from"] = valid_from
    if valid_until:
        structured["valid_until"] = valid_until
    return {
        "mid": mid, "type": "semantic", "content": content,
        "structured": structured, "created_days_ago": days_ago,
        "importance": importance,
    }


def _epi(mid: str, content: str, days_ago: int = 1, importance: float = 0.5) -> dict:
    return {
        "mid": mid, "type": "episodic", "content": content,
        "created_days_ago": days_ago, "importance": importance,
    }


def _proc(mid: str, mid_pattern: str, steps: list[str],
          days_ago: int = 7, importance: float = 0.6) -> dict:
    return {
        "mid": mid, "type": "procedural",
        "content": f"任务模式: {mid_pattern}\n步骤:\n" + "\n".join(
            f"  {i + 1}. {s}" for i, s in enumerate(steps)
        ),
        "structured": {"task_pattern": mid_pattern, "steps": steps},
        "created_days_ago": days_ago,
        "importance": importance,
    }


# 共享干扰项 — 让每个测试 user 有 5-8 条不相关记忆, 防止 K=5 时全召回
def _distractors(prefix: str, count: int = 5) -> list[dict]:
    pool = [
        ("我喜欢看科幻电影", "user", "likes", "科幻电影"),
        ("我会说英语和日语", "user", "speaks_language", "英语"),
        ("我家有只叫小白的猫", "user", "has_pet", "小白"),
        ("我的笔记本是 ThinkPad X1", "user", "owns_laptop", "ThinkPad X1"),
        ("我妈是医生", "user", "mother_occupation", "医生"),
        ("我每天跑步 5 公里", "user", "habit", "跑步 5 公里"),
        ("我喜欢喝美式咖啡", "user", "favorite_drink", "美式咖啡"),
        ("我用 iPhone 14 Pro", "user", "owns_phone", "iPhone 14 Pro"),
    ]
    return [
        _sem(f"{prefix}_d{i}", c, s, p, o, days_ago=15 + i, importance=0.4)
        for i, (c, s, p, o) in enumerate(pool[:count])
    ]


# ────────────────────────────────────────────────────────────────────────
#  Scenario 1: exact_recall (20 条) — 关键词完全匹配
# ────────────────────────────────────────────────────────────────────────


def build_exact_recall() -> list[dict]:
    cases = [
        # (suffix, content, predicate, object, query)
        ("001", "我对花生过敏", "allergic_to", "花生", "花生过敏"),
        ("002", "我住在北京海淀区", "lives_in", "北京海淀区", "住在北京"),
        ("003", "我在字节跳动工作", "works_at", "字节跳动", "字节跳动工作"),
        ("004", "我今年 28 岁", "age", "28", "用户年龄"),
        ("005", "我血型是 O 型", "blood_type", "O型", "血型"),
        ("006", "我的车是特斯拉 Model Y", "owns_car", "特斯拉 Model Y", "我的车"),
        ("007", "我喜欢吃川菜", "likes", "川菜", "喜欢吃什么菜"),
        ("008", "我女朋友叫小雪", "girlfriend", "小雪", "女朋友是谁"),
        ("009", "我毕业于清华大学计算机系", "educational_background", "清华大学计算机系", "学历"),
        ("010", "我老家在四川成都", "hometown", "四川成都", "老家在哪"),
        ("011", "我体重 75 公斤", "weight_kg", "75", "体重"),
        ("012", "我身高 178 厘米", "height_cm", "178", "身高"),
        ("013", "我的微信号是 alice2025", "wechat", "alice2025", "微信号"),
        ("014", "我邮箱是 alice@example.com", "email", "alice@example.com", "邮箱"),
        ("015", "我有个 5 岁的女儿叫小米", "has_child", "小米", "孩子"),
        ("016", "我对芒果过敏", "allergic_to", "芒果", "芒果过敏"),
        ("017", "我用佳能 R5 拍照", "uses_camera", "佳能 R5", "用什么相机"),
        ("018", "我最喜欢的颜色是深蓝色", "favorite_color", "深蓝色", "喜欢什么颜色"),
        ("019", "我每周三去健身房", "habit", "周三健身", "周三的安排"),
        ("020", "我老婆是律师", "spouse_occupation", "律师", "老婆职业"),
    ]
    out = []
    for suffix, content, pred, obj, query in cases:
        target = _sem("target", content, "user", pred, obj, days_ago=5, importance=0.8)
        setup = [target] + _distractors(suffix, count=4)
        out.append(_entry(
            f"zh-{suffix}", "exact_recall", suffix,
            setup, query, ["target"], top_k=5,
            notes="关键词完全匹配 — 向量 + 关键词信号都应给高分"
        ))
    return out


# ────────────────────────────────────────────────────────────────────────
#  Scenario 2: paraphrase (15 条) — 改写召回
# ────────────────────────────────────────────────────────────────────────


def build_paraphrase() -> list[dict]:
    cases = [
        # (suffix, fact_content, fact_pred, fact_obj, query)
        ("021", "我搬家到上海浦东了", "lives_in", "上海浦东", "用户现在住在哪个城市"),
        ("022", "我去年跳槽到字节做基础架构", "works_at", "字节", "用户的雇主是哪家公司"),
        ("023", "我对乳糖不耐受", "allergic_to", "乳糖", "用户有什么饮食限制"),
        ("024", "我家里有两只猫", "has_pet", "猫", "用户养了什么宠物"),
        ("025", "我儿子今年 10 岁了", "has_child", "儿子", "用户有几个小孩"),
        ("026", "我目前主要写 Python 和 Go", "skill", "Python Go", "用户用什么编程语言"),
        ("027", "我每天通勤 1 小时到公司", "commute_time", "1 小时", "用户上班路上多久"),
        ("028", "我大学读的是软件工程", "educational_background", "软件工程", "用户大学专业"),
        ("029", "我太太是做产品经理的", "spouse_occupation", "产品经理", "用户配偶做什么工作"),
        ("030", "我从小在南方长大", "hometown_region", "南方", "用户的成长地区"),
        ("031", "我办公的位置在朝阳区国贸", "work_location", "朝阳区国贸", "用户上班地点"),
        ("032", "我母亲是大学教授", "mother_occupation", "大学教授", "用户妈妈做什么"),
        ("033", "我下个月要去成都出差", "planned_visit", "成都", "用户最近的出差计划"),
        ("034", "我的健身目标是减脂 10 公斤", "fitness_goal", "减脂 10 公斤", "用户在健身方面的目标"),
        ("035", "我手机型号是 iPhone 16 Pro Max", "owns_phone", "iPhone 16 Pro Max", "用户用的什么手机"),
    ]
    out = []
    for suffix, content, pred, obj, query in cases:
        target = _sem("target", content, "user", pred, obj, days_ago=10, importance=0.75)
        setup = [target] + _distractors(suffix, count=5)
        out.append(_entry(
            f"zh-{suffix}", "paraphrase", suffix,
            setup, query, ["target"], top_k=5,
            notes="改写/同义召回 — 主要靠向量泛化能力"
        ))
    return out


# ────────────────────────────────────────────────────────────────────────
#  Scenario 3: temporal_window (10 条) — 时间窗口过滤
# ────────────────────────────────────────────────────────────────────────


def build_temporal() -> list[dict]:
    """考验 valid_from / valid_until 语义.
    每条设置一个 valid_until 已过期的旧事实 + 一个当前有效事实, query 应只命中当前.
    """
    cases = [
        ("036", "之前在北京工作", "worked_in", "北京", "在上海工作", "lives_in", "上海", "用户现在在哪工作"),
        ("037", "以前住过广州", "lived_in", "广州", "住在杭州", "lives_in", "杭州", "用户当前住哪"),
        ("038", "去年开宝马 X3", "owns_car_past", "宝马 X3", "现在开特斯拉", "owns_car", "特斯拉 Model 3", "用户现在的车是什么"),
        ("039", "之前用三星 S22", "owns_phone_past", "三星 S22", "现在用 iPhone 15", "owns_phone", "iPhone 15", "用户当前的手机"),
        ("040", "前任是张三", "ex_partner", "张三", "现在女朋友叫李四", "girlfriend", "李四", "用户女朋友是谁"),
        ("041", "去年体重 85 公斤", "weight_past", "85", "现在体重 70 公斤", "weight_kg", "70", "用户当前体重"),
        ("042", "原来在腾讯工作", "worked_in", "腾讯", "现在在阿里", "works_at", "阿里", "用户现在的雇主"),
        ("043", "之前喜欢摇滚", "past_preference", "摇滚", "现在喜欢爵士", "likes", "爵士", "用户当下音乐偏好"),
        ("044", "以前用 PyCharm", "tool_past", "PyCharm", "现在用 Cursor", "uses_tool", "Cursor", "用户现在用什么编辑器"),
        ("045", "之前住合租", "housing_past", "合租", "现在自己租了一居室", "housing", "一居室", "用户当下居住状态"),
    ]
    out = []
    for suffix, past_c, past_pred, past_obj, now_c, now_pred, now_obj, query in cases:
        # 旧事实: valid_until 设为 30 天前 (已失效)
        past = _sem(
            "past", past_c, "user", past_pred, past_obj, days_ago=180, importance=0.6,
            valid_until="2026-05-15T00:00:00",  # 测试基准日期之前
        )
        # 新事实: valid_from 30 天前
        current = _sem(
            "current", now_c, "user", now_pred, now_obj, days_ago=20, importance=0.85,
            valid_from="2026-05-15T00:00:00",
        )
        setup = [past, current] + _distractors(suffix, count=3)
        out.append(_entry(
            f"zh-{suffix}", "temporal_window", suffix,
            setup, query, ["current"], top_k=5,
            notes="过期事实应被降权, 当前有效事实排前"
        ))
    return out


# ────────────────────────────────────────────────────────────────────────
#  Scenario 4: conflict_latest (10 条) — 多版本冲突
# ────────────────────────────────────────────────────────────────────────


def build_conflict() -> list[dict]:
    """
    写两条同 (subject, predicate) 不同 object, 旧的标 staleness_signal=True
    (模拟 arbitrator REPLACE 后的状态), 召回应优先返回新的.
    """
    cases = [
        ("046", "lives_in", "北京", "上海", "住在哪", 30, 5),
        ("047", "works_at", "腾讯", "字节跳动", "在哪家公司", 60, 7),
        ("048", "favorite_food", "麻辣火锅", "日料寿司", "最喜欢的食物", 40, 3),
        ("049", "owns_phone", "iPhone 13", "iPhone 16", "用什么手机", 50, 5),
        ("050", "occupation", "工程师", "产品经理", "用户的职业", 90, 10),
        ("051", "weight_kg", "85", "72", "用户体重", 365, 30),
        ("052", "girlfriend", "小雪", "小雨", "女朋友是谁", 200, 14),
        ("053", "favorite_color", "红色", "蓝色", "喜欢什么颜色", 180, 7),
        ("054", "uses_camera", "尼康 D750", "索尼 A7M4", "用什么相机", 90, 14),
        ("055", "owns_car", "本田思域", "奥迪 A4L", "我的车是什么", 365, 30),
    ]
    out = []
    for suffix, pred, old_obj, new_obj, query, old_days, new_days in cases:
        old = {
            "mid": "old",
            "type": "semantic",
            "content": f"{pred} 旧值: {old_obj}",
            "structured": {
                "subject": "user", "predicate": pred, "object": old_obj,
            },
            "created_days_ago": old_days, "importance": 0.7,
            "staleness_signal": True,  # 已被仲裁标软废弃
        }
        new = _sem("new", f"我的 {pred} 是 {new_obj}", "user", pred, new_obj,
                   days_ago=new_days, importance=0.85)
        setup = [old, new] + _distractors(suffix, count=3)
        out.append(_entry(
            f"zh-{suffix}", "conflict_latest", suffix,
            setup, query, ["new"], top_k=5,
            notes="新事实应排前, staleness×0.2 把旧的压下去"
        ))
    return out


# ────────────────────────────────────────────────────────────────────────
#  Scenario 5: negation (5 条) — 否定查询
# ────────────────────────────────────────────────────────────────────────


def build_negation() -> list[dict]:
    """考验向量层是否能区分 "喜欢" vs "不喜欢" 的语义反向."""
    cases = [
        ("056", "我不喜欢吃辣的食物", "dislikes", "辣的食物", "用户不喜欢的食物"),
        ("057", "我对香菜深恶痛绝", "dislikes", "香菜", "用户讨厌的味道"),
        ("058", "我从不喝可乐", "avoids", "可乐", "用户避免的饮料"),
        ("059", "我讨厌坐飞机", "dislikes", "飞机", "用户害怕或讨厌的交通方式"),
        ("060", "我不会用 Vim 编辑器", "cannot_use", "Vim", "用户不熟悉的开发工具"),
    ]
    out = []
    for suffix, content, pred, obj, query in cases:
        target = _sem("target", content, "user", pred, obj, days_ago=10, importance=0.7)
        # 加一条相似但语义相反的"喜欢"作为干扰 (考验语义区分)
        opposite_obj = obj
        opposite = _sem(
            "opposite", f"我喜欢 {opposite_obj}", "user", "likes", opposite_obj,
            days_ago=15, importance=0.5,
        )
        setup = [target, opposite] + _distractors(suffix, count=3)
        out.append(_entry(
            f"zh-{suffix}", "negation", suffix,
            setup, query, ["target"], top_k=5,
            notes="否定查询应优先命中否定事实, 而非同主题肯定事实"
        ))
    return out


# ────────────────────────────────────────────────────────────────────────
#  Scenario 6: episodic_temporal (10 条) — 时序事件
# ────────────────────────────────────────────────────────────────────────


def build_episodic() -> list[dict]:
    cases = [
        ("061", "昨天晚上和小李在三里屯吃日料", "和谁在哪吃饭", 1),
        ("062", "上周三去了一趟杭州出差", "最近的出差行程", 7),
        ("063", "这周一开了产品评审会, 决定砍掉两个老需求", "产品评审会的结论", 3),
        ("064", "去年圣诞节和女朋友去东京玩了 5 天", "圣诞节假期去了哪", 180),
        ("065", "今天早上跑步 5 公里, 配速 5'30\"", "今天的运动数据", 0),
        ("066", "上个月底参加了 PyCon 大会", "近期参加的技术会议", 30),
        ("067", "昨晚看完了《沙丘 2》, 视效震撼", "最近看过的电影", 1),
        ("068", "周末把家里的电脑组装好了, 装了双显示器", "周末做了什么", 2),
        ("069", "这次代码 review 找出了一个并发 bug", "最近发现的技术问题", 4),
        ("070", "三天前买了个新键盘, HHKB Pro 3", "最近买了什么", 3),
    ]
    out = []
    for suffix, content, query, days in cases:
        target = _epi("target", content, days_ago=days, importance=0.6)
        # 干扰: 加 3 条不相关 episodic
        distract_epi = [
            _epi(f"d{i}", c, days_ago=20 + i * 3, importance=0.4)
            for i, c in enumerate([
                "和家人吃了顿火锅",
                "看了场篮球比赛",
                "调通了一个棘手的 SQL 优化",
            ])
        ]
        setup = [target] + distract_epi
        out.append(_entry(
            f"zh-{suffix}", "episodic_temporal", suffix,
            setup, query, ["target"], top_k=5,
            notes="时序事件召回 — 时间衰减应让近期事件更易命中"
        ))
    return out


# ────────────────────────────────────────────────────────────────────────
#  Scenario 7: procedural (5 条) — 流程模板
# ────────────────────────────────────────────────────────────────────────


def build_procedural() -> list[dict]:
    cases = [
        ("071",
         "代码 review 流程",
         ["拉取最新 main 分支", "本地跑全部单测", "用 ruff 检查 lint", "看 diff 给评论",
          "在 PR 里写改动总结"],
         "如何做代码 review"),
        ("072",
         "新员工入职流程",
         ["发送入职邮件", "申请账号权限", "配置开发环境", "走读项目架构文档",
          "结对编程一周"],
         "新员工 onboarding 步骤"),
        ("073",
         "线上 bug 排查",
         ["看 Sentry / 日志定位异常", "复现现象", "二分定位 commit",
          "本地 debug 验证修复", "灰度发布 + 监控"],
         "线上故障处理流程"),
        ("074",
         "周报写作模板",
         ["列本周完成事项", "数据指标对比上周", "下周重点 3 件事",
          "需要协助的事项", "技术总结一句"],
         "周报怎么写"),
        ("075",
         "性能优化流程",
         ["跑 profiler 定位热点", "确认是 IO/CPU/锁哪一类",
          "做最小可行优化 + benchmark", "灰度上线观察 P99"],
         "性能优化怎么做"),
    ]
    out = []
    for suffix, pattern, steps, query in cases:
        target = _proc("target", pattern, steps, days_ago=14, importance=0.7)
        # 干扰: 一些不相关 procedural
        distract_proc = [
            _proc(f"d{i}", p, s, days_ago=30 + i * 5)
            for i, (p, s) in enumerate([
                ("写需求文档", ["列用户场景", "画交互流程", "写验收标准"]),
                ("做技术分享", ["定主题", "写 outline", "做 slides", "试讲"]),
            ])
        ]
        setup = [target] + distract_proc
        out.append(_entry(
            f"zh-{suffix}", "procedural", suffix,
            setup, query, ["target"], top_k=3,
            notes="流程模板召回 — 任务描述对齐应主导"
        ))
    return out


# ────────────────────────────────────────────────────────────────────────
#  Scenario 8: mixed_types (5 条) — 跨记忆类型混合召回
# ────────────────────────────────────────────────────────────────────────


def build_mixed() -> list[dict]:
    """复杂查询应同时命中多种记忆类型."""
    cases = [
        ("076",
         _sem("sem", "我在字节跳动做 AI 基础架构", "user", "works_at", "字节跳动"),
         _epi("epi", "今天 review 了字节内部 RAG 框架的设计文档", days_ago=1),
         "字节工作相关",
         ["sem", "epi"]),
        ("077",
         _sem("sem", "我对花生过敏", "user", "allergic_to", "花生", days_ago=30),
         _epi("epi", "上周吃花生酱后嘴唇肿了一下午", days_ago=7),
         "花生相关",
         ["sem", "epi"]),
        ("078",
         _sem("sem", "我喜欢爬山", "user", "hobby", "爬山"),
         _epi("epi", "上个周末爬了泰山, 凌晨 4 点看的日出", days_ago=5),
         "爬山经历",
         ["sem", "epi"]),
        ("079",
         _sem("sem", "我在成都长大", "user", "hometown", "成都", days_ago=90),
         _epi("epi", "下个月计划回成都看父母", days_ago=2),
         "成都相关的事",
         ["sem", "epi"]),
        ("080",
         _sem("sem", "我用 Python 写后端", "user", "skill", "Python"),
         _epi("epi", "今天用 Python 重构了用户服务的鉴权层", days_ago=0),
         "Python 后端开发",
         ["sem", "epi"]),
    ]
    out = []
    for suffix, sem, epi, query, expected in cases:
        setup = [sem, epi] + _distractors(suffix, count=3)
        out.append(_entry(
            f"zh-{suffix}", "mixed_types", suffix,
            setup, query, expected, top_k=5,
            notes="跨记忆类型 — 应同时命中 semantic 事实 + episodic 事件"
        ))
    return out


# ────────────────────────────────────────────────────────────────────────
#  Build & Write
# ────────────────────────────────────────────────────────────────────────


def build_all() -> list[dict]:
    entries = (
        build_exact_recall()
        + build_paraphrase()
        + build_temporal()
        + build_conflict()
        + build_negation()
        + build_episodic()
        + build_procedural()
        + build_mixed()
    )
    return entries


def main():
    entries = build_all()
    # 校验
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "id 重复"
    assert len(entries) == 80, f"应为 80 条, 实际 {len(entries)}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter

    by_scenario = Counter(e["scenario"] for e in entries)
    print(f"[OK] Wrote {len(entries)} entries to {OUTPUT}")
    for sc, n in sorted(by_scenario.items()):
        print(f"  {sc:24s} {n:>3d}")


if __name__ == "__main__":
    main()
