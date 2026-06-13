"""Pattern Miner — 隐式行为模式挖掘 (MemoryMesh §5.3)

后台周期任务: 聚合 behavior_signals → LLM 归纳 → 生成 Implicit Memory.

挖掘流程 (参考 Honcho dialectic pattern inference):
  1. 按 (user_id, signal_type, context_tags) 分组聚合最近 N 条信号
  2. 频率阈值过滤: 同模式 >= MIN_OCCURRENCES 次才视为有意义
  3. LLM 归纳: 信号序列 → 自然语言 Implicit 偏好
       例: 5 次 regenerate_request in [code_review] context
         → "用户在代码 review 场景偏好简短直接的反馈"
  4. 以 source_type='inferred' (权重 0.60), type=IMPLICIT 写入 memory
  5. 在前端 /patterns Tab 可单独查看

与 Reflective Memory 关系:
  - Reflective = 显式用户画像 (从 Semantic facts 聚合) "用户住北京, 对花生过敏"
  - Implicit  = 隐式行为偏好 (从行为 signals 挖掘) "用户在代码场景喜欢简短回答"
  - 两者互补, 共同注入 Agent SystemPrompt
"""

from app.pattern.miner import mine_patterns_for_user
from app.pattern.signals import track_signal

__all__ = ["track_signal", "mine_patterns_for_user"]
