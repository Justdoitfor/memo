"""Entity Resolution — 实体消解系统 (P0)

判断 "小明" / "张小明" / "Mr. Zhang" 是否是同一个人,
通过向量相似 + LLM 判断实现消解, 合并相同实体的别名和三元组.
"""

from app.entity.resolver import EntityResolver, entity_resolver

# 自动注入到 SemanticMemory (打破循环依赖)
from app.memories.semantic import semantic_memory
semantic_memory.set_entity_resolver(entity_resolver)

__all__ = ["EntityResolver", "entity_resolver"]
