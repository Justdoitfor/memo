"""Procedural Memory — 任务模板与解决方法

特征:
  - "如何做某类任务"的可复用模式
  - 内容含: task_pattern (描述) + steps (步骤列表) + success_rate + last_used
  - 召回基于任务描述向量匹配
  - 按使用频率衰减 (recall_count 触发 importance ↑)
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger

from app.models import MemoryRecord, MemoryType
from app.storage import get_metadata, get_vector_store


class ProceduralMemory:
    """程序性记忆 — 任务模板存储."""

    def __init__(self) -> None:
        self._vector = get_vector_store()
        self._meta = get_metadata()
        logger.info("ProceduralMemory 初始化")

    async def write(
        self,
        user_id: str,
        task_pattern: str,
        steps: list[str],
        success_rate: float = 1.0,
        tags: list[str] | None = None,
    ) -> str:
        """注册一个任务模板."""
        content = f"任务模式: {task_pattern}\n步骤:\n" + "\n".join(
            f"  {i + 1}. {s}" for i, s in enumerate(steps)
        )
        record = MemoryRecord(
            user_id=user_id,
            type=MemoryType.PROCEDURAL,
            content=content,
            structured={
                "task_pattern": task_pattern,
                "steps": steps,
                "success_rate": success_rate,
            },
            importance=0.6,
            tags=tags or [],
        )
        await self._vector.add(record)
        await self._meta.upsert_memory(record)
        return record.id

    async def search(
        self, user_id: str, task_description: str, top_k: int = 5
    ) -> list[tuple[MemoryRecord, float]]:
        """根据任务描述召回历史可复用模板."""
        results = await self._vector.search(
            user_id=user_id,
            query=task_description,
            memory_types=[MemoryType.PROCEDURAL.value],
            top_k=top_k,
        )
        # 召回即"用了一次", recall_count + 1 + importance 微增 (异步, 不阻塞)
        for record, _ in results:
            new_count = record.recall_count + 1
            new_imp = min(1.0, record.importance + 0.02)
            await self._vector.update_metadata(
                record.id,
                user_id,
                {"recall_count": new_count, "importance": new_imp},
            )
        return results

    async def update_success_rate(
        self, memory_id: str, user_id: str, new_rate: float
    ) -> bool:
        """更新成功率 (Agent 执行完任务后回传)."""
        record = await self._meta.get_memory(memory_id)
        if not record or record.user_id != user_id:
            return False
        struct = dict(record.structured)
        struct["success_rate"] = new_rate
        struct["last_used"] = datetime.now().isoformat()
        record.structured = struct
        await self._meta.upsert_memory(record)
        return True


procedural_memory = ProceduralMemory()
