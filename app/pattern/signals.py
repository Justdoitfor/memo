"""信号收集 helper — 业务方通过 MCP track_signal 工具或 REST 调用."""

from __future__ import annotations

from loguru import logger

from app.models import SignalType
from app.storage import get_metadata


async def track_signal(
    user_id: str,
    signal_type: SignalType | str,
    context_tags: list[str] | None = None,
    memory_ids_in_context: list[str] | None = None,
    session_id: str | None = None,
    extra: dict | None = None,
) -> int:
    """记录一条行为信号 (异步, 不阻塞主流程).

    Args:
        signal_type: SignalType enum 或字符串 (regenerate_request / explicit_correction
            / format_preference / tool_selection / positive_feedback / topic_pivot)
        context_tags: 当时上下文标签 (e.g. ['code_review', 'python'])
        memory_ids_in_context: 本次召回涉及的记忆 ID 列表
        session_id: 可选会话 ID
        extra: 任意补充信息 (e.g. {"reason": "too verbose"})

    Returns:
        signal_id (DB 自增 ID)
    """
    st = signal_type if isinstance(signal_type, str) else signal_type.value
    # 验证合法性
    valid = {s.value for s in SignalType}
    if st not in valid:
        logger.warning(f"非法 signal_type '{st}', 跳过")
        return -1

    meta = get_metadata()
    sid = await meta.add_signal(
        user_id=user_id,
        signal_type=st,
        context_tags=context_tags,
        memory_ids_in_context=memory_ids_in_context,
        session_id=session_id,
        extra=extra,
    )
    logger.debug(f"[Signal] {st} user={user_id} tags={context_tags} → id={sid}")
    return sid
