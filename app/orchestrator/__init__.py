"""编排层入口

import arbitrator 触发它的副作用 — 把 ConflictArbitrator 注入 SemanticMemory.
否则 SemanticMemory 拿不到 arbitrator, 冲突时只能走启发式 fallback (且不写审计日志).
"""

from app import arbitrator as _arbitrator  # noqa: F401 — 副作用 import
from app.orchestrator.graph import orchestrator

__all__ = ["orchestrator"]
