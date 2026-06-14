"""Prompt loader 单元测试 — 版本化加载 / cache / 错误处理."""
from __future__ import annotations

import pytest

from app.core.prompt_loader import get_prompt_meta, load_prompt


class TestLoadPrompt:
    def test_load_arbitrator_v1(self):
        prompt = load_prompt("arbitrator", "v1")
        # 应该是 ChatPromptTemplate, 包含 system + human 2 个 message
        rendered = prompt.format_messages(
            user_id="u",
            field_semantics="unique",
            existing_facts="x",
            subject="user",
            predicate="lives_in",
            new_object="上海",
            confidence="0.9",
        )
        assert len(rendered) == 2
        # system message 包含 4 种 action 关键词
        sys_content = rendered[0].content
        assert "REPLACE" in sys_content
        assert "MERGE" in sys_content
        assert "VERSIONED" in sys_content
        assert "IGNORE" in sys_content
        # human message 应包含输入变量替换后的内容
        human_content = rendered[1].content
        assert "user" in human_content
        assert "lives_in" in human_content
        assert "上海" in human_content

    def test_load_missing_version(self):
        with pytest.raises(FileNotFoundError, match="prompt"):
            load_prompt("arbitrator", "v999")

    def test_load_missing_name(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt_xyz", "v1")

    def test_lru_cache(self):
        """同 (name, version) 应命中 lru_cache, 返回同一实例."""
        p1 = load_prompt("arbitrator", "v1")
        p2 = load_prompt("arbitrator", "v1")
        assert p1 is p2


class TestGetPromptMeta:
    def test_arbitrator_v1_meta(self):
        meta = get_prompt_meta("arbitrator", "v1")
        # v1.yaml 里 meta 应有 name / version / description
        assert meta.get("name") == "arbitrator"
        assert meta.get("version") == "v1"
        assert "description" in meta

    def test_missing_returns_empty_dict(self):
        meta = get_prompt_meta("nonexistent_prompt", "v1")
        assert meta == {}
