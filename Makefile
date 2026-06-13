.PHONY: help install test test-cov eval eval-cn lint format clean mcp

help:
	@echo "MemoCortex Makefile (MCP-Only)"
	@echo ""
	@echo "  install      uv 安装所有依赖(含 dev + eval)"
	@echo "  mcp          启动 MCP Server"
	@echo "  test         运行单元测试"
	@echo "  test-cov     运行测试 + 覆盖率"
	@echo "  eval         跑全套 eval(中文场景 + LongMemEval 子集)"
	@echo "  eval-cn      只跑中文冲突仲裁场景"
	@echo "  lint         ruff + mypy 静态检查"
	@echo "  format       black + ruff 自动格式化"
	@echo "  clean        清理运行时数据(谨慎)"

install:
	uv sync --all-extras

mcp:
	uv run python -m mcp_server.server

test:
	uv run pytest tests/unit -v

test-cov:
	uv run pytest tests/unit --cov=app --cov-report=term-missing --cov-report=html

eval:
	uv run python -m tests.eval.runner

eval-cn:
	uv run python -m tests.eval.runner --suite cn_scenarios

lint:
	uv run ruff check app mcp_server
	uv run mypy app --ignore-missing-imports || true

format:
	uv run black app mcp_server
	uv run ruff check --fix app mcp_server

clean:
	rm -rf data/chroma data/graph data/cold data/memocortex.db data/*.db-* logs/
	@echo "运行时数据已清理"