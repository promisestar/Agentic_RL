# EcomRLVE-Gym

Team LAMBDA submission for Hackathon 2026

Verifiable adaptive-difficulty e-commerce conversation environments for RL training.

## Requirements

- [uv](https://docs.astral.sh/uv/)（推荐）
- Python **≥ 3.13**（由 uv 根据 `.python-version` 自动拉取）

## Setup

```bash
# 安装依赖并创建 .venv（默认包含 dev 组：pytest / ruff）
uv sync

# 若还要跑 Gradio Demo
uv sync --extra space
```

常用命令：

```bash
# 在项目虚拟环境中运行脚本
uv run python scripts/train.py --collection C1 --episodes 10
uv run python scripts/run_debug.py smoke-test

# 交互式 Demo
uv run --extra space python space/app.py

# 增删依赖（会更新 pyproject.toml 与 uv.lock）
uv add <package>
uv add --dev <package>
uv remove <package>
```

`space/requirements.txt` 仍保留，供 Hugging Face Spaces 等仍依赖 `requirements.txt` 的部署场景使用；本地开发请优先使用 `uv`。
