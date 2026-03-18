# TASK-01: Project Scaffolding, Config, CLI

## Goal
Set up the full project skeleton so every subsequent task has a working foundation to import from.

## What to Build

### 1. pyproject.toml
All dependencies for the full project:
- Qlib: pyqlib, lightgbm, catboost, xgboost, torch, scikit-learn
- Agents: langgraph, langchain, langchain-anthropic, langchain-openai, anthropic, openai
- Execution: quantconnect-stubs, ccxt, cvxpy, pypfopt
- Data: yfinance, finnhub-python, sec-edgar-downloader, feedparser, newspaper3k
- Infra: pydantic, pydantic-settings, python-dotenv, pyyaml, typer, rich, loguru, httpx, plotly
- Dev: pytest, pytest-asyncio, pytest-cov, ruff, mypy, jupyterlab
- Optional: finrl, rdagent, python-telegram-bot, streamlit, transformers, peft

### 2. Configuration System
File: `src/utils/config_loader.py`

Pydantic BaseSettings models for the entire project. Must load from YAML + .env + environment variables.

Models needed:
- **QlibConfig**: provider_uri, region, date ranges (train/valid/test start+end), universe, model name+params, strategy params (topk, n_drop), backtest params (account, costs)
- **RDAgentConfig**: enabled, factor_library_dir, best_model_config_path, factor evolution params (iterations, min_ic, min_icir, budget), model evolution params, schedule
- **AgentConfig**: llm_provider (anthropic/openai/ollama), deep_think_model, quick_think_model, max_debate_rounds, temperature, per-agent enable flags, cost controls (max per analysis, cache TTL)
- **LeanConfig**: initial_cash, broker type, commission, slippage, rebalance frequency
- **RiskConfig**: all risk limits from CLAUDE.md
- **AppConfig**: combines all above + log_level + data_dir

### 3. Logger
File: `src/utils/logger.py`

Loguru-based: console with rich formatting, file rotation (10MB, 7 days), structured JSON option.

### 4. CLI Skeleton
File: `src/utils/cli.py`

Typer app with subcommands (initially just help text, implementation in later tasks):
- `aiquant setup` — data download
- `aiquant backtest` — run Qlib backtest
- `aiquant analyze TICKER` — run agent analysis
- `aiquant pipeline` — full pipeline
- `aiquant rd-agent mine-factors` / `optimize-model` / `library-status` / `validate-library`

### 5. Config Files
- `config/.env.example` — all API key placeholders
- `config/qlib_config.yaml` — full Qlib + RD-Agent config
- `config/agents_config.yaml` — LLM provider, models, agent settings, LangGraph workflow settings
- `config/lean_config.json` — execution and risk settings

### 6. Package Structure
All `__init__.py` files for: src, src/core, src/agents, src/execution, src/data, src/utils, tests, scripts

## Inputs
Nothing — this is the foundation.

## Outputs
- Installable package: `pip install -e ".[dev]"` succeeds
- Config loads: `python -c "from src.utils.config_loader import AppConfig; print(AppConfig())"` works
- Logger works: `python -c "from src.utils.logger import get_logger; get_logger('test').info('ok')"` works
- CLI renders: `python -m src.utils.cli --help` shows all subcommands

## Done Criteria
- [x] `pip install -e ".[dev]"` completes without errors
- [x] All config YAML/JSON files parse without errors
- [x] AppConfig loads with defaults when no .env exists
- [x] CLI help text shows all subcommands
- [x] Logger writes to console and file
- [x] `ruff check src/` passes clean
