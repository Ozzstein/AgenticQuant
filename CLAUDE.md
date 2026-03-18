# CLAUDE.md — AI Quant Stack

## What This Is
An open-source agentic AI investment platform combining quantitative finance with LLM intelligence for algorithmic trading across US equities and crypto.

## Tech Stack
- **Layer 1 (Quant Core):** Microsoft Qlib, RD-Agent, LightGBM/CatBoost/XGBoost
- **Layer 2 (Intelligence):** LangGraph (orchestration), LangChain (tools), Anthropic Claude (primary LLM), Pydantic (structured outputs)
- **Layer 3 (Execution):** Custom paper trader, QuantConnect LEAN bridge, CCXT (crypto)
- **Data:** yfinance, FinnHub, SEC EDGAR, CoinGecko, CCXT
- **Infra:** Python 3.11+, Typer CLI, loguru, pytest, Docker, Streamlit, Telegram

## Architecture

```
PERIODIC (weekly/monthly):
  RD-Agent → discovers factors → data/factor_library/
  RD-Agent → optimizes model  → config/best_model_config.yaml

DAILY PIPELINE:
  Market Data (equity + crypto)
    → LangGraph Agent Pipeline:
        Market Context
        → Macro Regime Detector (risk_on/neutral/risk_off/crisis)
          → Strategy Selector (regime-based/performance-weighted/bandit)
            → Qlib Features (using selected strategy's factor set)
              → ML Model (using selected strategy's model)
                → Predictions (ranked alpha scores)
                  → Top-N candidates (using selected strategy's entry rules)
                    → Parallel Analysts (fundamental, sentiment, technical)
                    → Risk Manager (veto power)
                    → Debate Loop (configurable rounds)
                    → Portfolio Strategist
                    → AnalysisResult (Pydantic)
                  → Portfolio Optimizer (using selected strategy's sizing method)
                    → Pre-Trade Risk Controls
                      → Order Execution (paper / Alpaca / CCXT)
                        → Strategy Performance Tracker (per-strategy P&L attribution)
                          → Reporting (Telegram + Dashboard)
```

## Key Schema

AnalysisResult (Pydantic):
- ticker: str
- decision: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
- confidence: float (0-1)
- target_price: float | None
- stop_loss: float | None
- position_size_pct: float
- time_horizon: str
- reasoning: str
- agent_reports: dict[str, str]
- risk_flags: list[str]
- catalysts: list[str]

## Code Conventions
- Python 3.11+: use `X | Y` unions, match statements, type hints on all signatures
- `pathlib.Path` over `os.path`
- Google-style docstrings on all public functions
- `loguru` for logging, never `print()` for operational output
- `ruff` for linting, line length 100
- All external API calls: retry 3x with exponential backoff
- LangChain tools: return < 2500 chars, catch all exceptions, return "TOOL_ERROR: {msg}" on failure
- Tests: pytest, no network/API calls in unit tests, use mocks

## Project Structure
```
QuantAgentLab/
├── CLAUDE.md
├── plan.md                 # Full build plan with phase status
├── pyproject.toml          # deps + aiquant CLI entry point
├── Dockerfile / docker-compose.yml
├── config/
│   ├── settings.yaml       # Default config (overridden by .env)
│   └── .env.example        # API key template
├── src/
│   ├── __init__.py         # version = "0.1.0"
│   ├── utils/
│   │   ├── config.py       # AppConfig(BaseSettings) + nested models, get_config() singleton
│   │   ├── logger.py       # Loguru: console INFO + file rotation (10MB/7d)
│   │   ├── exceptions.py   # AiQuantError hierarchy (DataPipelineError, AgentError, etc.)
│   │   ├── schemas.py      # ALL shared Pydantic v2 models (Decision, AgentReport, AnalysisResult, etc.)
│   │   └── cli.py          # Typer CLI: setup, backtest, analyze, trade, pipeline, monitor, optimize
│   ├── data/
│   │   ├── market_data.py      # @tool: get_stock_price, get_stock_history, get_options_data
│   │   ├── news_data.py        # @tool: get_company_news, get_market_news
│   │   ├── fundamental_data.py # @tool: get_financials, get_insider_trades, get_analyst_ratings, get_sec_filings
│   │   ├── earnings_calendar.py# @tool: get_earnings_calendar, get_earnings_estimate
│   │   ├── alternative_data.py # @tool: get_fear_greed_index, get_vix, get_sector_performance
│   │   └── crypto_data.py      # @tool: get_crypto_price, get_crypto_history, get_crypto_fundamentals, etc.
│   ├── core/
│   │   ├── data_pipeline.py    # DataPipeline: Qlib init, features, yfinance fallback
│   │   ├── crypto_pipeline.py  # CryptoPipeline: CCXT OHLCV, universe builder, Qlib format
│   │   ├── factor_engine.py    # FactorEngine: Alpha158 + custom numpy factors (RSI/MACD/BB/ATR/OBV/...)
│   │   ├── model_zoo.py        # ModelWrapper: LightGBM/CatBoost/XGBoost/Linear registry
│   │   ├── backtester.py       # WalkForwardBacktester + BacktestValidator (8 checks) + HTML report
│   │   ├── benchmark.py        # BenchmarkRunner: SPY/BTC/60-40/EW/Monte Carlo
│   │   └── rd_agent_runner.py  # RDAgentRunner: graceful stub for factor/model search
│   ├── agents/
│   │   ├── state.py        # TradingDeskState TypedDict with Annotated reducers
│   │   ├── llm.py          # get_llm() factory: anthropic/openai/ollama
│   │   ├── fundamental.py  # CFA analyst node
│   │   ├── sentiment.py    # Alt-data analyst node
│   │   ├── technical.py    # CMT analyst node
│   │   ├── risk_manager.py # CRO node with veto power
│   │   ├── strategist.py   # PM synthesis node → AnalysisResult
│   │   ├── graph.py        # build_trading_desk_graph(): parallel fan-out + debate + veto
│   │   └── debate.py       # DebateManager: multi-round agent debate
│   ├── execution/
│   │   ├── portfolio_optimizer.py  # risk_parity / mean_variance / black_litterman / HRP
│   │   ├── paper_trader.py         # FIFO avg-cost, slippage/commission, metrics
│   │   ├── risk_controls.py        # 6 pre-trade checks → RiskBreachError on violation
│   │   ├── signal_translator.py    # signals → target weights → orders (idempotent)
│   │   └── lean_bridge.py          # LEAN algorithm file generator
│   ├── strategies/
│   │   ├── base.py          # BaseStrategy ABC
│   │   ├── qlib_ml.py       # QlibMLStrategy: model scores → top-K signals
│   │   ├── agent_enhanced.py# AgentEnhancedStrategy: Qlib + LangGraph sizing
│   │   └── allocator.py     # MetaAllocator: rolling Sharpe-weighted capital allocation
│   └── monitoring/
│       ├── telegram_bot.py      # TelegramNotifier: alerts, daily report, trade notifications
│       ├── portfolio_monitor.py # PortfolioMonitor: health score (0-100), P&L attribution
│       └── scheduler.py         # PipelineScheduler (APScheduler): daily pipeline with lock file
├── scripts/
│   ├── run_backtest.py   # run / compare / validate
│   ├── run_agents.py     # analyze / batch / debate
│   └── run_pipeline.py   # run / paper-loop / setup
├── lean/
│   └── QuantConnectBridge.py   # LEAN AIQuantStackAlgorithm template
├── tests/
│   ├── conftest.py         # fixtures: mock_config, sample_ohlcv_df, mock_llm, sample_portfolio
│   ├── test_utils.py       # config, logger, schemas, exceptions (30 tests ✅)
│   ├── test_data.py
│   ├── test_crypto_data.py
│   ├── test_core_data.py
│   ├── test_core_factors.py
│   ├── test_core_models.py
│   ├── test_core_backtest.py
│   ├── test_core_benchmark.py
│   ├── test_agents.py
│   ├── test_graph.py
│   ├── test_execution.py
│   ├── test_strategies.py
│   ├── test_monitoring.py
│   └── test_integration.py  # @pytest.mark.integration
├── docs/
│   └── DOCUMENTATION.md
├── logs/                   # auto-created: aiquant_YYYY-MM-DD.log
└── outputs/                # auto-created: backtest results, paper portfolio, factor library
```

## Common Commands

```bash
# Install / setup
conda activate aiquant
pip install -e ".[dev]"
aiquant setup                        # initialize directories

# Testing
conda run -n aiquant python -m pytest tests/ -v                        # all unit tests
conda run -n aiquant python -m pytest tests/test_utils.py -v           # Phase 1 (30 tests ✅)
conda run -n aiquant python -m pytest tests/ -v -m "not integration"   # skip integration
conda run -n aiquant python -m pytest tests/test_integration.py -v -m integration  # with APIs

# CLI
aiquant analyze AAPL                 # multi-agent analysis
aiquant backtest --model LightGBM    # walk-forward backtest
aiquant trade status                 # paper portfolio status
aiquant pipeline run                 # run daily pipeline once
aiquant monitor                      # portfolio health check

# Scripts
conda run -n aiquant python scripts/run_backtest.py run --model LightGBM --topk 30
conda run -n aiquant python scripts/run_agents.py analyze AAPL
conda run -n aiquant python scripts/run_pipeline.py paper-loop

# Docker
docker build -t aiquant .
docker-compose up -d
```

## Key Outputs
- `outputs/backtest_*.json` — backtest results
- `outputs/paper_portfolio.json` — live paper trading state
- `outputs/factor_library.json` — RD-Agent discovered factors
- `outputs/best_model_config.yaml` — RD-Agent optimized model config
- `logs/aiquant_YYYY-MM-DD.log` — rotating log files

## Environment
- conda env: `aiquant` (Python 3.11)
- Config: `config/settings.yaml` + `config/.env` (API keys)
- `.env` files are blocked from direct edits — update `config/.env.example` instead

## Build Status
See `plan.md` for phase-by-phase status.
- Phase 1 (Foundation): ✅ Complete — 30/30 tests passing

## Risk Limits (Defaults)
- Max single equity position: 10%
- Max single crypto position: 5%
- Max sector concentration: 30%
- Total crypto allocation: 5-25%
- Min cash buffer: 5%
- Drawdown circuit breaker: -15%
- Daily turnover limit: 25%
- Long-only, no leverage
