# AI Quant Stack — Full Build Plan

## Status
- ✅ Phase 1: Foundation (30/30 tests passing)
- ⬜ Phase 2A: Equity Data Tools
- ⬜ Phase 2B: Crypto Data Tools
- ⬜ Phase 3A: Data Pipeline + Factor Engine
- ⬜ Phase 3B: Model Zoo + Walk-Forward Backtester + Validation + Benchmarks
- ⬜ Phase 4A: State + LLM Factory + Individual Agent Nodes
- ⬜ Phase 4B: Graph + Debate + Earnings Logic
- ⬜ Phase 5: Execution Layer
- ⬜ Phase 6: Multi-Strategy Framework
- ⬜ Phase 7A: Monitoring
- ⬜ Phase 7B: Scheduler + Scripts
- ⬜ Phase 7C: Docker
- ⬜ Phase 8: Integration, Documentation, Polish

## Context

The QuantAgentLab project directory had only `.claude/` metadata. Phase 1 was completed building the foundation:
- `pyproject.toml` — all deps, aiquant CLI, ruff/pytest config
- `src/utils/config.py` — AppConfig with nested models, singleton get_config()
- `src/utils/logger.py` — Loguru console + file rotation
- `src/utils/exceptions.py` — AiQuantError hierarchy
- `src/utils/schemas.py` — ALL Pydantic v2 models
- `src/utils/cli.py` — Typer CLI (stubs)
- `config/settings.yaml` + `config/.env.example`
- `tests/conftest.py` + `tests/test_utils.py` (30 tests)

**Architecture**: LangGraph + LangChain + Claude (not CrewAI)

**Environment**: conda env `aiquant` with Python 3.11
**Run tests**: `conda run -n aiquant python -m pytest tests/ -v`

---

## Task 1: Phase 2A — Equity Data Tools

### Files to create
- `src/data/market_data.py` — @tool: get_stock_price, get_stock_history, get_options_data (yfinance)
- `src/data/news_data.py` — @tool: get_company_news, get_market_news (FinnHub + RSS fallback)
- `src/data/fundamental_data.py` — @tool: get_financials, get_insider_trades, get_analyst_ratings, get_sec_filings
- `src/data/earnings_calendar.py` — @tool: get_earnings_calendar, get_earnings_estimate
- `src/data/alternative_data.py` — @tool: get_fear_greed_index, get_vix, get_sector_performance
- `tests/test_data.py` — all tools tested with mocked HTTP

### Patterns
- Use `from langchain_core.tools import tool` for @tool decorator
- Each tool returns STRING (agents consume text)
- Use `tenacity.retry` with `stop_after_attempt(3), wait_exponential(min=1, max=10)`
- On error: `return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"`
- Keep output under 2500 chars
- Check `config.finnhub_api_key` before using FinnHub API
- Import: `from src.utils.config import get_config`, `from src.utils.logger import get_logger`

### Verify
```bash
conda run -n aiquant python -m pytest tests/test_data.py -v
```

---

## Task 2: Phase 2B — Crypto Data Tools

### Files to create
- `src/data/crypto_data.py` — @tool: get_crypto_price, get_crypto_history, get_crypto_fundamentals, get_crypto_fear_greed, get_crypto_on_chain (CCXT + CoinGecko)
- `src/core/crypto_pipeline.py` — CryptoPipeline class: fetch_ohlcv, build_universe, fetch_universe_data, to_qlib_format
- `tests/test_crypto_data.py` — mocked CCXT/requests tests

### Patterns
- Same @tool patterns as 2A
- CCXT symbol: "BTC" → "BTC/USDT"
- CoinGecko free API (no key for basic): https://api.coingecko.com/api/v3/
- CoinGecko symbol mapping dict (BTC→bitcoin, ETH→ethereum, top 20)
- Crypto Fear & Greed: https://api.alternative.me/fng/

### Verify
```bash
conda run -n aiquant python -m pytest tests/test_crypto_data.py -v
```

---

## Task 3: Phase 3A — Data Pipeline + Factor Engine

### Files to create
- `src/core/data_pipeline.py` — DataPipeline: init_qlib(), get_dataset(), get_features(), yfinance_fallback(), merge_crypto_data()
- `src/core/factor_engine.py` — FactorEngine: builtin Qlib Alpha158 + custom numpy factors (RSI/MACD/BB/ATR/OBV/VWAP/momentum/vol/Garman-Klass/Parkinson), compute_factors(), evaluate_factors(), load/save factor library
- `src/core/rd_agent_runner.py` — RDAgentRunner: graceful ImportError stub, run_factor_search(), run_model_search(), saves outputs/factor_library.json + outputs/best_model_config.yaml
- `tests/test_core_data.py` — DataPipeline tests with mocked Qlib
- `tests/test_core_factors.py` — FactorEngine tests with synthetic OHLCV

### Factor implementations (pure numpy, no TA-Lib):
- RSI(14), MACD(12/26/9), Bollinger(20/2σ), ATR(14), OBV, VWAP
- Momentum: 1M/3M/6M/12M, Realized vol: 20d/60d
- Garman-Klass vol, Parkinson vol

### Verify
```bash
conda run -n aiquant python -m pytest tests/test_core_data.py tests/test_core_factors.py -v
```

---

## Task 4: Phase 3B — Model Zoo + Backtester + Validator + Benchmarks

### Files to create
- `src/core/model_zoo.py` — ModelWrapper: registry (LightGBM/CatBoost/XGBoost/Linear), lazy importlib, train/predict/save/load/get_feature_importance, load_rd_agent_config()
- `src/core/backtester.py` — WalkForwardBacktester (expanding window, per-fold retrain, top-K portfolio, transaction costs, returns BacktestResult) + BacktestValidator (8 checks) + generate_html_report() with Plotly
- `src/core/benchmark.py` — BenchmarkRunner: SPY buy-hold, BTC buy-hold, 60/40, equal-weight, Monte Carlo (1000 random portfolios, percentile rank)
- `tests/test_core_models.py`
- `tests/test_core_backtest.py`
- `tests/test_core_benchmark.py`

### BacktestValidator 8 checks:
1. Look-ahead bias (feature timestamps vs prediction dates)
2. Survivorship bias
3. Fill realism (order_size/daily_volume < 1%)
4. Cost sensitivity (0/5/10/20/50 bps, flag if Sharpe drops >50% from 0→10bps)
5. Concentration risk (max weight, HHI, position count)
6. Regime dependence (bull/bear via SPY 200d SMA, high/low vol via VIX)
7. Statistical significance (t-stat > 2.0, min 252 trading days)
8. Overfitting signals (OOS < 50% IS Sharpe → flag, Sharpe > 3.0 → flag)

### Verify
```bash
conda run -n aiquant python -m pytest tests/test_core_*.py -v
```

---

## Task 5: Phase 4A — Agent State + LLM Factory + Individual Nodes

### Files to create
- `src/agents/state.py` — TradingDeskState TypedDict with Annotated reducers
- `src/agents/llm.py` — get_llm() factory: anthropic→ChatAnthropic, openai→ChatOpenAI, ollama→ChatOllama
- `src/agents/fundamental.py` — CFA analyst node (DCF, comps, ROIC/FCF/debt). Tools: get_financials, get_sec_filings, get_analyst_ratings, get_insider_trades. Crypto: get_crypto_fundamentals
- `src/agents/sentiment.py` — Alt-data analyst node. Tools: get_company_news, get_market_news, get_fear_greed_index, get_earnings_calendar. Crypto: get_crypto_fear_greed
- `src/agents/technical.py` — CMT analyst node (RSI/MACD/BB/SMA/ATR/OBV). Tools: get_stock_history/get_crypto_history
- `src/agents/risk_manager.py` — CRO node with VETO power: Kelly sizing, VaR/CVaR, veto triggers (position > max weight, DD > threshold, earnings within 3 days + leveraged, liquidity insufficient, aggregate flags > 3)
- `src/agents/strategist.py` — PM synthesis: weighted combo (fundamental 35%, quant 25%, technical 20%, sentiment 10%, risk 10%), returns AnalysisResult
- `tests/test_agents.py` — mock ChatAnthropic, test each node independently

### State TypedDict fields:
- Input: ticker, asset_class, qlib_score
- Agents: agent_reports: Annotated[list[AgentReport], add]
- Debate: debate_round, debate_history, consensus_reached
- Risk: risk_flags, risk_veto (bool), risk_veto_reason
- Output: final_decision (AnalysisResult | None)
- Meta: errors (list[str])

### Verify
```bash
conda run -n aiquant python -m pytest tests/test_agents.py -v
```

---

## Task 6: Phase 4B — Graph + Debate

### Files to create
- `src/agents/graph.py` — build_trading_desk_graph(config) -> CompiledGraph
  - START → parallel: [fundamental, sentiment, technical] → risk_manager
  - Conditional: veto → END (HOLD), debate_needed → debate → strategist, else → strategist
  - Error resilience: node failure → append to errors, continue
  - Earnings awareness: if earnings within 5 days, inject flag
- `src/agents/debate.py` — DebateManager: each agent sees others' reports, revises, max N rounds, convergence when all within 20% confidence spread
- `tests/test_graph.py` — full graph execution, veto path, debate loop, error resilience, earnings flag

### Graph flow:
```
START ──┬── fundamental ──┐
        ├── sentiment   ──┼── risk_manager ──┬── [veto] → END
        └── technical   ──┘                  ├── [debate needed] → debate ─┐
                                             └── [synthesize] → strategist ←┘ → END
```

### Verify
```bash
conda run -n aiquant python -m pytest tests/test_agents.py tests/test_graph.py -v
```

---

## Task 7: Phase 5 — Execution Layer

### Files to create
- `src/execution/portfolio_optimizer.py` — PortfolioOptimizer: 4 methods (risk_parity, mean_variance with Ledoit-Wolf, black_litterman, HRP). Hard constraints: max 10% equity, 5% crypto, total crypto 5-25%, equity 50-90%, min 5% cash, long-only. from_agent_views() converts AnalysisResult confidence → BL views
- `src/execution/paper_trader.py` — PaperTrader: configurable slippage/commission bps, FIFO avg-cost, buy/sell, equity+crypto (24/7), metrics (Sharpe/drawdown/win-rate/Calmar), persists to outputs/paper_portfolio.json
- `src/execution/risk_controls.py` — PreTradeRiskControls: 6 checks (cash sufficiency, position concentration, liquidity >$1M ADV, market impact <1% volume, drawdown halt at -15%, daily turnover <25%). Violation → order rejected
- `src/execution/signal_translator.py` — SignalTranslator: signals → target weights → diff vs current portfolio → rebalancing orders (sells first, buys second). Idempotent: same signals + same portfolio = no new orders
- `src/execution/lean_bridge.py` — LeanBridge: generate LEAN algorithm file, signal file format
- `lean/QuantConnectBridge.py` — LEAN AIQuantStackAlgorithm template
- `tests/test_execution.py` — optimizer constraints, paper trader cycles, risk controls, idempotency

### Verify
```bash
conda run -n aiquant python -m pytest tests/test_execution.py -v
```

---

## Task 8: Phase 6 — Multi-Strategy Framework

### Files to create
- `src/strategies/base.py` — BaseStrategy ABC: name, generate_signals(date), get_performance_history(), get_metadata()
- `src/strategies/qlib_ml.py` — QlibMLStrategy: model scores → top-K → equal/score-weight signals (no agents)
- `src/strategies/agent_enhanced.py` — AgentEnhancedStrategy: Qlib top-K → LangGraph analysis → agent conviction sizes positions
- `src/strategies/allocator.py` — MetaAllocator: rolling 63d Sharpe-weighted / inverse-vol / risk-parity, diversification penalty for correlated strategies, floor 10%/cap 50%, rebalance on >5% drift
- `tests/test_strategies.py`

### Verify
```bash
conda run -n aiquant python -m pytest tests/test_strategies.py -v
```

---

## Task 9: Phase 7A — Monitoring

### Files to create
- `src/monitoring/telegram_bot.py` — TelegramNotifier: send_alert(), send_daily_report(), send_trade_notification(), send_risk_alert(). No-op if token not configured
- `src/monitoring/portfolio_monitor.py` — PortfolioMonitor: health score (0-100), concentration check, drawdown status, daily turnover, P&L attribution. Triggers Telegram on threshold breaches
- `tests/test_monitoring.py`

### Verify
```bash
conda run -n aiquant python -m pytest tests/test_monitoring.py -v
```

---

## Task 10: Phase 7B+C — Scheduler + Scripts + Docker

### Files to create
- `src/monitoring/scheduler.py` — PipelineScheduler (APScheduler): daily at 16:00 ET, lock file prevents concurrent runs, steps: fetch→retrain→predict→agent→optimize→risk→execute→report→notify
- `scripts/run_backtest.py` — run / compare / validate subcommands
- `scripts/run_agents.py` — analyze / batch / debate subcommands
- `scripts/run_pipeline.py` — run / paper-loop / setup subcommands
- `Dockerfile` — multi-stage python:3.11-slim, system deps, pip install
- `docker-compose.yml` — app + scheduler services, volumes: outputs/, logs/, config/

---

## Task 11: Phase 8 — Integration + Documentation + Claude Metadata Updates

### Files to create/update
- `tests/test_integration.py` — @pytest.mark.integration full pipeline tests
- Update `src/utils/cli.py` — wire stub commands to real implementations
- `README.md` — architecture diagram, quickstart, config reference, CLI reference
- `docs/DOCUMENTATION.md` — all three layers, data sources, methodology, deployment
- Update `.claude/agents/backtest-qa.md` — fix module paths to match new structure
- Update `.claude/agents/portfolio-monitor.md` — fix module paths
- Update `.claude/skills/*/SKILL.md` — update tool references
- Create `CLAUDE.md` — LangGraph architecture, commands, file structure

---

## Dependency Graph

```
Phase 1 (DONE)
  ├── Task 1: Phase 2A Equity Data
  └── Task 2: Phase 2B Crypto Data
        ├── Task 3: Phase 3A Data Pipeline + Factors
        │     └── Task 4: Phase 3B Models + Backtest
        ├── Task 5: Phase 4A Agent Nodes
        │     └── Task 6: Phase 4B Graph + Debate
        └── Task 7: Phase 5 Execution
              ├── Task 8: Phase 6 Strategies
              └── Task 9: Phase 7A Monitoring
                    └── Task 10: Phase 7B+C Scheduler + Docker
                          └── Task 11: Phase 8 Integration + Docs
```

## Critical Design Decisions

| Decision | Rationale |
|----------|-----------|
| LangGraph over CrewAI | Explicit state, conditional edges, parallel fan-out, structured output |
| Walk-forward backtesting | Simulates production, measures true OOS performance |
| riskfolio-lib | Battle-tested, CVXPY-based, all 4 methods native |
| CCXT for crypto | 100+ exchanges, uniform API, built-in rate limiting |
| Idempotent pipeline | Signal translator diffs portfolio; scheduler uses lock file |
| Factor library as JSON | RD-Agent writes factors; daily pipeline auto-loads |
