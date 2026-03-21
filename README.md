# QuantAgentLab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## What It Is

**QuantAgentLab** is an open-source agentic AI investment platform that combines Microsoft Qlib's
quantitative finance engine with LangGraph-orchestrated LLM intelligence. It covers US equities
and crypto, running a daily pipeline that moves from raw market data all the way through ML-scored
alpha signals, multi-agent debate, and portfolio construction to order execution — all without
requiring manual intervention once configured.

The platform is built on the principle that pure-quant and pure-LLM approaches each have blind
spots that the other compensates for. Qlib's walk-forward backtester and Alpha158 feature library
produce statistically rigorous rank signals; LangGraph agents then apply qualitative judgment
(macro regime, earnings catalysts, sentiment, risk veto) to size and filter those signals before
any capital is committed. The result is a system that is simultaneously auditable (every agent
decision is logged in JSON) and adaptive (LLM reasoning can incorporate breaking news that a
fixed-rule model cannot).

---

## Architecture

```
╔══════════════════════════════════════════════════════════════╗
║  PERIODIC  (weekly / monthly)                                ║
║                                                              ║
║   RD-Agent ──► discovers factors ──► data/factor_library/   ║
║   RD-Agent ──► optimizes model   ──► config/best_model_config.yaml ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  DAILY PIPELINE                                              ║
║                                                              ║
║  Market Data (equity + crypto)                               ║
║    └─► Qlib Features (Alpha158 + RD-Agent factor library)    ║
║          └─► ML Model (RD-Agent-optimized config if avail.)  ║
║                └─► Predictions (ranked alpha scores)         ║
║                      └─► Top-N candidates selected          ║
║                            └─► LangGraph Agent Pipeline:    ║
║                                  Market Context              ║
║                                  → Macro Regime Detector     ║
║                                    (risk_on/neutral/         ║
║                                     risk_off/crisis)         ║
║                                  → Parallel Analysts:        ║
║                                    · Fundamental (CFA)       ║
║                                    · Sentiment (alt-data)    ║
║                                    · Technical (CMT)         ║
║                                  → Risk Manager (veto)       ║
║                                  → Debate Loop (N rounds)    ║
║                                  → Portfolio Strategist      ║
║                                  → AnalysisResult (Pydantic) ║
║                            └─► Portfolio Optimizer           ║
║                                  (risk parity / BL / HRP)   ║
║                                  └─► Pre-Trade Risk Controls ║
║                                        └─► Order Execution   ║
║                                              (paper / LEAN)  ║
║                                              └─► Reporting   ║
║                                                    (Telegram ║
║                                                    + Dashboard) ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Project Structure

```
QuantAgentLab/
├── CLAUDE.md                    # AI assistant instructions
├── plan.md                      # Phase-by-phase build plan
├── pyproject.toml               # deps + aiquant CLI entry point
├── Dockerfile / docker-compose.yml
├── config/
│   ├── settings.yaml            # Default config
│   └── .env.example             # API key template
├── src/
│   ├── utils/
│   │   ├── config.py            # AppConfig(BaseSettings) singleton
│   │   ├── logger.py            # Loguru console + rotating file
│   │   ├── exceptions.py        # AiQuantError hierarchy
│   │   ├── schemas.py           # ALL shared Pydantic v2 models
│   │   └── cli.py               # Typer CLI entry point
│   ├── data/
│   │   ├── market_data.py       # yfinance tools
│   │   ├── fundamental_data.py  # FinnHub + SEC EDGAR tools
│   │   ├── news_data.py         # news tools
│   │   ├── crypto_data.py       # CoinGecko + CCXT tools
│   │   └── unified_data.py      # unified equity+crypto feed
│   ├── core/
│   │   ├── data_pipeline.py     # Qlib init, features, fallback
│   │   ├── crypto_pipeline.py   # CryptoPipeline OHLCV
│   │   ├── factor_engine.py     # Alpha158 + custom factors
│   │   ├── model_zoo.py         # LGB/CatBoost/XGB/Linear
│   │   ├── backtester.py        # WalkForwardBacktester
│   │   ├── benchmark.py         # SPY/BTC/60-40/EW benchmarks
│   │   └── rd_agent_runner.py   # RD-Agent factor/model search (9 modes)
│   ├── agents/
│   │   ├── state.py             # TradingDeskState TypedDict
│   │   ├── llm.py               # get_llm() factory
│   │   ├── fundamental.py       # CFA analyst node
│   │   ├── sentiment.py         # Alt-data analyst node
│   │   ├── technical.py         # CMT analyst node
│   │   ├── risk_manager.py      # CRO node (veto power)
│   │   ├── strategist.py        # PM synthesis → AnalysisResult
│   │   ├── graph.py             # build_trading_desk_graph()
│   │   └── debate.py            # DebateManager multi-round
│   ├── execution/
│   │   ├── portfolio_optimizer.py
│   │   ├── paper_trader.py
│   │   ├── risk_controls.py
│   │   ├── signal_translator.py
│   │   └── lean_bridge.py
│   ├── strategies/
│   │   ├── qlib_ml.py
│   │   ├── agent_enhanced.py
│   │   └── allocator.py
│   ├── monitoring/
│   │   ├── telegram_bot.py
│   │   ├── portfolio_monitor.py
│   │   └── scheduler.py
│   ├── dashboard.py             # Streamlit trading dashboard (portfolio, agents, factors)
│   └── rd_agent_dashboard.py    # Streamlit RD-Agent research dashboard (KB, factor library)
├── scripts/
│   ├── run_backtest.py
│   ├── run_agents.py
│   ├── run_pipeline.py
│   └── run_rd_agent.py          # RD-Agent CLI (9 commands)
├── tests/                       # pytest suite
├── notebooks/                   # Jupyter exploration notebooks
└── docs/                        # Extended documentation
```

---

## Getting Started

### Prerequisites

- [conda](https://docs.conda.io/en/latest/miniconda.html) or `mamba`
- API keys for Anthropic Claude (required), FinnHub (optional), Telegram (optional)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/youruser/QuantAgentLab
cd QuantAgentLab

# 2. Create the conda environment
conda create -n aiquant python=3.11 -y
conda activate aiquant

# 3. Install all dependencies (including dev extras)
pip install -e ".[dev]"

# 4. Initialise directories and verify the setup
aiquant setup

# 5. Run your first multi-agent analysis
aiquant analyze AAPL
```

The `aiquant setup` command creates `logs/`, `outputs/`, and `data/` directories and validates
that required config values are present.

---

## Configuration Reference

Copy `config/.env.example` to `config/.env` and fill in your keys. All values can also be
supplied as environment variables with the `AIQUANT_` prefix.

| Variable | Default | Required | Description |
|---|---|---|---|
| `AIQUANT_ANTHROPIC_API_KEY` | — | **Yes** | Claude API key (primary LLM) |
| `AIQUANT_OPENAI_API_KEY` | — | No | OpenAI fallback LLM |
| `AIQUANT_FINNHUB_KEY` | — | No | FinnHub market data (free tier) |
| `AIQUANT_TELEGRAM_TOKEN` | — | No | Telegram bot token for alerts |
| `AIQUANT_TELEGRAM_CHAT_ID` | — | No | Telegram chat/channel ID |
| `AIQUANT_ALPACA_KEY` | — | No | Alpaca paper/live trading key |
| `AIQUANT_ALPACA_SECRET` | — | No | Alpaca secret |
| `AIQUANT_COINGECKO_API_KEY` | — | No | CoinGecko Pro key (rate limits) |
| `AIQUANT_LLM_PROVIDER` | `anthropic` | No | `anthropic`, `openai`, or `ollama` |
| `AIQUANT_LLM_MODEL` | `claude-sonnet-4-6` | No | Model name for chosen provider |
| `AIQUANT_LLM_TEMPERATURE` | `0.1` | No | LLM temperature (lower = more deterministic) |
| `AIQUANT_DEBATE_ROUNDS` | `2` | No | Agent debate iterations |
| `AIQUANT_TOP_K_STOCKS` | `20` | No | Qlib candidates passed to agents |
| `AIQUANT_PAPER_INITIAL_CASH` | `100000` | No | Paper trader starting cash (USD) |
| `AIQUANT_MAX_POSITION_PCT` | `0.10` | No | Max single equity position |
| `AIQUANT_MAX_CRYPTO_PCT` | `0.05` | No | Max single crypto position |
| `AIQUANT_DRAWDOWN_LIMIT` | `0.15` | No | Portfolio circuit-breaker threshold |
| `AIQUANT_DAILY_BUDGET_USD` | `2.0` | No | LLM cost cap per day |
| `AIQUANT_LOG_LEVEL` | `INFO` | No | `DEBUG`, `INFO`, `WARNING` |
| `AIQUANT_QLIB_PROVIDER_URI` | `~/.qlib/qlib_data/cn_data` | No | Qlib data directory |

---

## CLI Reference

### `aiquant` subcommands

```bash
aiquant setup                         # Initialise dirs, validate config
aiquant analyze TICKER [--debate N]   # Full multi-agent analysis for one ticker
aiquant backtest [--model MODEL]      # Walk-forward backtest
aiquant trade status                  # Print current paper portfolio
aiquant trade reset                   # Reset paper portfolio to initial cash
aiquant pipeline run                  # Run daily pipeline once
aiquant pipeline paper-loop           # Run daily pipeline on schedule
aiquant monitor                       # Print portfolio health score
aiquant optimize                      # Run RD-Agent factor/model search
```

### `scripts/` usage

```bash
# Backtesting
conda run -n aiquant python scripts/run_backtest.py run --model LightGBM --topk 30
conda run -n aiquant python scripts/run_backtest.py run --with-rd-factors   # merge RD-Agent factors
conda run -n aiquant python scripts/run_backtest.py compare --models LightGBM,CatBoost
conda run -n aiquant python scripts/run_backtest.py validate --model LightGBM

# Agent analysis
conda run -n aiquant python scripts/run_agents.py analyze AAPL
conda run -n aiquant python scripts/run_agents.py batch --tickers AAPL,MSFT,GOOGL
conda run -n aiquant python scripts/run_agents.py debate TSLA --rounds 3

# Pipeline (idempotent — second run same day is a no-op)
conda run -n aiquant python scripts/run_pipeline.py run
conda run -n aiquant python scripts/run_pipeline.py paper-loop
conda run -n aiquant python scripts/run_pipeline.py setup

# RD-Agent research (factor/model/strategy discovery)
conda run -n aiquant python scripts/run_rd_agent.py library-status
conda run -n aiquant python scripts/run_rd_agent.py co-optimize --iterations 10
conda run -n aiquant python scripts/run_rd_agent.py mine-factors --iterations 20 --min-ic 0.02
conda run -n aiquant python scripts/run_rd_agent.py optimize-model --iterations 10
conda run -n aiquant python scripts/run_rd_agent.py multi-trace --traces 3 --iterations 5
conda run -n aiquant python scripts/run_rd_agent.py implement-paper ./papers/my_paper.txt
conda run -n aiquant python scripts/run_rd_agent.py evolve-strategies --iterations 10
conda run -n aiquant python scripts/run_rd_agent.py evolve-regime --iterations 10
conda run -n aiquant python scripts/run_rd_agent.py copilot-factor "momentum with rising earnings"
conda run -n aiquant python scripts/run_rd_agent.py copilot-strategy "mean reversion on earnings misses"
conda run -n aiquant python scripts/run_rd_agent.py copilot-model --source ./papers/model.txt
conda run -n aiquant python scripts/run_rd_agent.py ui --port 8080   # Streamlit research dashboard
```

---

## How the Agents Work

The LangGraph agent graph is assembled in `src/agents/graph.py` via `build_trading_desk_graph()`.
Every analysis begins with a shared `TradingDeskState` TypedDict that flows through the graph.

### Macro Regime Detector

Reads VIX level, yield-curve slope, and recent SPY return to classify the current environment
as one of four regimes: `risk_on`, `neutral`, `risk_off`, or `crisis`. All downstream agents
receive this classification and adjust their conviction thresholds accordingly.

### Fundamental Analyst (CFA persona)

Uses `get_financials`, `get_insider_trades`, `get_analyst_ratings`, and `get_sec_filings` tools
to assess balance-sheet quality, earnings trajectory, and management credibility. Outputs a
structured `AgentReport` with a verdict and supporting evidence.

### Sentiment Analyst (alt-data persona)

Combines `get_company_news`, `get_fear_greed_index`, and social-data proxies to gauge crowd
positioning. Specifically watches for divergence between price action and sentiment — extreme
pessimism in an uptrend is often a contrarian buy signal.

### Technical Analyst (CMT persona)

Computes RSI, MACD, Bollinger Bands, ATR, and OBV from `get_stock_history` data. Identifies
support/resistance levels and trend phase. Acts as an independent signal that can confirm or
reject the fundamental thesis.

### Debate Loop

`DebateManager` in `src/agents/debate.py` runs `N` configurable rounds (default 2) where each
analyst can rebut the others' reports. This surfaces genuine disagreement rather than groupthink
and often changes the final conviction score by ±0.1–0.2.

### Risk Manager (CRO / veto power)

Receives the consolidated analyst reports and the current portfolio state. Applies six pre-trade
checks (see Risk Management section). If any hard limit is breached the Risk Manager emits a
`VETO` that halts the position. No other node can override this veto.

### Portfolio Strategist (PM synthesis)

Aggregates all analyst reports, the debate transcript, and the risk verdict into a final
`AnalysisResult` Pydantic model with `decision`, `confidence`, `target_price`, `stop_loss`,
`position_size_pct`, `time_horizon`, and `reasoning` fields.

---

## Risk Management Framework

Risk controls are enforced at two layers: the Risk Manager agent (qualitative) and
`src/execution/risk_controls.py` (quantitative, hard limits).

| Limit | Default | Notes |
|---|---|---|
| Max single equity position | 10% | Hard limit; veto if exceeded |
| Max single crypto position | 5% | Crypto is higher volatility |
| Max sector concentration | 30% | GICS sector classification |
| Total crypto allocation | 5% – 25% | Range; adjusts with macro regime |
| Min cash buffer | 5% | Always keep 5% uninvested |
| Drawdown circuit breaker | −15% | Halts all new trades until reset |
| Daily turnover limit | 25% | Prevents over-trading |
| Leverage | Long-only | No shorting, no margin |

When the drawdown circuit breaker fires, the scheduler stops submitting new orders until an
operator manually confirms the state via `aiquant trade reset` or the portfolio recovers above
the threshold.

---

## Success Criteria Checklist

These are the target milestones for production readiness (TASK-07):

- [ ] Walk-forward Sharpe ratio > 1.0 on held-out test period
- [ ] Backtest validation suite returns zero FAIL verdicts
- [ ] Out-of-sample return beats random stock picker at 95th percentile
- [ ] Positive Sharpe in at least 2 of 4 macro regimes (risk_on, neutral, risk_off, crisis)
- [ ] 3+ consecutive months of paper trading with full audit trail
- [ ] Maximum portfolio drawdown remains below 15% during paper trading
- [ ] Agent reasoning manually reviewed for 20+ unique tickers
- [ ] 30+ consecutive daily pipeline runs without unhandled exceptions
- [ ] All monitoring channels working (Telegram alerts, Streamlit dashboard, log rotation)
- [ ] Total LLM API cost stays below €2/day average over rolling 30 days

---

## Expansion Roadmap

The following capabilities are planned but not yet implemented:

- **Short-selling** — integrate borrowing cost model; add SHORT signal to AnalysisResult
- **Options strategies** — use `get_options_data` tool already wired; implement delta-neutral sizing
- **Live trading via Alpaca** — the LEAN bridge supports it; requires Alpaca account verification
- **Additional data sources** — Quandl macro series, earnings call transcript NLP, satellite data
- **Multi-asset allocation** — extend MetaAllocator to commodities and fixed income ETFs
- **On-premise LLM** — Ollama provider is already wired; test with Llama-3 70B for cost reduction
- **Web3 on-chain data** — DeFi protocol TVL, whale wallet tracking via Dune Analytics

---

## Risk Disclaimer

**This software is for educational and research purposes only. It is not financial advice and
should not be used to make real investment decisions without independent professional guidance.**

Algorithmic trading involves significant risk of capital loss. Past simulated performance does
not guarantee future results. The authors and contributors accept no liability for financial
losses incurred through use of this software. Always consult a licensed financial professional
before deploying capital.

---

## License

MIT License — see [LICENSE](LICENSE) for full text.

Copyright (c) 2024 QuantAgentLab Contributors
