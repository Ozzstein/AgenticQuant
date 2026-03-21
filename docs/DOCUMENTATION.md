# QuantAgentLab — Technical Documentation

> Version 0.1.0 | Python 3.11+ | Last updated: 2026-03-21

---

## 1. Overview & Philosophy

QuantAgentLab sits at the intersection of two research traditions that rarely talk to each other:
systematic quantitative finance (Qlib, factor models, walk-forward validation) and large-language-
model reasoning (LangGraph orchestration, Claude Sonnet, structured Pydantic outputs).

**Pure-quant** systems are rigorous about avoiding look-ahead bias and measuring information
coefficients, but they are blind to narrative — an earnings restatement, a CEO change, or a
macro policy shift that alters the distribution of returns. **Pure-LLM** systems are expressive
and can reason about context, but they have no principled way to rank 5,000 stocks, no
statistical framework for evaluating their own historical accuracy, and no guard against
hallucinating financial data.

QuantAgentLab resolves this tension with a two-stage architecture:

1. **Stage 1 — Quantitative screen.** Qlib's Alpha158 features and ML models (LightGBM,
   CatBoost, XGBoost, Linear) rank the full universe by predicted alpha. Only the top-K
   candidates (default 20) advance to the agent layer. This keeps LLM cost bounded and focuses
   agent reasoning on genuinely interesting names.

2. **Stage 2 — Agent enrichment.** LangGraph orchestrates a structured deliberation among
   specialist analyst personas (CFA, CMT, alt-data), a debate loop, and a risk manager with
   veto power. The final `AnalysisResult` Pydantic model carries not just a `decision` but a
   full `reasoning` string, `agent_reports` dict, `risk_flags`, and `catalysts` — everything
   needed for an audit trail.

The platform is deliberately long-only, US-equities plus crypto, and paper-trading first. Real
money execution is possible via the LEAN bridge or Alpaca integration but is gated behind
explicit configuration.

---

## 2. Layer 1: Qlib Core

### Data Pipeline (`src/core/data_pipeline.py`)

`DataPipeline` initialises Qlib with a configurable `provider_uri` (default
`~/.qlib/qlib_data/cn_data`; US data via `qlib.data.dataset`). It exposes:

- `get_features(instruments, start, end)` — returns a multi-index DataFrame of Alpha158 features
  (158 price/volume/fundamental signals computed from OHLCV). Falls back to yfinance-derived
  features when Qlib data is unavailable.
- `load_factor_library()` — loads RD-Agent-discovered factors from
  `outputs/factor_library/factor_library.json` and merges them into the feature matrix.
  Pass `--with-rd-factors` to `run_backtest.py` to include these in the feature matrix.
- `create_dataset_with_custom_factors()` — convenience wrapper used by the daily pipeline.

### Alpha158 Feature Library (`src/core/factor_engine.py`)

`FactorEngine` computes the official Alpha158 set plus a custom extension layer:

| Custom Factor | Formula | Intuition |
|---|---|---|
| RSI-14 | Wilder smoothed avg gain / avg loss | Momentum mean-reversion signal |
| MACD Signal | EMA(12) − EMA(26), signal EMA(9) | Trend change detection |
| Bollinger %B | (price − lower) / (upper − lower) | Volatility-adjusted position |
| ATR-14 | Average True Range | Volatility sizing input |
| OBV | Cumulative volume × sign(return) | Accumulation/distribution |

### Model Zoo (`src/core/model_zoo.py`)

`ModelWrapper` provides a unified `.fit(X, y)` / `.predict(X)` interface over four backends:

- **LightGBM** — default; fastest training, handles missing values natively
- **CatBoost** — best on heterogeneous tabular data; slower but often highest IC
- **XGBoost** — strong baseline; good when regularisation is needed
- **Linear** — ridge regression; interpretable baseline

The `best_model_config.yaml` written by RD-Agent is loaded at runtime to override default
hyperparameters without code changes.

### Walk-Forward Backtester (`src/core/backtester.py`)

`WalkForwardBacktester` avoids look-ahead bias by never training on data that would not have
been available at prediction time. Key parameters:

- `train_window` — rolling training window in calendar days (default 252)
- `test_window` — out-of-sample test period per fold (default 63, one quarter)
- `embargo_days` — gap between train end and test start to prevent leakage (default 5)
- `n_splits` — number of folds (auto-computed from data length)

`BacktestValidator` applies 8 validation checks to each completed backtest. See Section 7 for
the full list.

---

## 3. Layer 2: LangGraph Agents

### State TypedDict (`src/agents/state.py`)

`TradingDeskState` is the shared state object that flows through the LangGraph graph. It uses
`Annotated` reducers (list append, dict merge) to safely accumulate partial updates from
parallel nodes:

```
TradingDeskState:
  ticker: str
  macro_regime: str | None          # risk_on / neutral / risk_off / crisis
  fundamental_report: AgentReport | None
  sentiment_report: AgentReport | None
  technical_report: AgentReport | None
  risk_verdict: str | None          # APPROVE / VETO
  debate_transcript: list[str]      # one entry per round per agent
  analysis_result: AnalysisResult | None
  risk_flags: list[str]
  messages: list[BaseMessage]
```

### Node: Macro Regime Detector (`src/agents/macro_regime.py`)

Reads VIX (via `get_vix`), 10Y-2Y yield spread (via `get_market_news` + parsed macro data),
and trailing 20-day SPY return to classify regime. Downstream agents receive the regime
classification and adjust their conviction thresholds:

- `risk_on` — agents are more willing to recommend BUY
- `neutral` — standard thresholds apply
- `risk_off` — agents require higher confidence before recommending BUY
- `crisis` — Risk Manager applies additional position size reduction

### Node: Fundamental Analyst (`src/agents/fundamental.py`)

CFA persona. Uses four LangChain tools: `get_financials`, `get_insider_trades`,
`get_analyst_ratings`, `get_sec_filings`. Assesses:

- Revenue and earnings trend (3-year CAGR)
- Balance sheet health (debt/equity, current ratio)
- Insider buying/selling signal
- Consensus analyst target vs current price
- Recent 8-K/10-K filings for material events

### Node: Sentiment Analyst (`src/agents/sentiment.py`)

Alt-data persona. Uses `get_company_news`, `get_fear_greed_index`, and sector performance
data. Flags:

- News sentiment score (positive/negative/neutral)
- Fear & Greed Index context (extreme fear = contrarian long signal)
- Relative sector strength

### Node: Technical Analyst (`src/agents/technical.py`)

CMT persona. Uses `get_stock_history` and `get_options_data`. Computes:

- Trend phase (uptrend/downtrend/range) via 50/200 SMA crossover
- RSI-14 overbought/oversold
- MACD divergence
- Key support/resistance levels
- Put/call ratio from options data

### Node: Risk Manager (`src/agents/risk_manager.py`)

CRO persona with unilateral veto power. Runs six pre-trade checks against the current portfolio
state and the proposed position. If any check fails, emits `VETO` and sets `risk_flags`. See
Section 8 for the complete check list.

### Node: Portfolio Strategist (`src/agents/strategist.py`)

PM persona. Synthesises all analyst reports and the debate transcript into a final
`AnalysisResult`. Explicitly reconciles disagreements: if fundamental is BUY and technical is
SELL, the strategist must explain the resolution in `reasoning`.

### Debate Loop (`src/agents/debate.py`)

`DebateManager` runs `N` rounds (configurable via `AIQUANT_DEBATE_ROUNDS`, default 2).

Each round: each analyst reads the others' latest reports and may update their verdict. The
full exchange is stored in `debate_transcript` for the audit trail. Debate typically converges
in 2 rounds; more rounds increase LLM cost without proportional information gain.

### AnalysisResult Schema (`src/utils/schemas.py`)

```python
class AnalysisResult(BaseModel):
    ticker: str
    decision: Decision          # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    confidence: float           # 0.0 – 1.0
    target_price: float | None
    stop_loss: float | None
    position_size_pct: float    # fraction of portfolio, 0.0 – 1.0
    time_horizon: str           # e.g. "3-6 months"
    reasoning: str
    agent_reports: dict[str, str]
    risk_flags: list[str]
    catalysts: list[str]
```

---

## 4. Layer 3: Execution

### Paper Trader (`src/execution/paper_trader.py`)

`PaperTrader` implements FIFO average-cost accounting with realistic frictions:

- **Commission** — configurable flat rate (default $1/trade) plus basis-points spread
- **Slippage** — percentage of trade value (default 0.05%)
- **Fill model** — instant fill at last price (conservative; does not simulate order book)

State is persisted to `outputs/paper_portfolio.json` after every trade. Metrics computed:
total return, annualised return, Sharpe ratio, max drawdown, win rate, profit factor.

### Portfolio Optimizer (`src/execution/portfolio_optimizer.py`)

Four optimisation modes selectable per run:

| Mode | Description | Best for |
|---|---|---|
| `risk_parity` | Equal risk contribution per asset | Diversified multi-asset |
| `mean_variance` | Markowitz efficient frontier | When return forecasts are reliable |
| `black_litterman` | BL with agent confidence as views | Blending quant + agent signals |
| `hrp` | Hierarchical Risk Parity (Lopez de Prado) | Robust to estimation error |

`black_litterman` mode maps each agent's `confidence` and `decision` to a BL view vector,
making the portfolio optimizer the natural integration point between agent output and capital
allocation.

### Risk Controls (`src/execution/risk_controls.py`)

Six pre-trade checks run before every order. A failed check raises `RiskBreachError`:

1. **Position size check** — proposed position must not exceed max single-asset limit
2. **Sector concentration check** — sector total must remain below 30%
3. **Crypto allocation check** — total crypto must stay within 5%–25% band
4. **Cash buffer check** — post-trade cash must stay above 5%
5. **Daily turnover check** — cumulative turnover today must stay below 25%
6. **Drawdown circuit breaker** — if portfolio drawdown exceeds −15%, block all new trades

### Signal Translator (`src/execution/signal_translator.py`)

Converts `AnalysisResult` objects into portfolio weight targets and then into concrete orders.
The translation is idempotent: running it twice with the same input produces the same orders,
preventing duplicate trades if the scheduler retries.

### LEAN Bridge (`src/execution/lean_bridge.py`)

Generates a valid QuantConnect LEAN Python algorithm file (`lean/QuantConnectBridge.py`) from
the current strategy configuration. This allows backtesting under LEAN's institutional-grade
cost model (bid/ask spreads, borrowing costs, margin) with zero code changes.

---

## 5. Data Sources

| Source | Access | Cost | Used for |
|---|---|---|---|
| yfinance | pip package | Free | OHLCV, options chains, basic fundamentals |
| FinnHub | REST API | Free tier (60 req/min) | Company news, analyst ratings, insider trades |
| SEC EDGAR | REST API | Free | 10-K, 8-K, 13-F filings |
| CoinGecko | REST API | Free (30 req/min) | Crypto prices, market caps |
| CCXT | pip package | Free | Crypto OHLCV via exchange APIs |
| Anthropic | REST API | Pay-per-token | LLM reasoning (primary) |
| OpenAI | REST API | Pay-per-token | LLM reasoning (fallback) |

All external calls use the project-standard retry wrapper: 3 attempts with exponential backoff
(1s, 2s, 4s). Every tool function returns `"TOOL_ERROR: {msg}"` on unrecoverable failure so
the LangGraph graph can continue rather than crashing.

---

## 6. Backtest Methodology

### Why Walk-Forward Matters

A standard train/test split optimises hyperparameters on the full training set and evaluates
on a fixed holdout. This has two failure modes common in finance:

1. **Temporal leakage** — if the split is not strictly chronological, future information leaks
   into training features.
2. **Regime mismatch** — a single holdout period may be unrepresentative of the full
   distribution of market regimes.

Walk-forward evaluation addresses both by rolling the training window forward in time, testing
on unseen quarters, and inserting an embargo gap to prevent leakage from high-autocorrelation
features.

### Single-Split vs Walk-Forward

| Metric | Single Split | Walk-Forward |
|---|---|---|
| Number of test periods | 1 | N (typically 8–12) |
| Look-ahead bias risk | High | Low (embargo enforced) |
| Regime coverage | Low | High |
| Computational cost | Low | N × higher |
| Statistical confidence | Low | High |

For this project, walk-forward is the **required** evaluation method. Single-split results are
reported for reference but are not used for go/no-go decisions.

### Embargo Days

The 5-day default embargo prevents the model from learning from features that overlap with the
test period. For example, a 20-day rolling mean computed on day T includes days T−19 through T.
Without an embargo, the last 20 days of training data would share information with the first
days of the test period.

---

## 7. Walk-Forward Validation Suite

`BacktestValidator` in `src/core/backtester.py` runs 8 checks after every walk-forward
evaluation. Each check returns `RELIABLE` or `NOT RELIABLE` plus a message:

| # | Check | RELIABLE condition |
|---|---|---|
| 1 | **Positive total return** | Total return > 0% |
| 2 | **Sharpe ratio** | Sharpe > 0.5 (target > 1.0) |
| 3 | **IC significance** | Information Coefficient > 0.02 |
| 4 | **Consistency** | Win rate across folds > 55% |
| 5 | **Drawdown control** | Max drawdown < −25% |
| 6 | **Turnover reasonable** | Annual turnover < 500% |
| 7 | **Benchmark beat** | Excess return vs SPY > 0% |
| 8 | **Factor decay** | IC is positive in latest fold (no decay) |

The summary verdict is `PASS` only if all 8 checks return `RELIABLE`. Any `NOT RELIABLE` check
should trigger investigation before deploying the model.

---

## 8. Risk Management Framework

### Pre-Trade Checks (quantitative layer)

These run in `src/execution/risk_controls.py` and raise `RiskBreachError` on failure:

| Check | Limit | Action on breach |
|---|---|---|
| Single equity position | 10% of portfolio | Reduce order to fit limit |
| Single crypto position | 5% of portfolio | Reduce order to fit limit |
| Sector concentration | 30% per GICS sector | Block order |
| Total crypto allocation | 5% – 25% of portfolio | Block order |
| Cash buffer | Min 5% post-trade | Block order |
| Daily turnover | Max 25% per day | Block order |
| Drawdown circuit breaker | −15% from peak | Block ALL new trades |

### Risk Manager Agent (qualitative layer)

The Risk Manager node in the LangGraph graph performs qualitative checks that are not easily
quantified:

- **Earnings proximity** — if earnings are within 5 days, flags event risk
- **Recent large move** — if stock moved >10% in past 5 days, flags momentum risk
- **Low liquidity** — if average daily volume < $5M, flags execution risk
- **High short interest** — if short float > 20%, flags squeeze risk
- **Concentration in correlated names** — if portfolio already holds a highly correlated
  position, flags diversification risk

The Risk Manager emits either `APPROVE` or `VETO`. A `VETO` is logged with the specific
reason and is visible in the Telegram daily report.

### Drawdown Circuit Breaker

When portfolio equity drops 15% below its rolling high-water mark, the circuit breaker engages:

1. All pending orders are cancelled (paper trader) or blocked
2. A Telegram alert is sent immediately
3. The scheduler stops submitting new pipeline runs
4. Recovery requires manual operator action: `aiquant trade reset --confirm`

---

## 9. Monitoring & Alerts

### Telegram Setup

1. Create a bot via `@BotFather` on Telegram, receive `TELEGRAM_TOKEN`
2. Get your chat ID by messaging `@userinfobot`
3. Set `AIQUANT_TELEGRAM_TOKEN` and `AIQUANT_TELEGRAM_CHAT_ID` in `config/.env`
4. Run `aiquant monitor` — you should receive a test message

### Alert Thresholds

| Event | Threshold | Message type |
|---|---|---|
| Portfolio drawdown | −5% | Warning |
| Portfolio drawdown | −10% | Urgent warning |
| Portfolio drawdown | −15% | Circuit breaker alert |
| Agent disagreement | ≥2 analysts disagree with majority | Advisory |
| API error rate | >10% of calls failing | System alert |
| LLM cost | >80% of daily budget | Cost warning |
| Pipeline failure | Any unhandled exception | Critical alert |

### Portfolio Monitor (`src/monitoring/portfolio_monitor.py`)

`PortfolioMonitor` computes a **health score (0–100)** based on:

- Drawdown component (0–40 points): max score at 0% drawdown, zero at −15%
- Diversity component (0–20 points): based on effective N (1/HHI) of positions
- Win rate component (0–20 points): rolling 30-day trade win rate
- Turnover component (0–20 points): inverted daily turnover (lower = healthier)

The health score is included in the daily Telegram report and the Streamlit dashboard.

### Scheduler (`src/monitoring/scheduler.py`)

`PipelineScheduler` uses APScheduler to run the daily pipeline at a configurable time (default
06:30 UTC). A lock file (`outputs/.pipeline.lock`) prevents overlapping runs. If the previous
run is still active when the scheduler fires, the new run is skipped and a warning is logged.

---

## 10. Audit Trail

Every `AnalysisResult` is serialised to JSON and appended to `outputs/analysis_log.jsonl`.
Each line is a complete record that answers the question: **"Why did the system recommend
buying X on date Y?"**

### JSON Schema

```json
{
  "timestamp": "2026-03-18T06:45:12Z",
  "ticker": "AAPL",
  "decision": "BUY",
  "confidence": 0.73,
  "target_price": 215.0,
  "stop_loss": 188.0,
  "position_size_pct": 0.07,
  "time_horizon": "3-6 months",
  "reasoning": "Strong FCF yield with improving margins. Macro regime neutral...",
  "agent_reports": {
    "fundamental": "P/E 27x, below 5yr avg 31x. Services revenue +18% YoY...",
    "sentiment": "Fear & Greed at 42 (Fear). Institutional buying in options...",
    "technical": "RSI 48, not overbought. 200-day SMA support at 185..."
  },
  "risk_flags": [],
  "catalysts": ["WWDC product launch", "Share buyback programme"],
  "macro_regime": "neutral",
  "debate_rounds": 2,
  "qlib_alpha_rank": 3
}
```

To query the audit log:

```bash
# All BUY decisions in the last 7 days
python -c "
import json, datetime
cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
with open('outputs/analysis_log.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['decision'] in ('BUY','STRONG_BUY'):
            print(r['timestamp'], r['ticker'], r['confidence'])
"
```

---

## 11. Dashboard Guide

The platform ships with two separate Streamlit dashboards:

### Trading Dashboard (`src/dashboard.py`)

The main operational dashboard for monitoring live/paper trading. Provides five pages:

| Page | Description |
|---|---|
| **Portfolio Overview** | P&L curve, health score, top positions |
| **Analysis Feed** | Latest AnalysisResult cards per ticker |
| **Agent Debate** | Full debate transcript viewer |
| **Factor Importance** | LightGBM feature importances |
| **Cost Tracker** | Daily LLM API cost vs budget |

```bash
conda run -n aiquant streamlit run src/dashboard.py
# Opens at http://localhost:8501
```

Data refresh rate is 60 seconds (controlled by `st.cache_data(ttl=60)`).

### RD-Agent Research Dashboard (`src/rd_agent_dashboard.py`)

A dedicated dashboard for monitoring the R&D loop results — factor discovery, knowledge base
state, and model optimisation history. Provides three pages:

| Page | Description |
|---|---|
| **Knowledge Base** | Last run date, tested/failed factor counts, discoveries table and IC bar chart |
| **Factor Library** | Full factor library (name, category, IC mean, ICIR, source), IC ranking chart |
| **R&D Loop Results** | Discoveries grouped by date, tested model configs, best model config YAML |

```bash
# Launch via CLI (recommended)
conda run -n aiquant python scripts/run_rd_agent.py ui --port 8080
# Opens at http://localhost:8080

# Or directly
conda run -n aiquant streamlit run src/rd_agent_dashboard.py --server.port 8080
```

Data refresh rate is 30 seconds (controlled by `st.cache_data(ttl=30)`).

For production deployments, serve both dashboards behind an authenticated reverse proxy
(nginx + basic auth) since they display portfolio positions and API cost information.

---

## 12. RD-Agent Research Loop

### What RD-Agent Does

`RDAgentRunner` (`src/core/rd_agent_runner.py`) is a Python-native research loop that
autonomously proposes, evaluates, and refines alpha factors, model hyperparameters, and full
trading strategies. All modes share a persistent knowledge base (KB) that prevents re-testing
known failures and accumulates discoveries across runs.

### All 9 Modes

| Mode | Type | What It Does |
|---|---|---|
| `co-optimize` | Autonomous | Joint factor-model alternation via multi-armed bandit |
| `mine-factors` | Autonomous | Factor evolution only — no model optimisation |
| `optimize-model` | Autonomous | LightGBM hyperparameter random search |
| `evolve-strategies` | Autonomous | Proposes, backtests, and saves full strategy definitions |
| `evolve-regime` | Autonomous | Evolves the HMM regime detector via simulated backtest comparison |
| `implement-paper` | Autonomous | Derives a trading factor from a research paper |
| `copilot-factor` | Interactive | Describe a factor in plain English → builds + evaluates |
| `copilot-strategy` | Interactive | Describe a strategy in plain English → formalizes + backtests |
| `copilot-model` | Interactive | Paper/file → detects model architecture, produces hyperparameter config |

### Co-Optimize Loop

The primary autonomous mode alternates between factor mining and model optimisation:

```
RDAgentRunner.co_optimize(iterations, budget, traces):
  1. Sample factor templates via epsilon-greedy bandit
  2. Simulate IC for each candidate factor
  3. Accept factors above min_ic threshold → append to factor library
  4. Sample random model hyperparameter configs from _PARAM_GRID
  5. Simulate Sharpe for each config
  6. Save best config to outputs/best_model_config.yaml
  7. Update KB (tested_factors, failed_factors, discoveries, tested_configs)
  8. Repeat for N iterations
```

### Copilot Modes (Interactive)

**`copilot-factor`** — natural language to evaluated factor:

```bash
python scripts/run_rd_agent.py copilot-factor "momentum factor with rising earnings"
```

Scans the description for keywords (momentum, rsi, macd, bollinger, atr, obv, volume), matches
to built-in factor templates, evaluates IC, and saves passing factors to the library.

**`copilot-model`** — paper or text file to model config:

```bash
python scripts/run_rd_agent.py copilot-model --source ./papers/my_paper.txt
```

Reads the file, detects model architecture keywords (lightgbm, catboost, xgboost, lstm,
transformer), produces a hyperparameter config, simulates Sharpe, and saves if Sharpe > 0.5.

**`copilot-strategy`** — natural language to strategy definition:

```bash
python scripts/run_rd_agent.py copilot-strategy "mean reversion on earnings misses"
```

### Factor Library

Accepted factors are saved to `outputs/factor_library/factor_library.json`. Each entry stores:
`name`, `expression`, `category`, `ic_mean`, `icir`, `source`, and `description`.

### Knowledge Base Persistence

The KB is stored in `data/rd_knowledge_base/kb.json`. It records:

- `tested_factors` — all factor names ever evaluated
- `failed_factors` — names that failed IC threshold (never re-proposed)
- `tested_configs` — model configs evaluated with their simulated Sharpe
- `discoveries` — accepted factors with date, IC, and category
- `last_run_date` — ISO date of the last successful run

The KB is loaded at the start of every run and saved at the end. Running `co-optimize` twice
with the same budget will yield different results because the second run starts with a populated
KB and explores new directions.

### CLI Quick Reference

```bash
python scripts/run_rd_agent.py library-status          # show KB + library counts
python scripts/run_rd_agent.py validate-library         # 3-way Sharpe comparison
python scripts/run_rd_agent.py co-optimize -i 10        # 10 factor+model iterations
python scripts/run_rd_agent.py mine-factors -i 20       # factor-only, 20 iterations
python scripts/run_rd_agent.py multi-trace -t 3 -i 5    # 3 parallel traces, 5 iters each
python scripts/run_rd_agent.py evolve-strategies -i 10  # evolve full strategies
python scripts/run_rd_agent.py evolve-regime -i 10      # evolve regime detector
python scripts/run_rd_agent.py reset-knowledge          # clear KB (irreversible)
python scripts/run_rd_agent.py ui --port 8080           # launch research dashboard
```

---

## 13. Pipeline Idempotency

The daily pipeline (`scripts/run_pipeline.py`) is idempotent: running it twice on the same
calendar day has the same effect as running it once. This prevents double-trades and duplicate
Telegram alerts when the scheduler restarts after a crash.

### How It Works

At startup, `_run_pipeline_once(mode)` checks `data/pipeline_state/last_run_{mode}.json`:

```json
{
  "date": "2026-03-21",
  "summary": { "tickers_analyzed": 10, "nav": 105230.0, ... }
}
```

If `date` matches today, the cached summary is returned immediately without fetching data or
calling any agents. At the end of a successful run, the state file is written with today's date
and the result summary.

Idempotency is intentionally skipped for `paper-loop` mode (which is a demo loop that runs
repeatedly by design).

### Resetting the Cache

To force a re-run on the same day (e.g., after changing configuration):

```bash
rm data/pipeline_state/last_run_backtest.json
```

Or run in `paper-loop` mode which bypasses the check.

---

## 14. API Cost Management

### Pricing Reference (as of early 2026)

| Model | Input (per MTok) | Output (per MTok) | Typical cost per analysis |
|---|---|---|---|
| claude-sonnet-4-6 | $3.00 | $15.00 | ~$0.05–0.10 |
| claude-haiku-3 | $0.25 | $1.25 | ~$0.005–0.01 |
| gpt-4o | $2.50 | $10.00 | ~$0.04–0.08 |
| gpt-4o-mini | $0.15 | $0.60 | ~$0.003–0.006 |

At 20 tickers per day with claude-sonnet-4-6, expected daily cost is $1–2. The daily budget
cap (`AIQUANT_DAILY_BUDGET_USD`, default $2.00) is enforced by `CostTracker`: once the cap is
hit, no new LLM calls are made and remaining tickers are skipped with a warning.

### Reducing Cost

- Reduce `AIQUANT_TOP_K_STOCKS` from 20 to 10 — halves LLM cost
- Switch to `claude-haiku-3` for sentiment and technical nodes (less reasoning required)
- Reduce `AIQUANT_DEBATE_ROUNDS` from 2 to 1
- Use `AIQUANT_LLM_PROVIDER=ollama` with a local model (zero API cost, lower quality)

### Cost Tracking

`CostTracker` logs every LLM call to `outputs/cost_log.jsonl` with timestamp, model, token
counts, and USD cost. The daily total is available via `aiquant monitor` and on the dashboard
Cost Tracker page.

---

## 15. Troubleshooting & Glossary

### Common Errors

| Error | Likely cause | Fix |
|---|---|---|
| `DataPipelineError: Qlib not initialised` | `provider_uri` path does not exist | Run `aiquant setup` or set `AIQUANT_QLIB_PROVIDER_URI` |
| `AgentError: LLM returned invalid JSON` | Model hallucinated malformed Pydantic output | Retry; if persistent, lower temperature |
| `RiskBreachError: drawdown limit exceeded` | Portfolio down >15% | Review portfolio; run `aiquant trade reset --confirm` |
| `TOOL_ERROR: rate limit` | FinnHub or CoinGecko rate limit hit | Wait 60s; retry logic should handle automatically |
| `SchedulerError: lock file exists` | Previous pipeline run still active | Check `outputs/.pipeline.lock`; delete if stale |
| `ImportError: qlib` | Qlib not installed in env | `pip install pyqlib` |
| `ConnectionError: Telegram` | Bot token or chat ID incorrect | Verify with `@userinfobot` and `@BotFather` |

### Glossary

| Term | Definition |
|---|---|
| **Alpha** | Return in excess of a benchmark, attributable to skill rather than market beta |
| **IC (Information Coefficient)** | Spearman rank correlation between predicted and realised returns; IC > 0.02 is considered useful |
| **ICIR (IC Information Ratio)** | IC mean divided by IC standard deviation; measures signal consistency |
| **Sharpe Ratio** | Annualised excess return divided by annualised volatility; > 1.0 is considered good |
| **Drawdown** | Peak-to-trough decline in portfolio value; max drawdown is the largest such decline |
| **Walk-Forward** | Evaluation methodology where the model is re-trained at each fold using only past data |
| **Regime** | Prevailing macro market environment; classified here as risk_on / neutral / risk_off / crisis |
| **Factor** | A quantitative signal used as a model feature, e.g. 5-day momentum or RSI-14 |
| **GICS** | Global Industry Classification Standard; used for sector concentration limits |
| **HRP** | Hierarchical Risk Parity; portfolio construction method robust to covariance estimation error |
| **Black-Litterman** | Bayesian portfolio model that combines market equilibrium with investor views |
| **Embargo** | Gap between training and test periods in walk-forward evaluation to prevent leakage |
| **Veto** | Risk Manager decision to block a proposed trade regardless of analyst consensus |
