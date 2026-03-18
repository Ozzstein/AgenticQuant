# TASK-08: Strategy Generation & Selection Layer

## Context
Tasks 01-07 are complete. The system has: Qlib core with walk-forward backtesting and validation, LangGraph multi-agent desk, execution layer with paper trader and portfolio optimizer, RD-Agent with co-optimization/copilots/UI, Docker deployment, monitoring, and dashboard.

**The gap**: there is no explicit strategy layer. Qlib produces alpha scores, agents produce buy/hold/sell decisions, but nobody decides *what kind of strategy to run*. The top-K dropout rule is hardcoded. There's no mechanism to say "switch from momentum to mean-reversion when volatility spikes" or "blend these three strategies and allocate capital based on recent performance."

## Goal
Add a strategy layer with two components:
1. **RD-Agent strategy research** (offline) — discovers and backtests full strategy logic, saves winners to a strategy library
2. **Runtime strategy selector** (daily) — picks which strategy or blend to run today based on current regime and recent performance

This follows the same pattern already established for factors: RD-Agent discovers offline → library → daily pipeline consumes.

## Depends On
All of Tasks 01-07 being functional.

## What to Build

### 1. Strategy Definition Schema

Define what a "strategy" means in this system. A strategy is more than a factor set or a model — it's the complete decision logic from signals to target portfolio.

A strategy must specify:
- Which factor set to use (Alpha158, RD-Agent library, a custom subset, or a combination)
- Which model to use (LightGBM, CatBoost, a specific RD-Agent-discovered architecture)
- Entry rules: under what conditions to open a position (top-K rank, agent consensus threshold, specific factor thresholds)
- Exit rules: under what conditions to close (rank drops below N, stop-loss hit, holding period exceeded, agent downgrades to sell)
- Position sizing method: equal weight, signal-proportional, Kelly, risk-parity within the strategy
- Rebalance frequency: daily, weekly, monthly
- Universe filter: which tickers this strategy applies to (all equities, large-cap only, crypto only, specific sectors)
- Regime applicability: which macro regimes this strategy is designed for (risk-on, risk-off, neutral, crisis, or all)

Store each strategy as a JSON/YAML file in `data/strategy_library/`. The schema must be a Pydantic model so it validates on load.

### 2. Built-In Baseline Strategies

Ship these as defaults in the strategy library so there's always something to run and compare against:

- **Momentum factor top-K**: rank by multi-horizon momentum score, hold top 30, rebalance weekly. The current default behavior, now formalized as a named strategy.
- **Value + quality**: rank by combined value (low P/E, high FCF yield) and quality (high ROE, low debt) factors, hold top 30, rebalance monthly. Lower turnover than momentum.
- **Mean-reversion**: rank by mean-reversion z-score (most oversold), hold top 20, rebalance daily, short holding period (5-10 days). Works best in range-bound markets.
- **Crypto trend-following**: for crypto universe only. Enter when price crosses above 20-day SMA with rising volume, exit when crosses below. Position size inversely proportional to ATR.
- **Defensive / risk-off**: activated when macro regime is risk-off or crisis. Reduces equity exposure to minimum, increases cash buffer to 30%+, crypto allocation drops to 0-5%, holds only low-beta large-cap stocks.
- **Agent-consensus**: no quantitative model — purely agent-driven. Run LangGraph analysis on a broad universe, take positions only where 4+ agents agree, size by confidence score. Slow and expensive but captures qualitative edge.

### 3. RD-Agent Strategy Research (extend `src/core/rd_agent_runner.py`)

Expand RD-Agent's role to propose and backtest full strategy logic, not just factors or models.

A new mode: `strategy-evolution`. The R&D loop:
1. RD-Agent examines the knowledge base (past factor/model results, regime history, backtest outcomes of existing strategies)
2. Proposes a new strategy hypothesis in natural language (e.g., "combine momentum and funding-rate factors for crypto, with a volatility filter that reduces exposure when 30-day realized vol exceeds 80th percentile")
3. Translates the hypothesis into a strategy definition (the Pydantic schema from section 1)
4. Implements any custom logic needed (new entry/exit rules, custom sizing logic)
5. Backtests the strategy using the walk-forward backtester from Task 03
6. Evaluates against the validation suite
7. If Sharpe > threshold and passes validation, saves to `data/strategy_library/`
8. Feeds results back to knowledge base for the next iteration

Also support a copilot mode: `copilot-strategy "description of your strategy idea"` — you describe a strategy in English, RD-Agent formalizes it into the schema, implements, backtests, and reports.

The key difference from factor/model evolution: the search space here is the *combination* of factors, models, rules, and regimes — not individual components. A strategy evolution might discover that "LightGBM with momentum factors works great in bull markets but should be replaced by mean-reversion with CatBoost in high-vol regimes." That's a meta-level insight that factor mining alone would never find.

### 4. Runtime Strategy Selector (`src/core/strategy_selector.py`)

This runs as part of the daily pipeline, between the macro regime detector and the Qlib prediction step. It decides what strategy (or blend) to execute today.

**Inputs**:
- Current macro regime (from the macro regime detector in the LangGraph pipeline)
- Strategy library contents (all strategies with their backtest metrics and regime applicability)
- Recent live performance of each active strategy (rolling 30/60/90-day Sharpe from paper trading)
- Current portfolio state

**Selection logic** (implement multiple methods, configurable):

- **Regime-based**: pick the strategy whose regime_applicability matches today's regime. Simplest. If multiple match, pick the one with the best walk-forward Sharpe for that regime.
- **Performance-weighted blend**: run multiple strategies simultaneously, allocate capital to each based on its recent rolling Sharpe. Strategies with higher recent Sharpe get more capital. Strategies in drawdown get reduced allocation. This is the multi-strategy allocator.
- **Bandit selector**: treat strategy selection as a multi-armed bandit problem. Explore new/untested strategies occasionally, exploit proven ones mostly. Thompson sampling or UCB.
- **Manual override**: allow the user to force a specific strategy via config or CLI flag. Always available as a safety valve.

**Output**: a StrategyAllocation — a dict mapping strategy names to capital allocation percentages, plus the merged parameter set (which factors, which model, which rules) that the downstream pipeline should use.

### 5. Pipeline Integration

Update the daily pipeline flow (in `scripts/run_pipeline.py`) to insert strategy selection:

```
Current flow:
  Data → Qlib predict → top-N → agents → optimizer → risk → execute

New flow:
  Data → Macro regime detect → Strategy selector → Qlib predict (using selected strategy's factors + model) → top-N (using selected strategy's entry rules) → agents → optimizer (using selected strategy's sizing method) → risk → execute
```

The strategy selector's output configures everything downstream:
- DataPipeline uses the strategy's specified factor set
- ModelWrapper uses the strategy's specified model
- Backtester uses the strategy's entry/exit rules and rebalance frequency
- Portfolio optimizer uses the strategy's sizing method
- Risk controls may have strategy-specific overrides (e.g., crypto strategy has tighter position limits)

### 6. Strategy Performance Tracking (`src/core/strategy_tracker.py`)

Track each strategy's live performance separately, even when running a blend.

Requirements:
- Attribute daily P&L to the strategy that generated each position
- Track per-strategy: rolling Sharpe, drawdown, win rate, turnover, correlation with other active strategies
- Detect when a strategy's live performance diverges significantly from its backtest (drift detection) — alert via Telegram if live Sharpe drops below 50% of backtest Sharpe over a 60-day window
- Store history in `data/strategy_performance/` for the strategy selector to use

### 7. CLI Commands

Add to `scripts/run_rd_agent.py`:
- `evolve-strategies --iterations 10 --budget 15.0` — run the strategy R&D loop
- `copilot-strategy "your strategy description"` — interactive mode

Add to `scripts/run_pipeline.py`:
- `--strategy momentum_topk` — force a specific strategy (manual override)
- `--strategy auto` — let the strategy selector choose (default)
- `--strategy blend` — run performance-weighted blend of all validated strategies

Add a new script `scripts/run_strategy.py`:
- `list` — show all strategies in the library with their backtest metrics
- `compare` — run walk-forward backtests for all strategies side-by-side, produce comparison report
- `detail STRATEGY_NAME` — show full details: definition, backtest results, live performance, regime fit
- `validate STRATEGY_NAME` — run the validation suite on a specific strategy

### 8. Update CLAUDE.md Architecture

The architecture flow becomes:

```
PERIODIC:
  RD-Agent co-optimize → factor library + model config
  RD-Agent evolve-strategies → strategy library  ← NEW

DAILY:
  Data update
    → Macro regime detect
      → Strategy selector (picks strategy based on regime + performance)  ← NEW
        → Qlib predict (using selected strategy's factors + model)
          → Top-N (using selected strategy's entry rules)
            → LangGraph agents
              → Portfolio optimizer (using selected strategy's sizing)
                → Risk controls
                  → Execute
                    → Strategy performance tracker  ← NEW
```

## Inputs
- Working pipeline from Tasks 01-07
- Macro regime detector output
- Strategy library (JSON/YAML files)
- Historical backtest results per strategy
- Live performance data (from paper trading)

## Outputs
- Strategy definition Pydantic schema
- 6 built-in baseline strategies in strategy library
- RD-Agent strategy evolution producing new strategies
- RD-Agent copilot-strategy for interactive strategy creation
- Runtime strategy selector choosing daily allocation
- Per-strategy performance tracking with drift detection
- Updated daily pipeline with strategy selection step

## Done Criteria
- [x] Strategy Pydantic schema validates all 6 baseline strategies without errors
- [x] All 6 baseline strategies load from `data/strategy_library/` and can be run individually
- [x] `python scripts/run_strategy.py list` shows all strategies with metrics
- [x] `python scripts/run_strategy.py compare` runs walk-forward on all strategies and produces a comparison table
- [x] `python scripts/run_rd_agent.py evolve-strategies --iterations 3 --budget 5.0` proposes at least 1 new strategy and backtests it
- [x] `python scripts/run_rd_agent.py copilot-strategy "momentum on crypto with volatility filter"` produces a valid strategy definition and backtest result
- [x] Strategy selector picks the correct regime-matched strategy (e.g., defensive strategy when regime = crisis)
- [x] Performance-weighted blend allocates more capital to higher-Sharpe strategies
- [x] `python scripts/run_pipeline.py --mode backtest --strategy auto` runs with strategy selection
- [x] `python scripts/run_pipeline.py --mode backtest --strategy momentum_topk` runs with forced strategy
- [x] Per-strategy P&L attribution tracks correctly when running a blend
- [x] Drift detection alerts when live performance diverges > 50% from backtest over 60 days
- [x] Pipeline still passes the validation suite from Task 03 with strategy layer inserted
- [x] `pytest tests/test_strategy.py` passes
