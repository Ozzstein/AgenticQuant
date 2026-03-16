# TASK-03: Walk-Forward Backtester & Validation Suite

## Goal
Make backtest results trustworthy. A single train/test split is unreliable. This task adds rolling-window walk-forward testing and automated integrity checks.

## Depends On
Task 02 (data pipeline, model zoo, backtester)

## What to Build

### 1. Walk-Forward Backtester (`src/core/walk_forward.py`)

Retrain the model every N months on an expanding or sliding window. Stitch all out-of-sample test periods into one continuous result.

Configuration: minimum training window (months), retrain frequency (months), test window length (months), expanding vs sliding mode, embargo period (days gap between train end and test start to prevent boundary leakage).

For each window: create a Qlib dataset with that window's dates, train model, predict on test segment, run backtest, record per-window metrics.

Track across windows: feature importance drift (how much top features change between retrains), rolling IC over time.

Generate HTML report: cumulative returns with vertical lines at each retrain point, per-window Sharpe bar chart, feature importance heatmap over time, drawdown chart with worst drawdowns annotated.

### 2. Backtest Validation Suite (`src/core/validation.py`)

Run automatically on every backtest result. Each check returns pass/fail/warning with explanation.

Checks:
- **Look-ahead bias**: verify no predictions use future data. Shuffle future returns and re-run — if performance barely changes, there's leakage.
- **Survivorship bias**: verify universe includes delisted stocks over the full period, not just current constituents.
- **Fill realism**: no fills outside day's high-low, order size < 1% of daily volume, no fills on zero-volume days.
- **Transaction cost sensitivity**: run at 0, 5, 10, 20, 50 bps. Flag if Sharpe drops > 50% from 0 to 10 bps.
- **Concentration risk**: flag single position > 15%, top 5 > 50%, single sector > 40%.
- **Regime dependence**: split test into bull/bear/high-vol/low-vol. Flag if Sharpe positive in only one regime.
- **Statistical significance**: t-stat of excess returns > 2.0, minimum 252 trading days, bootstrap confidence interval on Sharpe.

Output: per-check result (✅/⚠️/❌) with explanation + overall verdict (RELIABLE / NOT RELIABLE).

### 3. Benchmark Comparison (extend `src/core/backtester.py`)

Every strategy automatically compared against:
- Buy-and-hold S&P 500
- Buy-and-hold BTC
- 60/40 equity/bond
- Equal-weight universe
- Monte Carlo random picker: 1000 random portfolios of same size, report strategy's percentile rank. Must beat 95th percentile to have a statistical edge.

## Inputs
- BacktestResult from Task 02
- DataPipeline and ModelWrapper from Task 02

## Outputs
- WalkForwardResult: per-window metrics list, aggregate metrics, stitched return series, feature importance drift data
- ValidationReport: per-check pass/fail/warning, overall verdict
- Benchmark comparison table

## Done Criteria

- [x] Walk-forward generates at least 4 windows on 2015-2025 data with 6-month retraining
- [x] Stitched out-of-sample return series is continuous with no gaps
- [x] Embargo period creates a visible gap between train end and test start dates
- [x] Walk-forward HTML report renders with retrain vertical lines
- [x] Validation suite passes on a properly-configured backtest
- [x] Validation suite catches a deliberately-introduced look-ahead bias (e.g., using future close as a feature)
- [x] Transaction cost sensitivity correctly flags a high-turnover strategy
- [x] Random picker baseline runs 1000 portfolios and reports percentile
- [x] `pytest tests/test_validation.py` passes

## Implementation Notes (2026-03-16)

### New files created
- `src/core/walk_forward.py` — `WalkForwardRunner` class
- `src/core/validation.py` — `ValidationSuite` class
- `tests/test_walk_forward.py` — 7 unit tests
- `tests/test_validation.py` — 8 unit tests

### New schemas added to `src/utils/schemas.py`
- `WalkForwardResult` — enhanced walk-forward result with feature drift and rolling IC
- `CheckVerdict` — PASS / WARNING / FAIL enum
- `CheckResult` — single check result
- `ValidationReport` — 7-check report with RELIABLE / NOT RELIABLE verdict

### Test counts: 253 total (238 pre-existing + 15 new)

### Validation checks
| # | Name | Verdict Logic |
|---|------|---------------|
| 1 | look_ahead_bias | FAIL if shuffled Sharpe ≈ original AND Sharpe ≤ 0; WARNING otherwise |
| 2 | survivorship_bias | WARNING if ≤1 fold or <252 days; PASS otherwise |
| 3 | fill_realism | FAIL if `order_pct_volume > 0.01`; WARNING if no trades |
| 4 | cost_sensitivity | FAIL if Sharpe(10bps) < 0.5 × Sharpe(0bps); sweep [0,5,10,20,50]bps |
| 5 | concentration_risk | FAIL if max weight > 15%; WARNING if > 10% |
| 6 | regime_dependence | WARNING if positive in only one half; FAIL if negative in both |
| 7 | statistical_significance | PASS requires t-stat > 2.0, n ≥ 252, bootstrap 5th-pct Sharpe > 0 |
