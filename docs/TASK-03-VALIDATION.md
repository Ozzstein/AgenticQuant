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
- [ ] Walk-forward generates at least 4 windows on 2015-2025 data with 6-month retraining
- [ ] Stitched out-of-sample return series is continuous with no gaps
- [ ] Embargo period creates a visible gap between train end and test start dates
- [ ] Walk-forward HTML report renders with retrain vertical lines
- [ ] Validation suite passes on a properly-configured backtest
- [ ] Validation suite catches a deliberately-introduced look-ahead bias (e.g., using future close as a feature)
- [ ] Transaction cost sensitivity correctly flags a high-turnover strategy
- [ ] Random picker baseline runs 1000 portfolios and reports percentile
- [ ] `pytest tests/test_validation.py` passes
