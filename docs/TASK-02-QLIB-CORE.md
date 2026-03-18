# TASK-02: Qlib Core Pipeline

## Goal

Build the quantitative foundation: data ingestion, factor engineering, model training, and single-split backtesting.

## Depends On

Task 01 (config system, package structure)

## What to Build

### 1. Data Pipeline (`src/core/data_pipeline.py`)

Initialize Qlib, load stock universe, create datasets using Alpha158 or Alpha360 handler, split into train/valid/test segments using dates from config, expose DataFrames for downstream use.

Must also support merging custom factors (from the factor library, built in Task 06) with Qlib's built-in features via a `create_dataset_with_custom_factors()` method. For now this method can accept an optional DataFrame — the RD-Agent wiring comes in Task 06.

### 2. Factor Engine (`src/core/factor_engine.py`)

Compute custom alpha factors from OHLCV data: RSI, MACD, Bollinger Bands, ATR, On-Balance Volume, composite momentum score (multi-horizon, reversal-adjusted), volatility regime (short/long vol ratio), mean-reversion z-score.

Factor evaluation functions: Information Coefficient (IC), IC Information Ratio (ICIR), factor turnover, quintile return analysis, full factor report with t-stat.

### 3. Model Zoo (`src/core/model_zoo.py`)

Unified wrapper around Qlib models: LightGBM (default), CatBoost, XGBoost, Linear (Ridge), and others. All expose: train(dataset), predict(dataset, segment) → Series, evaluate(predictions) → metrics dict, get_feature_importance() → Series.

Must check for `config/best_model_config.yaml` (written by RD-Agent in Task 06) and use those hyperparameters if the file exists, otherwise use defaults from qlib_config.yaml.

### 4. Backtester (`src/core/backtester.py`)

Wrap Qlib's backtest with TopkDropout strategy. Compute: portfolio values, returns, Sharpe, max drawdown, Calmar, annual return, turnover, IC, ICIR. Generate HTML report with cumulative return chart, drawdown chart, monthly heatmap.

Support comparing multiple models side-by-side in a single report.

### 5. Data Setup Script (`scripts/setup_qlib_data.sh`)

Download US stock data to ~/.qlib/qlib_data/us_data with verification.

## Inputs

- QlibConfig from config system
- Qlib US data on disk

## Outputs

- DataPipeline produces DatasetH objects ready for training
- ModelWrapper produces prediction Series (alpha scores per ticker per date)
- Backtester produces BacktestResult with all metrics + HTML report

## Done Criteria

- [x] `bash scripts/setup_qlib_data.sh` downloads data successfully
- [x] DataPipeline initializes Qlib and creates an Alpha158 dataset
- [x] LightGBM trains on train segment and produces predictions on test segment
- [x] Predictions have correct shape: MultiIndex (date, instrument), float values
- [x] Backtest runs and produces Sharpe ratio, max drawdown, annual return
- [x] HTML report generates to ./reports/
- [x] Factor engine computes RSI and MACD without errors on sample data
- [x] IC computation returns ~1.0 for a perfect-foresight factor and ~0.0 for random
- [x] `pytest tests/test_core.py` passes
