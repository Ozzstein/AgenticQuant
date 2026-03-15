---
name: run-backtest
description: Run a Qlib backtest for QuantAgentLab with configurable model and parameters
disable-model-invocation: true
---

Run a backtest using the QuantAgentLab pipeline.

Parse the user's arguments to extract:
- `--model` or `-m`: model name (LightGBM, CatBoost, Linear). Default: LightGBM
- `--topk` or `-k`: number of top stocks to hold. Default: 30
- `--start`: backtest start date (YYYY-MM-DD). Default: 2020-01-01
- `--end`: backtest end date (YYYY-MM-DD). Default: 2023-12-31
- `--output` or `-o`: output format (rich, json). Default: rich

Then execute:
```
conda run -n aiquant python scripts/run_backtest.py run --model <model> --topk <topk>
```

After the run completes, display the key metrics (Sharpe ratio, max drawdown, annual return, total trades) in a formatted table using Rich.

If the backtest fails, show the error clearly and suggest common fixes:
1. Qlib data not initialized → run `python scripts/run_pipeline.py setup`
2. Missing model → check `src/core/model_zoo.py` REGISTRY
3. Date range too short → use at least 2 years of data
