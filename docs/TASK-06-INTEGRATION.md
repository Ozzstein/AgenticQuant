# TASK-06: Integration, RD-Agent Wiring, Pipeline Scripts

## Goal
Wire all three layers together into a working end-to-end pipeline. Integrate RD-Agent as a periodic offline component whose outputs feed into the daily pipeline. Build all CLI scripts.

## Depends On
Tasks 02, 03, 04, 05 (all layers must be functional)

## What to Build

### 1. RD-Agent Runner (`src/core/rd_agent_runner.py`)

RD-Agent is a PERIODIC component — runs weekly or monthly, NOT daily. It has two jobs:

**Factor discovery**: use LLM (via RD-Agent library) to propose factor hypotheses, implement them as Qlib expressions or pandas code, evaluate them (IC, ICIR, turnover, correlation with existing library factors). Factors passing thresholds (min_ic, min_icir, correlation < 0.7 with existing) are saved to `data/factor_library/` as JSON files containing: name, expression/code, description, ic, icir, turnover, discovered_date, passes_validation flag.

**Model optimization**: propose and evaluate model hyperparameter configurations. Best config saved to `config/best_model_config.yaml`.

**Report extraction**: read 10-K/10-Q PDFs, extract quantitative signals, evaluate as factors, save passing ones to library.

**Library management**: load all factors from library, return summary (count, avg IC, last discovery date), validate library with a dedicated backtest.

### 2. Pipeline Integration Points

The daily pipeline must wire these connections:

**RD-Agent → Qlib**: `DataPipeline.create_dataset_with_custom_factors()` loads `data/factor_library/`, computes each factor, merges with Alpha158 features. `ModelWrapper` checks for `config/best_model_config.yaml` and uses it if present.

**Qlib → Agents**: top-N tickers from model predictions are sent to the LangGraph pipeline with context: `{qlib_alpha_score, qlib_rank, factor_signals}`.

**Agents → Execution**: AnalysisResult decisions mapped to Signal objects. BUY/STRONG_BUY → buy signal with target weight from position_size_pct. SELL/STRONG_SELL → sell signal. HOLD → no signal.

**Execution → Reporting**: paper trader results formatted for logging and alerts.

### 3. CLI Scripts

**`scripts/run_backtest.py`**: Typer CLI. Commands: `run` (single backtest, optional --with-rd-factors to include factor library), `compare` (multiple models side-by-side). Uses rich for output tables.

**`scripts/run_agents.py`**: Typer CLI. Commands: `analyze` (one or more tickers, optional --with-qlib to include Qlib scores, --output json/rich), `debate` (verbose debate output).

**`scripts/run_pipeline.py`**: Typer CLI. Commands: `pipeline` with modes: backtest (historical), paper (one iteration with live data), paper-loop (continuous on schedule).

**`scripts/run_rd_agent.py`**: Typer CLI. Commands: `mine-factors` (run factor evolution), `optimize-model` (run hyperparam optimization), `extract-from-reports` (process financial reports), `library-status` (show library contents), `validate-library` (run 3-way backtest: Alpha158 only vs library only vs combined).

### 4. Full Pipeline Flow (`scripts/run_pipeline.py`)

Backtest mode:
1. Load config
2. Init data pipeline (include factor library if exists)
3. Create dataset
4. Train model
5. Predict on test set
6. Select top-N tickers
7. Run LangGraph agent analysis on each (with Qlib context)
8. Map AnalysisResult → Signals
9. Run through portfolio optimizer
10. Run through risk controls
11. Execute via paper trader
12. Log results, generate report

Paper mode: same flow but with live data (today's prices, today's news).

### 5. Notebooks

Create 4 Jupyter notebooks (as .py percent-format scripts):
- `01_qlib_quickstart.ipynb`: data → LightGBM → backtest → metrics
- `02_factor_mining.ipynb`: compute custom factors, evaluate IC, visualize quintile returns
- `03_agent_analysis.ipynb`: run LangGraph analysis on a ticker, display all agent reports
- `04_full_pipeline.ipynb`: end-to-end pipeline demo on 5 tickers

## Inputs
- All modules from Tasks 02-05
- Config from Task 01

## Outputs
- Working end-to-end pipeline in all three modes
- RD-Agent CLI with all subcommands functional
- Factor library integration verified
- 4 runnable notebooks

## Done Criteria
- [ ] `python scripts/run_backtest.py run --model LightGBM --topk 30` completes and prints metrics
- [ ] `python scripts/run_backtest.py run --with-rd-factors` includes factor library (or warns if empty)
- [ ] `python scripts/run_agents.py analyze NVDA` returns a valid AnalysisResult
- [ ] `python scripts/run_agents.py analyze NVDA --with-qlib` includes Qlib context in agent prompt
- [ ] `python scripts/run_pipeline.py --mode backtest` runs full pipeline and produces portfolio results
- [ ] `python scripts/run_rd_agent.py library-status` reports factor count (0 if empty)
- [ ] `python scripts/run_rd_agent.py validate-library` runs 3-way backtest comparison
- [ ] Factor library JSON files load correctly and merge into dataset
- [ ] Pipeline is idempotent: running twice with same data produces same trades
- [ ] All 4 notebooks execute without errors
- [ ] `pytest tests/test_integration.py` passes
