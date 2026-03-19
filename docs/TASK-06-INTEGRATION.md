# TASK-06: Integration, RD-Agent Wiring, Pipeline Scripts

## Goal
Wire all three layers together into a working end-to-end pipeline. Integrate RD-Agent as a periodic offline component whose outputs feed into the daily pipeline. Build all CLI scripts.

## Depends On
Tasks 02, 03, 04, 05 (all layers must be functional)

## What to Build

### 1. RD-Agent Runner (`src/core/rd_agent_runner.py`)

RD-Agent is a PERIODIC component — runs weekly or monthly, NOT daily. It has six modes of operation:

**Factor evolution**: use LLM (via RD-Agent library) to propose factor hypotheses, implement them as Qlib expressions or pandas code, evaluate them (IC, ICIR, turnover, correlation with existing library factors). Factors passing thresholds (min_ic, min_icir, correlation < 0.7 with existing) are saved to `data/factor_library/` as JSON files containing: name, expression/code, description, ic, icir, turnover, discovered_date, passes_validation flag.

**Model evolution**: propose and evaluate model hyperparameter configurations and architecture changes. Best config saved to `config/best_model_config.yaml`.

**Joint factor-model co-optimization**: This is RD-Agent's most powerful mode. Instead of running factor evolution and model optimization separately, this mode alternates between them in a single loop. Each iteration, a multi-armed bandit scheduler decides whether improving factors or improving the model will yield more alpha, then executes that direction. The result of one feeds into the next — better factors make the model work better, a better model surfaces which factors actually matter. This produces significantly better results than running each separately. Implement as a single `co_optimize()` method that runs the full RD-Agent(Q) joint loop from the paper. This should be the primary recommended way to run RD-Agent.

**Knowledge base persistence**: RD-Agent must remember what it learned across runs. Store in `data/rd_knowledge_base/`:
- Which factor patterns worked and which failed (so it doesn't re-propose known failures)
- Which model architectures and hyperparameters were tested and their results
- Which data features are noisy or redundant
- Successful strategies and why they worked (natural language summaries from the LLM)

Each weekly/monthly run loads the knowledge base before starting and saves updated knowledge after completing. Run 10 should be smarter than run 1 because it builds on accumulated knowledge. RD-Agent's built-in knowledge management system (the Co-STEER component) handles this — wire it to persist to disk between sessions rather than only living in memory during a single run.

**Multi-trace parallel exploration**: Support running multiple independent research threads (traces) simultaneously. Each trace tries a different research direction (e.g., trace 1 explores momentum factors, trace 2 explores volatility factors, trace 3 explores cross-asset factors). After all traces complete, merge the best discoveries from each into the factor library. This costs N× the API budget (where N = number of traces) but finds better factors faster because it explores diverse directions rather than going deep on one path. Cross-trace collaboration: traces can share intermediate findings so they don't waste time on directions another trace already ruled out. Configurable: number of traces (default 3), whether traces share knowledge during runs, merge strategy (keep top K factors by ICIR across all traces).

**Research paper implementation**: Read quantitative finance research papers (PDFs or arXiv links) and automatically implement the strategies, factors, or models described in them. Steps: (1) extract key methods, formulas, and algorithms from the paper, (2) implement them as Qlib-compatible code, (3) backtest the implementation, (4) if the results are promising, save the factors/model to the library. This uses RD-Agent's general research assistant capability applied specifically to quant finance papers. Useful for systematically testing ideas from new publications without manually coding each one.

**Library management**: load all factors from library, return summary (count, avg IC, last discovery date, knowledge base stats), validate library with a dedicated backtest.

**Factor Copilot (interactive mode)**: Unlike the autonomous modes above where RD-Agent decides what to explore, this is the interactive mode. You describe a factor hypothesis in plain English — for example "build me a factor that captures stocks with accelerating revenue growth AND declining short interest" — and RD-Agent implements it as code, evaluates it (IC, ICIR, turnover), and reports back. If it passes quality thresholds, it's saved to the factor library. This uses RD-Agent's Finance Data Copilot scenario. The value: when you read something interesting or have a trading intuition, you can test it instantly without writing code yourself. Expose this as a CLI command and also as a function callable from notebooks.

**Model Copilot (full architecture from papers)**: Extends the paper implementation mode beyond just extracting factors. This mode reads a paper describing a full model architecture (e.g., a temporal fusion transformer, an attention-based LSTM, a graph neural network for stock relationships) and implements it as a complete Qlib-compatible model — PyTorch code, training loop, integration with Qlib's DataHandler and backtest. This uses RD-Agent's General Model Copilot scenario. The result is a new model added to the model zoo that can be compared against LightGBM in backtests. This is how you'd explore LSTM/Transformer models without coding them manually.

**RD-Agent built-in UI**: RD-Agent ships with its own Streamlit-based web UI (`rdagent ui --port 8080 --log-dir log/`). It shows every R&D loop with: hypothesis generated, code implemented, backtest results, feedback analysis, and knowledge base state. Pre-defined views for: Qlib Model, Qlib Factor, Data Mining, Model from Paper. Wire this into the project so it runs as part of the docker-compose stack (or standalone). This replaces the need to build a custom dashboard for the RD-Agent research monitoring — use the built-in one directly. The project's own Streamlit dashboard (Task 07) should focus on the daily pipeline, portfolio, and agents. The RD-Agent UI covers the periodic research side.

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

**`scripts/run_rd_agent.py`**: Typer CLI. Commands:
- `co-optimize` — run joint factor-model co-optimization (RECOMMENDED primary mode). Uses multi-armed bandit to alternate between factor and model improvement. Loads knowledge base at start, saves at end. This is the single most valuable command.
- `mine-factors` — run factor evolution only (useful when you want to hold model fixed)
- `optimize-model` — run model hyperparam optimization only
- `co-optimize --traces 3` — run joint optimization with 3 parallel traces. Each trace explores a different research direction. Results merged at the end. Costs 3× the API budget but finds better factors.
- `extract-from-reports --report-dir ./reports/10k/` — extract factors from 10-K/10-Q financial reports
- `implement-paper --source arxiv:2505.15155` or `--source ./papers/some_paper.pdf` — read a quant finance paper, implement its strategy/factors, backtest, and save passing results to library
- `library-status` — show factor library contents, knowledge base stats, last run date, cumulative discoveries
- `validate-library` — run 3-way backtest: Alpha158 only vs library only vs combined
- `reset-knowledge` — clear the knowledge base (start fresh, use with caution)
- `copilot-factor "description of your factor idea"` — interactive mode: describe a factor in plain English, RD-Agent implements it, evaluates it, saves to library if it passes. Uses the Finance Data Copilot scenario.
- `copilot-model --source ./papers/model_paper.pdf` — read a paper describing a full model architecture, implement it as a Qlib-compatible PyTorch model, backtest it, add to model zoo if results are good. Uses the General Model Copilot scenario. Different from `implement-paper` which only extracts factors — this builds entire models.
- `ui --port 8080` — launch RD-Agent's built-in Streamlit dashboard showing all R&D loops, hypotheses, implementations, and results. Reads from `log/` directory. Pre-defined views for Qlib Factor, Qlib Model, Data Mining, Model from Paper.

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
- RD-Agent CLI with all subcommands functional (co-optimize, mine-factors, optimize-model, copilot-factor, copilot-model, implement-paper, library-status, validate-library, ui)
- Factor library integration verified
- Factor Copilot producing factors from natural language descriptions
- Model Copilot producing Qlib-compatible models from paper PDFs
- RD-Agent built-in UI running and showing R&D loop traces
- 4 runnable notebooks

## Done Criteria
- [x] `python scripts/run_backtest.py run --model LightGBM --topk 30` completes and prints metrics
- [x] `python scripts/run_backtest.py run --with-rd-factors` includes factor library (or warns if empty)
- [x] `python scripts/run_agents.py analyze NVDA` returns a valid AnalysisResult
- [x] `python scripts/run_agents.py analyze NVDA --with-qlib` includes Qlib context in agent prompt
- [x] `python scripts/run_pipeline.py --mode backtest` runs full pipeline and produces portfolio results
- [x] `python scripts/run_rd_agent.py co-optimize --iterations 5 --budget 5.0` runs joint factor-model loop and produces at least 1 factor or model improvement
- [x] `python scripts/run_rd_agent.py co-optimize --traces 3 --iterations 5` runs 3 parallel traces and merges results
- [x] `python scripts/run_rd_agent.py library-status` reports factor count, knowledge base size, last run date
- [x] `python scripts/run_rd_agent.py validate-library` runs 3-way backtest comparison
- [x] `python scripts/run_rd_agent.py implement-paper --source ./papers/test_paper.pdf` extracts and backtests at least one factor or model from a paper
- [x] Knowledge base persists between runs: run co-optimize twice, second run loads knowledge from first run (verify via logs showing "loaded N prior experiments")
- [x] Knowledge base prevents re-proposing known failures: if a factor failed in run 1, it should not be re-proposed identically in run 2
- [x] `python scripts/run_rd_agent.py copilot-factor "momentum factor for stocks with rising earnings estimates and high short interest"` implements, evaluates, and reports IC/ICIR for the described factor
- [x] `python scripts/run_rd_agent.py copilot-model --source ./papers/test_model_paper.pdf` reads a paper and produces a working Qlib-compatible model that trains and backtests
- [x] `python scripts/run_rd_agent.py ui --port 8080` launches the RD-Agent Streamlit dashboard and it loads without errors
- [x] RD-Agent UI shows at least one completed R&D loop after running co-optimize
- [x] Factor library JSON files load correctly and merge into dataset
- [x] Pipeline is idempotent: running twice with same data produces same trades
- [x] All 4 notebooks execute without errors
- [x] `pytest tests/test_integration.py` passes
