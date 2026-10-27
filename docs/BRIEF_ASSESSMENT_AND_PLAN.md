# QuantAgentLab repository and brief assessment

Assessment date: 2026-10-26; scope updated 2026-10-27. Repository baseline: `1d19552`. Source brief: *QuantAgentLab — System Overview & Technical Brief*, 15 pages, dated 2026-10-26.

The repository is worth building on. It contains substantial reusable components, but the brief describes a more complete system than the executable paths provide. The implementation target is the full brief, including both asset classes, all strategies/models/optimizers, the agent desk, native RD-Agent and custom research, all execution venues, and complete operations. The authoritative scope and acceptance contract is [the full system build plan](FULL_SYSTEM_BUILD_PLAN.md).

This document records the repository assessment and its verification evidence. No trading behavior was changed as part of the assessment or the scope revision.

## What to retain

- Python, Qlib/LightGBM, Pydantic, LangGraph, and Streamlit fit the brief's single-operator product.
- The separation between periodic research and daily trading is sound. Research should produce candidates; trading should consume a specific approved artifact version.
- Factor/model wrappers, the HMM detector, strategy definitions, broker protocol, audit utilities, and existing test fixtures provide useful starting points.
- The project already has Alpaca and CCXT adapters, advanced-order support, an ensemble, agent history, intraday scheduling, and transcript tooling. These need integration and verification rather than wholesale recreation.

The inventory contains 19,222 lines in 72 Python files under `src`, 2,301 lines under `scripts`, and 14,856 test lines. This broadly supports the brief's size estimate. Component breadth and test count do not establish end-to-end correctness.

## Findings that determine the build order

| Priority | Finding and evidence | Consequence |
|---|---|---|
| P0 | `core/backtester.py:278` selects top-K rows across an entire test window, then `:307` spreads the aggregate return uniformly across calendar days. `core/walk_forward.py` reuses that calculation. | These returns do not represent a chronological portfolio. Sharpe, drawdown, and research promotion based on them cannot support the brief's acceptance criteria. |
| P0 | `scripts/run_backtest.py:109` returns positive mock metrics on empty data, short datasets, training failure, or backtest failure. `scripts/run_pipeline.py` supplies a price of 100 for missing prices. | Failure can look like performance or valid trading input. Mocks belong in tests; normal runtime must report actual failure. |
| P0 | `execution/risk_controls.py:63` adds order notional for both buys and sells; its cash check uses 99% of cash rather than the configured NAV reserve. | Reducing exposure can be blocked, while the minimum cash requirement can be breached. |
| P0 | `scripts/run_pipeline.py:328` omits peak NAV, cumulative daily turnover, explicit asset class, and ADV. The captured portfolio is not refreshed after fills. Sector and aggregate crypto limits are absent from this gate. | The documented drawdown breaker and several portfolio limits are not reliably enforced by the daily path. Intraday uses similarly incomplete risk inputs. |
| P0 | `create_broker()` constructs a fresh `PaperTrader`; paper positions, cash, contingent orders, and history live in memory. The daily path attributes zero P&L to the active strategy. | Multi-day paper results, strategy performance selection, and restart recovery are not established. The roadmap's completed persistence claim is incorrect for this checkout. |
| P1 | `scripts/run_pipeline.py` uses fixed 2023–2024 config dates, ten hard-coded tickers, and average historical 20-day momentum. Strategy selection changes the candidate count; it does not wire the selected factors, model, sizing, exits, universe, or cadence through the run. | The main daily path does not implement the brief's ML → agents → optimizer architecture. |
| P1 | Every non-buy decision becomes `FLAT`, which liquidates existing holdings. Analysis failures become `HOLD`, then take this same route. Only shortlisted symbols receive prices. | HOLD, analysis failure, veto, and an explicit exit need distinct semantics; existing holdings outside the shortlist also require valuation and exit handling. |
| P1 | `core/validation.py` tests leakage by shuffling realized returns and recomputing Sharpe, and infers survivorship quality from fold/history length. Another validator in `core/backtester.py` uses different heuristics. | Permuting returns preserves mean and standard deviation; neither check demonstrates absence of leakage or survivorship bias. More folds do not repair a surviving-only universe. |
| P1 | `core/rd_agent_runner.py` has stub native RD-Agent entry points, custom LLM factor research, and simulated strategy/regime metrics. `evolved_momentum_0.json` is already marked validated with source `rd_agent_evolve_sim`. Some save paths bypass the factor backtest gate. | No single trustworthy promotion boundary exists. Simulated or incompletely validated artifacts can enter libraries used by selection. |
| P1 | `core/data_pipeline.py` computes some library expressions through keyword approximations; factor evaluation uses Python `eval`. Unknown expressions can become 20-day momentum in the fallback. | A factor evaluated in research can differ from the factor consumed later. Generated expressions also execute inside the host process. |
| P1 | `src/scheduler.py`, `monitoring/scheduler.py`, and `scripts/run_pipeline.py` implement different flows. The first has placeholder stages and references nonexistent `RiskControls`/configuration interfaces; the second reports zero orders. | There is no canonical operating cycle. A scheduler test that replaces all stages with no-ops does not validate the actual cycle. |
| P2 | Docker copies a missing `lean/` directory, omits strategy-library assets, and invokes the pipeline without its `run` subcommand. Dependency extras, environment names, and documentation differ. | Packaging and deployment need a reproducible smoke test. |

Additional integration work: the transcript node exists but is not registered in the main graph; the risk-manager node is currently a rule-based crisis/confidence/VIX check without portfolio or correlation inputs; memory is recent per-ticker JSONL history rather than semantic RAG; the LEAN bridge is a signal exporter/template rather than another `BaseBroker` implementation. The intraday scorer ranks nominal prices, not momentum. These distinctions should replace the current completion labels.

## Verification performed

A Python 3.11 environment and a separate source copy under `/tmp` were used for execution checks. The original application source was left unchanged.

- Installed the project's declared `dev` and `alpaca` dependencies. Unit run: **908 passed, 14 failed, 10 integration tests deselected**. The failures were missing Streamlit and a Parquet engine. Ten failures represent five crypto tests collected twice through `test_execution.py`.
- Added Streamlit and PyArrow only to the review environment. The four dashboard tests passed. PyArrow 26.0.0 could not import with the project's NumPy 1.26.4 (`pyarrow requires NumPy 2.0 or newer`). Pinning the review environment to PyArrow 19.0.1 allowed the remaining ten cases to pass. These were focused reruns, not a second full-suite run.
- Mocked integration run: **10 passed**. No live-provider or broker acceptance test was performed. Pytest socket blocking was enabled; warnings showed some nominal unit tests still attempted DNS lookups.
- Ruff over `src` and `scripts`: **4 findings**, consisting of three unused imports and one import-order issue.
- Qlib bundle execution, actual RD-Agent runs, container builds, and live model/broker connections remain unverified.

Small runtime probes independently reproduced the substantive issues:

| Input | Observed result |
|---|---|
| Sell all 100 shares of a $100 stock in a $100,000 portfolio | Rejected as a 20% post-trade position; the actual target is zero. |
| Buy $900 with $1,000 cash and $100,000 NAV | Passed, leaving only 0.1% of NAV in cash. |
| Recreate the paper broker after a purchase | Holdings count changed from one to zero. |
| Two dates with date-local top-1 returns of +10% and −20% | Backtest helper returned only `[0.1]`. |
| Run the backtest script with an empty data frame | Reported Sharpe 0.5 and total return 10%, with no daily returns. |
| Optimizer receives two zero raw weights | Fallback returned 47.5% per asset despite the default 10% cap. |

The passing tests are useful regression assets, but several assert interface shape or the current behavior rather than the financial or operational invariant.

## Evaluation of the brief

**Keep the architecture and two-stage research boundary.** Separating deterministic controls from LLM judgments is appropriate. The brief's native equity research and custom crypto research paths both belong in the complete delivery. The native RD-Agent CLI scenarios listed in the brief are supported by the [upstream project](https://github.com/microsoft/RD-Agent). Its [installation documentation](https://rdagent.readthedocs.io/en/latest/installation_and_configuration.html) confirms Docker-oriented execution and separate chat/embedding configuration. Pin the actual release and test its artifact contract rather than assume the latest workspace layout is stable.

**Revise “architecture complete and tested.”** A more accurate status is “component-rich prototype; core integration, statistical validation, and operational guarantees remain incomplete.” RD-Agent binding is only one of several substantial gaps. Retaining the current validation suite unchanged, as proposed on page 11, would preserve invalid checks.

**Make point-in-time data an explicit deliverable.** Qlib offers a PIT subsystem; using Qlib or date-indexed prices alone does not supply publication timestamps, historical revisions, delisted assets, or historical index membership. The [Qlib PIT documentation](https://qlib.readthedocs.io/en/latest/advanced/PIT.html) describes storing publication dates and successive versions. Data quality and coverage must be demonstrated for the chosen US universe.

**Treat LLM value as a hypothesis.** Keep historical quant evaluation separate from prospective agent evaluation. Archived inputs can test workflow replay, but do not erase knowledge in a modern model's weights. Compare quant-only, quant-plus-veto, and full-agent portfolios going forward using identical prices, costs, and risk policies.

**Strengthen the statistical gates.** IC 0.03, ICIR 0.5, Sharpe above 1, and a random-picker percentile are proposed thresholds, not proof of durable edge. Preserve an untouched holdout and a record of all research trials. Repeated selection inflates performance expectations; [Bailey and López de Prado's Deflated Sharpe Ratio paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) explains the multiple-testing problem. Add an appropriate uncertainty/selection-bias assessment before promotion.

**Resolve specification differences before wiring them.** The brief uses six-month walk-forward windows; config defaults to three. The defensive description specifies 30% cash; its JSON specifies 15% minimum cash. Research IC defaults are 0.02/0.3 rather than 0.03/0.5. A 5% crypto minimum also conflicts with a no-crypto defensive mode: make the effective lower bound zero when crypto is disabled. The agent-consensus strategy is an explicit exception to the quant-shortlist rule and requires a complete model-free implementation with a defined voting and risk contract.

**Specify trading time and failure behavior.** The brief's 08:00 UTC order submission precedes the US regular session, while the monitoring config defaults to 16:00 US/Eastern. Separate signal preparation, order submission, and fill assumptions using an exchange calendar. “Graceful failure” should mean continued monitoring and reconciliation, with new risk blocked when required data or state is unavailable. It cannot guarantee that an external outage never interrupts a rebalance.

**Treat cost and hardware figures as budgets to verify.** Free-tier data coverage, candidate counts, debate frequency, token use, and research retries are not measured here. Record actual use and peak memory before committing to the brief's monthly cost and VPS sizing. Unknown model prices must not silently count as zero cost.

## Complete implementation plan

[The full system build plan](FULL_SYSTEM_BUILD_PLAN.md) defines mandatory workstreams, dependencies, real integration evidence, and the brief's go-live gates. It replaces the earlier reduced-scope implementation sequence and estimate. The source findings above remain applicable to the full build.

[Workstream C](FULL_SYSTEM_BUILD_PLAN.md#c-quantitative-core-chronological-backtesting-and-validation) now explicitly assigns Qlib the research role and QuantAgentLab's portfolio replay engine the authoritative strategy-evaluation role. Requirements C1–C7 cover the backtester and walk-forward rebuilds, consolidated validation, removal of fabricated success, reproducible reporting, and reconciliation with the persistent paper simulator in F. LEAN remains an execution bridge; QuantGPT is parked. These are planned replacements with acceptance evidence, not completed implementation.
