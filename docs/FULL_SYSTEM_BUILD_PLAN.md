# QuantAgentLab full system build plan

Date: 2026-10-27. Scope authority: the complete 15-page *QuantAgentLab — System Overview & Technical Brief* and the user's instruction to deliver the full system. Repository assessment: [verified findings](BRIEF_ASSESSMENT_AND_PLAN.md).

Integration assessment: [finance MCP recommendations and acceptance requirements](FINANCE_MCP_ASSESSMENT.md). These recommendations fit within the complete scope below; MCP adoption does not replace quantitative validation, durable execution, or the required research workflows.

Accepted integration selection: [MCP integration decision](MCP_INTEGRATION_DECISION.md). OpenBB, FMP, restricted Alpaca observation tools, and Valyu are selected for implementation. FRED/ALFRED supplies macro data through OpenBB or a direct adapter that preserves vintages. DexPaprika and QuantConnect MCP remain conditional choices.

Accepted decision-model direction: [Jev integration decision](JEV_INTEGRATION_DECISION.md). Implement a shared typed decision service for the agreed classification, evidence-checking, retrieval/debate support, and text-feature workflows. Existing subtask replacements and trading influence require measured acceptance evidence; qualitative macro labeling is conditional. The full agent desk and quantitative/execution capabilities remain required.

## Current progress — 2026-11-03

This document defines **what must be delivered**. The
[implementation evidence register](BUILD_REQUIREMENTS.md) records **what is
implemented, tested and still outstanding**, by group and subgroup. Its current
checkpoint is integrated code `4d898ed`; this documentation reconciliation adds
no new runtime acceptance. The original scope and release thresholds below are
unchanged. Dated implementation reports are historical evidence, not competing
roadmaps.

| Group | Current position | Detailed progress and remaining work |
|---|---|---|
| A — Foundations | Partial; actual project Docker smoke verified | [A: environments, contracts, MCP, CLI, packaging](BUILD_REQUIREMENTS.md#a-foundations) |
| B — Data | Partial; sufficient income authority still missing | [B: sources and B7 income evidence checklist](BUILD_REQUIREMENTS.md#b-data-and-source-authority) |
| C — Quantitative core | Local paths tested; ensemble consumer integration and statistical acceptance open | [C: models, replay, walk-forward and validation](BUILD_REQUIREMENTS.md#c-quantitative-core-and-validation) |
| D — Regimes/strategies | Local implementations tested; full source/prospective acceptance open | [D: HMM, strategies, selectors and sleeves](BUILD_REQUIREMENTS.md#d-regimes-strategies-and-allocation) |
| E — Agent desk | Local authority/budget/action tests; measured provider quality open | [E: desk, veto, history and cost](BUILD_REQUIREMENTS.md#e-agent-desk) |
| F — Execution | Selected actual LEAN paper scenarios verified; broader acceptance and features open | [F: venues, settlement/groups and F5 runtime matrix](BUILD_REQUIREMENTS.md#f-risk-accounting-and-execution) |
| G — Research | Local registry/evaluation paths and RD offline container smoke; actual research jobs open | [G: custom research and each native RD workflow](BUILD_REQUIREMENTS.md#g-research-and-rd-agent) |
| H — Operations | Local cycle/scheduler/interface paths; fresh complete cycle blocked | [H: cycle, scheduling, monitoring and deployment](BUILD_REQUIREMENTS.md#h-operating-cycle-and-operator-interface) |
| I — Release | Not accepted | [I: full audit, external and prospective release checklist](BUILD_REQUIREMENTS.md#i-system-acceptance-and-release) |
| J — Jev supplement | Shared service implemented locally; measured integrations incomplete | [J: individual accepted J1–J7 requirements](BUILD_REQUIREMENTS.md#j-jev-decision-service-cross-cutting) |

No entire group is complete. Checked milestones in the register apply only to
the named scenario; synthetic paper execution does not establish authenticated
broker acceptance or strategy performance. Maintain the register as checkpoints
land, preserving source pins and distinguishing implementation, offline tests,
actual runtime tests and prospective release evidence.

## Verdict and completion contract

Build the complete system on this repository. Retain useful components and replace implementations that cannot meet the specification. This is a substantial engineering project involving statistical correctness, data engineering, orchestration, execution, research automation, and operations.

The release scope includes both US equities and crypto; every model, strategy, optimizer, research workflow, execution venue, and operational guarantee described in the brief. Work is ordered by dependencies, but completing an early workstream does not complete the assignment. An equity-only pipeline, paper-only execution adapter, single strategy, mocked research engine, or disconnected component is not the finished deliverable.

Every required capability must have:

1. A working implementation and explicit input/output contracts.
2. A production entry point connected to the shared system.
3. Durable state, meaningful failure behavior, and audit evidence where applicable.
4. Deterministic correctness tests and actual integration evidence for its external dependencies.
5. Operator documentation and an observable result in the CLI/dashboard.

Mocks and synthetic fixtures belong in tests. Runtime failures must not produce fabricated prices, random performance, substituted factors, or successful placeholder results. Paper trading is an actual simulator using market data, realistic fills/costs, persistent positions, and reconciled accounting.

## System architecture

One application service owns the operating pipeline. CLI commands, the scheduler, and Streamlit invoke the same services and read the same persisted records.

The trading flow is:

`versioned equity/crypto data → macro regime → strategy or strategy blend → model rankings or explicit agent-consensus selection → agent desk → portfolio optimization → deterministic risk → approved order intents → venue routing → reconciliation → attribution and reporting`

The research flow is:

`scheduled or operator-directed research → isolated proposal/code/experiment loop → candidate artifacts → independent validation → versioned approved library → atomic activation → subsequent trading runs`

Research cannot submit orders or mutate the active trading state directly. Trading pins artifact versions for the duration of a run and continues independently when the research host is unavailable.

Use a transactional ledger for runs, orders, fills, cash, positions, portfolio snapshots, risk state, and strategy attribution. A single execution owner with SQLite transactions is a viable initial deployment topology; the correctness contract must not depend on process memory. Store large market/research datasets as immutable versioned artifacts. Persist every account, currency, instrument, and venue identity needed to aggregate equity and crypto exposure without double counting.

Define shared schemas for `RunContext`, `MarketSnapshot`, `StrategySpec`, `ModelArtifact`, `AgentDecision`, `PortfolioTarget`, `RiskContext`, `OrderIntent`, `Fill`, `ResearchCandidate`, and `ValidationReport`. Record data/configuration/code/artifact versions throughout. Normalize units, timestamps, session calendars, confidence scales, and instrument identifiers at boundaries.

Add `DecisionQuestion`, `DecisionRequest`, and `DecisionResult` contracts for an application-owned Jev `DecisionService`, separate from the chat-model factory. Preserve probabilities, provider confidence and its meaning, evidence quality, and policy outcomes as distinct fields. Migrate legacy confidence records explicitly; numerical rescaling does not make unlike quantities comparable. The service has no execution or promotion authority.

## Mandatory workstreams

### A. Reproducibility and application foundations

Brief coverage: pages 11–14; all entry points.

- Lock compatible Python/application dependencies, with separate runtime, development, dashboard, broker, Qlib, and RD-Agent environments as needed. Correct the missing Parquet dependency and verify actual native-library imports.
- Make configuration precedence explicit and tested. Validate required settings, separate historical evaluation dates from current operating dates, and expose a redacted effective configuration.
- Add an application-owned MCP connection manager and pinned server/tool manifests for the selected integrations, with separate runtimes where needed, private credential configuration, and exact operation/argument permissions.
- Implement the shared Jev adapter with pinned model/SDK/question versions, typed outputs, source-aware caching and recorded replay, bounded budgets/deadlines/retries, explicit unavailable states, and versioned confidence-schema migration. Follow the [decision service contract](JEV_INTEGRATION_DECISION.md#shared-decision-service-contract).
- Repair Docker builds, service commands, strategy/library assets, runtime libraries, persistent volumes, startup health checks, and clean-host setup.
- Consolidate duplicate schedulers and conflicting validation paths. Establish one package/service contract for all five CLI areas: backtesting, agents, pipeline, research, and strategy management.
- Add CI for correctness, component contracts, complete pipeline replay, CLI execution, lint, and deployment builds. Eliminate duplicate test collection and unmanaged test artifacts.

Acceptance: a clean host can install, build, start, and inspect the complete application using the documented commands. Configuration errors are explicit; missing external prerequisites remain visible instead of silently disabling required features.

### B. Complete equity, crypto, and contextual data layer

Brief coverage: pages 1–5, 7, 11, 13–14.

- Build reproducible US-equity ingestion with OHLCV, adjusted and tradable price conventions, corporate actions, historical universe membership, delistings, stable instrument IDs, and a verified US Qlib bundle.
- Implement crypto ingestion for the supported exchanges: OHLCV, volumes/liquidity, funding rates, open interest, BTC dominance, market benchmarks, token/protocol context, and currency conversion for portfolio accounting. Preserve the distinction between perpetual-derived features and long-only spot execution.
- Connect market/news/financial/filing/event data through the brief's providers: yfinance, FinnHub, SEC EDGAR, CoinGecko, and CCXT. Cover analyst and insider/institutional information where used by the agent tools. Integrate the repository's earnings transcript tools with a real configured source.
- Record observation time, publication/availability time, source, revision, adjustment policy, quality checks, and snapshot identity. Do not use present-day fundamentals or index members as historical facts.
- Implement the shared evidence contract across direct and MCP adapters. Connect selected OpenBB/FMP capabilities and FRED/ALFRED vintages; preserve complete documents and raw responses outside agent context, with bounded cited retrieval and no duplicate tools for the same capability.
- Supply Jev with bounded passages from complete stored documents. Integrate news/transcript labels and retrieval relevance/conflict judgments with immutable source IDs and publication cutoffs; preserve missing and contradictory evidence. Jev does not determine historical availability or replace the data providers.
- Implement caching, rate limits, retries, freshness/coverage requirements, market calendars, and health reporting. Missing essential features must block the affected strategy with an explanation; they cannot be substituted with unrelated factors.

Acceptance: both asset classes feed research and trading from traceable data; revised and unavailable data obey as-of constraints; equity sessions and 24/7 crypto operation are handled correctly. Historical data access is an actual project dependency, including paid data if required to satisfy coverage.

### C. Quantitative core, chronological backtesting, and validation

Brief coverage: pages 3–4, 10, 14–15.

The following rebuilds are mandatory delivery requirements. The current backtester is not an accepted performance baseline; its fold-wide selection, smoothed returns, and successful mock fallbacks must be replaced before its results can support research promotion or go-live decisions.

#### C1. Backtesting engine responsibilities

- Use Microsoft Qlib for factor/model research and the in-loop backtests of native RD-Agent equity experiments. Configure and version the US universe, data bundle, calendar, benchmark, splits, and execution assumptions. Qlib experiment metrics are research evidence; they do not independently authorize activation.
- Build QuantAgentLab's chronological portfolio replay and validation engine as the authoritative evaluator of complete strategies and combined US-equity/crypto portfolios. Consume versioned model predictions and strategy artifacts, then run allocation, optimization, deterministic risk, order intents, simulated fills, accounting, and attribution through the application's shared contracts.
- Keep real crypto factor/model evaluation on the custom research path specified in workstream G. A native crypto Qlib scenario remains an optional extension, not a prerequisite for complete crypto backtesting.
- Keep LEAN's required role as the operational execution bridge in workstream F. It is not selected as the primary historical backtesting engine. QuantGPT is parked and is not a dependency of this architecture.

#### C2. Complete factor and model evaluation

- Implement Alpha158 and the named custom factors: RSI, MACD, Bollinger bands, ATR, OBV, reversal-adjusted composite momentum, and volatility regime ratio. Implement crypto funding-rate mean reversion, open-interest divergence, BTC dominance, and beta-adjusted momentum.
- Connect LightGBM, CatBoost, XGBoost, ridge regression, and IC-weighted ensembles to training, artifact storage, inference, and comparison. Fit preprocessing and ensemble weights using only eligible training/validation data.
- Add the Jev text-feature adapter defined in J5. Version source passages, questions, model responses, transformations, and feature availability. Compare these features with existing models through the same chronological evaluation; question/feature selection and calibration must remain inside eligible development windows. Account for pretrained-model knowledge through the prospective controls in I.

#### C3. Rebuild chronological portfolio replay and accounting

- Replace the portfolio-return mechanics in `src/core/backtester.py` and their reuse in `src/core/walk_forward.py`. Select instruments independently at each eligible decision time; never select the top rows across a whole future test window or manufacture a daily path by spreading an aggregate return.
- Replay point-in-time universe membership, features, predictions, regimes, strategy/blend choices, optimizer targets, and risk decisions in timestamp order. Preserve instrument identity through feature/label joins. A signal may only fill at a subsequent eligible event under its documented availability and latency assumptions.
- Account explicitly for positions, cash, reserved cash, orders, partial fills, fees, spread/slippage, rebalance turnover, corporate actions, delistings, currency conversion, and strategy attribution. Model order-type behavior, market closures, liquidity/participation limits, and unavailable prices; ambiguous intrabar fills require a declared conservative policy. Use equity sessions and continuous crypto calendars without fabricating observations.
- Share portfolio, risk, order, fill, and ledger logic with the rebuilt paper simulator in workstream F. Historical replay uses an isolated account/run namespace, a simulation clock, and simulated execution; it cannot submit broker orders or modify operational accounts. Compute NAV and returns from reconciled balances and observed prices, including an explicit stale-valuation policy.
- Exercise every required strategy, selection mode, and optimizer with valid asset-specific cases. Historical replay of agent decisions is a workflow diagnostic; claims about the agent contribution require the prospective controls in workstream I.

#### C4. Rebuild walk-forward training and evaluation

- Implement expanding-window walk-forward evaluation using six-month OOS windows by default, label-horizon-aware purging/embargo, separate training/validation/test segments, and a fresh fitted model per fold. Preprocessing, feature selection, hyperparameters, ensemble weights, and regime fitting must not consume the test window.
- Stitch only genuine OOS ledger observations, without overlap, duplicate dates, interpolation, or zero-filled failed folds. Carry holdings and cash across consecutive evaluation windows; any required liquidation/reset must be explicit and charged. Training-window expansion cannot rewrite already recorded OOS decisions.
- Freeze split definitions, data snapshots, code/environment versions, model artifacts, random seeds, strategy configuration, and execution assumptions per run. Mark insufficient data, missing provenance, and incomplete folds explicitly; an incomplete evaluation cannot pass promotion gates.

#### C5. Rebuild metrics, benchmarks, and validation

- Compute IC/ICIR from aligned predictions and realized labels at the appropriate cross-sectional dates; derive portfolio Sharpe, drawdown, turnover, and attribution from the actual NAV/fill history. Declare the sampling calendar, annualization, cash-return, and benchmark conventions for equity, crypto, and mixed portfolios. Undefined metrics remain undefined.
- Evaluate turnover-based costs at multiple levels, including the brief's 10 bps equity and 20 bps crypto assumptions, with clear one-way/round-trip conventions and no duplicate charging of modeled costs.
- Implement SPY, BTC, 60/40, equal-weight, and 1,000 random-selection benchmarks. Match dates, investable universe, portfolio size, constraints, rebalance cadence, and trading costs; avoid degenerate sampling on small universes.
- Consolidate `BacktestValidator` in `src/core/backtester.py` and `ValidationSuite` in `src/core/validation.py` into one authoritative validation service. Replace shuffled-return Sharpe and fold-count survivorship heuristics with actual data/split provenance and adversarial tests. Validate fill realism, concentration, cost sensitivity, regime dependence, and statistical significance with reproducible pass/warning/fail evidence.
- Enforce the brief's IC above 0.03, ICIR above 0.5, and correlation-below-0.7 factor gates. Add a registry of all research trials, untouched holdouts, and uncertainty/multiple-testing assessment. Require the same independent validation and promotion boundary for Qlib/RD-Agent, custom crypto, and strategy/regime research outputs; missing critical evidence prevents promotion.

#### C6. Remove fabricated success and unify reporting

- Remove `_mock_backtest_result` and every runtime success fallback in `scripts/run_backtest.py`. Empty data, training failure, invalid accounting, and backtest errors must produce a failed/incomplete run and a non-success CLI exit status, with diagnostics rather than invented metrics. Synthetic results remain confined to explicitly identified test fixtures.
- Route CLI, research promotion, scheduler, and dashboard through the same replay/validation services. Persist the run manifest, timestamped signals, model versions, orders/fills, balances, NAV, fold evidence, benchmark results, and validation report so an operator can reproduce a reported result.
- Version the result schema and distinguish legacy invalid/unverified reports from rebuilt-engine results. Re-evaluate existing candidate artifacts through the new gates before eligibility; prior placeholder or simulated performance cannot carry over as validation.

#### C7. Required acceptance evidence

1. Independent hand-calculated multi-date, multi-instrument examples reconcile trades, cash, fees, holdings, NAV, and drawdown. Include the reproduced date-local selection failure, a losing period, an equity corporate action, and a crypto calendar case.
2. Future-data perturbations cannot alter past decisions. Deliberately leaked features, overlapping labels, future universe membership, and unavailable fundamentals fail the relevant checks; shuffled returns or additional folds cannot confer a false pass.
3. Identical initial state, timestamped inputs, order rules, and simulated fills produce matching replay/paper balances and attribution within declared tolerances. Restart/resume preserves those results and prevents duplicate economic events.
4. All four model families and the ensemble execute real historical evaluations; both asset classes, all six strategies, and all five optimizers have appropriate replay evidence. Benchmarks and cost sensitivity run on the same eligible evaluation periods.
5. Missing data, failed training, failed folds, and unavailable essential evidence cannot produce successful reports or promoted artifacts. Actual Qlib/RD-Agent integration is exercised; mocks alone do not establish acceptance.

Historical quantitative performance and historical agent workflow diagnostics must be labeled separately. Present-day LLM knowledge cannot be removed merely by supplying archived inputs. Prospective quant-only, quant-plus-veto, and full-agent portfolios remain required to measure the agent contribution; the agent desk remains part of the complete delivery.

### D. Regimes, all six strategies, and allocation

Brief coverage: pages 7–8.

- Complete the four-state Gaussian HMM with the eight specified signals: VIX level/change, 10y–2y yield slope, actual constituent breadth, realized volatility, momentum, dollar trend, and credit-spread proxy. Implement rolling one-year normalization, state labeling, confidence gating, and durable minimum-duration state.
- Train/evaluate historical regimes causally. Persist model versions and regime transitions; inference with insufficient data must be distinguishable from neutral.
- Keep the HMM as the authoritative regime model. A Jev replacement for the agent's quick qualitative macro-labeling call is conditional auxiliary evidence under J7; it cannot silently replace or override the HMM.
- Implement momentum top-K, value/quality, mean reversion, crypto trend, defensive, and agent consensus as executable strategy specifications. Supply each strategy's actual features, universe, model policy, entry/exit rules, sizing, cadence, and risk constraints.
- Implement regime-matched selection, performance-weighted blending, bandit selection, and manual override. Base allocation on real attributed performance; constrain exploration to eligible validated candidates.
- Combine strategy sleeves into a coherent portfolio with traceable capital allocation, netted intents, attribution, and portfolio-level risk. Handle incompatible signals, overlapping positions, strategy transitions, and holdings removed from a universe.
- Define agent consensus explicitly as the brief's model-free exception: a configured eligible universe, named voting roles, at least four agreeing recommendations, and an absolute risk veto. Record which recommendations derive from others rather than implying independent votes.

Acceptance: every strategy changes downstream behavior in a complete run; changing strategy cannot merely change a display label. Each selection mode, rebalance frequency, defensive transition, and manual override is exercised. Strategy implementation completeness does not imply every candidate earns activation under the performance gates.

### E. Complete agent desk

Brief coverage: pages 2, 4–6, 12–13.

- Connect all five specialist roles to real tools and typed outputs: fundamental, sentiment, technical, risk manager, and strategist. Use equity or crypto tools appropriately, including business/token quality, news, events, technical context, and portfolio/correlation information.
- Wire the selected MCP evidence tools into the application graph and shared services. Allow approved Alpaca observation operations only; broker mutations remain exclusive to the execution owner. Enforce permissions beneath generic tool dispatch as well as direct calls.
- Implement the LangGraph fan-out/fan-in, two configurable challenge rounds, revised reasoning, absolute risk veto, and final synthesis with quantitative context. The risk role must have the data and reasoning capability described in the brief; three static conditions are insufficient.
- Preserve decisions, confidence, target price, stop loss, position-size suggestion, horizon, reasoning, full reports, catalysts, risk flags, debate transcript, and evidence provenance.
- Give BUY/HOLD/REDUCE/EXIT/ABSTAIN and veto precise portfolio semantics. Analysis failure does not silently become liquidation. The optimizer can adjust sizing; deterministic controls remain binding on every order.
- Integrate existing earnings/transcript analysis and useful prior-analysis history. Enforce as-of constraints on retrieved history during replay. Complete alternative configured LLM-provider routing without hard-coded provider-specific model substitutions.
- Connect Jev use cases J1–J4/J6: news and transcript classification, claim/source verification, evidence and memory relevance, and semantic disagreement checks. Replace only evaluated subtasks; retain full analyst reasoning, the configured debate process, risk review, and binding vetoes. Wire the currently disconnected transcript node and persist all judgments with their source/model/question provenance.
- Rebuild confidence-based routing and aggregation around the distinct schema meanings. Never treat rescaled Jev confidence, analyst conviction, and HMM probabilities as equivalent or use model confidence directly as a sizing instruction.
- Enforce token, request, retry, latency, and run/day spend budgets. Record actual usage and pricing version; unknown cost is visible. Isolate untrusted source material from operational instructions and tool permissions.

Acceptance: actual provider-backed equity and crypto analyses produce complete auditable records; debate affects synthesis; veto survives contradictory outputs; missing/stale sources produce explicit bounded behavior. Historical quant tests and prospective agent-value evaluation remain separately identifiable.

### F. Portfolio construction, risk, and every execution path

Brief coverage: pages 6–7, 12, 15.

- Implement and connect all five optimizers: equal-risk-contribution risk parity, mean-variance with Ledoit-Wolf covariance shrinkage, Black-Litterman, HRP, and maximum diversification. Use actual returns/covariances and validated views. Validate constraints again after solver fallback, rounding, and netting.
- Enforce the complete long-only/unlevered policy: 10% equity position, 5% crypto position, 30% sector, regime-adjusted aggregate crypto, 5% minimum cash, $1M ADV, less than 1% ADV order participation, 15% drawdown buy halt, and 25% daily turnover. Include fees, reserved cash, pending orders, and projected exposures.
- Persist positions, cash, cost basis, fills, pending/cancelled orders, contingent protection, NAV/high-water marks, turnover, and strategy P&L. Correct buy/sell arithmetic and allow valid risk-reducing orders under a buy halt.
- Complete the market-data-driven paper simulator, Alpaca equity adapter, CCXT adapters for Binance and Kraken, and an operational QuantConnect LEAN bridge behind equivalent execution/status/reconciliation contracts. LEAN acceptance requires actual LEAN execution, not only writing an algorithm file.
- Rebuild `src/execution/paper_trader.py` around persistent accounting and the shared replay contracts in C3. Use the C7 replay/paper reconciliation cases to verify fill handling, balances, attribution, and restart continuity; do not maintain a second incompatible definition of portfolio returns.
- Route equity and crypto orders to eligible configured venues with aggregate risk accounting. Define exclusive ownership of each trading book so direct adapters and LEAN cannot execute the same intent twice.
- Handle lot/tick sizes, minimum notional, market/limit orders and existing bracket/trailing-stop support, supported time-in-force, partial fills, cancellations, rejects, fees, and late broker acknowledgements.
- Use stable order-intent/client IDs, reconcile ambiguous timeouts before resubmission, and recover after crashes at every submission/fill boundary. Do not claim network-level exactly-once delivery; demonstrate deduplicated economic intents and broker reconciliation.

Acceptance: each venue has actual connection/contract evidence using its supported verification environment; both asset classes complete order-to-ledger round trips. Crash, duplicate-run, partial-fill, stale-price, insufficient-balance, and venue-outage exercises preserve invariants. Live execution code is part of delivery; committing real capital remains subject to the brief's go-live criteria.

### G. Native RD-Agent and real custom evolution

Brief coverage: pages 8–11.

- Bind a pinned Microsoft RD-Agent runtime through an adapter that renders scenarios, invokes real jobs, handles logs/results, supports cancellation/timeouts, and reports resource/spend use. Configure Linux/Docker execution, chat/JSON support, embeddings, persistent CoSTEER knowledge, and durable workspaces.
- Connect Valyu's current hosted MCP for cited source discovery and research, with verified source entitlements and bounded search/research spending. Preserve evidence with hypotheses; external reports still require actual experiments and independent validation before promotion.
- Deliver every named workflow: `fin_factor`, `fin_model`, `fin_quant`, `fin_factor_report`, `general_model`, `ui`, and `health_check`. Include actual PDF-folder factor extraction and paper-URL model implementation, with source provenance.
- Configure US Qlib data, instruments, calendar, benchmark, time segments, and execution assumptions explicitly. Independent project validation must not inherit a reference scenario's market assumptions unchecked.
- Implement whole-strategy evolution and regime-detector evolution as the project's own research workflows. Replace random Sharpe/drawdown values with real controlled experiments against an existing baseline.
- Keep crypto research operational through the custom research path described in the brief while native RD-Agent quant research uses its verified equity scenario. A native crypto Qlib scenario is an extension; crypto trading, crypto factors, and real crypto evaluation remain required in this release.
- Import all native/custom/copilot/paper-derived outputs through the same quarantine and promotion boundary. Validate the exact implementation later used for inference; eliminate expression approximation and host execution of arbitrary generated code.
- Store candidate lineage, dataset/splits, code/environment hashes, trial outcomes, validation results, and lifecycle state. Promote only eligible artifacts by an atomic manifest update; retain rollback and ensure a failed research run cannot damage trading.
- Extend hypothesis generation to propose bounded semantic questions; use Jev to compute candidate text features for existing model experiments. Track all question revisions and trials, keep code/memo generation with the research models, and enforce the common validation/promotion boundary for J5 outputs.

Acceptance: actual runs exercise each named workflow, persistent knowledge survives restart, and generated artifacts undergo reproducible independent evaluation. Both custom evolution loops run real comparisons. Rejected, simulated, or incomplete candidates cannot become active.

### H. Full operating cycle, monitoring, and operator interface

Brief coverage: pages 1, 3, 11–14.

- Implement the entire unattended cycle: data refresh, scheduled retraining, inference, agent review, portfolio construction, risk, orders, reconciliation, attribution, and reporting. Research runs on its independent cadence. Complete existing intraday operation with actual return-based signals and the same risk/ledger services.
- Persist stage/checkpoint status and use atomic run ownership. Deduplicate by account, strategy/version, and scheduled decision window; a date-only key is insufficient for intraday operation.
- Handle exchange holidays, DST, missed schedules, long-running jobs, restarts, expired inputs, and partial rebalance outcomes. Continue reconciliation and monitoring during outages; prevent unsafe new exposure.
- Deliver working trading and research Streamlit views: holdings, cash, P&L, allocations, run health, data freshness, model/factor performance, agents/debates, risk decisions, orders/fills, research jobs, candidates, validation, and active versions.
- Connect strategy overrides, supported operational controls, and configuration through the shared services. Show the reason a job or strategy is unavailable. No dashboard section is satisfied by static sample data.
- Deliver actual Telegram daily reports and separate drawdown, drift, risk, and system-failure alerts. Include predictions, agent reasoning, before/after holdings, blocked orders, fills, and costs in structured audit records.
- Add deployment/upgrade/rollback instructions, backup and restore procedures, secret/configuration handling, health checks, and measured CPU/RAM/disk/latency/spend. Test the brief's proposed host sizing and budget rather than assuming they hold.
- Expose Jev model/question/policy versions, source-linked judgments, abstentions, latency, cache use, errors, and measured spending. A Jev outage blocks affected new exposure when required; it does not imply neutral evidence or liquidation, and deterministic protection/reconciliation continues.

Acceptance: the deployed system runs through both market calendars without operator intervention, reports from actual ledger/research records, and passes recovery and alert-delivery exercises. An operator can trace a historical trade from evidence through decision, risk checks, submission, and reconciliation.

### I. Full-system acceptance and release evidence

Brief coverage: pages 14–15 and all cross-component promises.

- Maintain requirement IDs A–I with individual subrequirements tied to implementations, actual runtime entry points, tests, and evidence artifacts. No requirement is complete solely because its module exists or its mocked unit tests pass.
- Run complete real-data equity and crypto workflows for all six strategies, all model/optimizer variants, each selection mode, and every venue integration. Use risk- and calendar-appropriate cases rather than attempting meaningless model/asset/venue combinations.
- Exercise full research-to-promotion-to-inference lineage and rollback. Verify operating runs pin a consistent artifact version while research promotes the next version.
- Run matched prospective quant-only, quant-plus-veto, and full-agent evaluation portfolios to measure the agent contribution. These are measurement controls; the agent desk remains part of the delivered system.
- Freeze/version evaluation configurations, define paper/backtest discrepancy tolerances in advance, and perform reasoning review across at least 20 names.
- Apply the [Jev acceptance and activation requirements](JEV_INTEGRATION_DECISION.md#acceptance-and-activation-evidence): labeled task accuracy, probability calibration, adversarial/option-order cases, actual provider calls, deterministic replay, failure/rollback behavior, and measured operating benefits. Compare matched full-system runs with/without Jev; its integration cannot bypass the existing research or live-capital gates.

The brief's go-live gates remain mandatory: walk-forward Sharpe above 1 after the specified realistic costs; zero validation failures; above the 95th percentile of the matched random baseline; positive Sharpe in at least two regimes; three months of paper behavior consistent with predefined expectations; paper maximum drawdown below 15%; sound manual reasoning review; 30 consecutive daily runs without unrecoverable failure; verified alert delivery; documented operating spend within budget. Historical statistical evidence must include the stated t-statistic above 2 over at least 252 trading observations, alongside the controls for repeated trials.

Acceptance has two distinct facts: whether all required system capabilities work, and whether the evaluated configuration has earned live-capital eligibility. Both are reported explicitly. Complete software can reject an unprofitable strategy. A failed performance gate triggers further research or a clear rejection, never a lowered threshold or invented success.

## Dependency sequence and release boundaries

Begin with shared contracts, reproducibility, data provenance, chronological accounting, and the transactional ledger. These establish the meanings of prices, returns, risk, and execution state used everywhere else. Complete the C3/C4 replay and split rebuilds with the shared paper accounting in F before accepting strategy performance; C5/C6 validation and failure reporting must be connected before any research candidate can be promoted. C7 supplies the mandatory acceptance evidence for these rebuilds.

Then complete the model/factor framework, all strategies/regimes, all agents, and all optimizer/venue paths against those contracts. Research binding depends on valid experiments and artifact contracts; promotion depends on independent validation. Scheduler, CLI, and dashboard integration accompanies each capability rather than being postponed to a final cosmetic pass.

Finish with complete deployed-system integration, failure recovery, resource/cost measurement, and the required prospective observation. Intermediate demonstrations are checkpoints, not reduced releases or permission to drop remaining scope. Every workstream A–I is required for completion.

The build may verify one path before the others to expose contract mistakes early. It must continue through the entire capability set. Equity-first verification does not redefine the deliverable as equity-only, and paper-first verification does not remove live venue adapters from scope.

## Interpretation of ambiguous requirements

- The brief takes precedence over contradictory code defaults: six-month walk-forward windows, IC above 0.03, ICIR above 0.5, and 30% defensive cash.
- The 5–25% crypto allocation band applies when the selected regime/strategy permits crypto. Defensive/crisis policy can require zero; constraints are resolved before optimization.
- Preparation completes on the brief's pre-open schedule. Order submission and fill timing must be explicit per venue/session; queued orders cannot be accounted as pre-open fills. Crypto uses its own continuous session calendar.
- Agent consensus is the brief's explicit model-free strategy. Other strategy paths retain quantitative ranking before the agent desk.
- Native RD-Agent equity research and functioning custom crypto research satisfy the research split stated on page 11. The custom strategy/regime loops still require real evaluation and common promotion gates.
- Research success means producing and evaluating traceable experiments. “Self-improving” does not mean every run must discover a better strategy; the system must preserve the current version when evidence does not justify a replacement.

## Effort and external prerequisites

The previous 7–11-week estimate described a narrower delivery and does not apply to this scope. A preliminary allowance for the complete build is **20–32 engineer-weeks**, before refinement through workstream-level design: A 1–2, B 3–5, C 3–4, D 2–3, E 2–3, F 3–5, G 3–5, H 2–3, I 1–2. These are effort estimates, not elapsed-time promises or measured coding-agent throughput. Data/venue access and discovered integration defects can change them.

The three-month prospective observation is a real calendar requirement. It can overlap remaining independent integration work once the observed configuration is complete and frozen; material strategy changes require a new comparable evaluation window.

Required external inputs are usable historical/contextual data, LLM chat and embedding access, Linux/Docker research infrastructure, configured Alpaca/exchange/LEAN access, and the operator's notification destination. Credentials must be provisioned privately when needed. Accounts, spending, and live-capital actions are not assumed merely from this planning request. Missing access is reported as a concrete unverified dependency, not replaced with a mock and marked done.

The selected MCP integrations also require provider entitlements, including FMP transcript access and Valyu source/research access. Validate those against the required coverage and operating budget. Apply the [MCP acceptance requirements](FINANCE_MCP_ASSESSMENT.md#required-acceptance-evidence) in workstream I, including actual provider calls, historical-cutoff checks, failure recovery, permission enforcement, and measured operational use.

Jev additionally requires provisioned TypeSafe access, permitted use of the submitted source material, a verified pinned model/SDK, and measured budget/latency/quality acceptance. Account provisioning and live provider evaluation remain unverified. Refine effort estimates to include the shared service, confidence migration, labeled datasets, and proposed integrations; no Jev cost or development-time saving is assumed in the preliminary estimate.

The full brief uses Streamlit and a single operator. A different frontend, multi-user SaaS, marketplace, or mobile product is outside this specification; none is needed to deliver its complete trading and research system.

## Next implementation action

The register and shared contracts now exist. Continue from the
[current ordered action list](BUILD_REQUIREMENTS.md#current-next-actions-sequence-not-replacement-scope):

1. Correct and test the LEAN stop-market overload, then exercise actual DAY,
   trailing, nonzero-fee and partial-fill scenarios with honest simulation limits.
2. In parallel when evidence is available, establish IBKR income coverage and
   payment/deduction linkage; preserve the source gate until sufficient authority exists.
3. Run bounded native RD research with configured provider/model access and
   explicit spending limits; offline help/import smoke is already verified.
4. Complete separately reviewed ensemble consumer integration, discrepancy
   settlement and bracket/OCO ownership.
5. Carry every remaining A–I and accepted J requirement through source-complete
   integration, external acceptance and the unchanged prospective release gates.

This ordering is a dependency-based work queue, not a replacement or reduction
of the full plan. The register contains subgroup IDs, evidence and open items.
