# Accounting and backtesting implementation evidence

This lane implements C3–C7 and the durable accounting/risk portion of F. It is an integration checkpoint toward the full plan, not full-system acceptance. Synthetic fixtures are confined to tests. No provider/venue acceptance or real capital trade has been performed.

## Stable accounting boundary

`src.domain.portfolio` contains immutable Decimal contracts: `AccountSpec`, `OrderIntent`, `Fill`, `PriceMark`, `ExecutionBar`, `CorporateAction`, `Holding`, `PortfolioSnapshot`, `InstrumentRisk`, `RiskPolicy`, `FillPolicy`, and `RankedSignal`. Import directly from this module. Account IDs start with `paper:` or `backtest:`. All new contract clocks require timezones.

`PortfolioLedger(path)` creates/reopens a SQLite ledger. `create_account(AccountSpec)` is idempotent only for identical initial conditions. `submit`, `apply_fill`, `cancel`, `mark`, `corporate_action`, `snapshot`, `history`, `fills`, and `reconcile` implement persistent order/cash/holdings accounting. `atomic()` serializes writes and supports nested savepoints. Duplicate identifiers with changed contents fail. Cash and basis use exact decimal text; weighted-average basis includes entry fees. Sales allocate basis proportionately. This is not FIFO tax-lot accounting.

`PaperExecution(ledger, account_id, FillPolicy(), RiskPolicy())` owns risk-checked submission and observed-open simulation. `submit(intent, metadata)` and `open_bar(bar, metadata)` apply projected risk including pending buy exposure, reservations, fees, holdings, sector, crypto aggregate, cash reserve against NAV, ADV, strict participation, lot/minimum notional, drawdown, and daily turnover. Orders cannot execute at their decision instant. `close_bar` records marks and expiry. Partial fills share opening-volume capacity. Repeated bar processing cannot reuse liquidity after restart. Open-only limit/stop handling intentionally makes no intrabar high/low ordering assumption.

`execution_bar_from_market` is the explicit adapter over the data lane's `MarketBar`; it invokes `require_execution_price()` and transfers Decimal values unchanged. Callers supply actual opening volume, explicit calendar session bounds/version, and FX rate. Whole-day volume is not used as opening liquidity. Equity sessions come from a versioned provider/calendar artifact; this lane does not invent a holiday calendar. Crypto explicitly uses `24/7`. Adjusted/total-return prices and perpetual instruments cannot become trade prices via the adapter.

Base-currency accounting supports foreign quote currencies via explicit quote-to-base FX marks/fills; no synthetic conversion prices are supplied. There is no multi-currency cash wallet yet. Corporate splits cancel resting orders before changing units; dividends and delisting proceeds are explicit base-currency per-share cash events. Split actions remove old protective price levels and require protection to be re-established.

`PaperTrader(initial_cash, ..., ledger_path=..., account_id='paper:default', base_currency='USD')` exposes the canonical engine as `.execution` and `.ledger`. Its legacy `execute_order(Order, prices)` is explicitly an observed-current-quote compatibility adapter, not historical replay. Legacy naive timestamps use the host timezone. It records fills using the same ledger and persists bracket/trailing protection. Operating callers must migrate to typed instrument/risk/event contracts; the legacy schema lacks complete risk metadata. A directly constructed legacy session without a path receives an isolated on-disk temporary database; `create_broker(config)` always supplies the stable configured output path.

Requested coordinator config: `paper_ledger_path` derived from `output_dir/paper-ledger.sqlite3`, `paper_account_id='paper:default'`, `paper_base_currency='USD'`, and six-month expanding OOS defaults. No shared config/schema/dependency files were edited. Risk policies use fractions; legacy RiskConfig uses percentages.

## Executed checks at accounting contract checkpoint

From this worktree with `PYTHONPATH="$PWD"`:

```
/tmp/quantagentlab-review-venv/bin/python -m pytest -q tests/test_durable_accounting.py tests/test_paper_trader.py tests/test_paper_advanced.py --basetemp=/tmp/quantagent-accounting-pytest
```

Result: **30 passed**. Includes hand-calculated fees/cash/basis/P&L, rollback on unaffordable fill, partial/duplicate fills, reservations, actual subprocess restart and deduplication, sell-to-zero, NAV reserve, pending position exposure/crypto cap/buy halt, opening-liquidity restart, split/dividend/delist reconciliation, crypto weekend, stale marks, double-sell/cross-account failure, persistent legacy paper state, and contingent quote execution.

Changed-file Ruff passes (five import-order findings auto-fixed before commit). Existing old tests inspecting `_contingent` now inspect durable `protected_tickers`; average-cost expectation now includes entry commissions.

## Remaining at this checkpoint

Chronological replay, six-month purged walk-forward, benchmarks/validation and explicit CLI failure behavior are being implemented next. All five optimizer end-to-end proofs, actual Alpaca/CCXT/LEAN execution/reconciliation, actual historical provider/calendar/open-liquidity/ADV/universe inputs, Qlib/RD-Agent integration, promotion trial/holdout evidence, complete strategy/regime/agent pipeline integration, and prospective observation remain mandatory. An optimizer zero-weight fallback was corrected to respect position caps; this alone does not verify all optimizers.

## Chronological replay / research checkpoint

The next checkpoint replaces `src/core/backtester.py`, `walk_forward.py`, `benchmark.py`, `validation.py`, and `scripts/run_backtest.py`. `_mock_backtest_result`, fold-wide ranking and constant daily smoothing are removed. Label-only backtests fail explicitly. The retained `_top_k_period_returns` method is a date-local, zero-cost label diagnostic only; runtime performance never uses it.

### Runtime inputs and entry points

`PortfolioReplay(ledger_path).run(...)` requires keyword arguments:

- `context: RunContext` in backtest mode, matching the isolated `account: AccountSpec`.
- `bars: list[ExecutionBar]`, `metadata: list[InstrumentRisk]`, `universes: list[UniverseEligibility]`.
- Exactly one of `signals: list[RankedSignal]` or `targets: list[PortfolioTarget]`. A target carries `universe_version`, target/strategy/artifact/data IDs, decision time and typed instrument weights. Multiple strategy sleeves must be explicitly netted into a target before replay.
- Optional corporate `actions`, `topk`, `FillPolicy`, `RiskPolicy`, explicit daily `valuation_times`, `samples_per_year`, `valuation_calendar`, and provenance evidence references.

`UniverseEligibility` carries `universe_version`, `data_snapshot_id`, `effective_at`, `available_at`, and unique `instrument_ids`. Both effectiveness and documented availability must precede decisions. A later-known membership snapshot cannot qualify a historical signal/positive target. Calendar and universe correctness still depend on real source artifacts; merely asserting a version does not certify the provider.

Replay persists immutable input records, bars, targets, orders, fills, account state, daily NAV, run manifest and result. It processes observed closes/actions/opens/decisions in timestamp order. Same-boundary decisions cannot fill that open. New targets cancel old resting rebalance intents. Pending sales do not fund purchases; rejected new orders appear in decisions/diagnostics. Market gaps trigger risk checks again. Evaluation ends with valued holdings and explicit cancellation of unfilled remainders, without a synthetic terminal liquidation. Initial capital is included in the NAV and drawdown path.

`WalkForwardRunner.run(features, labels, ..., label_end_times=..., feature_available_at=..., bars=..., metadata=..., universes=..., context=..., account=..., ledger_path=..., artifact_dir=..., universe_version=...)` requires sorted, unique `(decision_at, instrument_id)` indexes, identical label/availability alignment, timezone-aware clocks and finite eligible data. Defaults: 12 initial training months, 1 validation month, six-month OOS windows, 5 calendar-day embargo. Train and validation labels must end before their respective segment cutoff. The boundaries are half-open. Final partial windows are marked `incomplete`, never zero-filled or promoted.

A fresh `model_factory()` instance fits each fold; `train_with_validation(X_train,y_train,X_validation,y_validation)` is used if the model exposes it, otherwise `train` or sklearn `fit`. Prediction is performed independently per decision date. Preprocessing belongs inside the fresh model, fitted only on eligible training data. The actual fitted object is serialized and hashed; changing an existing model artifact fails. All fold predictions feed one continuous OOS replay, preserving cash/holdings across boundaries. Feature/model/universe/label hashes, exact segment bounds and row counts are retained. Labels unresolved at the run cutoff are excluded from IC.

`ReplayResult` is schema `2.0`, engine `chronological-ledger-v1`. It contains status, run/account IDs, frozen manifest, nullable `ReplayMetrics`, initial-inclusive NAV/timestamps/snapshots, actual daily returns (one fewer than NAV), fill-level fees/cash changes/realized P&L/notional/strategy IDs, decisions, fold evidence, date-local IC, reconciliation and diagnostics. Annualized Sharpe uses mean daily return / sample standard deviation, zero cash return and explicit annualization. Undefined metrics remain null. Costs are one-way, charged once in fills. Attribution currently identifies the executing strategy; complete overlapping-sleeve ownership/P&L requires the common netting/strategy integration.

The real CLI is:

```
PYTHONPATH="$PWD" /tmp/quantagentlab-review-venv/bin/python -m scripts.run_backtest run --input /path/to/actual-bundle.json --ledger /path/to/isolated-run.sqlite3 --output /path/to/result.json
PYTHONPATH="$PWD" /tmp/quantagentlab-review-venv/bin/python -m scripts.run_backtest compare --input /path/to/actual-oos-bundle.json --result /path/to/result.json --output /path/to/comparison --simulations 1000
PYTHONPATH="$PWD" /tmp/quantagentlab-review-venv/bin/python -m scripts.run_backtest validate --result /path/to/result.json --ledger /path/to/isolated-run.sqlite3 --costs /path/to/comparison/costs --benchmarks /path/to/comparison/benchmarks.json --output /path/to/validation.json
```

The JSON bundle uses the exact contract fields above; `load_inputs(Path)` is the public parser. A `walk_forward` object may instead contain `rows` with `decision_at`, `instrument_id`, numeric `features`, `label`, `label_end_at`, `feature_available_at`; it also specifies `model_name`, `universe_version` and optional split settings. Do not include precomputed signals/targets in a training bundle. After training, comparisons require the stored OOS signal artifact and exact OOS execution span, rather than the entire development dataset.

Missing inputs, failed models and accounting errors exit **1** with a separate failure JSON; incomplete evaluation, unavailable benchmark coverage or rejected validation exits **2**. Successful replay exits **0**, which denotes completed accounting, not promotion eligibility. The canonical package CLI/research/scheduler/dashboard are coordinator-owned and must adopt these entry points; their old label-only API cannot silently continue.

### Benchmarks and one validation service

`BenchmarkRunner.compare_replay` reruns the ledger with identical timestamps, costs, risk policies, date-local random-selection universe and top-K. It persists all 1,000 random NAV paths, seeded at 42 by default. When top-K spans the universe or all random Sharpes are undefined/identical, the percentile is unavailable, never fabricated as 50. SPY, BTC/USD, SPY/AGG 60/40 and equal-weight paths use supplied real bars/metadata; absence is explicit. Named baselines are constrained sleeves with remaining capital in cash. The 60/40 ratio is preserved when scaling to position limits. Equal-weight uses the full date-local universe and explicitly discloses any breadth difference from top-K; random selection matches top-K. `cost_sensitivity` performs actual fill reruns at paired equity/crypto one-way costs 0/0, 10/20, 20/40 bps.

`ValidationSuite.validate(result, ledger_path=..., cost_results=..., benchmark=...)` is the single authority; `BacktestValidator` delegates to it only to preserve legacy return shape. It rejects legacy results, incomplete runs, unbound ledger evidence, altered metrics/fills/manifests, unpurged splits, missing provenance, insufficient matched cost/benchmark evidence, weak IC/ICIR/correlation gates, missing trial/holdout records, and absent causal regime evidence. It does not shuffle returns or infer survivorship quality from fold count. Statistical evidence uses a reproducible moving-block bootstrap with trial-count correction and rejects unresolved/insufficient bounds.

Durable validation checks the exact canonical manifest hash, persisted run/status/result, complete fill/order/economic content, journal-derived balances and persisted snapshots. All recomputable metrics are recalculated, including Sharpe, annual return, volatility, Sortino, Calmar, win rate, turnover, fees and fill count. Checks are explicit branches, not production `assert` statements that disappear under `python -O`.

Evidence references currently required in `manifest.provenance`: `universe_history_id`, `availability_audit_id`, `calendar_artifact_id`, `corporate_actions_id`, `max_existing_factor_correlation`, `regime_evaluation_artifact`, and `trial_registry={artifact_id,trials,untouched_holdout_id}`. These fields preserve evidence linkage; the coordinator's quarantine registry must independently load/hash the artifacts, count all attempted trials, protect holdouts and verify their content before promotion. A reference string alone does not establish artifact quality.

### Executed verification

Combined command from this worktree, without network access:

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" AIQUANT_OUTPUT_DIR=/tmp/quantagent-accounting-final/outputs AIQUANT_LOG_DIR=/tmp/quantagent-accounting-final/logs /tmp/quantagentlab-review-venv/bin/python -m pytest -q tests/test_durable_accounting.py tests/test_core_backtest.py tests/test_walk_forward.py tests/test_core_benchmark.py tests/test_validation.py tests/test_paper_trader.py tests/test_paper_advanced.py tests/test_risk_controls.py tests/test_portfolio_optimizer.py tests/test_order_manager.py tests/test_broker_protocol.py tests/test_signal_bridge.py --disable-socket --basetemp=/tmp/quantagent-accounting-final-pytest
```

Result before the final validation-review regression: **168 passed in 24.37s**. Additional final targeted command: `pytest -q tests/test_validation.py tests/test_core_backtest.py --disable-socket --basetemp=/tmp/quantagent-accounting-validation-pytest` with this worktree on PYTHONPATH result: **15 passed in 0.62s**, including the new metric/manifest/full-fill tamper cases. Changed-file Ruff passes. No dependency was installed into the shared environment.

Acceptance cases now include the exact +10% then −20% date-local selection diagnostic; an actual NAV path 10,000 → 10,000 → 11,000 → 8,800 (−12%, max drawdown −20%); multi-instrument fees/cash/basis; future feature and universe perturbations; label overlap rejection; six-month split boundaries; cross-fold cash/holdings continuity; actual fitted Ridge/LightGBM/CatBoost/XGBoost/ensemble code on synthetic test data; all 1,000 random ledger paths; actual named benchmark and cost replay; partial/duplicate fills; process restart and actual process exit during an uncommitted fill; exact paper/replay economic-state parity; equity split/dividend/delist; weekend crypto; FX basis/valuation; stale prices; pending sector/crypto risk; strict participation boundary; limits/FOK/cancellation; and failed training/data/incomplete CLI outcomes.

### Remaining full-plan work and integration requirements

- Real historical provider evidence, documented point-in-time universes, source/calendar versions, opening liquidity, ADV/sector/lot metadata, corporate actions and FX coverage are external/data-lane prerequisites. All financial data in these tests is explicitly synthetic. No live historical-provider performance claim is made.
- Coordinator must adopt v2 results, typed paper submission, nullable metrics, the explicit bundle loader and the quarantine validation binding in the canonical CLI/pipeline/research/dashboard. `src/core/factor_backtester.py` still needs its coordinator-owned migration from label-only calls. Old integration tests exercising successful mock/label-only behavior must migrate to actual bundles.
- All five optimizers, all strategies/regimes and sleeves need real connected replay acceptance. The zero-weight optimizer fallback is fixed, and existing optimizer unit tests pass; this is not the full F optimizer proof.
- Actual Alpaca, Binance/Kraken CCXT and LEAN connection/order/reconciliation evidence, venue ownership, ambiguous timeout recovery, exchange tick/minimum rules and typed contingent-order lifecycle remain mandatory F follow-on. The quote facade's durable protective orders are not proof of all typed venue capabilities.
- Multi-currency cash wallets, full strategy-sleeve cost ownership and economic attribution, intrabar advanced order policy and execution after funding-dependent target sells require further full-plan integration. Current behavior is explicit base-currency conversion, executing-strategy attribution and conservative open-only fills; no pending sale is treated as settled cash.
- Qlib/RD-Agent real experiment evidence, registry/holdout integrity, calibration/multiple-testing acceptance, complete operating recovery and prospective three-month observation remain required. Legacy invalid reports cannot become approved artifacts.
