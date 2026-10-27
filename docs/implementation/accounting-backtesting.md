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
