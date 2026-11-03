# Actual LEAN paper engine, 2026-11-03

> Follow-up: the shutdown failure described below was subsequently resolved in
> the locally patched pinned image. See [clean-exit verification](lean-shutdown-fix-2026-11-03.md).
> This report preserves the original failing evidence.

## Result

**Paper-order execution verified; complete runtime acceptance remains blocked.**
The actual source-built LEAN engine initialized our production algorithm/port,
received its own synthetic ticks, submitted one canonical GTC market buy through
`QCAlgorithm.MarketOrder`, emitted actual Submitted/Filled OrderEvents, and
reconciled the canonical ledger. Execution compliance was PASS with lineage
`lean-paper`. A fresh read-only ledger connection after process exit again found
one fill and PASS, with no blocking codes.

The process then exited **139** during `PythonInitializer.Shutdown`:

```text
System.InvalidOperationException: GIL must always be released, and it must be released from the same thread that acquired it.
   at Python.Runtime.Py.GILState.Finalize()
   at System.GC.RunFinalizers()
```

A separate minimal QCAlgorithm (AddCrypto, then Quit on OnData; no project
imports, execution bridge or orders) reproduced the same shutdown exception and
exit status. This isolates the failure to the tested runtime combination; it
is not proof that every upstream distribution has this defect. No upstream
shutdown code was patched and no exception/exit status was suppressed.

## Runtime/source evidence

- Public QuantConnect/Lean source revision:
  `3b2bf19e597805cc917dd199ed9ee7d10d5ceb84` (2026-11-02).
- Image: `quantagentlab-lean:source-3b2bf19`, actual ID
  `sha256:c73e5ed201c54b4c6c410df778a033e5ef1cdc3ec9dc0314129bd96366fc21a0`.
- Actual engine assembly version 2.5.0.0, Python 3.11.16, .NET 10 SDK source build.
- SDK resolved digest:
  `sha256:de4fe90a98845d75910b9769662465c84de1a934665c431279d1382b2bddc02b`.
- Dependency base: previously tested project image `quantagentlab:smoke-a233bb5`;
  current execution source mounted read-only and bound by `prepare_project`'s
  implementation hashes. The committed reproduction Dockerfile pins the SDK
  digest resolved during the actual tag-based build, but keeps the local project
  base tag explicit; it is not a fully reproducible public image lock.
- Only upstream source addition: a test-only `FakeDataQueue` subclass setting
  two data points/sec/symbol. No economic or engine shutdown modification.
- LiveMode=true with actual **PaperBrokerage** and
  BacktestingTransactionHandler, fixed synthetic account/cash, network disabled.
- No authenticated/public market-data call, provider key, live-capital permission,
  host .NET installation or new Python dependency.
- All exact test containers removed; image retained. Final container
  `adf71213ae810f75623fd528f6813e91b10f1d7e76bf08e24d7356d42e4114eb`
  had exit139 and cleanup exit0. Control and earlier attempts also cleaned up.

Persistent local evidence (private parent):
`~/.local/share/quantagentlab/runtime-smoke/lean-20261103/` contains console logs,
launch/inspection records, actual SQLite ledgers, configurations, original
implementation manifests, decimal/tick assertions, filled-run result, no-project
control, build log and per-file SHA256 inventory. These contain synthetic account
artifacts only; they are not production source authority. No report/private
credential was read. Test values use upstream's clock-driven synthetic prices,
not a deterministic historical series.

## Compatibility corrections established by the real engine

1. CLR enum `.ToString()` preserves names such as `Crypto` and `Filled`;
   Python `str(enum)` can instead produce `CRYPTO`/`FILLED`.
2. Algorithm-owned fields now use `_qal_` prefixes: `self.execution` collided
   with QCAlgorithm's native Execution property.
3. This runtime decodes `System.Decimal.Parse` and monetary properties into
   Python floats. Native constructor words retain System.Decimal. The boundary
   constructs exact 96-bit coefficients/scale and verifies the round trip;
   unrepresentable inputs fail without rounding. Compiled CLR expressions read
   **only decimal-typed properties** and format invariant text before returning
   to Python. Direct float admission remains forbidden.
4. Actual test: `100000.000000000000000000001` stored in a real Cash object was
   returned identically by the new boundary while direct Python property access
   was float. This validates precision beyond binary float, not just a small
   integer fixture.
5. Native DateTime properties are also decoded by Python.NET. Receipt clocks,
   data EndTime (with actual exchange timezone conversion) and event UtcTime now
   read original CLR ticks before decoding. Exact `638712864000000001` ticks
   project to `2025-01-01T00:00:00.000001Z`, using the existing forward-ceiling
   microsecond convention; no floating Unix timestamps.
6. Actual account currency is `Portfolio.CashBook.AccountCurrency` rather than
   a nonexistent `Portfolio.AccountCurrency` property.

The new helper is included automatically in existing full execution-module
manifest hashing. An intermediate formatting change correctly caused the prepared
manifest gate to reject initialization; a fresh preparation was used afterward.
No manifest/source/fee/cutoff guard was bypassed to obtain a fill.

## Validation and limits

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest \
 tests/test_lean_decimal.py tests/test_lean_runtime_enum.py \
 tests/test_lean_execution.py tests/test_lean_day.py tests/test_lean_trailing.py \
 tests/test_lean_binance_references.py tests/test_signal_bridge.py \
 --disable-socket --allow-unix-socket -q
```

**121 passed in 5.45s**, no skips. Ruff/check-format and diff checks also run on
changed files. Offline fake-port tests explicitly inject their numeric property
reader; they do not masquerade as real CLR precision tests. Actual engine
precision/tick assertions live in `environments/lean-smoke/smoke_algorithm.py`.

Scope: one GTC crypto market order with zero runtime fee, actual synthetic
PaperBrokerage. Clean shutdown, restart/recovery, cancellation, partial fills,
nonzero fee variants, actual DAY/calendar and trailing overload acceptance,
real broker credentials and production market data remain unverified. Existing
DAY unit tests do **not** establish compatibility of its remaining native
DateTime conversion path with this runtime. No sufficient income authority,
financial strategy quality or release readiness is claimed.

Next runtime task: resolve the independently reproduced Python/GIL shutdown
failure using a reviewed compatible runtime or narrowly reviewed upstream fix;
repeat the same diagnostic requiring clean process exit, then expand acceptance
one scenario at a time.
