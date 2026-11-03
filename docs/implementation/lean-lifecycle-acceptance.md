# Actual LEAN resting-limit cancellation acceptance

## Scope and result

On 2026-11-03 the network-disabled, synthetic **PaperBrokerage** engine accepted
a canonical GTC buy limit, left it resting, and canceled that exact ticket through
`execution.cancel`. Actual callbacks were `Submitted`, `CancelPending`, `Canceled`.
The original client ID/tag and native ticket survived cancellation; there were
no fills, fees, holdings, or cash changes. Canonical reconciliation and a fresh
read-only ledger reopen both returned execution compliance **PASS**, lineage
`lean-paper`. Process exit and exact-container cleanup must separately be zero;
`result.json` alone is insufficient.

Base checkout: `bc48d273eac9080267330924f0ca0e642718f0d4`. Runtime image:
`sha256:94ee09ba94c575561b11bc19cc6f5406f507d1f2fa436449b13ea3532612b2d2`.
The locally patched shutdown runtime and upstream source pin are described in
[shutdown acceptance](lean-shutdown-fix-2026-11-03.md) and the
[original smoke report](lean-runtime-smoke-2026-11-03.md). No image rebuild,
installation, credentials, provider request, or live capital was used.

## Real-engine compatibility correction

The first run reached the durable submit attempt but had no actual ticket or
callbacks; its outbox became `ambiguous` and kept its reservation. It timed out
at the fixture's internal 50-second limit, then quit cleanly. Its engine exit0
is **not** cancellation acceptance. No fabricated cancellation or reservation
release was used to recover that run.

The pinned `Algorithm/QCAlgorithm.Trading.cs` decimal overload (line 410) is:

```csharp
LimitOrder(Symbol symbol, decimal quantity, decimal limitPrice,
           bool asynchronous = false, string tag = "", IOrderProperties orderProperties = null)
```

The production adapter omitted the boolean before the client tag. The narrowly
scoped fix supplies `True`, as the existing market submission does. Only the
limit branch of `lean_runtime.py` changed. The next run, freshly prepared against
that source, submitted and canceled successfully. Other overloads, including the
similarly shaped stop-market path, are outside this acceptance and remain
unverified; they were not changed speculatively.

The shared economic gates, execution authority, ledger, domain contracts,
shutdown patch, feed and smoke files were not modified. Monetary reads and
construction still use the existing exact CLR decimal boundary. The inherited
runtime checks also preserve `100000.000000000000000000001` through native cash
storage and the original `638712864000000001` CLR clock ticks.

## Acceptance fixture and verification

Owned files under `environments/lean-smoke/order_lifecycle_*` provide:

- A preparer calling the unchanged smoke preparer, including production
  `prepare_project` and a fresh execution manifest; it copies the baseline
  initialization checks and substitutes the lifecycle algorithm.
- One quantity-1 buy limit at USD `1.00`, below the actual synthetic price,
  prepared with the real observed reference price and a USD `1` estimated fee.
  Dispatch runs through the production operating bridge. The canonical reserve
  is based on reference price plus estimated fee, not the USD `1` limit price.
- Snapshots at preparation, first actual Submitted ticket, at least two seconds
  later before cancellation, and after actual cancellation reconciliation.
- A bounded Docker launcher using the fixed socket and empty Docker config,
  unique name, creation timeout30s, engine timeout150s, and finally-protected
  removal by that unique name even when creation acknowledgement is uncertain.
- A fail-closed result/inspection reader which reopens SQLite read-only and
  independently requires original identity, zero fills, exact reserve release,
  unchanged cash, empty open orders, native cancel event, compliance and exits.

Reproduce from this checkout using an existing pinned public LEAN source tree:

```bash
export PYTHONPATH="$PWD"
PY=/home/ozzstein/AgenticQuant/.venv/bin/python
RUN=/tmp/qal-lifecycle-UNIQUE/run
$PY environments/lean-smoke/order_lifecycle_prepare.py \
  --source /tmp/qal-lean-build-source --output "$RUN"
# Keep the parent private; permit the image's UID10001 to write the run directory.
chmod 700 "$(dirname "$RUN")"
chmod -R a+rwX "$RUN"
$PY environments/lean-smoke/order_lifecycle_launch.py --run "$RUN" --checkout "$PWD"
$PY environments/lean-smoke/order_lifecycle_evidence.py --run "$RUN"
```

Each actual container has network `none`, all capabilities dropped,
`no-new-privileges`, memory/memory-swap4GiB, CPUs2, PIDs256, BLAS/OMP threads1,
read-only own-checkout `/app/src`, and a fresh synthetic `/smoke`. No server,
external feed, or authentic account is involved. No other container is removed.
Preparation and execution must be sequential with no production source changes
between them. The launcher intentionally does not implement retries.

`order_lifecycle_recorded.json` retains a successful actual result as a regression
input, with synthetic IDs only. Tests checking that file are evidence-reader
contract tests; they do not constitute a new engine run. Negative tests reject
changed ticket/tag, a tiny residual reservation, any fill, a missing native cancel,
or an undisclosed custom fill model. A separate fake-port test and an exact
pinned-overload signature regression are also explicitly offline.

## Partial-fill feasibility

**Not exercised or claimed.** Runtime observation identifies
`QuantConnect.Orders.Fills.ImmediateFillModel`. At upstream pin
`3b2bf19e597805cc917dd199ed9ee7d10d5ceb84`:

- `Common/Securities/Crypto/Crypto.cs` constructs `ImmediateFillModel`.
- `Common/Orders/Fills/ImmediateFillModel.cs` adds no override to `FillModel`.
- `Common/Orders/Fills/FillModel.cs:697` forwards `LimitFill` to
  `InternalLimitFill` with `order.Quantity`; crossed buy/sell branches set
  `OrderStatus.Filled` and assign the complete quantity. It has no volume-driven
  partial execution on this path.
- `Brokerages/Paper/PaperBrokerage.cs` inherits `BacktestingBrokerage` and scans
  algorithm model fills; this feed provides synthetic prices, not an external
  matching engine's partial-execution stream.

Increasing quantity or waiting longer cannot establish partial-fill acceptance
with that model. Existing fake-port partial receipt tests validate canonical
accounting only. A future separate bounded custom FillModel fixture could emit
staged model fills through actual engine events, but must be labeled **custom
simulation**; it would not establish default-model or real-broker liquidity
behavior. No custom FillModel was installed or silently substituted here.

## Limits

This acceptance covers one resting synthetic crypto GTC limit cancellation only.
It does not prove partial fills, restart recovery, DAY/trailing overloads,
nonzero fees, authentic broker behavior, trading efficacy, or release readiness.

Offline regression initially ran 130 cases: **125 passed; 5 failed** in the
existing `tests/test_lean_day.py` mock, whose `LimitOrder` callable accepts the old
five arguments. Those tests need their fake signature updated to the pinned
six-argument runtime signature, including the boolean. That file is outside this
lane's ownership and was left unchanged; production was not adapted to preserve
an incorrect mock. The owned new suite passed all nine cases. See the final
validation/evidence entry below for the scoped repeat and actual run records.

## Final evidence and validation

Exactly three engine runs were used, each bounded at150s; all returned engine
exit0, cleanup0, no timeout and no OOM. Only runs2 and3 satisfy acceptance:

| Run | Exact container ID | Outcome |
| --- | --- | --- |
| 1, original adapter | `409718f50b182fe38f782f2b3c6bf36a84df38f38b92eb0aa46013df8bec6eb1` | No native ticket; ambiguous dispatch; internal timeout, clean quit |
| 2, limit overload fixed | `19a675f33d54388488a9632a5bafb4aa6352fb1fbbefb844b78868e7628e2b52` | Actual resting cancellation; exact reserve `101.954674342293486` released; PASS |
| 3, final formatted source | `d85267e511b13c624fc957fe0ae82ea49ba2d9ec4edf48a9e7b60ecab5cda605` | Repeated actual cancellation; exact reserve `101.891749146984376` released; PASS |

Run3 observed Submitted at `2026-11-03T13:09:04.060534+00:00`, with the same
open order still present at `13:09:07.021485+00:00`. Available cash moved from
`99898.108250853015624` back to `100000`; the reserve became exactly zero.
Ticket1 and client tag `7ccd497a-72b6-5022-af17-addc956ce0ce` were unchanged.
Actual events1/2/3 were Submitted/CancelPending/Canceled; all fill quantities and
fees were zero. Different reserves between runs reflect the existing wall-clock
synthetic price generator, not rounding or a deterministic historical replay.

Private persistent evidence:
`~/.local/share/quantagentlab/runtime-smoke/lean-lifecycle-20261103-a2dbc79d/`.
Its `initial`, `fixed`, and `final` directories contain original preparation
manifests, configurations, SQLite ledgers, engine logs, container inspections,
exits, cleanup and subsequent absence checks. `final/verification.json` records
the read-only reopen PASS. The final manifest's implementation hashes were
checked against this checkout after execution. Fixture/source snapshots and
`sha256.json` retain the evidence inventory. None of the three unique containers
remained on the fixed daemon after cleanup.

The scoped offline repeat excluding the out-of-scope DAY mock suite passed
**109 tests in4.30s**, including all nine lifecycle regressions, decimal, enum,
execution, trailing, Binance references and signal bridge tests. Sockets were
disabled, with only Unix sockets allowed. The earlier full selection's five DAY
mock failures remain explicitly unresolved. Ruff lint/format and `git diff
--check` also pass for owned Python files and the narrow runtime change.
