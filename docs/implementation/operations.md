# Operating control and monitoring

The operating store coordinates work; `PortfolioLedger` remains the sole economic authority. `OperationsStore` archives immutable run manifests, exact inputs and stage outputs, commits an attempt before work, and records failed or uncertain completion explicitly. A completed stage replays its verified receipt. A process crash does not authorize repeating an external call: recovery must use the original frozen input and the component's reconciliation API. Kernel process locks serialize work across independent processes.

`ScheduleSupervisor` persists UTC deadlines, uses declared local time zones and actual XNYS session dates, and archives an input bundle before invoking its handler. Missing bundles wait only until the configured deadline, then become missed runs with alerts. Failed or interrupted runs block their schedule until explicit recovery names the exact input hash. Restarting does not erase failures or schedule duplicate successful jobs.

`src.dashboard` and `src.rd_agent_dashboard` are read-only projections of the actual ledger, research registry, evidence archive, operating receipts and `AgentCallBudget`. Unknown usage retains its charged reservation and is displayed separately from cost calculated from reported tokens at configured rates. Missing stores or marks remain unavailable. Research monitoring shows development outcomes and holdout consumption, not holdout measurements in development views. There are no trading or activation controls in the dashboard.

Configuration paths live under `AppConfig.operations` and default to the configured output directory. Explicit approved baseline and model hashes default to empty; file presence and candidate JSON flags never authorize trading. The account uses `paper_ledger_path` and `paper_account_id` consistently.

## Canonical application and scheduler entry points

`src.operations.application.paper_cycle_service` constructs the configured evidence,
operations, research, ledger and attribution stores. `execute_cycle` enforces the
configured account. `scripts.run_pipeline`, `aiquant pipeline`, and `aiquant trade`
delegate to this service; the retired independent schedulers and simulated
pipeline outputs no longer execute. `aiquant optimize --run-id ID` reads an actual
completed optimizer receipt, and `monitor` reads configured operational state.

The explicit commands are:

```sh
python -m scripts.run_operations build-cycle --input preparation.json --output cycle.json
python -m scripts.run_operations prepare-cycle --template template.json --macro-dataset macro.json --contextual-sources contextual.json --income-authorities income-authorities.json --refresh-run-id REFRESH --run-id CYCLE --scheduled-for 2026-10-27T21:15:00Z --output cycle.json
python -m scripts.run_operations cycle --input cycle.json
python -m scripts.run_operations dispatch-pending --input cycle.json
python -m scripts.run_operations schedule register --input schedule.json
python -m scripts.run_operations schedule serve --lane operating
python -m scripts.run_operations schedule serve --lane research
```

These examples are operator invocation shapes, not executed or approved trading
jobs. `CyclePreparationRequest` carries the exact template, source-task recipe,
completed refresh ID, scheduled/run identities, breadth snapshot, historical
membership and its sources, and explicit income authority references. `build-cycle`
calls the actual completed-refresh data builder, preserves the original macro
training prefix, forwards current contextual sources, and archives the preparation
receipt. Manual `prepare-cycle` takes an already source-bound macro dataset.
Fresh templates need the explicit income policy and all thirteen economic arrays.
Missing positive coverage remains a block; the CLI cannot bypass it.

`ScheduleSpec.execution_lane` selects independent operating/research worker
processes; its default is omitted in serialization to preserve older hashes.
Scheduled input is a closed typed job: `paper_cycle`, `dispatch_pending`,
`data_refresh`, or `research_trial`. The exact scheduled timestamp must match the
durable claim. The scheduler archives input bytes before execution and never
reruns an interrupted external call blindly. Research jobs execute only previously
registered development trials; holdout admission and promotion are not scheduled
permissions. Missing outcomes are distinguished from corrupt outcomes, so
corruption cannot authorize a new execution. A rolling producer of fully sourced
future job inputs and full intraday orchestration remain unfinished.

Verification before the income gate: 53 app/CLI/scheduler/registry/accounting tests
passed in 127.23 seconds, including actual configured full-cycle receipt replay.
After the gate, 34 isolated scheduler/registry/CLI checks passed; the new configured
CLI rejection check plus eleven native inventory-fee cases passed in 56.10 seconds.
The positive full-cycle application case is retained and now has the same missing
income-source acceptance dependency as H's full suite. This is not a current
whole-suite success claim.

Verification at this checkpoint: 25 offline tests passed across operating-store subprocess death/restart, cross-process locking, immutable inputs, alert delivery uncertainty, schedule DST/session boundaries/recovery, configuration, and actual Streamlit page rendering. Authenticated notifications, unattended provider refresh, complete cycle composition and external venue operation require their subsequent integration checks; dashboard rendering is not evidence of those executions.

`NotificationPlan` declares Telegram or SMTP destinations and environment-variable credential references. `dispatch_notifications` uses the same durable alert outbox, bounds work, and records transport acceptance separately from recipient delivery. Telegram sends plain text through the fixed official endpoint; SMTP requires TLS and checks every recipient's acceptance. An ambiguous call keeps its reservation and never retries automatically. `resolve_delivery` archives an explicit operator review, with distinct confirmed/abandoned/retry-authorized states and an expected attempt number. Nine offline delivery/store tests passed; no external notification was sent. Implementations were checked against [Telegram sendMessage](https://core.telegram.org/bots/api#sendmessage) and the [Python SMTP API](https://docs.python.org/3/library/smtplib.html).

## Bounded operational source refresh

`python -m scripts.run_operations refresh --input refresh.json --store outputs/operations.sqlite3 --evidence outputs/evidence.sqlite3`
executes an explicit `RefreshPlan`. Each `FetchTask` pins a selected provider, operation,
canonical arguments, instrument where required, hard deadline and HTTP request cap.
Direct FMP, ALFRED, Finnhub, SEC, CoinGecko, Alpaca read APIs, Yahoo and public
Binance/Kraken CCXT observations are connected to the existing evidence adapters.
MCP calls require a deployed `ServerManifest` with exact version/schema pins and
application read allowlist membership; generic calls are reauthorized to their
actual tool before budget reservation.

The operating store commits daily fetch-count and maximum MCP-cost reservations
before dispatch. Adapter fetches and HTTP requests are separate counters. Direct
subscription/entitlement cost is unmeasured, never reported as measured zero.
Every fetch runs in a separate bounded worker process; an SDK/MCP timeout kills
the worker and retains its reservation. Uncertain or failed stages do not retry
on restart. Successful stages reuse immutable receipts without another request.
Optional missing sources remain explicit gaps; required unavailable sources fail
the run and emit durable alerts. Successful completion advances a new context to
its measured completion time, preserving the original requested context.

`CaptureStore` measures UTC at each archive insertion and bounds current-source
availability by that actual receipt. Only pinned ALFRED vintage and SEC accession
adapters retain their documented earlier publication availability. Full raw,
normalized and document bytes are preserved and verified. Current downloads
cannot become eligible under the earlier request cutoff.

`aiquant operations status` reads actual configured stores; `operations notify`
uses an explicit notification plan and durable pending alerts. `operations
resolve-delivery` requires the exact alert/channel/attempt and archived operator
review evidence before confirmed, abandoned or retry-authorized recovery.
No notification or credentialed provider call was made during implementation.
Offline tests cover real SQLite/restart receipts, actual FMP adapter parsing with
an HTTP boundary fixture, measured availability, durable caps, generic MCP
permissions, actual process termination and missing credential redaction.

## Binance native reference refresh

`FetchTask(provider='operating', operation='binance_notional_reference', ...)`
requires a canonical instrument and strict integer `average_minutes >= 0`.
Its only other argument is `network='mainnet'|'testnet'` (default `mainnet`).
The selected network fixes the HTTP origin; no caller URL, price, method or
source callback is accepted. It captures the actual native reference, or its
documented absence and exact average-price/individual-last-trade fallback, under
the same total request allowance. This is a public read operation.

Use the completed refresh's actual context and exact source IDs with
`freeze_binance_reference`, then `create_evidence_backed_binance_execution`.
The frozen resolver rechecks full source envelopes/raw bytes on every native
preflight. Paper accounts require Testnet sources; live accounts require mainnet
sources and retain the separate execution authorization gate. Missing, stale,
future, corrupted or wrong-network evidence cannot authorize an order. A refresh
or a source binding grants no live-capital authority.

Coordinator verification: 119 offline source/refresh/venue-boundary tests passed
in 19.42 seconds, including all four actual refresh-to-certification combinations
(mainnet/Testnet, native reference/documented fallback), Decimal fidelity, stale
rejection and completed-refresh restart without repeated HTTP. HTTP transports
use synthetic wire fixtures; no vendor calls or authenticated acceptance occurred.

## Income source admission

Operating refresh permits two additional closed GET operations:
`alpaca_cash_dividend_sources` and `alpaca_account_dividend_sources`. Both require
an explicit instrument and `started_at`/`ended_at`; the account operation also
requires the exact canonical account and provider account UUID. Request budgets,
worker deadlines, actual capture clocks and immutable raw bytes apply unchanged.
The final `economic_bundle` may take explicit `income_ids` from preceding task
receipts. It appends the five canonical income source arrays, including sourced
empty arrays, and the completed-refresh reader independently reparses them.

`verify_paper_provenance` accepts `income_policy`, `required_started_at`,
`required_ended_at`, and `income_authorities`. The cycle owns the required interval
from the previous committed account anchor (or account opening) through the actual
cutoff. Income-aware inputs bind `cash_dividends`, `dividend_payments`,
`dividend_distributions`, `action_coverage`, and `paper_dividend_policies` with
same-named `_id` artifact references. Each source pointer names its exact array.

`src.operations.income` uses a closed native parser allowlist. It rejects arbitrary
normalized authority claims, altered arrays, account/scope substitutions, future
availability and gaps in the required interval. Alpaca's process-date corporate
actions and creation-date activities currently establish observations and unknown
coverage only. Neither payable dates nor net dividend activities authorize an
inferred payment, zero deductions, or complete effective-event coverage. This
path intentionally blocks fresh income-aware cycles with only those sources.
Legacy immutable completed receipts remain distinct from new admissions.

Positive income source acceptance remains unfinished. The source review considered
[IBKR accruals](https://www.ibkrguides.com/reportingreference/reportguide/change%20in%20dividend%20accrualsfq.htm)
and [Statement of Funds](https://www.ibkrguides.com/reportingreference/reportguide/statement%20of%20fundsfq.htm):
an accrual reversal can mean correction, cancellation or payout, and a cash ledger
alone does not establish exact action/component linkage. DTCC's
[published message specifications](https://www.dtcc.com/products-and-services/asset-services/corporate-actions-processing/ca-documentation)
are a candidate for event/payment authority; no DTCC parser, stream completeness,
subscription access or account-level deduction applicability is claimed. An XML
message name or a fabricated fixture is not sufficient evidence. A supported
positive wire parser and its coverage authority must land before this gate opens.

Verification: 36 offline tests passed in 5.48 seconds across root income proof,
actual native adapter parsing, bounded operating refresh, source artifacts and
opening migration. HTTP wires are synthetic; no source verifier is replaced.
Tests prove rejection of unsupported income authority, not positive income-window
acceptance. Separately, 13 source/ledger checks passed in 1.69 seconds including
the sleeve-publication fence preventing retroactive dividend mutations.

Operational opening prints use `verify_paper_provenance(..., receipts=inputs.receipts,
rule_sources=plan.rule_sources)`. Exact source envelopes bind actual archive
observation/availability clocks and opening quantities. Only this paper-only path
permits an execution receipt after the occurred opening print. The historical
research verifier still rejects that late source. Calendar, universe and metadata
gates retain their original causal cutoffs. Venue rules also require exact source
contents. Receipt verification alone never authorizes a fill: `PaperVenue.settle_open`
separately requires durable acceptance before the open, original admitted risk
metadata and chronological accounting. Already published NAV is not silently
restated. Thirty-one source/venue/validation tests passed offline in 3.63 seconds.

## Operating source dependencies

`FetchTask(provider='operating', ...)` exposes the closed operations in
`src.operations.operating_fetch.OPERATIONS`: Alpaca calendar/bars/reference/trades/
corporate actions, Kraken and Binance bars/reference/trades/calendar,
`current_universe`, and `economic_bundle`. These adapters own the measured archive
clock inside their normalized economics and receive the underlying `EvidenceStore`.
Wrapping them in `CaptureStore` would invalidate their exact source clocks.
The existing worker deadline applies; the request cap is shared across all hosts
used by one operation, including Alpaca plus FMP reference capture.

`source_arguments` contains `FetchSourceRef(argument, task_id, kind, selection)`.
Each reference must name an earlier task in the same immutable refresh plan.
For example, `calendar_id` selects one `operating_calendar`, `bar_ids` selects all
`bar` records, and `reference_ids` selects all `operating_reference` records from
the declared task receipts. Scalar arguments require `selection='one'`; arrays
require `'all'`. Missing, duplicate, ambiguous or changed sources fail. The stage
receipt binds the original task, resolved task and every dependency envelope
hash. Failed calls retain reservations and do not automatically repeat.

The final `economic_bundle` operation derives exact source bindings and archives
the availability audit before the refresh completion clock. It takes explicit
instruments, market start and source IDs (or preceding task references). It does
not invent missing actions, liquidity, metadata or coverage. Use
`load_operating_sources(operations, evidence, refresh_run_id)` to recover the
bindings from the authoritative completed manifest, stage receipt and transitive
source proof for `PaperCycleService.prepare`. Completed refreshes replay their
original receipts even after the original freshness window expires; each new
fetch admission still enforces the actual request-age limit. Do not edit old
immutable refresh plans to add dependencies; admit a new run identity.

Verification: 33 offline refresh/operating adapter tests passed, followed by 20
refresh tests covering the final reader and replay after freshness expiry.
Provider wires in these tests are synthetic. Current ADV freshness, genuine
opening-event composition and mixed-calendar operating acceptance remain separate
checks; these receipt tests do not establish them or authenticated entitlements.

## Opening-only operating events

New paper operating plans use `inputs.opening_events: tuple[OpeningEvent, ...]`
with `inputs.bars=()`. The event contains the actual raw print price, opening
quantity, FX and session identity, without future OHLC or completed volume.
`expected_artifacts` replaces `execution_bars` with `opening_events` and its
`opening_events_id` reference; calendar and opening-liquidity proofs derive from
those actual events. The completed feature `MarketSnapshot` remains separate
and cannot exceed the operating cutoff. Existing completed-bar plans retain
their prior evidence convention; ambiguous mixtures are rejected.

Every event needs an exact observed receipt. Full event price and opening
liquidity must both bind subtrees of the **same** receipt envelope, including
its content hash and availability. A second source containing the same event ID
cannot substitute a price. Calendar, metadata, membership and historical
research cutoffs remain unchanged. Source-bundle loading accepts the two exact
eight-kind versions, with no mixing.

`PaperVenue.settle_open(event, metadata_by_intent, observed_at=..., available_at=...,
as_of=..., evidence_id=..., evidence_hash=...)` delegates to the shared
`PaperExecution.open_event` primitive. Durable pre-open acceptance, original
admission metadata, consumed liquidity and published NAV chronology still govern
execution. Completed bars can explicitly reference an already consumed opening
event without filling twice. Authenticated completed market closes are marked
separately; `PaperVenue.expire_orders(as_of=...)` handles canonical expiry at the
close boundary. No completed-bar fields are manufactured from an opening print.

Verification: 82 offline source/opening/venue/durable-accounting/sleeve/replay
artifact tests passed in 22.14 seconds. These include settlement evidence before
session close, exact receipt/price/quantity/FX tampering, unrelated price-envelope
rejection, restart and shared opening-capacity conservation. The full mixed
operating cycle and external source acceptance are separate requirements.

## Source identity across opening-format upgrades

`prepare_opening_source(store, venue, event, receipt, as_of=...)` authenticates
both the current event and every unmapped historical receipt for that instrument
against complete archived native provider responses. Provider, stable instrument,
native trade ID and exact UTC nanoseconds identify the print independently of
wire version or feed alias. The helper checks the original receipt's entire
opening economics and archive clocks before binding any migration, and commits
all prior mappings atomically. An unprovable old receipt blocks migration.

Pass the returned identity to `PaperVenue.settle_open(source_identity=...)`.
The venue decides whether the print is already consumed; aliases may add their
own documentary receipt after a published NAV but cannot add fills, marks or
liquidity. No identity is inferred from matching prices or rounded event times.
H's old-anchor branch must use this path rather than silently dropping a new ID.
The direct engine also rejects omission of identity once source-bound history
exists. Legacy records without preserved raw print membership cannot be upgraded
by supplying a guessed tuple.

Verification: 29 offline source-identity tests were exercised; the two final
root migration regressions passed in 1.31 seconds after correcting test API
arguments. They use actual adapter parsing over synthetic provider wires, a
seven-unit partial fill, a v1/v2 alias after published NAV, restart, exact retained
remaining quantity, and price/hash substitution failures before mapping writes.
No authenticated provider acceptance is implied.
