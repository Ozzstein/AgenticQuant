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

Current acquisition checklist and candidate disposition:
[income source acceptance](income-source-acceptance.md).

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

## Native perpetual history refresh

Closed `provider="ccxt"` tasks support `funding` for explicit Binance or
Kraken Futures perpetual instruments, and `open_interest` for Binance. Kraken
Spot is not a derivative source. Arguments are exactly `since`, `end` (required
timezone-aware ISO instants), optional `timeframe`, `limit`, and `max_pages`.
Unknown endpoints/parameters, spot symbols, unsupported periods, and future end
cutoffs are rejected before network work. The total request cap includes SDK
contract loading and every native history page; page count is additionally capped
by that budget. The existing worker still enforces the overall wall deadline.

The native adapter receives the underlying EvidenceStore because it stamps its
own measured post-response clocks and immutable source links. Refreshed context
advances beyond these actual receipts. Exact native decimal scalars and contract
units independently reparse through `verify_derivative_record`; no assumption of
eight-hour funding or historical point-in-time availability is introduced.
Coverage remains explicitly uncertified, including Binance's limited OI history.

Coordinator verification: 37 offline refresh/source tests passed in 4.75 seconds,
including real stores/parsers, metadata-plus-page budget exhaustion, receipt
clock/source verification, and completed-run replay without another request.
The separate 45-test derivative/context/direct-adapter/factor suite passed in
1.25 seconds. HTTP/SDK boundary fixtures do not establish vendor acceptance.

### Deployment entry-point audit — 2026-10-27

The application Docker default and Compose scheduler now call the canonical
`src.scheduler serve --lane operating` command. A separate research scheduler
consumes the durable research lane; the manual app defaults to read-only status.
The previous `--loop` and `run --mode paper` commands were retired and could not
start the current services. README/deployment instructions now describe explicit
manifests, actual services, immutable stores and the unresolved income/runtime
gates, removing unsupported cost/performance and automatic-fallback claims.

With Python sources pinned at `fd54c9f`, `uv build --offline
--no-build-isolation --out-dir /tmp/quantagentlab-root-fd54c9f-dist` built both sdist
and wheel. An isolated extracted-wheel CLI smoke check passed, including packaged
`scripts` and all six JSON strategy specs. Compose YAML commands were checked
against actual CLI help; lane separation/manual profile/loopback dashboard were
asserted. This is not Docker execution. The existing root editable install still
had an old `src`-only package mapping; it needs reinstalling after the active full
suite, whereas the newly built wheel correctly contains `scripts`.

The user confirmed no sufficient corporate-action/custodial income source has
been selected. Fresh source-complete acceptance therefore remains blocked; the
retained positive-cycle requirements are not replaced by mock completeness.

### Full pinned audit and canonical migrations — 2026-10-27

The full Python source snapshot `fd54c9f` completed 1,886 tests in 1,139.76s:
1,851 passed, 17 failed, 18 setup errors, zero skips. The XML/log are retained at
`/tmp/quantagentlab-root-fd54c9f-full.{xml,log}`. Fourteen failures referenced retired
interfaces: ticker/model-label execution, mutable KB globals, old regime commands
and the legacy `--broker` selector. Three failures plus all 18 setup errors depend
on genuine missing positive income policy/coverage; those acceptance cases remain
enabled and unchanged.

After integrating lane checkpoints through `8f556cf`, 378 focused tests passed in
48.36s across decision evaluation/history/memory, actual later-clock desk
snapshots, historical sleeves, LEAN original-order recovery, crypto context and
accounting. Canonical `run_agents` now forwards the actual EvidenceStore.
Retired integration tests now exercise real typed inputs, source/usage-backed
quarantine, exact feature formulas, read-only ledger status and actual CLI replay,
with explicit rejection of removed parallel routes. The migration exposed a real
CLI artifact problem: restart serialized nested manifest keys in a different
order and overwrote result bytes. Backtest output now uses canonical sorted JSON
and the existing immutable writer. The 62 migration/core/artifact tests passed in
21.91s, including unchanged result and SQLite bytes across CLI restart.

The root editable package was reinstalled offline from the existing lock after
the full audit; the installed `aiquant` entry point now imports packaged scripts
without a PYTHONPATH override. No dependency versions changed. Root source Ruff
and diff checks pass. These focused counts are not a whole-repository green claim.

### Dated dominance refresh and configured credentials

The closed CoinGecko dispatch now exposes `current_dominance` (optional
`tier="demo"|"pro"`) and `dominance_history` (`tier="pro"`, optional `days` from
`"1"|"7"|"14"|"30"|"90"|"180"|"365"|"max"`, default `"30"`). These are global
tasks without an instrument. Keys, URLs and arbitrary method arguments cannot
enter task manifests. Both history requests share the admitted HTTP budget and
the adapter writes to the underlying EvidenceStore, retaining its measured
capture clocks and exact transitive source hashes. A historical observation
captured now does not become historical decision-time knowledge.

Read transports now consume the documented application configuration:
`AIQUANT_FMP_API_KEY`, `AIQUANT_FRED_API_KEY`, `AIQUANT_FINNHUB_API_KEY`,
`AIQUANT_COINGECKO_API_KEY`, `AIQUANT_SEC_USER_AGENT`, and nested
`AIQUANT_ALPACA__API_KEY` / `AIQUANT_ALPACA__API_SECRET`. Existing unprefixed vendor
environment names remain fallback inputs. Resolved values enter only the native
transport; they are not added to tasks, evidence manifests or operation receipts.

Root integration on top of `0f0cac9`: 87 tests passed in 10.83s with
`tests/test_operations_dominance_refresh.py`, `test_operations_refresh.py`,
`test_operations_operating_refresh.py`, `test_operations_crypto_refresh.py`,
`test_config_resolution.py`, and `test_crypto_dominance_bindings.py`, using
`--disable-socket --allow-unix-socket`. Actual adapters/stores ran over synthetic
HTTP transport. Evidence includes both native origins/header keys, exact cap
ratios and clocks, request-budget exhaustion retaining raw evidence, immutable
restart, configured FMP/Alpaca credentials and receipt redaction. No provider
access or source sufficiency is claimed.

### Verified registry monitoring

`ResearchRegistry(path, read_only=True)` opens an existing canonical path with
SQLite `mode=ro` and `query_only`, without schema creation, host-control
reservation or initialization. Mutations reject explicitly. `read_snapshot()`
keeps all nested readers on one database transaction, including concurrent writer
admissions. Candidate and trial readers verify their immutable identity and
admission snapshot. `library-status` uses this read-only mode.

The dashboard/operating status uses `research.monitoring.research_snapshot`.
Development outcomes pass the authoritative restricted reader; holdout outcomes
are not exported and a changed trial-purpose projection cannot disclose them.
Historical activation receipts are separated from current resolver eligibility;
legacy native-origin activations remain blocked. Native rows verify admission,
launch, original container and the full recovery chain before exporting only
identity/state. They never export mount/token paths, inspection/gateway bodies,
artifact inventories or result blobs. The last recorded absence remains uncertain
and cannot imply completion. The monitor does not contact Docker or providers.

On top of `a15f8e4`, 43 registry/monitor/dashboard/lifecycle tests passed in
13.08s, including unchanged database bytes, query-only permissions, concurrent
snapshot reads, current-directory independence, holdout/candidate/outcome
tampering, old native activation and absent/corrupt native authority. The full
native discovery/recovery/supervisor plus original portfolio/advice and numeric
registered-experiment integration then passed 96 tests in 181.37s, including
actual fits, paired regimes, callback ledgers and changed-cost accounts. The
transport/source/refresh integration after the HTTP cap separately passed
159 tests in 43.01s. Actual local authorities and synthetic transport only.

### Full archived audit, 2026-10-28

The next full suite ran against an immutable `git archive` of
`a0ec996acd360d7caad2edd0be56b35e53fa1332` at
`/tmp/quantagentlab-root-audit-a0ec996-pc3v4196`, using the locked main Python
environment and `PYTHONPATH` pointing only to that archive:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/ozzstein/AgenticQuant/.venv/bin/python -m pytest tests -q \
  --disable-socket --allow-unix-socket \
  --junitxml=/tmp/quantagentlab-root-a0ec996-full.xml
```

Result: 2,103 tests, **2,081 passed, 3 failed, 18 setup errors, 1 skipped**, two
warnings, 1,107.76s. Full log `/tmp/quantagentlab-root-a0ec996-full.log` and JUnit
above retain exact results. All failures/errors are the enabled positive fresh
H/application acceptance cases requiring explicit sufficient income authority.
The operator has selected no source; that gate remains closed. The prior fourteen
retired-interface failures are resolved in this whole-suite run.

The archive excluded the separately installed RD-Agent environment, so
`test_native_rd_adapter.py::test_real_pinned_templates_render_explicit_us_scenario`
skipped there. Running that exact test in the main checkout with its installed
pinned environment passed in 2.03s. No archive result was relabeled; no Docker or
provider execution is implied. Before the full audit, canonical trailing-stop
integration passed 248 stop/venue/remote/LEAN/ledger/sleeve/income/artifact tests in
33.57s. This is still not a source-complete or deployed release.

### Qlib daily-grid integration

Data checkpoint `86fb043` integrated as root `3a501c6`. Root command
`python -m pytest tests/test_qlib_daily_contract.py tests/test_local_quant_flow.py
tests/test_native_discovery_exposure.py -q --disable-socket --allow-unix-socket`
passed all 50 tests in 13.78s, including actual Qlib Alpha158/Ridge. Intraday or
duplicate rows cannot overwrite a daily instrument/session, and unresolved
internal sessions cannot silently become NaN samples. Full XNYS boundaries and
strict source clocks are checked before writing. No membership outside supplied
endpoints is inferred. Existing valid layout/schema remains unchanged; malformed
published source grids fail without rewriting their files. Native admission
continues its separate full snapshot-to-binary reconstruction. File hashes alone
are not presented as that numerical proof.

### Frozen evaluation and native trailing integration, 2026-10-28

Accounting `badabba` integrated as `2cdef9e`: 192 root native trailing,
frozen-reference, LEAN compatibility, venue, ledger, inventory, sleeve and stop
tests passed in 18.09s. LEAN `13b89f1` integrated as `800d475`: 183 focused LEAN
trailing/runtime/native recovery/reference/signal/venue/ledger tests passed in
15.10s. These are offline protocol checks, not actual SDK/vendor acceptance.
LEAN DAY/calendar expiry and contingent/OCO remain open.

Jev `1b753cb` integrated as `528c8db`: 270 experiment, evaluated-policy, contract,
service, workflow, desk, history, memory and desk-clock tests passed in 50.56s.
The installed `aiquant decision evaluate schema` delegates to the exact same
`EvaluationManifest` CLI. Use `freeze|run|report|inspect` with explicit
`--store` and `--evidence`; report inspection uses `--kind report`. Freeze needs
an operator-reviewed source/label manifest; no production examples or efficacy
passes are supplied. Every cohort is admitted before dispatch under one immutable
budget, selection/calibration cannot publish, unknown earlier retry costs remain
unknown, and report completeness is distinct from release eligibility. No
automatic publication or registry promotion occurs. Root report consumers must
call `ExperimentService.inspect` / `verify_report`, not trust a JSON pass flag.
The subsequent audit reproduced a shadow semantic label path into debate prompts
and undelivered-document coverage claims; owned fixes are active and those
boundaries are not claimed accepted.

### Rolling refresh templates

`ScheduleSpec` accepts exactly one immutable `refresh_template` or legacy
`input_directory`. `RollingRefreshTemplate(plan, bindings)` keeps all provider,
instrument, source-reference and budget arguments fixed; only listed closed
temporal arguments can bind scheduled/actual invocation time with nonpositive
offsets. The seed's prior snapshot approvals are prohibited. The generated run
ID binds schedule/window/template, its context uses actual invocation time, and
the full input is archived transactionally before existing scheduled dispatch.
Recovery reuses the exact original plan and normal refresh stage receipts. This
does not re-date a source or make captured history usable under an old cutoff.
Old schedule serializers omit the new absent key, retaining original identities.

100 root rolling/scheduler/refresh/CLI/store/config tests passed in 23.35s with
TCP disabled (Unix self-pipe allowed). Actual FMP parsing and evidence/operations
stores over synthetic HTTP verify two windows, later archive availability,
canonical application dispatch, restart deduplication and a lost scheduler
completion after durable refresh without refetch. Closed derivative/date windows,
old identity shape, expired windows and invalid bindings are covered. Initial
two derivative fixture failures used an invalid zero-width seed window; corrected
to a real positive interval, with no runtime gate relaxation. Log:
`/tmp/quantagentlab-root-rolling-integration.log`. This closes rolling fetch-plan
materialization, not full source-complete unattended build/retrain/operation.

### Expiry, prompt authority and native indicative FX

`d65e637` integrated as `489c9fe`: 267 root expiry/native/LEAN/reference/ledger/
sleeve/replay/validation tests passed in 40.49s. `expire_orders` returns current
outbox states, not a list of successful cancellations. Only an undispatched
prepared intent can expire locally; actual global receipts are reconciled before
native cancellation, with uncertain/incomplete reservations retained. Eligibility
is `[eligible_at, expires_at)`. Post-deadline real fills remain evidence requiring
explicit reconciliation, never retimed into compliant fills.

`7639ba6` integrated as `e2d5dd2`: 262 root prompt-projection, desk, history,
memory, experiment, policy and registered historical-advice/sleeve/cost tests
passed in 90.62s. Shadow citation semantics no longer reach model-visible debate
extras. Eligible semantic entries retain exact observation/request/source-span/
model/question/policy/publication identities and are reverified at prompt time.
Legacy unisolated prompts remain audit records, not newly certified memory.
Delivered-document coverage is the next separate owned checkpoint.

FX `56a61d0` integrated as `4f1745b`. Root closed refresh adds only
`provider='operating', operation='kraken_fx_reference'`, exact crypto
`USDT/USD` instrument with quote `USD` and venue `kraken`, and explicit native
`market_id`. There is no default market ID, source lookup, user endpoint, asset
version, trade count or caller price. Actual fixed-origin AssetPairs/Assets/Trades
requests share existing refresh request/deadline caps. The result archives all
three raw authorities and one derived FX record with original observed time and
measured availability. `FetchSourceRef(argument='fx_evidence_id', task_id=...,
kind='fx', selection='one')` pins the exact earlier task's source. The original
cutoff cannot use a later capture. Source and consumer verification use the same
native parser, at the actual valuation/open cutoff and its explicit age bound.

Root source/consumer/factory/refresh integration passed 116 cases; initial new
dispatch tests had one test-only exception mismatch (direct fetch raises the
documented ProviderError, while the child-worker wrapper uses OperationError).
After correcting that expectation, all 13 closed-dispatch tests passed in 0.44s,
including actual source normalization and certification, exact dependency/restart,
wrong identity/arguments and shared request exhaustion with no derived FX proof.
Logs: `/tmp/quantagentlab-root-fx-integration.log` and
`/tmp/quantagentlab-root-fx-dispatch.log`. HTTP responses are synthetic; no vendor
calls or market-availability claim. The 300-second operating ceiling never
relaxes the 30-second fee bound. Current-FX revaluation of historical quote
turnover keeps its existing explicit convention; neither historical daily USD
conversion nor executable FX/multicurrency settlement is established.

LEAN `f710dc0` integrated as `1088cad`: 207 root DAY/trailing/runtime/native
recovery/reference/expiry/signal/venue tests passed in 16.07s. Actual
DayTimeInForce replaces GoodTilDate; the subscribed security calendar and timezone
must reproduce the admitted expiry on both preflight and submit, and original
ticket order time binds restart proof. Unsupported brokerage/asset/calendar,
arbitrary intraday expiry, legacy missing proof and unowned contingent tickets
reject. This remains an offline protocol result; direct native DAY calendar
certification, discrepancy settlement, grouped orders and actual LEAN runtime
acceptance are separate outstanding requirements.

### Delivered source coverage and native calendar authority

Jev `8ce0439` integrated as `c12da5e`. Root forwards the exact serialized
`source_coverage` from `load_desk_sources` into `analyze_instrument` and persists
that same object in canonical CLI output. The allocator gives each role/document
an initial span before long-document tails and reports exact delivered hashes,
eligible/omitted/partial IDs and exclusions. Stored availability is separate
from delivery status; omitted documents block new exposure and partial delivery
cannot substantiate numeric evidence completeness. Original graph/history
validation checks spans against immutable full documents.

120 root coverage/projection/workflow/desk/history/memory/clock/original-portfolio/
registered-advice/cost/CLI tests passed in 66.28s. The additional CLI regression
uses actual stored long filing and short contrary-news fixtures, verifies both
reach the graph boundary, retains partial status and exact output equality;
all seven canonical agents CLI cases passed in 2.11s. Only provider/model boundaries
are substituted in these tests; they do not establish semantic efficacy.

Accounting `e48faa1` integrated as `42d25d3`. 169 root native Alpaca DAY, LEAN
DAY/trailing/recovery, expiry, venue, ledger and frozen-reference cases passed in
14.36s. The original bounded native calendar raw receipt is bound to the
original direct/LEAN submission claim and reverified before native fill reports;
restart cannot replace it with a newly selected session. Actual provider/SDK
acceptance remains unavailable.

Data's reproduced captured-vintage IC ensemble issue and missing canonical
parent artifact/strategy-operating admission are tracked explicitly in C2. The
next approved narrow fix preserves strict historical defaults and validates
the entire retrospective weight-selection population against the actual fit
cutoff before fitting. It does not by itself make an ensemble an approved
strategy or operating model. The live catalog was rechecked and this task was
explicitly assigned Codex GPT-6.1-Sol, xhigh reasoning, Standard tier; root remains
Astra/high and the accounting/Jev authority lanes remain Astra/xhigh/Standard.

### Temporal ensemble integration and completed scheduler recovery

Data `21e78e7` integrated as `7520957`. All 108 root ensemble temporal-policy,
temporal-model, ensemble, actual local Qlib and walk-forward tests passed in
42.21s. Captured fitting now checks the complete indexed population at the actual
fit cutoff and purges training outcomes by their explicit economic end before
the validation boundary. Internal predictions remain non-contextual retrospective
weight selection, labeled as such; strict historical publication purging remains.
The canonical parent artifact and strategy/H recipe admission are still pending.
Log: `/tmp/quantagentlab-root-ensemble-temporal.log`.

A root scheduler regression interrupted execution after the completed tick
receipt committed but before advancing the schedule cursor. Previously, restart
could dispatch again within the lag window or misreport the completed job as
missed afterward. Recovery now verifies both original archived input and completed
receipt, advances only the cursor, and returns the original receipt without
reading a replacement input or executing the handler. Missing/corrupt/incomplete
receipt authority blocks advancement. Tests cover both lag cases, changed inbox
contents, real RefreshService/adapter/store execution with synthetic HTTP,
unchanged first invocation clock and exactly one fetch, and invalid authority.
All 49 scheduler/rolling-refresh/refresh/CLI/store/monitoring cases passed in
14.80s, offline; Ruff and diff checks passed. Log:
`/tmp/quantagentlab-root-scheduler-cursor.log`.

The new whole-suite audit is pinned to immutable archive `7520957` and therefore
predates this cursor fix. Its final outcome is recorded below. Jev's separately
reproduced known-usage overrun admission gap is assigned to its existing budget
store/service authority; no financial efficacy or positive income acceptance
is implied by these local checks.

### Full integration audit at 7520957 and publication fencing

Immutable checkout `/tmp/quantagentlab-root-audit-7520957-ydktqsm3`, full commit
`7520957eb738e16ea8884990609202d4db16293e`. Exact command from that checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/ozzstein/AgenticQuant/.venv/bin/python -m pytest tests -q \
  --disable-socket --allow-unix-socket \
  --junitxml=/tmp/quantagentlab-root-7520957-full.xml
```

Result: **2,434 passed, 3 failed, 18 setup errors, 1 skipped, 2 warnings**, 2,456
collected, in **1,255.70s**. Log `/tmp/quantagentlab-root-7520957-full.log` and
the JUnit file above retain the exact output. Every failure/error was inspected:
the two fresh H cycle/preparation cases and canonical application completed-cycle
case reject missing explicit sourced income policy; 18 dependent H fixture
setups fail at the same gate. No test was disabled or positive income authority
invented. The one native template test skips because its separately installed
environment is not in `git archive`; its earlier main-checkout result remains
distinct and is not actual Docker execution. Both warnings are intentional
blocked-socket probes in MCP tests. This is not a whole-system green result or
financial/prospective release acceptance.

Accounting `cb89665` integrated afterward as `c92e037`. New ordinary fills now
check published canonical snapshots, legacy sleeve observations and full sleeve
receipts at or after execution, regardless of whether a later observation clock
was supplied. Exact duplicate fills return their original receipt first.
Publication checks and monetary changes share the existing ledger transaction;
no exception economics or caller waiver is introduced. All 87 root fill-fence,
historical-sleeve, durable-ledger, venue, replay-artifact and original advice
portfolio tests passed in 23.93s. Log:
`/tmp/quantagentlab-root-fill-publication.log`. H's same-clock close/open batch
publishes its period only after those economic events; an already published
earlier-cycle snapshot cannot be silently amended by a newly arrived fill.

Repository-wide production/script lint (`ruff check src scripts`) passes. A
separate wider `ruff check tests` found 48 lint findings (mostly import ordering
and unused imports, plus ambiguous local names). The CI production lint scope
is unchanged; the test tree is not represented as lint-clean. Changed-file lint
and diff checks pass. No authenticated provider, Docker, LEAN or capital calls
were made during this audit.

### Jev usage authority and read-only reporting

Jev `84aa388` integrated as `01aff7a`. Actual response usage is now retained
atomically with the original attempt, independently of semantic acceptance.
Admission charges the greater of reservation and verified reported usage, retains
unknown prior charges, and permanently fences a violated run and its original UTC
reservation day. Restart or a differently named run cannot clear that day fence.

Root reporting calls only `src.decision.store.read_usage_summary` against the
canonical `output_dir/decisions.sqlite`; it does not duplicate charge arithmetic.
The operations status CLI and Streamlit view expose verified known cost, nullable
total cost, conservative charges and sanitized breach identities. Missing or
corrupt authority is unavailable, never zero. These are configured-rate charges,
not provider invoices. The reader creates no store and does not mutate evidence.

Root's broad decision/experiment/history/graph run passed 296 cases with two
historical advice fixture setup errors: that fixture froze the service response
clock but omitted the new store-level measured usage clock. Adding the store to
the fixture's existing clock patch preserves actual runtime clock checks. The
final reporting/dashboard/CLI/advice run passed all 13 cases in 45.11s:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests/test_decision_usage_reporting.py \
  tests/test_dashboard_smoke.py tests/test_operations_cli.py \
  tests/test_registered_agent_strategy.py -q --disable-socket --allow-unix-socket
```

The five new reporting cases exercise actual DecisionService/DecisionStore with
synthetic HTTP responses, authority tampering, exact read-only file preservation,
CLI sanitization and real Streamlit rendering of breach/unknown totals. Logs:
`/tmp/quantagentlab-root-jev-usage.log` and
`/tmp/quantagentlab-root-usage-reporting-advice.log`. Production/script and changed
test lint plus diff checks pass. The full-suite audit above predates this change;
the positive income/full-cycle and external acceptance gaps remain unchanged.

### Opening publication parity and staged discrepancy review

H `6ee5a48` integrated as `093fa4f`: actual source-backed midnight crypto close
and next-open events are ordered before one sleeve publication, even with reversed
caller order. The grouped case ends with three units and exact NAV 9991; completed
cycle publication blocks a new late opening. This exercises the economic boundary
directly and does not bypass fresh-cycle income admission.

That regression also exposed a distinct sleeve-only case: fill arithmetic refused
the late fill, but the paper simulator could already have marked the print and
rejected the order. Accounting `882e166`, integrated as `3cbcda4`, now shares the
ledger's exact base/legacy/full-sleeve publication predicate at the opening's
pre-mutation gate. Original retries and authenticated consumed-print aliases
remain ahead of that gate. Entire database state remains unchanged on rejection.

All 74 selected root opening/fill publication, source-identity, opening-event,
venue and H midnight/expiry tests passed in 7.23s, with 37 unrelated H cases
deselected (not a full H success claim). Exact command:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests/test_opening_publication_fence.py \
  tests/test_fill_publication_fence.py tests/test_opening_source_identity.py \
  tests/test_opening_events.py tests/test_venue_execution.py \
  tests/test_operating_cycle.py \
  -k 'not test_operating_cycle or crypto_same_clock or open_only_close_mark_expires' \
  -q --disable-socket --allow-unix-socket
```

Log: `/tmp/quantagentlab-root-open-publication.log`. Production/script and changed
test lint plus diff checks passed. Only synthetic native HTTP fixtures were used.

Discrepancy proposal `2fef8f0` integrated as documentation-only `4ea07e9` and was
reviewed for staged implementation. Stage one authorizes closed immutable types,
pure retained-source reexport, metadata-only detection/assessment and the public
read-only `PortfolioLedger.execution_compliance(account_id)` audit. It does not
authorize reachable settlement, an expiry exception, group economics or a waiver.
Permanent compliance FAIL must survive factual reconciliation, report application,
restart and deletion of a derived discrepancy index. Missing source closure fails.

Before economic settlement can be enabled, accounting must wire the distinct
ValidationSuite check and native prepare/claim/LEAN gates; H must check current
authority before new-risk stages and dispatch, including partial resumes. Root
must bind operator-selected current ledger authorities for research eligibility
and recursively recheck native execution ancestry at activation, resolve,
rollback and status. A frozen evaluation PASS or a newly named empty account is
not a clearance. Existing AgentAuthorityPaths supplies source-account ledgers for
advice verification, but registry current-authority binding is still design work.
Historical completed receipts remain immutable; cancellation, capture and factual
inspection remain possible. No settlement enablement or release acceptance is
claimed by this checkpoint.

The decision-budget repair does not close the separate generative-call budget.
A subsequent owned audit reproduced valid token counters being discarded by
`AgentCallBudget.finish` when a model identity or the other counter is missing;
the gateway also checks model identity before accounting. Another run on the
same admission day can therefore call again despite the known token lower bound.
The bounded repair is assigned to the existing budget/receipt/gateway authority:
retain valid counters and unknown pricing separately, enforce original-day/run
fences, preserve legacy receipts, and expose a verified read-only summary. Root
generative monitoring and native completion will be integrated against that
contract; current nonnull-counter/charged-cost aggregation is not sufficient for
wrong-model pricing. No generative budget closure or actual spend claim is made
until that checkpoint and its consumer tests pass.

### Pure native order-term prerequisite

Accounting `a0db90a` integrated as `9edfa14`. The unchanged Alpaca, Binance and
Kraken order/intent predicates now share private pure functions in
`execution/native_order_terms.py`. The actual report paths retain their original
calendar, cumulative quantity, source capture and economic behavior; no adapter
or transport is constructed by these predicates. This extraction is a prerequisite
for independent retained-source verification, not an audit PASS or settlement API.

All 181 root cases passed in 15.46s across `test_alpaca_venue`,
`test_alpaca_day_calendar`, `test_binance_venue`, `test_kraken_venue`,
`test_lean_day`, `test_lean_trailing`, `test_lean_execution` and
`test_lean_binance_references`, with `--disable-socket --allow-unix-socket`,
`PYTHONDONTWRITEBYTECODE=1` and BLAS/OMP threads set to two. Log:
`/tmp/quantagentlab-root-pure-native-terms.log`. Production/script Ruff and diff
checks pass. Native/provider responses remain synthetic offline fixtures; no
authenticated or actual LEAN acceptance is implied.

### Copilot preserves rejected-model usage for accounting

The copilot previously replaced an actual wrong-model response with an empty
usage object before `AgentCallBudget.finish`, hiding valid counters from the
upcoming durable budget repair. It now passes the original raw response through
while still rejecting the model mismatch. The existing regression observes the
real finish call, verifies both original counters/model/content, then delegates
to the actual budget implementation and checks unavailable output, retained
reservation and no retry. `SimpleNamespace` remains required elsewhere.

All 56 `test_research_copilot.py` and `test_copilot_cli.py` cases passed in 29.09s
with sockets disabled and Unix self-pipes allowed; changed-file Ruff and diff
checks passed. Log: `/tmp/quantagentlab-root-copilot-raw-usage.log`. This caller
repair does not claim the separate generative-budget day-fence implementation
is already integrated; no wrong-model pricing or evaluator acceptance is added.

### Canonical ensemble artifact stage one

Data `e136011` and `5d887c8` integrated as `15f61b3` and `1503b3a`. Stage one
adds closed compositions, the parent/actual child and IC-selection artifacts,
five-frame identities, a bounded ZIP_STORED primary/sidecar pair and explicit
build/file-proof/load APIs. Loader checks all members and metadata before any
child deserialization; joblib remains limited to application-owned bytes whose
external hashes are authenticated by the consumer. File identity alone is not
source, fitting or approval authority. The existing single-family serialization
and H frame-hash convention remain compatible.

The root run passed all **236 cases in 299.19s**, zero skips:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests/test_canonical_ensemble.py \
  tests/test_ensemble_temporal_policy.py tests/test_temporal_models.py \
  tests/test_local_quant_flow.py tests/test_domain_common.py \
  tests/test_ensemble.py tests/test_core_models.py tests/test_walk_forward.py \
  -q --disable-socket
```

Log: `/tmp/quantagentlab-root-ensemble-artifacts.log`. This includes all 88 new
canonical cases, actual four-family fits, retained selection models, malformed
container/pre-deserialization rejection, frozen old serializer comparison,
real local Qlib/Alpha158/Ridge and canonical walk-forward ledger tests. Root
also removed one test-only first-party import separator exposed by the complete
checkout; production/script and changed-test Ruff plus diff checks pass.

Stage-two consumers are unchanged: no StrategySpec/H/ExperimentPlan composition
admission, canonical WF two-file provenance or evaluator comparison relaxation
has been introduced. Fresh CatBoost fits retain their actual varying native GUID
and train-finish metadata, so their real payload hashes can differ despite equal
local predictions. Frozen-parent reload/republication is byte-identical; fresh
four-family byte reconstruction is not claimed. Native semantic-comparison
design remains a separate review prerequisite. No metadata stripping, new
source/income authority, authenticated provider or release acceptance is implied.

### Generative usage authority and current native budget consumers

Jev `02115fb` integrated as `0d71c79`. Generative budgets now preserve independent
valid counters, unknown prices and original reservation floors, with immutable
run/original-UTC-day breach authority. The root reporting adapter delegates to
`src.agents.receipts.read_usage_summary` in read-only mode; the dashboard separates
known lower-bound cost, nullable exact cost and conservative charge. Missing or
corrupt authority is unavailable, never measured zero. Unknown provider model
text is not exposed by this public projection.

Copilot retains original response usage, propagates `AgentBudgetError.status`
without inventing an overrun, and compares pricing against the verified original
admission before attaching priced generation provenance. This also rejects a
changed caller price when an actual overrun takes status precedence.

Native admission, original finalization and recovery require the authenticated
v2 run/day summary, exact request/model/cap identities, consistent public
projections and a measured fresh audit clock. Final success requires both scopes
to have no pending, incomplete, unpriced or breached usage and to remain within
all explicit caps. Known bills can be lower than reserved charges; the invalid
`charged == actual + uncertain` equation is removed. Actual uncertainty/breaches
are archived as failure evidence; they never become successful discovery.
Preflight and final fetches reject stale same-day receipts, future clocks and
scope substitutions. Existing completed authoritative native results remain
immutable and readable; fresh v1-only receipts cannot authorize new completion.

Root offline evidence (TCP disabled, Unix event-loop self-pipes only):

- Initial consumer integration: **140 passed in 176.58s**.
- Final native recovery/supervisor plus actual gateway accounting: **74 passed
  in 69.53s**, including reservation-floor success, original/day breach failure,
  stale receipt rejection and immutable original-result retention.
- Actual gateway/reporting and copilot pricing/status regressions: **77 passed
  in 26.18s**.
- Actual budget authority, gateway/deployment, dashboard/CLI and existing decision
  reporting: **129 passed in 31.40s**.

Commands used `.venv/bin/python -m pytest ... -q --disable-socket
--allow-unix-socket` with `PYTHONDONTWRITEBYTECODE=1`, OMP/OpenBLAS threads2.
Logs are `/tmp/quantagentlab-root-generative-consumers.log`,
`/tmp/quantagentlab-root-generative-native-verified.log`,
`/tmp/quantagentlab-root-generative-extra-final.log` and
`/tmp/quantagentlab-root-generative-authority-final.log`. New provider-bound tests
use actual SQLite/ResearchGateway with HTTPX MockTransport; Docker boundary tests
exercise the actual supervisor/registry with isolated protocol fixtures. Initial
new-test fixture errors and stale synthetic receipt callbacks were corrected;
no runtime authority, deadline or provider-usage check was relaxed. Production
and changed-test Ruff plus diff checks pass. These overlapping counts are not a
whole-suite green result. No authenticated provider, Docker, financial efficacy,
positive income-source or release acceptance is implied.

A final public receipt re-export check now retains the verified `statuses`
projection, so the bounded archived v2 receipt itself passes the same verifier.
Malformed non-text audit clocks are rejected explicitly. All **22 actual
consumer tests passed in 4.12s** (`test_generative_usage_consumers.py`, same
offline environment); log `/tmp/quantagentlab-root-generative-reexport.log`.

### Native retained-source compliance: read-only phase one integrated

Accounting `45c143a`/`dc119ba` integrated as `8a0e17e`/`0b0b1cb`.
Pure retained Alpaca/Binance/Kraken/LEAN exporters preserve first observation
versus first complete execution authority and reject unsupported valuation
closure or source revisions. The public canonical `execution_compliance` reader
has no historical `as_of` escape hatch and does not trust mutable report/applied
flags or the presence/absence of discrepancy indexes. Detection only appends
immutable metadata; no settlement or expiry exception is reachable.

Root source/audit/venue/durable/publication/historical-sleeve verification passed
**113 tests in 18.22s**, zero skips, log
`/tmp/quantagentlab-root-execution-authority.log`. Strict no-file-mutation tests
use closed DELETE-journal archives; live WAL read consistency is preserved,
including SQLite's possible coordination-file creation. No `immutable=1`
shortcut is used to claim live authority.

Accounting validation `555a61f` integrated as `2ebf571`. Its separate required
compliance check applies even to accounting-only artifact validation; arithmetic
reconciliation cannot clear an execution violation. Root validation/artifact/
benchmark/callback verification passed **24 tests in 281.13s**, and actual
registered model/advice experiments passed **10 tests in 293.93s**. Logs:
`/tmp/quantagentlab-root-compliance-validation.log` and
`/tmp/quantagentlab-root-compliance-registered.log`. All runs disabled TCP and
allowed Unix event-loop self-pipes only. These results precede the next registry
locator/ancestor gate changes, which remain separately under implementation.

Data's source-pinned CatBoost comparator feasibility note `192920a` integrated
as `9b8df4e` is documentation only. No comparator/parser, model-byte rewriting,
consumer widening or exact-evaluator relaxation has been implemented.
