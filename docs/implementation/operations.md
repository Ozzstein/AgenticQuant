# Operating control and monitoring

The operating store coordinates work; `PortfolioLedger` remains the sole economic authority. `OperationsStore` archives immutable run manifests, exact inputs and stage outputs, commits an attempt before work, and records failed or uncertain completion explicitly. A completed stage replays its verified receipt. A process crash does not authorize repeating an external call: recovery must use the original frozen input and the component's reconciliation API. Kernel process locks serialize work across independent processes.

`ScheduleSupervisor` persists UTC deadlines, uses declared local time zones and actual XNYS session dates, and archives an input bundle before invoking its handler. Missing bundles wait only until the configured deadline, then become missed runs with alerts. Failed or interrupted runs block their schedule until explicit recovery names the exact input hash. Restarting does not erase failures or schedule duplicate successful jobs.

`src.dashboard` and `src.rd_agent_dashboard` are read-only projections of the actual ledger, research registry, evidence archive, operating receipts and `AgentCallBudget`. Unknown usage retains its charged reservation and is displayed separately from cost calculated from reported tokens at configured rates. Missing stores or marks remain unavailable. Research monitoring shows development outcomes and holdout consumption, not holdout measurements in development views. There are no trading or activation controls in the dashboard.

Configuration paths live under `AppConfig.operations` and default to the configured output directory. Explicit approved baseline and model hashes default to empty; file presence and candidate JSON flags never authorize trading. The account uses `paper_ledger_path` and `paper_account_id` consistently.

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
