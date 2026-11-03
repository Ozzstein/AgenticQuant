# Bounded actual LEAN recovery acceptance — 2026-11-03

Partial acceptance: completed-order retries do not duplicate submission or fills,
and a fresh PaperBrokerage engine fails closed on unresolved submission history.
**Accepted-pending-ticket restart remains blocked by a runtime overload defect.**
No production execution module or shared smoke fixture was changed.

Source baseline: `bc48d273eac9080267330924f0ca0e642718f0d4`.
Image ID: `sha256:94ee09ba94c575561b11bc19cc6f5406f507d1f2fa436449b13ea3532612b2d2`.
See [shutdown baseline](lean-shutdown-fix-2026-11-03.md) and
[original runtime report](lean-runtime-smoke-2026-11-03.md).
The harness identifies Codex / gpt-6-astra / high effort. Standard service tier
was requested but is not independently exposed by the available runtime evidence.
No subagents were started.

## Actual engine observations

Exactly three engine starts; no install, rebuild, provider connection, credentials,
live capital, push or merge. All used the fixed local Docker socket, empty Docker
config, network none, 4 GiB memory/swap, 2 CPUs, 256 PIDs, capabilities dropped and
no-new-privileges. Actual inspections record these settings and the read-only
`/app/src` mount from this worktree. Each attached engine had a 150-second outer
bound. All three exact container names were removed and subsequently inspected as
absent. No external timeout or OOM occurred.

1. Initial fixture sizing failure: 0.001 units violated minimum quote notional
   before submission. Exit **1**, duration about 15 seconds, cleanup **0**.
   Container `c4c4482b841dc90051fd691649f4f1dd13b3a2beb67fc96580af9d015bca6f25`.
   This was a fixture mistake, not recovery evidence.
2. Corrected origin: one unit filled through the production bridge. Three
   completed-order `dispatch` retries plus reconciliation retained exactly one
   native order, one fill, one submission claim and the same original ticket
   binding. A later one-unit buy limit at 1 produced a durable submission claim
   but **no native limit ticket**; outbox became `ambiguous`. The fixture reached
   its own 60-second observation cutoff and quit normally. Exit **0**, total
   container duration about 75 seconds, cleanup **0**. Exit0 alone does not mean
   the intended pending-ticket scenario succeeded.
   Container `1bc343346c271c78e57c6722287719bafb002383237fd0ac805f132259cd022f`.
3. Fresh engine from a SQLite backup of that real original ledger: initialization
   rejected with `LEAN paper restart has unresolved accepted orders; original
   proof required`. The diagnostic records and re-raises the original exception;
   it does not bypass initialization. Exit **1**, about 12 seconds, cleanup **0**.
   Container `32a8219f138772ffad632d974b2eca7c41422d318830cc405d8f979651d9a819`.

The fresh runtime ID differs from the original. Its actual native order list is
empty. Original ticket bindings, both submission claims, fill, outbox, reports
and pending reservation remain unchanged. The unresolved limit has **no ticket
binding**, so no original accepted-ticket ownership is claimed for it. Its
reserved cash remains exactly `101.761999450653214`, remaining quantity 1. A
read-only post-exit comparison confirms identical binding, outbox, report and
fill tables. The quarantine here is initialization refusal with preserved
`ambiguous` state; it is not successful recovered execution.

## Concrete defect for the coordinator

`src/execution/lean_runtime.py`, `LeanAlgorithmPort.submit`, calls:

```python
self.algorithm.LimitOrder(symbol, quantity, limit_price, client_id, properties)
```

The pinned LEAN source (`Algorithm/QCAlgorithm.Trading.cs:410`) requires:

```csharp
LimitOrder(Symbol symbol, decimal quantity, decimal limitPrice,
           bool asynchronous = false, string tag = "", IOrderProperties orderProperties = null)
```

The fourth positional argument is therefore a client string where the runtime
expects a boolean. The int/double overloads also require that boolean. This is a
source-confirmed mismatch and the likely cause of the observed claimed attempt
without a ticket. The production dispatcher catches submission exceptions and
later reconciliation overwrites the detail with `lookup unresolved; never safe
to blindly resubmit`; the precise thrown exception was not retained by this
fixture, so its class/message are not asserted. The StopMarketOrder call has the
same positional pattern, but was not exercised. Reported only: no economic rule,
authority verifier, timestamp or production overload was changed here.

Ownership inspection also found the expected safeguards: durable submission
claims in `LeanVenue._claim`, runtime-bound lookup rejection in `LeanVenue.lookup`,
and refusal of unresolved orders in `LeanAlgorithmPort.restore_paper_state`.
Fresh PaperBrokerage cannot recreate native original ticket/history authority.

## Evidence and reproduction

Private directories (synthetic artifacts only):

- `/home/ozzstein/.local/share/quantagentlab/runtime-smoke/recovery-bc48d27-20261103-7c92`
- `/home/ozzstein/.local/share/quantagentlab/runtime-smoke/recovery-bc48d27-20261103-corrected-7c92`

They retain engine console/results, manifests, actual fixture main.py, SQLite
ledgers, source hash maps, source HEAD, exact create command, image/host inspection,
exit/cleanup/absence proof and SHA256 inventories. Original checkpoints are
SQLite backups with no receipt/table edits. Config and production implementation
manifest are copied intact to the restart; the actual production verifier runs.
The execution source mounted in all three runs is this checkout, not the parent
repository. No timestamp backdating or source-authority bypass was used.

Owned fixtures:

- `recovery_algorithm.py`: original fill, repeated completed dispatch, intended
  pending limit and bounded observation; records initialization rejection.
- `recovery_driver.py`: new private preparation and up to three bounded starts;
  refuses automatic restarts if the accepted-pending checkpoint was not reached.
- `recovery_resume_pending.py`: one explicitly selected restart from the retained
  original timeout/ambiguous ledger, requiring completed origin cleanup and a new
  destination. Does not rerun the origin.

Actual corrected command used the root interpreter and checkout PYTHONPATH:

```bash
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python \
 environments/lean-smoke/recovery_driver.py \
 --source /tmp/qal-lean-build-source \
 --output /home/ozzstein/.local/share/quantagentlab/runtime-smoke/recovery-bc48d27-20261103-corrected-7c92 \
 --skip-completed-restart
```

After the driver correctly stopped without the accepted-pending checkpoint,
`recovery_resume_pending.py --output <same-directory>` performed the third and
final engine start. Do not reuse these evidence directories for another run.
The initial sizing failure consumed one of the three allowed starts. The
completed-only fresh-engine branch in the driver was therefore **not executed**.

## Validation and limits

Five read-only actual-evidence assertions passed:

```bash
QAL_LEAN_RECOVERY_EVIDENCE=/home/ozzstein/.local/share/quantagentlab/runtime-smoke/recovery-bc48d27-20261103-corrected-7c92 \
 PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python -m pytest \
 tests/test_lean_runtime_recovery_acceptance.py --disable-socket --allow-unix-socket -q
```

These assertions launch no engines and skip without the explicit private evidence
path. They validate this observed ambiguous-attempt outcome, not successful
accepted-ticket recovery. Two separate existing **synthetic fake-port** tests
also passed: `durable_attempt_recovers_lost_ack_without_new_ticket` and
`new_runtime_cannot_invent_restart_fill_history_or_resubmit` in
`tests/test_lean_execution.py`. They are not actual-engine acceptance. Ruff and
diff checks passed for owned files.

Next step: coordinator corrects/reviews the LimitOrder positional overload, then
prepares a fresh manifest and new private evidence under a separately authorized
run budget. Require an actual Submitted pending limit ticket before shutdown;
retry the completed-only checkpoint in a fresh engine, and require pending-paper
restart to reject without resubmission or altered original ownership/reservation.
No live/native broker recovery, successful original pending-ticket recovery,
process-kill crash injection, partial-fill restart or production readiness is
established by this task.
