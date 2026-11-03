# Paper restart: restore canonical balances after brokerage setup

Resolved the completed-only restart balance blocker reproduced in the integrated
recovery run. No reconciliation rule, order ownership, pending-history refusal,
source authority, fill economics or live-account setting was relaxed.

## Lifecycle correction

Pinned LEAN source `3b2bf19e597805cc917dd199ed9ee7d10d5ceb84` calls algorithm
Initialize, loads brokerage cash/holdings, then PostInitialize. Its Python wrapper
calls the base PostInitialize directly, so overriding that Python method is not
the right hook. AlgorithmManager invokes Python OnWarmupFinished after setup,
including when no warmup is configured. Source inspected locally:
`Engine/Setup/BrokerageSetupHandler.cs`, `Engine/AlgorithmManager.cs`, and
`AlgorithmFactory/Python/Wrappers/AlgorithmPythonWrapper.cs`.

The algorithm now:

1. Validates local PaperBrokerage/account identity and reconciled original ledger
   in Initialize. Unresolved accepted orders still reject there, before any
   runtime balance restoration.
2. Restores cash and holdings once in OnWarmupFinished, after LEAN's configured
   starting cash has been loaded. Restoration repeats the original-history
   check. Its existing immutable `paper-restore` receipt is retained.
3. Creates the execution bridge only after successful restoration. A duplicate
   warmup-finished callback cannot reset balances; a data callback before ready
   fails closed. Live accounts keep their original separate path and never
   receive paper balance writes.

QCAlgorithm.SetCash is locked after setup, so restoration sets the existing
CashBook account-currency amount using the exact CLR decimal transport. It first
requires the actual account currency to match and its conversion rate to equal
one. It does not fabricate conversion rates or alter canonical cash. Existing
crypto base inventory and holdings restoration remain in effect. This is local
paper initialization, not a periodic balance repair or permission to modify
actual broker cash.

## Actual engine verification

Same existing network-disabled image, no rebuild:
`sha256:94ee09ba94c575561b11bc19cc6f5406f507d1f2fa436449b13ea3532612b2d2`.
Fresh production manifests and exactly three bounded engine starts:

| Scenario | Engine exit | Cleanup | Result |
| --- | --- | --- | --- |
| Origin fill + accepted pending limit | 0 | 0 | Original fill and both exact ticket bindings retained |
| Completed-only fresh engine | 0 | 0 | Runtime ready; cash/holdings reconcile; repeated completed dispatch has no new native order or fill |
| Accepted-pending fresh engine | 1 | 0 | Expected initialization refusal; original bindings, reservation and receipts unchanged |

Containers respectively:
`43df46f4df978b57b4acf5fae87f3d62987a89ac9780fb8ba14ffa2fe1344d61`,
`6db9013d06608875a8f4e6521625d7266255cf3168cde5e33c42248080495189`,
`a9a56b18f4b3126656611d2d21c03dd6f4386129211b6e891035d13f82fc11e3`.
All confirmed removed; no timeout or OOM.

Completed-only restart observed exact USD cash `99894.46868774948926495`, matching
the retained ledger, and BTC inventory/holding1. Runtime health was `ready` with
no error; balance reconciliation returned true and execution compliance PASS.
One paper-restore receipt belongs to the new runtime. Three dispatch retries
retained one original fill and claim, while the fresh engine's order list stayed
empty. Read-only post-exit checks verify the same ledger state. Original source
manifest hashes match every current execution implementation file.

Private synthetic evidence:
`~/.local/share/quantagentlab/runtime-smoke/recovery-cash-fix-20261103-root/`.
It retains original source map/manifests, configurations, SQLite backups,
`restart-ready.json`, engine logs, inspections, cleanup/absence and SHA256 inventory.
No credentials, real broker connections or public market-data calls were used.

Reproduction from repository root:

```bash
PYTHONPATH="$PWD" .venv/bin/python environments/lean-smoke/recovery_driver.py \
 --source /tmp/qal-lean-build-source --output /absolute/new/private/directory
```

The completed-only fixture now explicitly requires ready health and canonical
balance reconciliation before writing its retry result. The evidence reader keeps
the historical pre-fix blocked case distinct from this newly healthy result.

## Regression validation

**145 passed in 5.24s**, no skips, with TCP disabled and Unix sockets allowed:

```bash
QAL_LEAN_RECOVERY_EVIDENCE=/home/ozzstein/.local/share/quantagentlab/runtime-smoke/recovery-cash-fix-20261103-root \
 PYTHONPATH="$PWD" .venv/bin/python -m pytest \
 tests/test_lean_paper_initialization.py tests/test_lean_decimal.py \
 tests/test_lean_runtime_enum.py tests/test_lean_execution.py tests/test_lean_day.py \
 tests/test_lean_trailing.py tests/test_lean_binance_references.py \
 tests/test_signal_bridge.py tests/test_lean_runtime_lifecycle.py \
 tests/test_lean_runtime_recovery_acceptance.py --disable-socket --allow-unix-socket -q
```

Nine new initialization tests cover ordering, single restoration, blocked early
callbacks, failed restoration, live/nonlocal denial, unresolved original orders,
and incorrect account currency/conversion rejection. They are isolated lifecycle
unit tests; the three runs above provide actual CLR/engine evidence. Ruff and
format/diff checks passed.

This resolves completed-only local paper restart readiness. It does not recover
lost active PaperBrokerage history, admit pending orders after restart, or prove
live-native broker recovery. Partial fills, nonzero fee acceptance, DAY/trailing
runtime paths and broader release/source-income requirements remain separate.
