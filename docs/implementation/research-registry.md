
## Model-free historical advice verification

`ResearchRunner(..., authorities=AgentHistoryAuthorities(...))` and
`ExperimentEvaluator(..., authorities=...)` use operator-selected original desk,
Jev, budget and portfolio authorities. CLI `rd-agent run --agent-authorities
paths.json` loads `AgentAuthorityPaths`; candidate plans cannot select these paths,
and missing stores are rejected before any empty authority is created.

Independent recipe verification reexecutes the registered advice strategy from
actual exported receipts, original HMM/portfolio proofs and current replay
holdings. Every output result field, economic trace, source/model artifact and
opportunity byte is compared; only filesystem placement is normalized after
content verification. A successful reproduction issues an in-process typed
`AdviceValidationProof` bound to the canonical manifest, decision trace and exact
sorted decision/receipt/original-snapshot identities. The validator does not accept
a JSON proof or a boolean success flag. Missing authorities remain a failed check.

Model-free observations carry no fabricated model folds, predictions, IC, ICIR or
factor correlation. The authoritative advice check replaces only those numerical
model checks. Actual ledger reconciliation, costs, matched 1000 controls, calendars,
source eligibility, regime evidence, holdout, multiplicity and performance gates
remain required. Advice is held fixed in cost counterfactuals; actual current
ledger holdings determine sizing. This evaluates recorded advice, not hypothetical
new provider calls under counterfactual portfolios.

Verification: 9 offline tests passed in 293.45s through real graph generation
receipts (provider fixtures), source/budget/history stores, original portfolio
proof, actual model-free replay, three cost ledgers, 1000 persisted random controls,
independent reconstruction and typed advice validation. The fixture is explicitly
synthetic and cannot activate; authenticated provider/historical/statistical
acceptance is not established by these tests.

## Current execution authority binding

The root follow-on adds an immutable per-evaluation locator manifest in the
existing registry, outside the old `Evaluation` serialization. The evaluator
selects the original replay ledger, the exact operator-selected portfolio stores
behind archived advice receipts, and the original paired baseline where present.
Required scope is reconstructed from typed archived replay/manifest, full decision
receipts and paired proof; candidate JSON cannot nominate a clean substitute or
supply a PASS. Each locator retains canonical path and ordinary file identity,
account ID, canonical account-spec/immutable execution-ownership hashes, original
audit identity and exact source hashes. Original ownership hashes are derived by
the canonical accounting reader, not duplicated SQL in the registry.

Promotion, current active/resolve, rollback and status traverse all retained
trials of the candidate and its ancestors and reread every required canonical
account authority without an `as_of` parameter or cached PASS. Missing, changed,
noncompliant or unresolved authority blocks current eligibility, including when
an immutable earlier evaluation passed. Original activation/outcome bytes remain
available to historical monitoring. Legacy evaluations without a locator cannot
acquire one retrospectively through an idempotent outcome retry. A normal repeat
with the same identity retains the original locator receipt instead of replacing
its measured creation/audit clock. Legitimate appended history may change the
current authority-prefix hash; it must not change account or execution ownership.

A path moved/replaced by backup or restoration is not silently reauthorized.
Operator-led relocation/recovery remains a separate explicit migration contract;
there is no mutable path override at resolve time. These locators identify local
authoritative stores, not authenticated provider access or an external backup
attestation. No settlement, source parser, model composition or evaluator metric
relaxation is part of this gate. The canonical reader's required
`account_spec_hash` and `execution_binding_hash` fields are integrated from
accounting `87b9165` (root `e3d5ed9`), alongside source reexport, current audit,
validation, native claim and H uncached gates. No local locator establishes
provider entitlement or permits factual economic settlement.

Root verification: 111 claim/source/audit/paper/opening tests passed in
10.46s; 206 native/LEAN/DAY/trailing/expiry/frozen-reference tests passed in
19.22s; 70 registry/monitor/ancestor/locator/identity tests passed in 18.11s.
The latter included one test-only Decimal serialization warning, corrected to
use an actual Decimal. Final identity/admission/audit/locator verification passed
67 tests in 10.54s without warnings. H independently verified 23 actual
compliance/midnight/expiry/passive cases against exact `e3d5ed9` in 8.97s.
All commands use root `.venv/bin/python -m pytest -q --disable-socket
--allow-unix-socket` with `PYTHONDONTWRITEBYTECODE=1` and bounded BLAS threads;
no provider, Docker, credential or capital action occurred.

Actual `test_research_evaluation.py` + `test_agent_research_evaluation.py` passed
6 tests in 357.17s, including real generation/source/history/ledger receipts,
three cost ledgers and 1000 benchmark controls. The original advice account is
then given a retained unproved native footprint: mathematical reconciliation
continues to pass, current eligibility fails, and the frozen evaluation remains
unchanged. `test_research_execution_proof.py` passed 2 tests in 115.85s, including
actual model/regime reconstruction and the same late-failure check against the
original paired baseline account. The scheduled research holdout-before-outcome
access regression passed separately in 2.94s. These are synthetic protocol and
accounting fixtures, not statistically accepted trading candidates.

Tests used the fully integrated root runtime, including source/audit/validation
and H/native gates through `e3d5ed9`, plus this root locator change; later parent
commits `c29847b` and `7db2817` add only attributed lane evidence. Logs:
`/tmp/quantagentlab-root-canonical-claim-gates.log`,
`/tmp/quantagentlab-root-canonical-native-gates.log`,
`/tmp/quantagentlab-root-execution-locators.log`,
`/tmp/quantagentlab-root-current-authority-final.log`,
`/tmp/quantagentlab-root-current-authority-evaluation.log`, and
`/tmp/quantagentlab-root-current-authority-paired.log`.
Ruff passes for all `src`/`scripts` and changed tests; diff checks pass. Existing
fresh income-source failures remain enabled and are not waived by this gate.
