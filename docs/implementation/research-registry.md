
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
