# Data and quantitative lane evidence

Worktree branch: `build/data-quant`; base `6844b89`. This is an integration checkpoint toward the complete A–I plan, not a full-system acceptance claim.

## B: immutable evidence and market boundary

`src/domain/market.py` defines Decimal-valued `MarketBar` and `MarketSnapshot` tied to `RunContext`, `Instrument`, and `EvidenceRecord`. Bar intervals are half-open; publication is distinct from bar close. Execution adapters must call `require_execution_price()` and use raw equity/spot bars. Perpetual bars are features only. Snapshots reject future/incomplete bars, unknown availability, duplicate conflicting prices and missing provenance.

`src/data/evidence/store.py` exposes `EvidenceStore(path)`, append-only `put`, `query(as_of=...)`, complete `document(evidence_id, as_of=...)`, exact `passage(evidence_id,start,end,as_of=...)` and aggregate-bounded `retrieve`. Passage offsets count Unicode characters, not bytes. Each passage carries its full `EvidenceRecord` and document hash. SQLite stores immutable raw response bytes, full documents, normalized payloads and content-addressed snapshots transactionally. All historical retrieval enforces the common documented-availability rule; later ingestion does not disqualify documented backfill. Publication/revision times remain separate envelope fields. Explicit revision links can supersede a same-provider fact; differing sources and unlinked contradictory revisions remain visible. Coverage distinguishes missing from ineligible and conflicting data. No retrieved passage grants execution authority.

`SecurityMaster` resolves evidence-backed, interval-scoped symbol aliases and historical universe membership; it never substitutes today's members. Historical coverage must actually be ingested. A membership result is not proof of a complete universe dataset; upstream coverage acceptance remains required.

Validation (2026-10-27):

```sh
PYTHONPATH="$PWD" /tmp/quantagentlab-review-venv/bin/python -m pytest tests/test_evidence_boundary.py tests/test_domain_common.py -q --basetemp=/tmp/quantagentlab-data-quant-tests
# 21 passed
/tmp/quantagentlab-review-venv/bin/ruff check src/domain/market.py src/data/evidence tests/test_evidence_boundary.py
# All checks passed
```

These are offline invariant tests with test-only fixtures. Provider entitlements, real source completeness, historical US universe/delistings and all release gates are unverified. Direct/MCP ingestion, faithful feature/model flow, US Qlib execution, D causal HMM/strategies/allocation and external integration work continue in subsequent commits. Shared dependency/config/CLI integration belongs to the coordinator.

## C2/G: exact factors and temporal model artifacts

`src.data.features.expressions.evaluate_expression(expression, frame)` is the single bounded AST interpreter for inference and research. Unsupported expressions, negative shifts, centered/global calculations, arbitrary attributes/calls and invalid panel identities raise `ExpressionError(ValueError)`. Supported Qlib operators are Ref, Delta, Mean, Sum, Std, Var, Min, Max, EMA, trailing Rank, Corr, Cov, Abs/Log/Sign/If/Greater/Less. The safe pandas subset includes shifts, differences, returns with no fill, trailing rolling aggregations, ewm mean, abs/cumsum/clip. There is no host eval. Formula names cannot override expressions. The coordinator must replace its research evaluators' host eval with this API.

`DataPipeline` uses this interpreter, fails explicitly on invalid libraries/Qlib failures, merges custom features by date AND instrument, and requires explicit instruments for direct Yahoo fetches. `FactorEngine` preserves positional alignment across multiple instruments per date and adds the documented equal-weight 3/6/12-month momentum minus last-month reversal specification. Crypto beta estimation uses consistent covariance/variance and only earlier observations; crypto volatility annualizes at 365.

`feature_dataset(MarketSnapshot, specs, horizon=...)` emits identity-preserving features/labels with both availability series and snapshot/specification hashes. `ModelWrapper.train(..., context=RunContext, feature_available_at=..., label_available_at=..., feature_spec_hash=..., data_snapshot_hash=...)` checks temporal eligibility and rejects revised inputs unavailable at their original decision date. Predict accepts explicit context/availability for temporal inference. Median imputation and standardization are fitted only on training data. All four actual model families execute offline; no substitute models. Saved joblib payloads include transforms and require versioned hash sidecars. Temporal artifacts carry environment, parameters, code, inputs, feature schema and training/label windows. Loading joblib is restricted by convention to trusted application-owned artifacts; raw provider/research payloads must not be loaded. Calls without temporal provenance remain generic diagnostics and cannot perform contextual inference. ICIR derives from actual per-date cross-sectional ICs; undefined statistics are NaN.

Validation:

```sh
PYTHONPATH="$PWD" /tmp/quantagentlab-review-venv/bin/python -m pytest tests/test_feature_fidelity.py tests/test_temporal_models.py tests/test_evidence_boundary.py -q --basetemp=/tmp/quantagentlab-data-quant-tests-2
# 28 passed in 2.07s; includes actual LightGBM/CatBoost/XGBoost/Ridge training and artifact roundtrips
```

Coordinator review fixes: snapshots check ALL supplied evidence, not just bar-linked records; every SQLite connection closes after commit/rollback.
