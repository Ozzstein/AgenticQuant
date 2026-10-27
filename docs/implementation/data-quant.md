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
