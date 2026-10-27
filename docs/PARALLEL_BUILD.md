# Parallel build coordination

Started: 2026-10-27. Coordinator thread: `67e24814-0156-42fb-8442-ebe1563b1d38`.

Active isolated worktrees (base checkpoint `6844b89`):

| Thread | Branch | Stable deliveries integrated by coordinator |
|---|---|---|
| `mcp:4cdc3e7e-410b-476a-96ab-11c9a386f3d5` | `build/accounting-backtesting` | `693c768` → `db05c24`: portfolio contracts, ledger, paper execution |
| `mcp:c6065855-9311-421f-94df-3e66d1cda3f1` | `build/data-quant` | `67c5086` → `717c908`: market contracts and evidence store |
| `mcp:8f3ff1c4-5616-4111-a59d-9a3821348bb0` | `build/jev-agents` | `4cac664` → `575d1dc`: decision and agent-action contracts |
| Coordinator | `build/full-system` | Integration, configuration, reproducibility, research and operation |

The [full system build plan](FULL_SYSTEM_BUILD_PLAN.md) is the completion authority. Read the [verified assessment](BRIEF_ASSESSMENT_AND_PLAN.md), [MCP decision](MCP_INTEGRATION_DECISION.md), and [Jev decision](JEV_INTEGRATION_DECISION.md) before implementation. Intermediate deliveries are integration checkpoints, not reduced releases. No mocks, invented metrics, or silent feature substitution may satisfy a runtime requirement.

## Ownership

Each implementation thread runs in its own T3-bound worktree from the same local checkpoint. It commits its own work and sends the coordinator the commit IDs, test evidence, interface notes, and remaining requirements. Only the coordinator integrates branches. Do not edit another checkout or reset/rewrite shared history.

| Lane | Owned implementation | Initial complete delivery | Follow-on responsibility |
|---|---|---|---|
| Accounting and backtesting | `src/execution/`; `src/core/backtester.py`, `walk_forward.py`, `validation.py`, `benchmark.py`; `scripts/run_backtest.py`; new `src/domain/portfolio.py`; corresponding tests | Reconciled durable ledger/paper accounting, corrected risk invariants, chronological replay and walk-forward results, one validation path, explicit CLI failures | Remaining F venue/optimizer paths and C acceptance; coordinate crypto/equity calendars with data lane |
| Data and quantitative models | `src/data/`; `src/core/data_pipeline.py`, `crypto_pipeline.py`, `factor_engine.py`, `crypto_factors.py`, `model_zoo.py`, `ensemble.py`, `macro_regime.py`, `strategy_selector.py`, `strategy_tracker.py`; `src/strategies/`; new `src/domain/market.py` and `src/data/evidence/`; corresponding tests | Versioned evidence storage and as-of retrieval, real equity/crypto adapters with explicit coverage/failure, faithful feature/model data flow | Selected data/MCP integration, US Qlib scenario, all D strategies/regimes and quantitative model variants |
| Jev and agent desk | `src/agents/`; new `src/decision/`, `src/domain/decisions.py`; corresponding tests | Actual Jev transport and replayable typed service, confidence semantics, source-linked classification/verification nodes, full graph integration | All E roles/debates/reports, Jev J1–J6 evaluation; J7 remains conditional |
| Coordinator | `src/domain/common.py`, shared package exports, `src/utils/`, configuration, dependencies/locks, CI/Docker; research orchestration/proposer/evaluator; scheduler/pipeline/dashboard/monitoring; requirement register and acceptance | Shared interfaces, reproducible checks, branch integration, removal of cross-system failures | A, G, H, I and remaining cross-lane integration through full-system completion |

Each lane may create isolated modules/tests under its owned paths. Shared-file edits require coordination: send the proposed change to the coordinator instead of competing over `src/utils/schemas.py`, config, dependency files, package exports, or top-level CLI wiring. A lane may create a directly runnable package CLI within its owned package; the coordinator connects the canonical CLI. Existing imports remain usable until their consumers migrate explicitly. Record requirements needing another lane; do not fake the dependency.

## Shared boundary available at launch

`src.domain.common` supplies immutable versioned `Instrument`, `RunContext`, and `EvidenceRecord` contracts:

- Instrument identity is explicit and stable; a display ticker is insufficient. Asset classes are `equity`, `etf`, and `crypto`; legacy enum adapters must translate explicitly.
- All instants are timezone-aware and normalized to UTC. Run clocks and configuration hashes are explicit; no ambient current-time defaults are allowed in historical decisions.
- Evidence distinguishes observation, documented availability, ingestion, requested historical cutoff, revision, quality, and immutable raw-content hash. `require_available(as_of)` rejects future/unknown/inferred/unusable evidence, while permitting documented historical backfill ingested later.
- `content_hash()` identifies canonical serialized contract content. These contracts establish boundaries, not provider verification, data completeness, or a complete ledger.

Portfolio contracts belong to the accounting lane, market bars/features to the data lane, and decision contracts to the agent lane. Import sibling modules directly; send changes to common contracts/package exports to the coordinator. Until another contract lands, use explicit narrow adapters against existing schemas, not duplicate implementations presented as final integration.

## Evidence and reporting

Each lane maintains `docs/implementation/<lane>.md` with requirement IDs, implemented entry points, exact commands/results, integration prerequisites, and unresolved findings. Unit fixtures may be synthetic; mark external-provider checks separately. An unavailable credential, dataset, model, venue, or Docker host is an unverified dependency, never a successful mock fallback. Do not expose secrets or provision paid accounts. No live-capital trades are authorized by this build request.

Use private worktree test environments. A review environment exists at `/tmp/quantagentlab-review-venv`; it may be used read-only to run tests from the current worktree, but do not install into or mutate it from parallel threads. Run Python through its absolute path with the current worktree on `PYTHONPATH`; do not test the old copied source in `/tmp/quantagentlab-review-source`. Override shared `/tmp/aiquant_test` outputs to unique lane-local temporary paths where applicable.

Before committing a delivery, run relevant invariant tests and lint on changed Python. Correct tests that assert the known broken semantics; do not delete checks to claim success. Return commits and remaining work without pushing or opening a PR unless the coordinator requests it. Send milestones or contract questions to the coordinator with T3 thread messaging; do not launch more threads independently.

## Completion accounting

The coordinator tracks the entire A–I scope. Passing unit tests proves only the exercised invariants. Real provider/venue calls, all required strategy/model/optimizer paths, deployment/recovery, research promotion/rollback, and the brief's prospective observation remain distinct acceptance requirements. No thread may label the full system complete from its own component result.
