# Full-system implementation evidence register

Authority: [full build plan](FULL_SYSTEM_BUILD_PLAN.md). Coordination: [parallel build](PARALLEL_BUILD.md). Updated 2026-10-27.

Status values: `pending`, `in_progress`, `implemented_unverified`, `verified`, `rejected`. Verification applies only to the stated evidence and does not imply live-capital eligibility. This register begins at workstream/subsystem level; lane records must expand individual requirements as they are implemented.

| Requirement | Owner | Status | Implementation / evidence | Remaining acceptance |
|---|---|---|---|---|
| A: reproducibility, shared contracts, configuration, CI, packaging | Coordinator | in_progress | `src/domain/common.py`; temporal/provenance tests in `tests/test_domain_common.py` | Lock environments; integrate domain contracts; canonical config/CLI; clean-host builds and CI |
| B: US equity/crypto data, point-in-time evidence, provider/MCP access | Data/quant | in_progress | `src/domain/market.py`, `src/data/evidence/`; initial six evidence-boundary invariants pass | Historical universes/revisions/corporate actions; actual source coverage, entitlements and failure behavior |
| C1–C7: quant research, chronological replay, walk-forward, validation, reporting | Accounting/backtesting + data/quant | in_progress | Reproduced defects documented in assessment | All rebuilt-engine correctness and real-data acceptance; all model families and matched benchmarks |
| D: HMM, six strategies, selection/blending, attribution | Data/quant + accounting | in_progress | Existing modules require causal integration | Every strategy/mode changes portfolio behavior; risk-compliant allocation and actual attribution |
| E: full agent desk and MCP evidence tools | Jev/agents + data | in_progress | `src/domain/decisions.py`; typed agent actions; shared AnalysisResult 2.0 migration | Five real roles, required debates, binding veto, complete reports and provider evidence |
| F: five optimizers, complete risk policy, durable paper/live execution | Accounting/backtesting | in_progress | `src/execution/ledger.py`, `simulator.py`; 30 focused accounting checks pass; broader regressions tracked | Reconciliation, crash/partial-fill handling, every venue including operational LEAN |
| G: native RD-Agent and real strategy/regime/crypto research | Coordinator + data/agents | pending | Existing stubs/simulated outputs are not accepted | Every named native workflow; real custom experiments; quarantine/validation/promotion/rollback |
| H: unattended operation, dashboard, alerts, monitoring, recovery | Coordinator | pending | Existing divergent entry points are not accepted | Shared deployed cycle, visible ledger/evidence, operational recovery and measured budget |
| I: complete integration and release gates | Coordinator | pending | No full-system acceptance yet | Real integrations, all variants, research lineage, prospective comparison, three-month paper and other gates |
| J1–J6: selected Jev workflows | Jev/agents + data/research | in_progress | Accepted [Jev decision](JEV_INTEGRATION_DECISION.md) | Typed service, source linkage, accuracy/calibration, actual calls, replay, failure behavior and measured value |
| J7: auxiliary macro labeling | Jev/agents + data | pending | Conditional extension only | Demonstrate incremental value; HMM remains authoritative |

No performance, broker, Qlib, RD-Agent, or Jev provider acceptance is claimed by this bootstrap. Record executed test commands/results in the owning lane's evidence file rather than inferring completion from file existence.
