# Full-system implementation evidence register

Scope authority: [full system build plan](FULL_SYSTEM_BUILD_PLAN.md), with the
accepted [MCP](MCP_INTEGRATION_DECISION.md) and [Jev/J](JEV_INTEGRATION_DECISION.md)
decisions. **Updated 2026-11-03 against integrated code `4d898ed`.** This is the
current progress register; dated implementation reports preserve historical
results. This documentation update does not rerun runtime or full-suite tests.

## How to read progress

- `offline_verified`: the stated local/deterministic boundary was exercised; no
  authenticated, statistical, prospective or deployment acceptance is implied.
- `runtime_verified`: the explicitly named actual runtime scenario passed. It
  does not certify every path in its parent group.
- `partial`: implementation/composition or required acceptance remains.
- `external_unverified`: actual external access/lifecycle evidence is missing.
- `pending`: not implemented/accepted in the required form, or conditional work
  whose eligibility has not been established.
- `[x]` below checks off **only the named bounded milestone**. `[ ]` remains open.
  No whole group is declared complete and no percentage is invented.

**Overall release: not accepted.** Sufficient income-source coverage still blocks
fresh complete operating-cycle acceptance. No row authorizes trading or promotion.
Old full-audit failures remain visible; later focused test counts are not additive.

## Group-level map

| Group | Scope | Overall progress | Principal remaining boundary |
|---|---|---|---|
| [A](#a-foundations) | Foundations, configuration, MCP, packaging | partial | Complete clean-host/deployed service lifecycle and credentials |
| [B](#b-data-and-source-authority) | Equity/crypto/context/evidence/income data | partial | Real coverage/entitlements; sufficient income authority |
| [C](#c-quantitative-core-and-validation) | Factors/models, replay, WF, metrics | partial | Ensemble consumer admission, actual histories and statistical acceptance |
| [D](#d-regimes-strategies-and-allocation) | HMM, six strategies, selectors, sleeves | partial | Source-complete end-to-end and prospective performance evidence |
| [E](#e-agent-desk) | Analysts, debate, veto, history and budgets | partial | Authenticated quality, calibrated comparisons and operational benefit |
| [F](#f-risk-accounting-and-execution) | Optimizers/risk, ledger, venues, LEAN | partial | Remaining runtime/venue cases, settlement and contingent groups |
| [G](#g-research-and-rd-agent) | Registry, custom evolution, native RD | partial | Actual bounded native research workflows and independent acceptance |
| [H](#h-operating-cycle-and-operator-interface) | Scheduling, monitoring, dashboard, alerts | partial | Fresh complete cycle, unattended operation, delivery/restore exercises |
| [I](#i-system-acceptance-and-release) | Full integration and go-live evidence | not accepted | Current full audit, external acceptance and prospective release gates |
| [J](#j-jev-decision-service-cross-cutting) | Accepted cross-cutting Jev integrations | partial | Measured labeled/provider/prospective evidence; J7 conditional |

## Current next actions (sequence, not replacement scope)

1. **F5.7:** correct/test the known stop-market positional-overload issue against
   the pinned engine, preserving durable claims; then **F5.8** actual DAY/trailing.
2. **F5.9/F2:** nonzero-fee and partial-fill runtime scenarios. The default crypto
   paper model does not produce partial fills; explicitly label any custom model
   as simulation and retain separate real-broker acceptance requirements.
3. **B7**, in parallel when evidence is available: preserve query configuration
   and capture, inspect actual rows, establish coverage/payment/deduction linkage,
   then implement/review sufficient source authority. Do not buy unspecified
   subscriptions or weaken the existing source gate.
4. **G5**, with explicit configured provider/model/embedding access and spend
   authorization: one bounded real research job, then the remaining workflows.
5. **C2/F2/F4:** separately reviewed ensemble consumer integration, factual
   discrepancy settlement and bracket/OCO ownership; these remain implementation
   work, not external credentials alone.
6. **H/I/J:** source-complete end-to-end paper operation, full current audit,
   measured comparisons and the unchanged prospective release requirements.

## A. Foundations

| ID | Status | Implementation / local evidence | Remaining acceptance |
|---|---|---|---|
| A1 environments | partial | Root `uv.lock`, isolated RD environment, actual native model/Qlib imports; Docker/WSL restored; source-built pinned LEAN with reviewed local shutdown patch | Independent clean-host install, complete deployed services and every supported runtime |
| A2 contracts/config | offline_verified | `src/domain/`, `src/utils/config.py`; identity, temporal, precedence and Decimal roundtrip tests | Deployment/secrets lifecycle |
| A3 MCP | offline_verified | `data/evidence/mcp.py`; actual local HTTP SDK, exact manifests, generic-dispatch permission tests | Deployed server/schema pins, credentials, entitlements |
| A4 application/CLI | partial | Canonical backtest/agents/research/operating services, scheduler aliases, retired-interface test migration and immutable replay output | Fresh source-complete success; complete rolling intraday operation |
| A5 packaging/CI | partial | Wheel/sdist and installed CLI checks; actual project Docker build + environment/Parquet + CLI smoke ([evidence](implementation/docker-smoke-2026-11-02.md)) | Clean-host reproducibility, CI and persistent service/upgrade/backup lifecycle; image smoke is not deployment acceptance |

### A5 runtime/packaging milestones

- [x] Project Docker build and network-disabled environment/Parquet/CLI smoke.
- [x] RD isolated image build/import/template/help smoke (G5.1).
- [x] Pinned source-built LEAN plus bounded local shutdown patch; actual paper scenarios (F5).
- [ ] Independent clean-host deployment, service lifecycle, CI, backup/restore and resource acceptance.

## B. Data and source authority

| ID | Status | Implementation / local evidence | Remaining acceptance |
|---|---|---|---|
| B1 evidence | offline_verified | `data/evidence/`; immutable raw/full documents/revisions/spans and cutoff tests | Production retention and entitled history |
| B2 US equity/Qlib | partial | Actual US binary bundle/Alpha158/Ridge; unique full-XNYS daily rows and complete internal session grids validated before publication; 50 root daily/local/native tests | Historical universes, delistings, corporate actions and genuine benchmarks |
| B3 crypto/context | partial | Native funding/OI, dated CoinGecko current/Pro dominance and exact BTC/perpetual contextual bindings; closed refresh preserves native clocks; streamed decoded-response cap | Actual funding/OI/dominance/protocol and non-USD histories, Pro/PIT entitlements and documented OI-history limits |
| B4 vendor/MCP data | partial | FMP/FRED/ALFRED/Yahoo/Finnhub/SEC/CoinGecko/Alpaca parsing and provenance | Credentialed schemas/entitlements, transcripts and document coverage |
| B5 actual prints/ADV | offline_verified | Raw print IDs/ns, aliases, exact native references and VWAP turnover; certified indicative USDT/USD with fixed three-source proof and closed refresh | Authenticated feeds, historical FX and remaining native constraints; indicative FX is not cash conversion |
| B6 source preparation | offline_verified | `data/operating_plan.py`; completed task receipts, dated seven-series macro inputs, immutable training prefix and contextual whitelist | Approved actual source recipes/historical breadth and rolling unattended preparation |
| B7 income authority | partial | Native Alpaca gap parser plus `operations/income.py`; IBKR paper Flex retrieval, read-only intake/canonical inspection and reviewed descriptor-safe local capture/resume; unsupported positive/complete claims rejected | **No sufficient positive income/complete-stream parser admitted**; crypto scope also missing |

### B7 income evidence subgroups — no sufficient positive authority yet

- [x] B7.1: operator-reported paper Flex API retrieval; diagnostic capture reader,
  safe inspection CLI and descriptor-safe capture/resume tests implemented.
- [x] B7.2: query-scope diagnostic preserves original bytes/clocks and explicitly
  retains authority gaps. Selected UI fields are scope observations, not rows.
- [ ] B7.3: independently establish account/query/window coverage and actual
  accrual → cash payment → tax/fee/correction linkage from returned evidence.
- [ ] B7.4: implement and independently verify a sufficient positive income and
  complete-stream source parser; no existing Flex summary grants this authority.
- [ ] B7.5: demonstrate fresh required operating/replay acceptance, including
  required crypto/source scope. Paper evidence cannot certify live entitlement.

Next operator action and exact safe command: [IBKR checklist](implementation/ibkr-next-evidence-checklist.md).
Source assessment: [income sufficiency](implementation/income-source-assessment.md).
No required subscription product has been established.

## C. Quantitative core and validation

| ID | Status | Implementation / local evidence | Remaining acceptance |
|---|---|---|---|
| C1 evaluator roles | offline_verified | Actual host US Qlib research and authoritative chronological ledger replay; explicit bundle CLI; RD image packaging verified separately | Actual bounded RD/Qlib research jobs and entitled historical inputs (G5); LEAN is an execution bridge, not the selected historical evaluator |
| C2 factors/models | partial | Bounded AST engine, four families/train-only transforms; temporal IC selection; closed canonical ensemble parent, child/selection artifacts and two-file proof; 236 root artifact/model/Qlib/WF tests | Reviewed native volatility comparison, dual-file consumer provenance and explicit strategy/operating ensemble admission; actual entitled histories |
| C3 replay/accounting | offline_verified | Shared Decimal ledger, actual events/fills/targets/callback snapshots, costs and sleeves; durable/artifact suites | Broader actual actions/FX/delistings and venue parity |
| C4 walk-forward | offline_verified | Six-month default, actual label-end purge, fresh artifacts and OOS reconstruction; `test_walk_forward` | Full history and all production variants |
| C5 metrics/benchmarks | offline_verified | Nullable canonical metrics, durable costs/named/1,000 random controls independently reconstructed | Real statistical gates, nondegenerate universe, multiplicity and untouched holdouts |
| C6 proof/reporting | offline_verified | Immutable Replay v2, full event/fill/NAV recomputation, source/execution proofs; income-aware inputs, OOS callback filtering and independent source gate | Sufficient positive income authority and full reporting |
| C7 acceptance | partial | Hand arithmetic, future perturbation, corruption, namespace, restart and cost tests | Actual historical/provider/prospective cases; no manufactured performance |

### C2 ensemble integration subgroups

- [x] C2.1: four individual model families and train-only transformations locally exercised.
- [x] C2.2: policy-aware temporal IC selection and closed canonical ensemble
  parent/child/selection artifacts with two-file proof locally verified.
- [ ] C2.3: reviewed implementation of a bounded native-volatility comparator;
  CatBoost creation metadata is retained, not stripped to manufacture equality.
- [ ] C2.4: canonical dual-file WF/consumer provenance and real outcome-end forwarding.
- [ ] C2.5: explicit full-composition StrategySpec/H recipe/factory opt-in and
  actual combined consumer evaluation. Parent artifact existence alone is insufficient.

Evidence/design boundaries: [data/quant note](implementation/data-quant.md).

## D. Regimes strategies and allocation

| ID | Status | Implementation / local evidence | Remaining acceptance |
|---|---|---|---|
| D1 HMM | offline_verified | Eight sourced signals, constituent breadth, persistent causal filter, actual model-byte/source reconstruction | Long genuine membership/macro histories and strategy performance |
| D2 six strategies | offline_verified | Six immutable specs, actual four-model and model-free consensus execution | Fresh source-complete cycle and real statistical eligibility |
| D3 four selectors | offline_verified | Actual attributed selection, exact holdings/netting, per-sleeve signal/data identities | Sufficient prospective performance history |
| D4 attribution | offline_verified | `SleeveBook` and `AttributedPerformanceStore`; actual fill/flow ownership, funded epochs and exact daily chains | Long prospective reconciliation and all venues |

## E. Agent desk

| ID | Status | Implementation / local evidence | Remaining acceptance |
|---|---|---|---|
| E1 full desk | partial | Seven report paths, two challenge rounds, exact citations, HMM/portfolio boundary; actual later-clock ledger and sleeve review snapshot persisted before CRO | Authenticated providers/tools and measured quality |
| E2 actions/veto | offline_verified | Canonical actions, derived consensus, hard veto, HOLD/ABSTAIN ownership and REDUCE/EXIT invariants | Prospective contribution comparisons |
| E3 cost/history | partial | Jev and generative immutable usage authority, independent partial/wrong-model counters and permanent run/original-day fences; sole-reader reporting, native v2 budget gates and certified recursive model-visible history | Authenticated usage/pricing, latency and measured operational benefit |

## F. Risk accounting and execution

| ID | Status | Implementation / local evidence | Remaining acceptance |
|---|---|---|---|
| F1 five optimizers | offline_verified | Actual LW covariance, deterministic HRP, BL views, feasible constrained fallback and exact target conversion | Real histories and all venue fee/lot interactions |
| F2 durable execution | offline_verified | Exclusive owner, causal pre-I/O claims, partial/ambiguous receipt recovery, opening capacity identity; expiry preserves uncertain reservations; base/sleeve publication fences; pure retained-source exporters and uncached current compliance gates in validation, H and native submission | Authenticated lifecycle, explicit post-deadline settlement and unsupported reconciliation |
| F3 Alpaca | external_unverified | Typed bounded REST/activity adapter, actual FILL/FEE parsing, trailing and account/order reconciliation; original native DAY calendar proof bound to direct/LEAN submit claims | Actual paper account, bracket/contingent and full lifecycle |
| F4 crypto venues | partial | Typed Binance/Kraken spot adapters, native gross settlement, actual inventory-fee ledger/sleeves, bounded signing/native predicates; exact native trailing terms and frozen references | Contingent/OCO, expiry/settlement and authenticated/testnet acceptance |
| F5 LEAN | partial | Actual pinned engine, exact CLR money/clocks, clean shutdown, market fill, resting-limit cancellation, completed-only restart readiness, and pending-history refusal verified; detailed checklist below | Remaining order/fee scenarios, active-order recovery and authenticated broker acceptance; synthetic PaperBrokerage is not vendor acceptance |
| F6 income ledger | offline_verified | Sourced terms/receivables, original sleeve owners, actual posted/local-paper distributions, taxes and publication fences | Sufficient source authority; mergers/spinoffs/due-bills/foreign conventions blocked |

### F2/F3/F4 economic and external acceptance subgroups

- [x] F2.1: pure retained native source reexport, metadata-only discrepancy
  detection and read-only current execution-compliance audit locally tested;
  validation, registry ancestry and new-risk submission gates integrated.
- [ ] F2.2: source-backed factual discrepancy settlement. No reachable economic
  exception/expiry waiver is enabled by the read-only audit checkpoint.
- [x] F3.1: Alpaca native-shaped offline signing/activity/DAY/trailing contracts tested.
- [ ] F3.2: authenticated Alpaca paper/native lifecycle and supported live contracts.
- [x] F4.1: Binance/Kraken native-shaped offline signing, exact fee/ledger proofs,
  frozen reference checks and native trailing terms tested.
- [ ] F4.2: external Binance/Testnet and Kraken verification appropriate to each
  venue's available environment; no verified Kraken spot sandbox is assumed.
- [ ] F4.3: canonical bracket/OCO ownership, shared reservations, child revisions,
  sibling races and closed native routes. They require reviewed settlement handling.

Evidence: [accounting/backtesting](implementation/accounting-backtesting.md).
Actual broker integration remains separate from synthetic LEAN paper testing.

### F5 LEAN actual-runtime acceptance matrix

| ID | Milestone | Progress | Evidence / limit |
|---|---|---|---|
| F5.1 | Pinned engine startup, exact money/ticks, shutdown | [x] runtime_verified | [Runtime](implementation/lean-runtime-smoke-2026-11-03.md), [local GIL patch + exit0](implementation/lean-shutdown-fix-2026-11-03.md); not universal upstream certification |
| F5.2 | GTC crypto market order → actual event → ledger | [x] runtime_verified | Synthetic feed/PaperBrokerage, one fill, reconciliation/compliance PASS |
| F5.3 | Resting limit cancellation | [x] runtime_verified | [Actual cancellation twice](implementation/lean-lifecycle-acceptance.md), exact ticket/client, zero fills, reserve released; limit overload fixed |
| F5.4 | Completed-only fresh-engine restart | [x] runtime_verified | [Post-setup restoration fix](implementation/lean-paper-restore-fix-2026-11-03.md), ready health, exact cash/holdings, no duplicate order/fill |
| F5.5 | Unresolved accepted pending order after restart | [x] runtime_verified (refusal only) | [Recovery evidence](implementation/lean-recovery-acceptance.md): initialization refuses, original binding/reservation retained; this is a safety test, not recovered execution |
| F5.6 | Successful active-order recovery | [ ] external_unverified | Original authoritative ticket/history required; fresh PaperBrokerage cannot invent it |
| F5.7 | Stop-market native overload | [ ] pending | Source review found the analogous missing asynchronous argument; no runtime acceptance/fix claimed |
| F5.8 | DAY/calendar and trailing actual engine paths | [ ] partial | Offline contracts exist; actual CLR/runtime cases remain |
| F5.9 | Nonzero fees and partial fills | [ ] partial | Offline accounting exists; no actual fee/partial scenario accepted. Default ImmediateFillModel full fills do not prove partial execution |
| F5.10 | Native brokerage through LEAN | [ ] external_unverified | Exact credential/account/history agreement and broker-specific acceptance remain; no live-capital authorization implied |

Latest bounded checkpoint: `4d898ed`, **145 tests passed in 5.24s** plus three
actual engine starts. Origin and completed restart exit0; accepted-pending
refusal exit1 is expected, not a successful recovery. All cleanup0. This is not
a full-suite run or an all-venues release certificate.

## G. Research and RD-Agent

| ID | Status | Implementation / local evidence | Remaining acceptance |
|---|---|---|---|
| G1 registry/lifecycle | offline_verified | Immutable candidates/trials, independent validation; original replay/advice/paired-account locators and current ancestor compliance for activation/resolve/rollback/status; full mount exposure, opaque-source restrictions and native lineage quarantine | Real statistical acceptance, explicit authority-store relocation/recovery and native internal multiplicity/implementation audit |
| G2 custom factor/model/crypto | offline_verified | Actual fit/artifact/ledger execution and independent reconstruction | Actual data, every variant and multiple-testing/holdout gates |
| G3 strategy/regime/advice | offline_verified | Actual callback economics, same-trial paired HMM; archived advice verifies exact original base ledger and immutable historical full-sleeve observations | Real prospective receipts, complete controls and statistical gates |
| G4 research copilot | offline_verified | Full-document chunk coverage, exact claims, deep memos and durable generation/budget receipts | Authenticated research quality and actual papers |
| G5 native RD | partial | Supervisor/gateway/watchdog, frozen source/mount lineage, immutable create/bind/start/recovery and CLI; actual offline Docker imports, eight upstream hashes, five templates and CLI/factor help passed ([evidence](implementation/rd-runtime-smoke.md)) | All seven actual research workflows, provider/model/embedding access and explicit budget, source-specific document authority, candidate import/evaluation and persistent knowledge acceptance |

### G5 native RD subgroups

- [x] G5.1: pinned offline Docker image imports, source/hash/template checks,
  upstream CLI and factor help; [actual smoke evidence](implementation/rd-runtime-smoke.md).
- [x] G5.2: local supervisor/gateway/exposure/immutable container binding and
  recovery boundaries tested; opaque documents retain unknown exposure restrictions.
- [ ] G5.3: actual `health_check` with the required external runtime/provider configuration.
- [ ] G5.4: actual `fin_factor` job and independently evaluated candidate.
- [ ] G5.5: actual `fin_model` job and independently evaluated candidate.
- [ ] G5.6: actual `fin_quant` workflow.
- [ ] G5.7: actual `fin_factor_report` with authoritative document exposure.
- [ ] G5.8: actual `general_model` paper/document workflow with source lineage.
- [ ] G5.9: actual `ui` workflow, predecessor source linkage and persistent knowledge restart.
- [ ] G5.10: full generated-artifact import/quarantine/evaluation/promotion-to-inference acceptance.

Help commands and rendered templates are not executions of those seven named
workflows. Provider credentials, entitlements and finite spend approval remain
explicit prerequisites; no successful candidate or performance is fabricated.

## H. Operating cycle and operator interface

| ID | Status | Implementation / local evidence | Remaining acceptance |
|---|---|---|---|
| H1 operating cycle | partial | Shared six-strategy models/HMM/desk/optimizer/ledger, actual mixed-calendar recovery verified before income guard | Fresh positive acceptance now blocked by insufficient income source; retained tests fail visibly |
| H2 scheduler/entry points | partial | Same application services, frozen jobs, independent lanes, immutable rolling refresh templates with measured invocation cutoffs and exact-plan recovery; completed-receipt cursor recovery avoids redispatch after interruption | Rolling source-complete build/retrain/cycle composition and full intraday production operation |
| H3 dashboard/alerts | partial | Verified read-only registry snapshots, historical/current eligibility distinction, narrow native recovery state; actual ledger/usage; durable notification outbox | Authenticated delivery, resource monitoring, backup/restore and deployment |

### H operating/deployment acceptance subgroups

- [x] H1.1: offline stage receipts, later-clock review snapshots, uncached
  execution-compliance gates, calendar/expiry/passive-action and safe recovery cases.
- [ ] H1.2: fresh source-complete six-strategy cycle and complete intraday operation.
- [x] H2.1: shared entry points, scheduling/deduplication and persisted cursor recovery locally tested.
- [ ] H2.2: rolling refresh/retrain/research/trading composition across real calendars and outages.
- [x] H3.1: persisted read-only dashboard/alerts/usage paths implemented locally.
- [ ] H3.2: actual Telegram/notification delivery, deployment measurements,
  durable volumes, upgrade/rollback, backup and restoration exercises.

## I. System acceptance and release

| ID | Status | Implementation / local evidence | Remaining acceptance |
|---|---|---|---|
| I1 full integration | partial | Latest **full** pinned audit remains `7520957`: 2,434 passed, 3 failed, 18 setup errors, 1 skipped; failures/errors retain fresh income-source dependency. Later bounded checkpoints include 145 LEAN tests at `4d898ed` | Rerun whole-repository integration at current source after resolving required dependencies; do not substitute focused passes for full-suite acceptance |
| I2 external release | external_unverified | Project/RD container smoke and selected actual synthetic LEAN scenarios now verified; explicit blocked/unavailable states retained | Clean-host deployed lifecycle, credentialed source/venue/native research runs, alert delivery, operational measurements and labeled agent calibration |
| I3 prospective release | pending | Thresholds retained in full plan | Three-month paper/required run count, quant-only/veto/full-agent comparison and release decision |

### I release checklist — unchanged acceptance thresholds

- [ ] I1.1: current-source full integration audit, with every failure/error resolved by sufficient evidence.
- [ ] I1.2: real-data equity/crypto cases for all required strategy/model/optimizer/selection/venue variants.
- [ ] I1.3: complete research → independent evaluation → promotion → pinned inference → rollback lineage.
- [ ] I2.1: credentialed data/agent/broker/RD acceptance and operational alert/resource/spend evidence.
- [ ] I3.1: walk-forward Sharpe >1 after specified costs; zero validation failures;
  above the 95th percentile of matched random controls; positive Sharpe in at
  least two regimes; t-statistic >2 over at least 252 trading observations, with trial controls.
- [ ] I3.2: three months of paper behavior against frozen expectations, paper max
  drawdown <15%, 30 consecutive daily runs without unrecoverable failure, and
  sound manual reasoning review across at least 20 names.
- [ ] I3.3: matched prospective quant-only / quant-plus-veto / full-agent comparisons,
  verified alerts and documented spend within budget; explicit release decision.

Software capability acceptance and live-capital eligibility are distinct. Neither
is established by the latest synthetic LEAN scenarios.

## J. Jev decision service (cross-cutting)

J is the accepted extension linked from the full plan, not a replacement for E,
H or the authoritative HMM in D. Source: [integration/activation decision](JEV_INTEGRATION_DECISION.md).

| ID | Subgroup | Progress established | Remaining acceptance |
|---|---|---|---|
| J0 | Shared service/evaluation/budget authority | partial: typed immutable observations, evaluated-policy proof, bounded experiment CLI, source-backed calibration/pairs, durable run/day overrun fences and certified memory/prompt isolation locally tested | Authenticated model/SDK/pricing, incumbent baseline and measured efficacy/latency/cost; no approved production cohort shipped |
| J1 | News classification | partial: shared source/question/model-bound workflow | Labeled actual-source/provider accuracy and replacement benefit |
| J2 | Transcript classification | partial: shared typed transcript workflow and evidence boundaries | Actual transcript entitlement/coverage and measured labeled quality |
| J3 | Claim/source verification | partial: deterministic reference/quote checks separated from eligible semantic judgments | Adversarial/provider efficacy; semantic outputs cannot clear a hard veto |
| J4 | Retrieval/history relevance | partial: certified history, exact delivered spans, omissions/conflicts tracked | Measured relevance/recall benefit against incumbent retrieval; no completeness from omitted documents |
| J5 | Text features | partial: source/question/model/transformation and generation-time availability contracts | Actual chronological feature evaluation, selection controls and prospective benefit |
| J6 | Debate/risk support | partial: source-bound semantic projection, shadow isolation and veto-preserving integration | Matched debate/risk quality, calls saved and prospective contribution |
| J7 | Auxiliary macro labels | pending/conditional | Demonstrated incremental value; HMM remains authoritative and cannot be silently replaced |

Evidence: [agents/Jev/H implementation](implementation/jev-agents.md). Synthetic
protocol cohorts and offline budget/corruption tests do not establish financial
quality or authenticated efficacy.

## Historical audit and evidence record

The counts below retain their original pins and dates. They are not current
full-suite results. Use the group/subgroup tables above for current status.

Evidence: [accounting/backtesting](implementation/accounting-backtesting.md),
[data/quant](implementation/data-quant.md), [agents/Jev/H](implementation/jev-agents.md),
[root operations](implementation/operations.md). Counts identify their exact pinned
source states; do not sum them into a whole-repository green claim.

The [income source assessment](implementation/income-source-assessment.md) records
the exact DTCC SR2025 schemas reviewed and the still-missing account, deduction,
delivery and complete-window authorities. No positive parser is admitted from
generic ISO fields or those insufficient component formats.

Full pinned audit at `fd54c9f`: 1,886 tests, **1,851 passed, 17 failed, 18 setup
errors**, zero skips, in 1,139.76s. Fourteen failures were retired CLI/integration
contracts; their replacements and related replay tests subsequently passed
62 cases in 21.91s. The other three failures and 18 setup errors require fresh
positive income-complete cycles. They remain enabled and blocked. The operator
confirmed that no sufficient source has been selected; there is no positive
fixture bypass.

Subsequent integration evidence: 378 tests passed in 48.36s for current desk,
certified memory/evaluated policies, LEAN/sleeves and B3 producers; 19 original
portfolio/advice tests passed in 36.93s after G sleeve integration; 87 refresh and
credential tests passed in 10.83s; 159 transport/source/refresh tests passed in
43.01s after the streamed HTTP cap. These overlapping checks are not a full-suite
green claim. See the linked implementation notes for exact pins and commands.

Previous complete audit: immutable archive `a0ec996acd360d7caad2edd0be56b35e53fa1332`,
2,103 tests in 1,107.76s: **2,081 passed, 3 failed, 18 setup errors, 1 skipped**.
Every failure/error is the retained fresh positive income-policy/source dependency;
none was skipped or replaced with a fabricated positive authority. The one skip
was the separately installed RD-Agent environment, absent from `git archive`.
That exact pinned-template test subsequently passed in the main checkout in
2.03s using the installed isolated environment; this is template rendering, not
Docker/native job acceptance. Stop/remote/ledger/artifact integration separately
passed 248 tests in 33.57s. Original full-audit counts remain unchanged.

Latest complete audit: immutable archive `7520957eb738e16ea8884990609202d4db16293e`,
2,456 tests in 1,255.70s: **2,434 passed, 3 failed, 18 setup errors, 1 skipped**.
Every failure/error was inspected in the JUnit report and has the same explicit
fresh income-policy/source requirement as the previous audit. The three failed
tests require a complete six-sleeve cycle, source preparation, and canonical
application/CLI completed-cycle replay; the 18 setup errors depend on that fresh
cycle. No new runtime failure category appeared. The separate native environment
is absent from this immutable archive, accounting for the single template-test
skip; at that audit pin, Docker/LEAN acceptance was unavailable. Subsequent bounded
runtime evidence is recorded in A/F/G above; the historical full-audit counts
have not been rerun or upgraded. Two warnings record
deliberately blocked socket probes. No positive source fixture was substituted.

This audit predates scheduler cursor recovery `aa48c8c` (49 focused tests passed
in 14.80s) and fill publication fencing `c92e037` (87 focused tests passed in
23.93s). Production/script Ruff checks pass; the wider test-tree scan records
48 import-order/unused/ambiguous-name lint findings and is not claimed clean.
Logs and exact invocation are recorded in the root operations evidence note.
