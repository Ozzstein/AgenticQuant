# Jev and agent desk implementation evidence

Checkpoint date: 2026-10-27. Branch: `build/jev-agents`. This is an integration checkpoint in the complete [build plan](../FULL_SYSTEM_BUILD_PLAN.md), not full-system or live-capital acceptance. Contract commit: `4cac664`; service/desk implementation commit: `8f2d2de`. No other lane was merged, no dependency environment was modified, and no PR/push or real-capital operation was performed.

## Implemented contracts and entry points

- `src.domain.decisions`: immutable `DecisionQuestion`, `DecisionRequest`, `DecisionResult`, exact `DecisionSource` spans, threshold/version policy, typed specialist reports and `AgentDecision`. Source identity includes `EvidenceRecord`, raw/document/passage hashes, exact character offsets, documented availability, run/as-of/config/code identity. Source documents remain owned by the data lane.
- `src.decision.service.DecisionService(store).evaluate(request, timeout_seconds=30, mode="cache"|"refresh"|"replay", observation_id=None)` and async `aevaluate`. Synchronous entry rejects use inside an existing event loop. `DecisionStore(path)` persists requests, immutable observations, immutable HTTP attempts and durable operating reservations.
- `python -m src.decision request.json --store outputs/decisions.sqlite [--mode replay --observation-id ID]`. Input is the complete `DecisionRequest` JSON; credentials come privately from `TYPESAFE_API_KEY`, never command arguments. Exit 0 means a valid typed observation, which may still contain abstaining answers or be shadow-only. Missing credentials, replay miss, malformed inputs and provider failures exit 2. This is a judgment CLI, not an order or promotion command.
- `src.decision.evidence.source_from_passage` validates a data-lane `Passage` against the actual immutable `EvidenceStore.document` before use. `load_desk_sources(EvidenceService, instrument, context)` consumes the five real role bundles, deduplicates exact spans and returns both sources and coverage. The data lane retains ingestion, retrieval, security-master and entitlement authority.
- `src.agents.graph.analyze_instrument(instrument, context, *, sources, hmm_context=None, portfolio_context=None, qlib_context="", config=None, decision_service=None, policy=None, llm_factory=None, observation_mode="cache") -> AgentDecision`. `build_trading_desk_graph` exposes the same graph for canonical orchestration. The `observation_mode` controls Jev calls, not generative chat replay. Complete returned decisions are persisted to canonical desk history when memory is enabled.
- `analyze_ticker` is an explicit migrated export bridge: it requires the canonical instrument, context and sources and shared `AnalysisResult.agent_decision`. It rejects old ticker-only calls rather than inventing identity, current data or a successful HOLD result.
- `src.agents.contracts.hmm_context_from_result` consumes the data lane's `CausalRegimeResult`: available status, effective-state probability, model hash and explicit as-of. Insufficient data returns `None`; the graph blocks affected exposure. `PortfolioReviewContext` accepts read-only portfolio/correlation JSON, snapshot/as-of identity and externally computed deterministic vetoes. It does not duplicate accounting arithmetic.
- `src.agents.consensus.consensus_recommendation(decision, minimum_agree=4)` returns `ConsensusResult(action, agreeing_roles, derived_votes, reason)`. Five named roles are fundamental, sentiment, technical, risk manager and strategist. All must be available; at least four must agree. Strategist derivation is explicit. Missing roles, blocked BUY or fewer votes abstain; veto remains binding. The strategy owner supplies the eligible universe.

## Provider contract and operating behavior

Verified the official [documentation index](https://docs.typesafe.ai/llms.txt), [System One OpenAPI](https://api.typesafe.ai/openapi.json), [models](https://docs.typesafe.ai/models), [confidence definitions](https://docs.typesafe.ai/confidence) and [Python SDK changelog](https://docs.typesafe.ai/sdk/python/changelog) on 2026-10-27. Direct official HTTP transport uses `POST https://api.typesafe.ai/v1/systemone`, OpenAPI version `0.2.0`, pinned model `jev-1.14.0`, and installed/coordination-pinned `httpx==0.28.1`. Official SDK version was `0.7.5`; this implementation uses the official API directly and has no SDK dependency. Retrieved OpenAPI SHA-256: `3ca93b6e6b9aa8d7820531b78c66ed7a9798b11e6e163ad96b140fb3d75fbcf4`.

Choice uses a criteria mapping and returns a selected maximum-probability option, full distribution and provider confidence. Score uses an ordered zero-based rubric; the response legend must match and its score must equal the distribution's expected value. Noul returns a yes-probability without a confidence field. Envelope, answer IDs/types, finite bounds, distributions, score expectations and actual model identity are validated. Aliases are rejected. Wrong returned versions are explicit failures even in shadow mode.

Cache identity includes exact question/rubric content and order, model/policy, source revisions/availability, canonical state and historical context. The run ID is lineage rather than a semantic cache discriminator. Cache mode reuses the first compatible successful observation; refresh appends a new observation without rewriting it. Recorded replay selects a specified observation, or the first compatible observation including failures, and never calls Jev. Reuse references the original observation rather than charging its tokens again. Replay proves record reproducibility, not historical absence of pretrained knowledge.

Requests have bounded context bytes, attempts, rate spacing, per-attempt and caller deadlines, cancellation, and durable run/day request/token/spend reservations. Only transient statuses/transport failures retry within the deadline. Unknown prices block admission when a spend cap is enabled. Default Jev pricing records the reviewed model/date and USD 0.042 per million input tokens; output is free under that reviewed schedule. Reservations are conservative and remain charged after uncertain failures; observations distinguish actual reported usage/cost from unknowns. SQLite admission/audit I/O is synchronous; the deadline bounds provider work and retry admission, not arbitrary operating-system stalls.

Generative calls have separate durable reservations under `AgentConfig.budget_store_path`, a per-instrument analysis key inside the run, day limits, explicit model pricing/version/effective date, and actual `include_raw` token metadata. The shared analysis deadline reaches provider SDK timeouts; hidden SDK retries are disabled. Unknown pricing blocks calls. Unknown usage retains its reservation and is labeled `usage_unknown`, never measured zero. The obsolete factory cost callback was removed because its unknown-price/usage fallback records a false zero; canonical operating consumers should use `AgentCallBudget` records. No generative rates were invented in this lane. All pricing/config fields agreed with the coordinator are consumed; `analysis_timeout_seconds` defaults to 180 until the shared configuration is present.

`ConfidenceSemantics` separates subjective analyst conviction, evidence completeness, calibrated task accuracy and preserved legacy confidence. Jev probabilities/confidence stay on `Judgment`; no multiplication by 100 occurs. Legacy exports preserve an original legacy number when present; newly generated records carry a zero legacy placeholder labeled unvalidated and the real named quantities in `agent_decision`. Consumers must use the canonical fields, not interpret that placeholder as a probability or sizing instruction.

## Connected requirements

| Requirement | Executable behavior | Evidence / remaining acceptance |
|---|---|---|
| A / J typed infrastructure | Official HTTP transport, immutable contracts, source/time checks, cache/replay, policy/model pin, durable budgets, explicit unavailable/timed-out/malformed/stale/insufficient/version-mismatch/cancelled states | Deterministic tests pass; authenticated calls are unverified |
| J1 | Source-linked news tone, relevance and event classification in the graph's evidence node | Shadow observations; labeled accuracy and replacement acceptance remain |
| J2 | Management tone, explicit guidance direction, hedging and Q&A responsiveness; transcript synthesis registered in the graph | Full-document spans, exact quotes; actual transcript entitlement and provider evaluation remain |
| J3 | Exact source/hash/quote checks before semantic support/contradiction/insufficiency judgment; analyst/debate claims rechecked | Bad references and quotes never reach semantic provider; source support is not source truth |
| J4 | Passage relevance/conflict evaluation before specialists; material contrary, unknown and unavailable retrieved candidates are retained. Eligible prior history is assessed as untrusted context | Actual retrieval coverage and semantic reranking quality remain to measure |
| J5 | `compute_text_feature(workflow, sources, TextFeatureDefinition)` supports proposed Score/Noul questions with a separate availability Choice, normalized numeric transformation and complete lineage; `demand_weakness_feature` is a ready rubric | Missing features remain `None`; no training, validation or promotion bypass. Real six-month research comparison remains |
| J6 | Semantic discrepancy check around revised reports, two configurable full challenge rounds, portfolio/correlation-informed generative CRO review | Held-out conflict/false-block evaluation and authenticated complete runs remain |
| E | Fundamental, sentiment, technical, risk and strategist, plus earnings/transcript; explicit join barrier; revised full reasoning/citations; configured providers; canonical history; complete final reports and advisory targets/stops/size/horizon | Real-data equity/crypto runs and manual reasoning review remain |
| E / F boundary | HOLD retains; REDUCE and EXIT are explicit suggestions; failed/incomplete analysis abstains; veto cannot become approval or liquidation; missing technical evidence cannot be repaired with news during debate | Deterministic invariant tests; canonical portfolio mapping remains execution/coordinator ownership |
| J7 / D | HMM-only authoritative regime bridge; no Jev macro replacement activated | Conditional auxiliary macro extension still requires incremental-value evidence |
| H / I | Persisted judgments, sources, policies/models, errors, latency, usage and budget state exposed to callers; evaluation API for per-class precision/recall, abstention, Brier/log loss, reliability bins, material misses, false blocks and option variants | Dashboard/canonical operating integration, real measurements and prospective attribution remain |

Default Jev policy is **shadow**. Graph observations are persisted but only `accepted` judgments from an explicitly evaluated, task-specific policy may influence generative prompts or semantic gates. This does not claim any task accuracy or trading edge. Generative synthesis is never fabricated from classification labels. Exact citation/availability checks and deterministic vetoes apply regardless of Jev activation.

## Executed verification

All commands used this worktree on `PYTHONPATH`, the read-only `/tmp/quantagentlab-review-venv`, isolated `/tmp/quantagentlab-jev-*` outputs and no external sockets except read-only documentation retrieval. Synthetic examples exist only in tests. The async tests allow Unix-domain sockets for the event loop while denying network sockets.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" \
  /tmp/quantagentlab-review-venv/bin/python -m pytest \
  tests/test_agents_foundation.py tests/test_agents_memory.py \
  tests/test_agents_analysts.py tests/test_agents_graph.py \
  tests/test_agents_synthesis.py tests/test_transcript_nlp.py \
  tests/test_agent_desk_invariants.py tests/test_decision_contracts.py \
  tests/test_decision_service.py tests/test_decision_workflows.py \
  -q --disable-socket --allow-unix-socket \
  --basetemp=/tmp/quantagentlab-jev-final-tests
```

Result: **174 passed** (10.12 s). Tests cover Choice/Score/Noul, malformed/NaN/infinite/distribution/model failures, missing/future/stale/wrong-instrument evidence, cache invalidation for state/question/order/model/policy/as-of/revision, replay/refresh, retries/rate deadlines/cancellation, unknown price and restart budgets, untrusted-source endpoint isolation, exact quote-before-semantic checks, retained conflict evidence, all specialist paths, two rounds, explicit EXIT/HOLD/REDUCE/ABSTAIN, binding veto, failed-role no-liquidation, missing-role source coverage, canonical history availability, research missing-value semantics, CLI failures/replay and evaluation proper scores.

Existing agent regression tests were migrated to explicit source/context contracts. Former assertions for LLM replacing the HMM, invented neutral/40-confidence fallback, ticker-only success, veto-as-SELL and skipping debate based on incomparable confidence were replaced with assertions for the agreed behavior. Foundation/provider-routing, memory storage/format/concurrency and full-report checks remain. This is not a full repository suite result.

Additional contract checks, performed without merging or modifying another checkout:

- **2 passed** in `tests/test_decision_data_contract.py` against actual data-lane `EvidenceStore`, `EvidenceService` and market modules extracted from committed `e822504` into isolated test scratch and loaded explicitly with `importlib`. Exact source is still this worktree for all agent/decision/common code. The tests also run normally after the coordinator integrates that data commit; in an isolated lane without those modules they report an explicit dependency skip.
- **46 passed** (`test_agent_desk_invariants.py`, `test_agents_graph.py`) with the coordinator's committed `c791850` configuration/schemas loaded from isolated scratch. This verifies the actual `AnalysisResult` 2.0 bridge and agreed pricing fields without editing owned shared files.
- Ruff over all changed Python: **passed**. `git diff --check`: **passed**. `python -m src.decision --help`: **exit 0**. CLI missing-key/replay/invalid-request paths are exercised in the test command above.

Authenticated validation is separate: `TYPESAFE_API_KEY configured=False` in the execution environment. No authenticated Jev model call, live financial-provider entitlement, generative-model analysis or measured provider cost/latency acceptance is claimed.

## Coordinator integration and remaining complete-plan work

The coordinator already integrated `4cac664` and added shared `AnalysisResult` schema version `2.0`, optional `agent_decision`, and literal `legacy_unvalidated` confidence annotations; the numeric legacy fields were not rescaled. It also owns the resolved `httpx==0.28.1` lock and declared pricing/budget configuration. No new runtime dependency or shared-file mutation is required by this implementation commit.

Canonical CLI/pipeline/scheduler should resolve `Instrument` and `RunContext`, fetch a data `EvidenceService` bundle once, pass the authoritative HMM bridge and an accounting-owned `PortfolioReviewContext`, call `analyze_instrument`, then consume `action`, `new_exposure_blocked` and `risk_veto`. HOLD/ABSTAIN/VETO must never become FLAT/liquidation targets. Optimizer and deterministic execution gates remain authoritative. Update mixed-owner `tests/test_integration.py::test_run_agents_analyze`, which still mocks the obsolete ticker-only interface. Connect the returned records and named confidence quantities to the dashboard and audit/operating service. Research can call `compute_text_feature`; its actual splits, trials, fits and promotion gates belong to the research/data lanes.

Remaining full-plan acceptance: authenticated Jev and chat calls; real point-in-time news/transcript coverage and entitlements; independently labeled J1–J4/J6 financial evaluation with frozen thresholds, baseline comparisons, injection/ordering/ambiguity cohorts and measured reliability/latency/cost; actual J5 six-month walk-forward and holdout/trial validation; prospective matched quant-only/quant-plus-veto/full-agent and with/without-Jev attribution; manual reasoning review, operational recovery and observation gates. Recorded Jev replay is complete; fresh historical generative calls retain pretrained-knowledge caveats, and complete recorded desk decisions must be replayed from their persisted artifacts. J7 remains conditional. None of these external/statistical/operational requirements is satisfied by fixture accuracy or by the existence of this code.

## Native research gateway follow-on (G / H)

`src.decision.research_gateway` implements the requested separate provider gateway for the native RD-Agent supervisor. The gateway owns the provider credential; generated Python receives only a run-scoped bearer. The coordinator owns Docker/internal-network isolation, process lifecycle, secure credential injection, native LiteLLM settings/log suppression, and the pre/post-job receipt checks. Gateway code cannot establish those container boundaries by itself.

Runtime entry points:

```bash
python -m src.decision.research_gateway --config /run/gateway.json --check-config
python -m src.decision.research_gateway --config /run/gateway.json --host 0.0.0.0 --port 8787
```

The supervisor supplies `RESEARCH_GATEWAY_BEARER` and `OPENAI_API_KEY` privately to the gateway process. Only the scoped bearer goes to RD-Agent; it must never equal the upstream key. No credential CLI argument or configuration field exists. The bearer must be a supervisor-generated random value of at least 32 ASCII characters. The gateway stores only its SHA-256 digest, binds it to one immutable run/config identity, compares digests in constant time, and disallows reuse across runs in the persistent store. Restarting a run cannot change its caps, configured models/prices, expiry or bearer identity. Revocation is handled by stopping the gateway; expired runs reject new provider requests while receipts remain readable by the run bearer.

`GatewayConfig` is an immutable closed schema with required `run_id`, UTC `expires_at`, `budget_store_path`, `chat`, `embedding`, `max_requests`, `max_tokens`, `max_cost_usd`, `max_requests_per_day`, `max_tokens_per_day`, and `max_cost_per_day_usd`. Each `GatewayModel` requires exact `model`, `price_version`, public HTTPS `source_uri`, UTC `effective_at`, `input_usd_per_million`, and `output_usd_per_million`; `provider` is fixed to `openai`. Rates have no defaults. Embedding output-token pricing must explicitly be zero. Configurable bounds default to request timeout 120 seconds, request bytes 1,000,000, response bytes 16,000,000, input token reservation 100,000, output tokens 8,192, and four concurrent HTTP handlers. `config.content_hash()` is the normalized identity the supervisor verifies. Config validation prints only that hash and the run ID.

The authenticated HTTP API is deliberately bounded:

- `POST /v1/chat/completions` accepts text messages, configured model ID, bounded sampling/stop/seed/reasoning options and JSON response formats. It normalizes `max_tokens` to `max_completion_tokens`; it sends `stream=false`, `n=1`, `store=false` and standard `service_tier=default`. Streaming, tool/multimodal calls, arbitrary provider URLs, alternative models and unknown parameters fail before admission.
- `POST /v1/embeddings` accepts nonempty text or token-array batches with bounded dimensions and float/base64 output. Returned dimensions, indices and finite vector values are validated before returning features. No embedding substitute is generated on failure.
- `GET /v1/budget` is read-only and requires the same run bearer. It returns run/config hash, expiry, readiness, configured model/pricing provenance, run/day caps and remaining amounts, charged totals, actual reported token totals, uncertain reservations, observed model IDs and status counts. It excludes keys, bearer digests, prompts, generated content and filesystem paths. `ready` means provider presence and eligible configuration/minimum admission capacity; `authenticated_provider_validation` remains explicitly `unverified`. The supervisor must verify run/hash/model/caps and its required request headroom, then check the terminal receipt. This endpoint never tests credentials by making a paid call.

The fixed upstream routes are `https://api.openai.com/v1/chat/completions` and `https://api.openai.com/v1/embeddings`; redirects and environment proxies are disabled. Request bodies cannot override destinations or credentials. Exact configured provider model IDs must match returned IDs. This detects mismatches, but an embedding provider's unversioned public ID cannot prove unchanged underlying weights; no such claim is made. Native LiteLLM may use `openai/<model-id>` internally for provider selection; the HTTP body must contain the configured ID without that routing prefix.

Both endpoints use the existing `AgentCallBudget` SQLite admission transaction. UTF-8 payload bytes plus schema/envelope allowance bound input reservations, with the requested output cap reserved separately. Per-run and shared UTC-day request/token/spend caps are checked atomically before the HTTP call. No provider retry occurs inside the gateway; any supervisor/LiteLLM retry is a new admitted, charged attempt. A total asynchronous deadline bounds the entire upstream response, including slow streaming of the nonstreaming JSON body. The HTTP server bounds body sizes, thread count and socket read time; ambiguous framing, duplicate authorization/content lengths and transfer encoding are rejected.

Valid provider usage settles the reservation using the configured versioned rates. Receipt `cost_semantics=reported_tokens_at_configured_rates` describes usage valuation, not an invoice measurement; cached-token discounts are conservatively ignored. Unknown/malformed usage, transport failure, deadline expiry, redirects and process termination retain the original charge and remain distinct from measured zero. Missing usage returns `502 usage_unknown`; deadline expiry returns 504; budget admission failure returns 429. Model mismatch and output/total reservation overruns block further run admission. Actual overrun charges commit **before** the error is raised. Settlement is single-use and scoped to its run; a later failure cannot overwrite a settled call. Immutable gateway audit records retain endpoint/request/response hashes and statuses without storing prompt/output text.

Protocol verification used installed SDK type definitions and the official [chat completion API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) and [embedding API](https://developers.openai.com/api/reference/resources/embeddings/methods/create), checked on 2026-10-27. These establish wire and usage shapes; no model price, credential access or live-model behavior was inferred from them. There is no new dependency beyond the coordinator's pinned httpx/Pydantic environment and Python standard library.

Verification: **232 passed in 17.26 s** for the preceding 174-test command plus `tests/test_research_gateway.py` (58 new cases), using `--disable-socket --allow-unix-socket --basetemp=/tmp/quantagentlab-jev-followup-final-tests`. Provider transports use synthetic fixtures only in tests. Real HTTP parsing is exercised over Unix socket pairs; CLI validation and missing-bearer startup are exercised as subprocesses. Tests include atomic concurrent caps, cross-run day limits, immutable restart/cap/bearer identity, real route/header/body construction, source injection and arbitrary endpoint rejection, expiry, pricing/token/spend admission, outage/redirect/deadline paths, unknown usage, actual overrun persistence, model mismatch, malformed/NaN/overflow JSON, embeddings, private receipts, process-termination reservations, HTTP framing and CLI failures. Ruff and diff checks pass.

Authenticated gateway calls and end-to-end Docker/native RD-Agent jobs remain unverified. Network isolation, gateway image/provisioning, secure secret injection, RD-Agent retries/logging, full actual research evaluation and operational acceptance remain coordinator-owned requirements, not fixture-proven outcomes.

## Final canonical boundary correction

Consensus now treats the canonical desk action as a binding boundary: it cannot revive ABSTAIN or convert a final HOLD into a different actionable recommendation from earlier reports. Four of the five named roles must still agree, the votes must agree with the final action, duplicate role reports abstain, and strategist derivation must explicitly include its four required inputs. A risk-manager VETO remains binding even if an inconsistent aggregate omitted the flag. The return schema is unchanged.

A final history-save failure preserves the completed reports, debate, judgment lineage and qualitative veto. It returns VETO when already vetoed and otherwise ABSTAIN, always blocks new exposure, and exposes `decision_persistence_unavailable`. A legacy export, if present, is rebuilt from that gated result; it cannot retain the earlier recommendation. No private exception text is returned.

Verification: **63 passed in 8.00 s** (`tests/test_decision_workflows.py`, `tests/test_agent_desk_invariants.py`, `tests/test_agents_graph.py`) with the same read-only environment/PYTHONPATH and `--disable-socket --allow-unix-socket --basetemp=/tmp/quantagentlab-jev-boundary-tests`. Nine additional cases cover canonical abstention/HOLD against BUY/REDUCE/EXIT votes, duplicate/unattributed votes, explicit valid HOLD, risk-manager veto and persistence failure with/without a qualitative veto.

The actual coordinator `c791850` config/schema contract was also reloaded from the previously extracted read-only scratch modules: **48 passed in 7.64 s** for `test_agent_desk_invariants.py` and `test_agents_graph.py`, including the persistence-failure legacy export path. Ruff and `git diff --check` pass.

### Prospective consensus history proof (owned follow-on)

`src.agents.history.DeskHistoryStore(Path(config.output_dir) / "desk-history.sqlite3")`
now records every completed graph even when optional display memory is disabled.
`export(decision_hash, *, as_of, evidence_store, decision_store, budget_store_path)`
returns immutable `DeskHistoryReceipt`; `receipt.decision` is the existing
`AgentDecision` for `consensus_recommendation`. Export only accepts a decision
identity and reads the authoritative stores. A caller-supplied receipt-shaped
blob is never an admission input.

The receipt includes measured session start/completion/publication, exact source
records/document hashes/Unicode spans, complete specialist generation requests
and reports, pinned/actual models, immutable budget admissions and settlements
(with verified pricing, caps and run/day usage at admission), all Jev requests,
results, reuse dependencies, attempts and linked conservative reservations, and
HMM/portfolio/memory/quantitative input context. Export rechecks every source
against EvidenceStore, every call against the mutable accounting authority AND
its immutable settlement, and every Jev record against DecisionStore. It checks
report-generation linkage and exact quotes before returning the proof.

This is prospective evidence: graph completion/publication use measured UTC
clock time. Interpreting old text today cannot qualify at an earlier historical
signal. Old `DeskMemoryStore` rows have no such proof and are rejected; no
retroactive backfill or supplied publication timestamp is supported. Receipts
provide evidence for the existing consensus logic, never order authority.
`AgentCallBudget` now appends immutable admission/settlement records. Jev
reservations now link to observation ID and attempt. Missing old linkage is
explicitly ineligible for certification. Unknown usage and interrupted calls
retain their conservative charges. Report generation requires exact returned
model identity and known settled usage.

Offline verification: 110 existing service/desk/gateway tests passed in 17.87s;
9 new authoritative history tests passed in 6.01s, including complete 17-call
role/debate linkage, as-of exclusion, old-memory rejection, corrupted/missing
source/model/budget/Jev authorities and immutable triggers. History source tests
loaded the actual data-lane EvidenceStore read-only while every owned module
resolved from this worktree. Authenticated provider validation remains unverified.

### Grounded research copilot, full papers and development memos (G / H)

`src.research.copilot.ResearchCopilot` is the shared implementation behind the
rewritten `src.core.factor_proposer.FactorProposer` and
`src.core.research_analyst.ResearchAnalyst`. Constructor:

```python
ResearchCopilot(config, *, registry, evidence_store, code_hash,
                environment_hash, llm_factory=None, limits=None)
```

`code_hash` and `environment_hash` must identify already archived execution
artifacts. The generator never guesses a runtime identity from a source file or
active environment. Provider model IDs, verified pricing and run/day caps come
from the agreed AgentConfig. Unknown prices/credentials block admission. One
synchronous operation has a bounded end-to-end deadline, actual async provider
calls, SDK retries disabled, and durable AgentCallBudget reservations. Restart
cannot reset request/token/spend totals or change model/pricing/configuration
under the same run ID. Model mismatch retains the conservative charge and blocks
further admission for that run; it cannot be priced at the requested model's
rate. Uncertain usage is never recorded as zero.

Public operations:

- `propose(kind, description, *, context, sources=(), n=1, tested=(),
  parent_ids=(), allowed_fields=('open','high','low','close','volume'),
  selected_model='Linear', prior_memo_hash=None) -> ProposalOutcome`.
- `propose_from_papers(evidence_ids, *, context, kind, description, n=1,
  tested=(), parent_ids=(), allowed_fields=..., selected_model='Linear')`.
- `write_memo(*, context, sources=(), development_trial_ids=(),
  prior_memo_hash=None) -> MemoOutcome`.
- `load_memo(memo_hash, *, context) -> MemoDocument`, requiring actual recorded
  memo generation, settled budget identity, archived provider response, exact
  source quotations and publication before the supplied cutoff.

`kind` supports factor, model, strategy, regime and semantic_question. Factor and
model payload bytes are the exact canonical `{features, model_name,
model_parameters}` consumed by root ExperimentPlan, with FeatureSpec JSON,
selected concrete family and bounded scalar parameters. Ensemble has an empty
parameter object and uses the pinned four-family recipe. The data-owned bounded
AST validator rejects arbitrary Python, unknown inputs, future shifts, centered
windows and oversized expressions/windows without synthetic market inputs.
Strategy payload is the actual complete StrategySpec with research_candidate
origin. Regime payload is the actual data-owned HMMVariantSpec; it cannot replace
the HMM with text labels. Semantic payload is a versioned TextFeatureDefinition
with separate availability Choice and Score/Noul question. No payload contains
self-reported validation, performance, promotion or executable Python.

Sources are exact DecisionSource records checked against EvidenceStore's full
documents and raw blobs. Full paper processing reads every bounded chunk before
synthesis, archives all reads, retains source offsets, remaps exact quote spans
for synthesis, and reports paper_chunks_total/read. It does not silently truncate
papers. A missing/failed chunk prevents any candidate from being quarantined.
Paper ingestion/extraction and provider retrieval remain data-owned; this API
accepts immutable document IDs already ingested there.

Every request, complete prompt/schema, parsed response, actual provider usage,
pricing and GenerationObservation is archived as immutable registry blobs. A
ProposalReceipt identifies candidate/hash, exact payload/hash, rationale,
citations and provenance hash. ProposalOutcome explicitly reports available,
abstained, unavailable or partial quarantine storage. Successful candidates are
quarantined only; no trial registration, evaluation, holdout reading, activation
or execution method is called. There are no random/template/keyword candidates
or static success memos on errors.

Memo evidence may include exact sources and explicitly named recorded
**development** trial IDs. Registry.development_evaluation is the only outcome
reader. Unknown, future, synthetic or held-out outcomes are refused before a
provider call; recorded failed development trials remain usable as negative
findings. Memo hypotheses are labeled unvalidated. Memo availability is actual
generation time, so reuse must use a later eligible context/run; old-text
interpretation is never backdated. Arbitrarily archived memo-shaped JSON does
not establish generation provenance.

Core wrapper migration is intentional: old methods returning lists of legacy
FactorCandidate/Strategy/model dictionaries, string memos, mutable knowledge-base
text, keyword paper guesses or retry wrappers are gone. Root owns migration of
`tests/test_rd_agent.py` and canonical research CLI/orchestrator callers. New
wrappers require explicit context, source documents, registry and pinned execution
identities; they return the shared typed outcomes above. No shared config/schema
or dependency edit is needed.

Test integration uses actual root registry cbbfe14 and shared config/schemas,
data expression contract 553114f and actual EvidenceStore, StrategySpec and market
contracts loaded read-only. Regime tests load the actual
HMMVariantSpec from data checkpoint d3724f1; that exact module must be integrated
before regime proposal runtime is available. Owned modules always use
this worktree on PYTHONPATH. No sibling checkout, shared virtual environment or
root-owned file was changed.

Verification: **303 passed in 46.49s** across the entire owned agent/decision/
gateway/history/copilot suite plus both data contract tests, with actual shared
configuration/schema from cbbfe14. Final targeted verification after complete
prompt archival and malformed-metadata hardening: **111 passed in 18.82s**
(53 copilot cases + 58 gateway cases), using registry cbbfe14 and HMM module/
variant d3724f1 extracted read-only into isolated test scratch. Network sockets
were disabled, Unix sockets allowed only for async/HTTP framing tests. Ruff and
`git diff --check` passed. New cases cover all five payload kinds, exact model
parameters, quoted/full paper coverage, negative development findings, holdout/
future/synthetic exclusion, unknown sources/pricing, durable resumed caps,
config/model mismatch, source instructions/conflicts, output/usage bounds,
malformed usage metadata, no false success, immutable memo history and actual
core entry points. This is offline implementation verification, not authenticated
provider accuracy/cost acceptance or full-system performance validation.

## H paper operating cycle integration

Owned runtime is `src.operations.contracts` and `src.operations.cycle`. It composes the source store, approved model recipes, causal HMM, complete desk, canonical strategy selectors/optimizers, `SleeveBook`, `VenueExecution`, and daily attribution. This is a full-plan integration checkpoint; authenticated vendor, prospective performance, deployed scheduler/container, and live-capital acceptance are separate and are not claimed by offline provider-boundary fixtures.

Canonical entry points:

- `PaperCycleService(config, *, operations_store, evidence_store, registry, ledger, attribution_store, artifact_dir, decision_service=None, decision_policy=None, llm_factory=None, resolver=None)`.
- `prepare(template, *, refresh_run_id, run_id, scheduled_for, economic_sources, macro_dataset) -> PaperCyclePlan`. It requires the authoritative completed refresh manifest/report and its actual measured cutoff. The exact eight v2 source arrays are opening events, instrument metadata, universe history, corporate actions, paper venue rules, calendar, opening liquidity, and market availability audit. Opening events and liquidity must bind the same exact receipt envelope; actual price/quantity cannot come from unrelated documents. Operational plans use `opening_events` with `bars=()`. The explicit legacy v1 completed-bar variant remains separate; mixed variants reject. References are outside the immutable economic-input hash. Original raw sources, normalized pointers, clocks and rule contents are independently verified; opening receipts use the exact envelope hash and actual ingestion clock. Audit/bundle capture must finish before refresh completion.
- `run(plan, *, now=None) -> CycleResult` and `dispatch_pending(plan, *, now=None)`. The normal runtime uses UTC wall time. Explicit `now` is for an injected observation clock; model/desk completion is measured separately and cannot be backdated to the source close. The scheduler owns refresh, dispatch cadence and final notifications.
- `materialize_features(plan, *, training=False)` and `training_frame_hash(dataset, fitted_at)` support review of exact approved training recipes before admission. Numeric recipes include the frozen original training snapshot, feature/data/frame hashes, model parameters, actual label endpoints, code version and availability policy. Trusted artifacts verify the complete expected payload/sidecar set against immutable operating receipts before deserialization; reuse preserves the original fitted identity. Model recipes pin the current-at-fit eligible cohort with exact universe source bindings and state the resulting current-membership selection bias; they do not claim historical index membership.

Model and macro receipt `training_cutoff` is the approved temporal cutoff (the deterministic artifact’s `fitted_at`), not proof of physical existence. Each immutable creation receipt records its original run/stage/attempt, authoritative stage start, measured artifact completion and availability. The `model_availability` stage resolves the original attempt from OperationsStore and records original stage state/completion plus the current completed consumption stage’s times and output hash. Failed original attempts remain failed when a valid local artifact is recovered. The desk context is at least the physical model availability and stage completion; the final execution context follows actual desk completion. Reuse keeps the original creation receipt and training cutoff. Account-specific HMM state and per-recipe model locks prevent cross-account state progression and concurrent artifact writes.

All six StrategySpecs remain active in the composition, including the model-free named-role consensus exception. Selection supports regime, actual performance weighting, declared bandit exploration/exploitation and manual choice. All five optimizer methods use actual covariance samples and canonical risk limits, including passive ownership minima. Missing completed daily performance cannot enable exploitation. Cash allocation moves only actual free cash using explicit conserved ledger transfers; unselected and retiring sleeves retain actual invested ownership until an authorized exit. HOLD, ABSTAIN and VETO preserve actual ownership even when a quantitative strategy requests an exit. REDUCE caps the quantitative target at existing ownership independently of the exposure-block flag; explicit EXIT targets zero. A passive veto remains binding during explicit sleeve retirement.

The covariance policy `retrospective-lagged-intervals-1` estimates current risk from actual asset-specific completed intervals paired at observed equity closes (or observed crypto closes for a pure crypto book). Both endpoints must precede each anchor and satisfy the pinned endpoint-age bound. The same interval for any asset cannot add another sample. No price interpolation or forward-filled return is used. The receipt preserves every actual endpoint price/time, source envelope hash and availability, anchor, current estimation cutoff and sample hash. Current captures may estimate historical returns now; this is explicitly retrospective estimation, not a claim of historical source availability or a replacement for G/WF validation.

Account-level kernel locks cover economic mutation. Immutable per-stage inputs/results and complete account history prevent cross-run overlap and duplicate schedule execution. Completed runs return their original receipt. Pure/idempotent stages can recover; ambiguous generative work remains unresolved without authoritative reconciliation. Observed-opening retries recover original admission metadata from the committed venue receipt, including orders no longer pending. No dispatch is backdated to an observed print. New targets become eligible strictly after decision completion and only previously accepted orders may consume a later observed open. Completed source MarketBars supply actual close marks independently of opening prints. Canonical venue expiry runs at each observed economic timestamp; an unclosed bar never supplies a closing price or whole-session opening volume. Posted fees, fills, cash and exact ownership remain accounting-owned.

Attribution uses actual sleeve NAV periods and explicit end flows. Actual close anchors and intermediate decision timestamps are retained with real later availability. Positive funded epochs create an anchor rather than a fabricated return; incomplete days, zero-NAV hiatuses and insufficient performance remain ineligible. Daily counts come exclusively from the data lane's exact-close aggregation.

The canonical CLI, scheduler, source-refresh dispatch and notification delivery are coordinator-owned. The template must include held/retiring instruments as well as the current candidate universe so their prices and exit identities remain available. USD base/quote instruments use exact FX=1 for independent close marks; nonbase close valuation blocks unless an authoritative close-FX contract is supplied. This service does not invent that conversion. Generic nonbase close-FX integration remains a full-plan follow-on. Ambiguous generative/network outcomes require explicit reconciliation; they are not automatically retried.

Shared integration baseline: root `f108b3cb9e957faf8eee7aecee7b184bd846945b`, extracted read-only into isolated test scratch. All owned modules resolve from this worktree; no sibling source or environment was modified. The private environment is `/tmp/quantagentlab-jev-cycle-venv`. The final test command uses `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`, that environment’s Python, and `/tmp/quantagentlab-jev-cycle-contracts/run_tests.py tests/test_operating_cycle.py -q --disable-socket --allow-unix-socket --basetemp=/tmp/quantagentlab-jev-cycle-tests10`. The driver only resolves shared imports from the exact extracted root commit; after integration the equivalent command is `python -m pytest tests/test_operating_cycle.py -q --disable-socket --allow-unix-socket` with an isolated basetemp.

Verification covers two actual mixed-calendar operating cycles, all six strategy outputs, four trained model families, real causal HMM fitting/reuse, all four selectors from 26 actual funded daily ledger returns, all five optimizers, DST/weekend covariance lineage without repeated interval samples, authoritative refresh construction, source/receipt corruption and missing coverage, trusted-model tampering, duplicate schedules, kernel locks, uncertain-provider nonretry, interruption after a committed opening fill, exact replay, physical model availability, independent close marks, partially filled order expiry and long Decimal conservation. Every full desk run uses actual graph nodes and archived history/budget/Jev records with mocked provider boundaries only. Synthetic markets, prices and provider responses exist only in tests; no credentialed API success is inferred from those fixtures.

Final owned checkpoint verification: **28 passed in 201.00s**, with network sockets disabled (Unix sockets allowed for the async provider-boundary fixture). Ruff check, Ruff format check and `git diff --check` passed. Earlier full physical-clock suite: 22 passed in 211.42s; earlier opening-only integration: 21 passed in 198.75s. These are deterministic implementation tests, not authenticated provider, historical-source entitlement, prospective statistical, deployed scheduler or live-capital acceptance. No dependency/shared schema/config changes were made. The source-level v1/v2 same-print migration fence is a coordinated accounting/data/root follow-on and is not claimed by this checkpoint.

Recovery freshness follow-on: a resumed incomplete run rechecks the actual invocation clock even when `verified_inputs` is already complete. Expired inputs cannot reach account mutation or provider admission. Whole-run completed replay remains a read of its original immutable receipt. The targeted regression used the actual full-cycle fixture and a committed OperationsStore verification checkpoint: **1 passed in 117.24s** (`-k partial_resume_rechecks_freshness`, same pinned shared source and offline flags); Ruff and diff checks passed.
