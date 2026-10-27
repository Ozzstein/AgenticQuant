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

## B/A/E: direct adapters, MCP and usable CLI

Direct entry points under `src/data/adapters/`:

- `FMPAdapter.fetch`: stable statements/ratios/analyst/insider/institutional/filing/earnings/transcript routes. Full response bytes and complete transcript text survive bounded retrieval. Event dates do not invent revision availability; these captures are usable from receipt.
- `FredAdapter.observations`: paginated FRED/ALFRED realtime intervals, all vintages retained. Day-resolution vintage availability uses the end of its US/Eastern date; missing `.` observations are explicitly unusable. Revisions remain visible for explicit resolution.
- `CCXTAdapter.bars/derivatives`: explicit canonical venue/instrument, spot versus perpetual checks, bounded pagination, raw candles and separate funding/OI capabilities. `YFinanceAdapter.bars`: unadjusted SDK output, actual XNYS sessions and separately recorded corporate actions. Yahoo's SDK output is identified as parsed output, not claimed to be raw HTTP bytes. Current historical downloads are not silently treated as archived vintages.
- `FinnhubAdapter.news`, `SECAdapter.submissions/filing`, `CoinGeckoAdapter.token/global_context`, `AlpacaReadAdapter.inspect`. SEC full accession documents preserve documented acceptance from an evidence-backed index; amendments are distinct accessions. Alpaca only account/position/order GET observations are reachable.

`ReadHTTP` enforces bounded attempts/count/timeouts and rate spacing; errors distinguish credentials, entitlement, rate limits, malformed payload, provider outage and missing data without printing keys/URLs with credentials. External provider coverage is not inferred from successful fixture parsing. Bulk paths need deployment-specific coverage/entitlement exercises. Yahoo/FMP historical captures requiring genuine vintage provenance fail temporal training rather than fabricate publication dates.

`MCPManager` supports actual MCP 2.3 Streamable HTTP connections, server identity/version and exact schema-hash verification, fixed argument policies, tool deduplication by capability, per-run evidence reuse and request/spend reservation budgets. Generic dispatch is unwrapped and checked locally and is never sent as an unrestricted remote call. Approved Alpaca observation names cannot be expanded to submission/cancellation/configuration/position mutations by a manifest. Failed verification evicts the connection. Unknown actual costs remain unknown. Selected tool results are stored completely as received; this does not assert that a remote server returned a complete original document. Prefer the full direct FMP ingestion for transcripts.

`server_sources.json` pins reviewed OpenBB/FMP/Alpaca repository commits and identifies current Valyu's hosted endpoint. These are source pins, **not verified deployment manifests**: deployed server version/schema pins and provider extensions/entitlements must be captured and supplied as `ServerManifest`s before tool calls. Verified runtime versions: `mcp==2.3.0` (uses bundled `httpx2`), `exchange-calendars==4.13.2`, `pyqlib==0.9.7`. The coordinator owns the dependency lock.

Primary implementation references: [Alpaca exact tool registry](https://github.com/alpacahq/alpaca-mcp-server/blob/d8dac43f3e17594e8879bac3b5b333196ea69aac/src/alpaca_mcp_server/tool_registry.py), [OpenBB MCP configuration](https://github.com/OpenBB-finance/OpenBB/tree/7df7d0c5592bf25be88e87e130a05c71e243d2d8/openbb_platform/extensions/mcp_server), [FMP API catalog](https://github.com/imbenrabi/Financial-Modeling-Prep-MCP-Server/blob/fe4e0a6eb4d07c81c0f79322263a4011afb13656/docs/API_REFERENCE.md), [FRED realtime observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html), [Valyu current hosted service](https://mcp.valyu.ai/).

`EvidenceService.source_bundle(role,instrument,context)` returns source coverage and bounded exact passages without network access. `market_snapshot` resolves duplicate identical observations, reports conflicting prices and explicit expected-session gaps, and retains canonical identity. The legacy transcript tool now stores the complete response before excerpting, cites its store ID and redacts provider errors; unresolved legacy ticker identity does not become canonical evidence for typed agents.

Executable interface: `python -m src.data --store /owned/path/evidence.sqlite --help`. Subcommands: `import-archive`, `query`, `document`, `snapshot`, `train`, `predict`, `qlib-export`, `qlib-alpha158`, `fmp-transcript`. Archive imports consume caller-supplied records/raw files with hashes; there is no runtime generator of sample market data. Context/instrument/feature/scenario inputs are validated JSON contracts. Failures return nonzero status. The coordinator must connect these services to the canonical application cycle/CLI.

## C1/C2: US Qlib, local numeric flow and ensemble

`src/data/qlib_us.py`: `export_bundle(snapshot,path)` writes immutable actual Qlib daily binaries/calendar/instruments and a content-hashed manifest with the entire evidence snapshot. Alpha158 requires actual VWAP; typical-price substitution is rejected. One raw/adjusted convention must be selected explicitly; raw factor=1 is declared raw. `USQlibScenario` pins US region, XNYS calendar, canonical-to-Qlib instrument mapping, benchmark, splits, label horizon and cost convention. `verify_bundle` checks files, coverage and label-horizon purging in actual sessions. `alpha158_dataset` runs real Qlib with train-only robust normalization and explicit labels. This does not claim a production US universe/delisting dataset has been obtained.

IC ensembles split by distinct dates, purge overlapping training outcomes, compute date-local held-out IC and fail when no model has positive/defined validation IC. Equal/custom modes are explicit, never silent error substitutes. Every child artifact retains its fitted preprocessing; the parent manifest pins all child hashes. Current models cannot activate merely by writing a research YAML file.

Consolidated Python 3.11 offline run (exact selected files in command below): **209 passed, 1 skipped in 10.93s**. The skipped Qlib test passes in the separate Python 3.12 environment. Synthetic sources are confined to `tests/data_quant_fixtures.py` and other tests.

```sh
PYTHONPATH="$PWD" /tmp/quantagentlab-review-venv/bin/python -m pytest tests/test_evidence_boundary.py tests/test_direct_evidence_adapters.py tests/test_mcp_evidence_permissions.py tests/test_feature_fidelity.py tests/test_temporal_models.py tests/test_local_quant_flow.py tests/test_core_data.py tests/test_core_models.py tests/test_core_factors.py tests/test_crypto_factors.py tests/test_crypto_pipeline.py tests/test_ensemble.py tests/test_transcript_data.py -q --basetemp=/tmp/quantagentlab-data-quant-checkpoint
PYTHONPATH="$PWD" /tmp/quantagentlab-data-quant-venv/bin/python -m pytest tests/test_local_quant_flow.py::test_alpha158_actual_qlib_local_bundle -q --basetemp=/tmp/quantagentlab-data-quant-qlib
# 1 passed in 2.64s: actual US Qlib/Alpha158, 300x158 train, 96x158 test, actual Ridge
PYTHONPATH="$PWD" /tmp/quantagentlab-data-quant-venv/bin/python -m pytest tests/test_mcp_local_transport.py tests/test_mcp_evidence_permissions.py -q --basetemp=/tmp/quantagentlab-data-quant-mcp
# 12 passed in 1.65s: actual SDK over localhost + permission/manifest/cache/budget cases
```

Outstanding full-plan evidence: real authenticated provider/MCP calls and deployed pins, historical US/crypto coverage including delistings and adjustments, complete text-feature integration, native RD-Agent US overlay and independent validation, D causal HMM/six strategies/allocation, actual historical comparative performance and all I prospective gates. No paid accounts, live orders, external deployments, push or PR were performed.

## Offline socket permissions checkpoint

The actual MCP protocol test permits only localhost/127.0.0.1 connections via `pytest.mark.allow_hosts` and asserts external TCP is rejected. The async permission test uses a per-test Unix-only fixture (pytest-socket 0.7 has no Unix-only marker) and asserts AF_INET socket creation is rejected. No suite-wide network exception is introduced.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" /tmp/quantagentlab-data-quant-venv/bin/python -m pytest tests/test_direct_evidence_adapters.py tests/test_mcp_local_transport.py tests/test_mcp_evidence_permissions.py -q --disable-socket --basetemp=/tmp/quantagentlab-data-quant-socket
# 23 passed; actual MCP SDK localhost transport, offline provider parsing, permission/cache/budget invariants
```

## D: causal authoritative regime contract

`src.core.macro_regime.CausalHMMRegimeDetector(state_path, normalization_window=252, min_regime_days=5, confidence_threshold=.6, n_init=10, n_iter=100)` provides `fit(raw, context, *, available_at, evidence, source_ids)` and `predict(...) -> CausalRegimeResult`. Inputs have the eight ordered `MACRO_SIGNALS`, timezone-aware chronological dates, a per-row availability Series, and a per-row tuple-of-evidence-IDs Series. Each linked record must have been available at that row's decision cutoff. `macro_signals(raw, constituent_prices, membership)` computes actual DGS10 minus DGS2 and historical constituent breadth (explicit boolean membership and complete 200-session price coverage); Yahoo IRX and index-above-average approximations were removed.

Actual four-state Gaussian HMM fits use eligible data, rolling trailing normalization and multiple seeded initializations. Inference uses the forward filter, not backward smoothing. Training-prefix identity and already emitted prefixes cannot change. SQLite persists filtered probabilities and minimum-duration counters transactionally; restarting cannot reset duration. Missing models/data or initial low confidence produce `status='insufficient_data', regime=None`, never invented neutral. Available results carry context, regime, all four probabilities, confidence of the **effective** held state, model_hash, trained_through, input_hash, evidence_ids and held_reason. The agents bridge should reject nonavailable results. Save/load uses a payload integrity hash and a JSON manifest with training/source/code/environment/config identity; joblib is trusted application-owned material only.

`tests/test_causal_regime.py`: **6 passed under --disable-socket**, actual HMM fit/filter plus future perturbation, revised source/training rejection, historical source cutoffs, restart and held-state confidence, missing-vs-neutral, actual breadth/slope. The legacy `HMMRegimeDetector` remains an explicitly diagnostic API and must not be used by production orchestration; its ambient proxy fetch fails. Its prior numerical tests are separate from acceptance of the causal path. Registry-gated evolved regime activation and real historical coverage/performance remain coordinator integration/acceptance requirements.

## D: executable strategies, allocation and attribution

`src/strategies/specifications/*.json` contains six complete immutable `StrategySpec` declarations, loaded by `baseline_specs()`. Each pins actual exact expressions, model family (agent consensus alone has no model), eligible asset classes, rebalance cadence, numeric executable parameters, ranking/entry/exit policy, top-K, sizing and sleeve limits. `evaluate_strategy(spec, FeatureDataset, RunContext, *, universe, instruments, holdings, signal_at, regime, model, holding_sessions, last_rebalanced_at, consensus, resolver, approved_baseline_hashes)` consumes accounting's existing `src.domain.portfolio.UniverseEligibility`, existing common identities, actual saved `ModelWrapper` artifacts and the authoritative causal HMM. It returns a DataFrame of target_weight/action/reason/score with source/spec/model/universe/context attributes; the accounting lane owns the canonical target/order/portfolio adapter. No duplicate portfolio or agent-decision schema was added.

Momentum ranks reversal-adjusted 3/6/12-month momentum, weekly top 30, with positive actual model predictions. Value/quality uses dated PE/ROIC/FCF, monthly; mean reversion uses actual Wilder RSI/z-score with a three-session exit; crypto uses MA/volume confirmation, the actual funding/OI/dominance/beta feature inputs and inverse volatility sizing; defensive uses low beta/quality/volatility, at least 30% sleeve cash and no crypto. Agent consensus consumes the agent lane's named/derived-vote `ConsensusResult`, preserving HOLD/ABSTAIN/VETO and implementing explicit partial REDUCE and EXIT. Missing essential observations block new affected exposure; universe removal is an explicit exit. Final portfolio-level caps, sector limits, liquidity, turnover and fills remain binding downstream accounting responsibilities.

`require_strategy_eligible` accepts exact shipped code baselines for research/backtest. Operational use requires explicitly approved baseline hashes, or `ResearchRegistry.resolve('strategy:'+name, payload_hash=spec.content_hash(), as_of=context.as_of)` returning the exact activated artifact. Code baseline approval is distinct from research validation. Arbitrary JSON/YAML `validated:true` never grants eligibility. The compatibility `StrategySelector` loads only shipped declarations and no longer runs synthetic smoke performance; production must use the new contextual API.

`select_allocations` implements regime matching, positive observed Sharpe weighting, reproducible seeded eligible-candidate bandit exploration and explicit manual choice. Future/unattributed/insufficient performance cannot qualify a candidate; absent positive metrics do not silently produce equal weights. `blend_sleeves` nets weights with sleeve contributions, retains binding vetoes and rejects omitted existing holdings. Caller must supply explicit retiring-sleeve output and persist rebalance/session ages from actual replay/ledger events.

`AttributedPerformanceStore(path).record_nav(...)` appends reconciled Decimal opening/closing sleeve NAV, explicit end-period flow, period/availability times and immutable accounting event ID/hash. It is a reporting projection, not another position/order ledger. Duplicate IDs are idempotent only for identical payloads. `metrics(as_of=..., annualization=252|365)` computes cash-flow-correct returns/drawdowns including the initial NAV and source lineage; missing daily sessions make Sharpe undefined instead of filling gaps. Intraperiod flows must be split by the authoritative accounting caller.

`NumericEvidenceBinding` and `materialize_evidence_fields` provide scalar joins from actual immutable provider evidence at EACH historical row cutoff, preserving explicit revisions, conflicts, missing coverage, source identities and freshness. Spot-to-perpetual bindings must declare a separate source instrument and perpetual provenance. `feature_dataset(..., evidence_fields=...)` combines their availability/hashes with the market feature pipeline. This is used by the same exact strategy features and model transforms; it grants no derivative execution authority.

Validation:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" /tmp/quantagentlab-data-quant-venv/bin/python -m pytest tests/test_contextual_features.py tests/test_executable_strategies.py tests/test_feature_fidelity.py tests/test_temporal_models.py tests/test_local_quant_flow.py -q --disable-socket --basetemp=/tmp/quantagentlab-data-quant-d-checkpoint
# 41 passed in 5.17s; actual models/features, US Qlib, all six strategies and four allocation modes
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" /tmp/quantagentlab-review-venv/bin/python -m pytest tests/test_strategy.py -q --basetemp=/tmp/quantagentlab-data-quant-strategy-legacy
# 29 passed in 0.21s
```

These are local deterministic integration/invariant exercises with synthetic sources confined to tests, not historical performance or full application acceptance. Root must connect the real ledger NAV projection, target adapter, canonical workflow, registry and CLI; accounting retains chronological replay and final risk. Provider entitlements, historical constituent/delisting data, real historical performance and prospective acceptance remain unverified. New owned research strategy/regime experiment adapters follow this checkpoint and must use the root registry/evaluator lifecycle.

## J5: actual text-feature storage boundary

`src.data.features.text.store_text_feature(store, feature: src.decision.features.TextFeature, instrument, context, *, sources: tuple[DecisionSource], ingested_at)` verifies the actual immutable source records, document hashes and Unicode spans, then persists the complete Jev feature object as a `text_feature` observation. Question/model/policy/transformation identities and source/observation IDs are retained. Effective availability is the maximum of documented source availability and **actual generation time**. A newly generated interpretation of an old transcript cannot enter historical training. Missing value/status stays missing, never zero; numeric bindings can select a pinned question/model/transformation for the regular evidence-feature pipeline. Pretrained knowledge and independent/prospective validation remain required.

`tests/test_text_feature_boundary.py` passed **1 test under --disable-socket** using the actual agent contracts read from commit `8f2d2de` in a test-only in-memory import overlay. The lane checkout was not merged or changed; all owned implementation imports came from this worktree. The integrated repository runs the same test directly. No authenticated Jev call or statistical acceptance is claimed by this storage exercise.

## Exact static validation and sleeve provenance follow-up

`validate_expression(expression, allowed_fields) -> None` uses the same bounded parser/interpreter as numeric evaluation with empty typed operands. It checks supported operators, fields, arity, causal windows and scalar bounds without synthetic observations. `ExpressionError` is the public rejection contract; it does not assert data coverage or numerical quality.

`blend_sleeves(..., current_weights=..., current_contributions=...)` now requires reconciled actual sleeve ownership when multiple sleeves hold positions. HOLD/ABSTAIN/VETO preserve that actual ownership despite allocation changes; REDUCE cannot increase it. Vetoes suppress proposed additions without shrinking preserved holdings. Context, signal cutoff, universe version and data snapshot must agree before netting. Output attributes retain `context`, `allocations`, `data_snapshot_hash`, `universe_version`, `signal_at`, `strategy_hashes`, `model_artifact_hashes`, each complete `sleeve_provenance`, `contributions` and `vetoed`. No duplicate portfolio schema is introduced.

Verification: `tests/test_feature_fidelity.py tests/test_executable_strategies.py` under `--disable-socket`: **31 passed in 3.81s**, including all baseline expressions, rejected noncausal/unsafe syntax, mismatched sleeve identity, immutable provenance and passive holdings under allocation changes.
