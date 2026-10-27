# Research rebuild evidence

The original simulated IC, Sharpe, model optimization and strategy/regime promotion paths are being replaced. No randomly generated result may activate a candidate. Full native, custom, copilot and paper-derived scope remains required.

## Registry boundary

`src.research.registry.ResearchRegistry(path)` persists content-addressed artifacts, immutable candidates, preregistered development/holdout trials, outcomes and versioned activations. Holdouts are single-use on attempted execution, including failed attempts. Renaming an interval does not make it untouched. Failed and unfinished trials remain in family-wide significance counts. Promotion uses compare-and-swap generation checks; rollback creates a new generation.

`resolve(slot: str, *, payload_hash: str, as_of: datetime) -> Activation | None` resolves only an exact historical activation. Strategy callers use `strategy:<name>` and the complete `StrategySpec.content_hash()`. YAML `validated` flags do not grant eligibility. Research activation is not live-capital authorization.

Registry lifecycle tests are synthetic invariant tests. Registry outcome insertion is an internal evaluator capability; the public CLI does not accept user/generated metric claims as an evaluation. The independently computing evaluator and executable experiment adapters are separate integration requirements.

## In-progress experiment integration

`ResearchInputs` binds an archived `MarketSnapshot`, explicit replay account/context, execution bars, metadata, universe histories, corporate actions, costs/risk policy and six-month walk-forward settings. `ExperimentPlan` fixes feature and reference-feature specifications and model identity. `ResearchRunner` registers a trial before fitting fresh models, persists actual OOS replay and performs matched cost/random/named benchmark reruns. `ExperimentEvaluator` checks input/model/source contents as well as ledger validation and computes IC, ICIR, correlation and a centered moving-block test. Missing evidence blocks activation.

Source-artifact audit requires exact normalized source contents and raw hashes. Provider publication claims/entitlements still need real acceptance. Synthetic tests are explicitly marked and cannot activate.

## Native runtime

Separate `environments/rd-agent/` environment pins RD-Agent v1.0.0 at `484776c211e4fbbeef03e0ec00d6bbee7362a4f4`, Qlib 0.9.7 and CPU PyTorch 2.14.1. CPU PyTorch availability was checked against the [official release announcement index](https://dev-discuss.pytorch.org/c/release-announcements/27) and the actual pinned package was installed/imported separately.

The native adapter checks pinned upstream template hashes, substitutes explicit US universe/benchmark/cost/split settings and contains all generated code in an outer container. It exposes fin_factor, fin_model, fin_quant, fin_factor_report, general_model, ui and health_check. Native discovery metrics remain quarantined and do not replace the ledger evaluator. Source documents are staged into narrow read-only mounts. Provider credentials stay in the separate metered gateway; generated code receives only a bounded run bearer on an internal network.

Actual native CLI import/help, template rendering and locked dependencies were exercised locally. Docker Desktop WSL integration is unavailable on this host; image execution, native authenticated jobs, job recovery/knowledge reuse, generated-model ingestion and provider acceptance remain unverified or in progress. A template test does not establish that native workflows complete.

## Integrated executable research checkpoint (2026-10-27)

Canonical `aiquant backtest`, `agents`/`analyze`, `data`, `decision` and `rd-agent` commands now delegate to the same typed implementations as their direct modules. Ticker-only analyst calls and label-only backtests fail with missing-input errors. The old RD runner's random IC/Sharpe and simulated model/strategy/regime promotions were removed. Prior tests asserting simulated success were replaced with explicit-boundary tests and actual fitted-model/chronological-ledger tests.

`ExperimentPlan.model_parameters` is an immutable tuple of parameter pairs. Candidate payload is exactly `{features: [...FeatureSpec JSON], model_name: ..., model_parameters: {...}}`. Each fold constructs a fresh supported model with these overrides and version-pinned defaults; ambient saved YAML cannot change the recipe. Ensemble fixes all four actual model families and IC weighting. Source verification requires `universe_history_id`, `availability_audit_id`, `calendar_artifact_id`, `corporate_actions_id`, `execution_bars_id`, `opening_liquidity_id` and `instrument_metadata_id`. Exact normalized archived bar values are reconstructed. Opening liquidity/FX must be available at the open; instrument metadata must be available at its declared cutoff.

Regime input evidence binds `ResearchInputs.content_hash()` and is established before the experiment, without future portfolio returns. The regime adapter independently reconstructs observations from archived macro/breadth data and actual HMM models; the evaluator then attributes subsequent ledger return intervals and archives the derived proof. That adapter and generalized strategy/regime execution are a separate, still-active lane. Until its verifier is available, that certification check fails closed.

`ResearchRegistry.development_evaluation(trial_id)` is a restricted read-only memo interface: it refuses holdout trials before accessing their outcomes and verifies lineage and referenced artifacts. Multiplicity is conservatively registry-wide, including failed/unfinished attempts, so changing a family label cannot reset it. `trial_count_at_admission(trial_id)` fixes the count included in a run manifest; activation applies the current count. Reserved holdout overlap follows underlying security identities across changed run contexts/source revisions. Exact holdout contents and interval remain fixed.

Native admission additionally requires explicit `gateway_limits` and `gateway_day_limits` (requests, tokens, cost_usd); both must exactly match the private receipt. Model IDs, scopes, faults and expiry are checked before Docker execution. Native LiteLLM models use `openai/<wire-model-id>`. UI requests require `ui_source_run` and copy only hash-verified logs from that prior run. CPU PyTorch 2.14.1+cpu, Qlib 0.9.7 and PyTables 3.11.1 imports passed in the separate locked native environment. Native source-package discovery supports the installed namespace package. Upstream CPU/thread settings are adapted explicitly. Docker execution remains unverified because the engine is unavailable here.

Observed coordinator checks: 77 offline checkpoint integration tests passed (actual strategies/models/replay, text availability, venue recovery and desk boundaries); 72 migrated research/native/canonical CLI/regime tests passed; 32 subsequent registry/source-proof/native tests passed and cover value substitution, restricted development access, renamed-family multiplicity and changed-context holdout reuse. Wheel build succeeded and contains all six strategy JSON specifications, the native source pin manifest and canonical CLI scripts. These are software/invariant checks, not historical profitability, authenticated vendor, venue or prospective trading acceptance.
