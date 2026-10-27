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
