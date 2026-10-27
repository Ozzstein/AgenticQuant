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

Regime input evidence binds `ResearchInputs.content_hash()` and is established before the experiment, without future portfolio returns. The integrated regime adapter independently reconstructs observations from archived macro/breadth data and actual HMM models; the evaluator then attributes subsequent ledger return intervals and archives the derived proof. An initial, unexposed zero-return cash interval remains explicitly unclassified rather than receiving a future regime label.

`ResearchRegistry.development_evaluation(trial_id)` is a restricted read-only memo interface: it refuses holdout trials before accessing their outcomes and verifies lineage and referenced artifacts. Multiplicity is conservatively registry-wide, including failed/unfinished attempts, so changing a family label cannot reset it. `trial_count_at_admission(trial_id)` fixes the count included in a run manifest; activation applies the current count. Reserved holdout overlap follows underlying security identities across changed run contexts/source revisions. Exact holdout contents and interval remain fixed.

Native admission additionally requires explicit `gateway_limits` and `gateway_day_limits` (requests, tokens, cost_usd); both must exactly match the private receipt. Model IDs, scopes, faults and expiry are checked before Docker execution. Native LiteLLM models use `openai/<wire-model-id>`. UI requests require `ui_source_run` and copy only hash-verified logs from that prior run. CPU PyTorch 2.14.1+cpu, Qlib 0.9.7 and PyTables 3.11.1 imports passed in the separate locked native environment. Native source-package discovery supports the installed namespace package. Upstream CPU/thread settings are adapted explicitly. Docker execution remains unverified because the engine is unavailable here.

Observed coordinator checks: 77 offline checkpoint integration tests passed (actual strategies/models/replay, text availability, venue recovery and desk boundaries); 72 migrated research/native/canonical CLI/regime tests passed; 32 subsequent registry/source-proof/native tests passed and cover value substitution, restricted development access, renamed-family multiplicity and changed-context holdout reuse. Wheel build succeeded and contains all six strategy JSON specifications, the native source pin manifest and canonical CLI scripts. These are software/invariant checks, not historical profitability, authenticated vendor, venue or prospective trading acceptance.

## Managed native gateway lifecycle

`GatewayDeployment` binds the complete `GatewayConfig`, a digest-pinned image,
loopback receipt port, external bridge network, memory limit and startup deadline.
Its container budget path is exactly `/state/budgets.sqlite3`. The gateway uses a
private internal network with alias `gateway`; only the gateway also joins the
external bridge. Native generated code receives a run bearer and has no provider
key mount or external network. Both images must include the current runtime code.

```bash
python -m scripts.run_rd_agent managed-native \
  --request native-request.json --deployment gateway-deployment.json \
  --provider-key-file /private/keys/research-provider \
  --gateway-state /private/quantagent-gateways --workdir /work/quantagent-jobs
```

`gateway-start` and `gateway-stop` expose the same explicit lifecycle for operator
reconciliation. Each accepts `--deployment` and `--gateway-state`; start also
requires `--provider-key-file`. Files contain no credential fields. The provider
file must be an owned ordinary file with mode 0600 or stricter; symlinks fail.
The private state directory must have mode 0700. No credential is passed in
Docker arguments or container configuration. The container entrypoint loads
private files into the gateway process only and stops it at the immutable expiry.

Every run in one gateway state directory shares the same budget database and
immutable daily cap policy. Restart preserves the run bearer, original full
configuration, reserved/unknown charges and actual usage. Starting a new run does
not reset daily spending. The manager checks the exact image, entrypoint, mounts,
permissions, published address and network attachments before reusing a container.
Job, Qlib and document mounts must be disjoint from the private gateway state tree.
The manager always attempts to stop its exact owned container after a managed job;
a cleanup failure is recorded without replacing the original execution error.

The container-local watchdog enforces elapsed time and counted output bytes,
including child stdout/stderr, even if the host Docker client dies. Admission,
scenario and request files are read-only mounts. Completed native runs persist a
final authenticated gateway receipt with actual/reserved/unknown usage; missing
or inconsistent final evidence produces `audit_unavailable`, never a successful
discovery record. Outputs remain quarantined regardless of native exit status.

Verification: 93 offline tests passed across gateway deployment, native adapter,
gateway protocol/budgets and research CLI. Tests exercise real local child process
timeouts/output bounds, actual shared SQLite admission across runs, immutable
restart, mount/network drift and failure cleanup; Docker uses explicit boundary
fixtures. Docker execution, native workflow acceptance and authenticated provider
usage remain unverified on this host because the Docker engine is unavailable.

`NativeRequest.knowledge` optionally pins one factor and one model CoSTEER file
from prior completed native runs. Each `NativeKnowledgeRef` binds `source_run`,
the exact source `result.json` SHA-256, `artifact` (`factor` or `model`) and its
SHA-256. The supervisor checks the prior request/result/actual file identities,
copies only those bytes into a read-only input mount, and sets the pinned native
CoSTEER settings to save new knowledge under `/work/knowledge/`. No host process
deserializes native pickle files. Source knowledge remains discovery provenance,
not independent performance or activation evidence. A malicious-pickle sentinel
test confirms host staging only copies bytes; changed source files are rejected.
The 23 native/deployment tests pass offline; actual CoSTEER workflow reuse still
requires the unavailable Docker/provider acceptance run.

## Registered strategy, paired regime and copilot integration

`rd-agent register --kind strategy|regime` parses the owned `StrategyExperimentPlan` or `RegimeExperimentPlan`. Actual fresh models, frozen contextual evidence, causal HMM observations and holdings-aware callbacks feed the same ledger engine. `ResearchRunner` binds the callback benchmark to the original ledger through `ledger_path`, preserves the opportunity artifact, and supplies durable cost artifacts to evaluation.

The evaluator independently re-executes strategy/regime recipes from frozen sources, refits actual models, and compares all results, including folds, decisions, fills, NAV and opportunity bytes. Only verified filesystem placement may differ. `Evaluation.execution_verification_hash` archives that proof. A regime experiment runs its declared candidate and baseline in the same trial, with separate ledgers and identical numeric model bytes. `paired_evaluation_hash` binds both results, the original baseline ledger, common valuation intervals and recomputed excess returns. Its significance test uses those paired excess returns. Missing baseline proof, nonpositive improvement or incomplete windows blocks activation. An incomplete development run can still be inspected and reproduced without being certified.

All canonical walk-forward callers pass actual feature-specification and data-snapshot hashes, feature availability and label availability. The actual four-family IC-weighted regression uses a declared validation fraction with enough observations; insufficient evidence is not replaced by equal weights. Unversioned diagnostic fitting cannot be certified.

`rd-agent copilot propose --request ... --registry ... --evidence ... --code ... --environment ... --output ...` exposes typed source-grounded factor/model/strategy/regime/text-question proposals. `paper_evidence_ids` selects full-document reading with complete chunk coverage. `rd-agent copilot memo` reads only verified recorded development outcomes. Exact execution-code/environment artifacts are archived before generation; generated candidates remain quarantined. Missing credentials/pricing/sources yields an archived unavailable outcome and exit 2, not a template candidate or placeholder memo. Late generation cannot be reused under an earlier research cutoff.

Additional coordinator checks: 53 research/source-proof/ensemble/copilot-CLI/public-wrapper tests passed; 56 copilot and CLI tests passed; independently reconstructed strategy and paired-regime tests passed, including fold and return substitution rejection. No authenticated provider calls or statistical profitability acceptance are claimed.
