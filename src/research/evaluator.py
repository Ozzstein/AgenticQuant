"""Research outcomes are computed from frozen inputs and persisted ledger evidence."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from src.core.backtester import ReplayResult, canonical_hash, performance_metrics
from src.core.validation import ValidationSuite
from src.core.walk_forward import frame_hash
from src.data.evidence.store import EvidenceStore
from src.domain.common import VersionedContract
from src.domain.market import FeatureSpec
from src.research.datasets import ResearchInputs
from src.research.evidence import verify_provenance
from src.research.registry import (
    Evaluation,
    RegistryError,
    ResearchRegistry,
    canonical,
    content_hash,
)
from src.utils.schemas import CheckResult, CheckVerdict


class ExperimentPlan(VersionedContract):
    inputs: ResearchInputs
    features: tuple[FeatureSpec, ...] = Field(min_length=1)
    # Frozen reference catalog; empty catalog cannot pass a correlation check.
    reference_features: tuple[FeatureSpec, ...] = ()
    model_name: str = "Linear"
    model_parameters: tuple[tuple[str, int | float | str | bool | None], ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def bounded_model(self):
        if self.model_name not in ('Linear','LightGBM','CatBoost','XGBoost','Ensemble'):
            raise ValueError('unsupported fitted model family')
        if len(dict(self.model_parameters)) != len(self.model_parameters):
            raise ValueError('duplicate model parameter')
        if self.model_name == 'Ensemble' and self.model_parameters:
            raise ValueError('ensemble uses pinned per-family defaults')
        if len(dict(self.provenance)) != len(self.provenance):
            raise ValueError('duplicate provenance key')
        return self

    def candidate_payload(self) -> dict:
        return {'features':[f.model_dump(mode='json') for f in self.features],
                'model_name':self.model_name,'model_parameters':dict(self.model_parameters)}

    def model_factory(self):
        from src.core.ensemble import EnsembleModel
        from src.core.model_zoo import ModelWrapper
        from src.utils.config import AppConfig, QlibConfig
        # Trials cannot consume ambient YAML/model-selection overrides.
        config = AppConfig(_env_file=None).model_copy(update={
            'qlib':QlibConfig(model_name=self.model_name,model_params=dict(self.model_parameters))})
        if self.model_name == 'Ensemble':
            return EnsembleModel(model_names=['LightGBM','CatBoost','XGBoost','Linear'],
                                 weighting='ic_weighted',config=config)
        return ModelWrapper(self.model_name,config)


def load_experiment_plan(registry: ResearchRegistry, trial_id: str):
    """Only owned executable plan types may cross the registry boundary."""
    trial = registry.trial(trial_id)
    kind = registry.candidate(trial.candidate_id).kind
    if kind == 'strategy':
        from src.research.strategy_experiments import StrategyExperimentPlan
        return StrategyExperimentPlan.model_validate_json(registry.blob(trial.inputs_hash))
    if kind == 'regime':
        from src.research.regime_experiments import RegimeExperimentPlan
        return RegimeExperimentPlan.model_validate_json(registry.blob(trial.inputs_hash))
    return ExperimentPlan.model_validate_json(registry.blob(trial.inputs_hash))


def block_bootstrap_p_value(returns: list[float], *, samples: int = 10000, seed: int = 42) -> float | None:
    """One-sided centered moving-block bootstrap test for positive mean returns."""
    values = np.asarray(returns, dtype=float)
    if len(values) < 60 or not np.isfinite(values).all() or samples < 1000:
        return None
    observed = float(values.mean())
    if observed <= 0:
        return 1.0
    if np.ptp(values) == 0:
        return None
    centered = values-observed
    block = max(2, round(len(values)**(1/3)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(samples):
        starts = rng.integers(0, len(values)-block+1, size=math.ceil(len(values)/block))
        sample = np.concatenate([centered[s:s+block] for s in starts])[:len(values)]
        exceed += float(sample.mean()) >= observed
    return (1+exceed)/(samples+1)


class ExperimentEvaluator:
    def __init__(self, registry: ResearchRegistry, evidence_store: EvidenceStore):
        self.registry, self.evidence_store = registry, evidence_store

    def evaluate(self, trial_id: str, result: ReplayResult, *, ledger_path: str | Path,
                 artifact_dir: str | Path, completed_at: datetime, cost_results=None, cost_artifacts=None,
                 benchmark=None, paired_baseline=None) -> Evaluation:
        trial = self.registry.trial(trial_id)
        plan = load_experiment_plan(self.registry,trial_id)
        inputs = plan.inputs
        candidate = self.registry.candidate(trial.candidate_id)
        if candidate.kind in ('strategy','regime'):
            from src.research.strategy_experiments import materialize_strategy_plan
            def materialize(specs):
                return materialize_strategy_plan(plan,self.evidence_store,specs=specs)
        else:
            materialize = inputs.materialize
        checks = []

        def verify(name, operation):
            try:
                value = operation()
            except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
                checks.append(CheckResult(name=name, verdict=CheckVerdict.FAIL, explanation=str(exc)))
                return None
            checks.append(CheckResult(name=name, verdict=CheckVerdict.PASS, explanation="Exact archived content verified."))
            return value

        def require(condition, message):
            if not condition:
                raise RegistryError(message)

        def frozen_inputs():
            require(trial.dataset_hash == inputs.snapshot.content_hash(), "trial dataset differs from exact snapshot")
            require(trial.split_hash == content_hash(canonical(inputs.split_manifest())), "trial split policy differs")
            require(result.manifest["context"] == inputs.context.model_dump(mode="json"), "replay context differs")
            expected = {"metadata_hash": canonical_hash([m.model_dump(mode="json") for m in inputs.metadata]),
                        "universe_hash": canonical_hash([u.model_dump(mode="json") for u in inputs.universes]),
                        "fill_policy": inputs.fill_policy.model_dump(mode="json"),
                        "risk_policy": inputs.risk_policy.model_dump(mode="json"),
                        "topk": inputs.topk, "samples_per_year": inputs.samples_per_year,
                        "valuation_calendar": inputs.valuation_calendar}
            require(all(result.manifest.get(k) == v for k, v in expected.items()), "economic inputs differ from trial")
            data = materialize(plan.features)
            provenance = result.manifest["provenance"]
            require(all(provenance.get(k) == frame_hash(v) for k, v in {
                "features_hash": data.features, "labels_hash": data.labels,
                "feature_availability_hash": data.feature_available_at, "label_end_hash": data.label_available_at,
            }.items()), "training/feature/label lineage differs")
            require(bool(result.cross_sectional_ic), "OOS predictions missing")
            first = min(pd.Timestamp(t) for t in result.cross_sectional_ic)
            bars = sorted([b for b in inputs.bars if b.closed_at >= first],
                          key=lambda b: (b.opened_at, b.instrument.instrument_id))
            require(result.manifest["bars_hash"] == canonical_hash([b.model_dump(mode="json") for b in bars]),
                    "OOS execution bars differ")
            start = min(b.opened_at for b in bars)
            account = inputs.account.model_copy(update={"opened_at": start-pd.Timedelta(microseconds=1)})
            require(result.manifest["account"] == account.model_dump(mode="json"), "OOS account differs")
            actions = [a for a in inputs.actions if a.effective_at >= start]
            require(result.manifest["actions_hash"] == canonical_hash([a.model_dump(mode="json") for a in actions]),
                    "OOS corporate actions differ")
            require(all(trial.evaluation_start <= pd.Timestamp(t) < trial.evaluation_end
                        for t in result.cross_sectional_ic), "OOS predictions outside preregistered interval")
            return data

        data = verify("registered_inputs", frozen_inputs)

        def candidate_binding():
            payload = self.registry.blob(candidate.payload_hash)
            require(payload == canonical(plan.candidate_payload()), "candidate recipe differs from executed plan")
        verify("candidate_recipe", candidate_binding)

        def source_proof():
            refs = dict(plan.provenance)
            require(all(result.manifest["provenance"].get(k) == v for k, v in refs.items()), "source references changed")
            return verify_provenance(self.registry, self.evidence_store, inputs, refs)
        verify("source_contents", source_proof)

        def actual_models():
            hashes = []
            for fold in result.walk_forward_results:
                payload = (Path(artifact_dir)/f"fold-{fold.fold_id}.joblib").read_bytes()
                require(content_hash(payload) == fold.model_version, "fitted model bytes differ from fold artifact")
                hashes.append(self.registry.archive(payload))
            require(bool(hashes), "no fitted model artifacts")
            return tuple(hashes)
        model_hashes = verify("actual_model_contents", actual_models) or ()

        execution_hash, paired_hash, paired_returns = None, None, None
        if candidate.kind in ('strategy', 'regime'):
            from src.research.execution_proof import verify_execution
            proof = verify('registered_recipe_reproduction', lambda: verify_execution(
                self.registry, self.evidence_store, trial_id, result,
                artifact_dir=artifact_dir, baseline=paired_baseline))
            if proof is not None:
                execution_hash = proof['execution_hash']
                if proof['paired'] is not None:
                    verify('paired_complete_runs',lambda: require(
                        proof['paired']['candidate_status'] == proof['paired']['baseline_status'] == 'complete',
                        'paired regime runs must both have complete registered OOS windows'))
                    paired_returns = proof['paired']['excess_returns']
                    proof['paired']['unadjusted_p_value'] = block_bootstrap_p_value(paired_returns)
                    paired_hash = self.registry.archive(canonical(proof['paired']))

        def trial_count():
            provenance = result.manifest["provenance"].get("trial_registry", {})
            count = self.registry.trial_count_at_admission(trial_id)
            require(provenance.get("artifact_id") == self.registry.trial_snapshot(trial_id), "trial snapshot reference differs")
            require(provenance.get("trials") == count, "trial count differs from durable registry")
            require(provenance.get("untouched_holdout_id") == trial.holdout_id and trial.purpose == "holdout",
                    "promotion requires a consumed preregistered holdout")
        verify("trial_registry_contents", trial_count)

        corr = None
        if data is not None and plan.reference_features:
            references = materialize(plan.reference_features)
            dates = {pd.Timestamp(t) for t in result.cross_sectional_ic}
            mask = data.features.index.get_level_values(0).isin(dates)
            correlations = []
            for col in data.features:
                for ref in references.features:
                    value = data.features.loc[mask, col].corr(references.features[ref], method="spearman")
                    if np.isfinite(value):
                        correlations.append(abs(float(value)))
            corr = max(correlations) if correlations else None
        reported = result.manifest["provenance"].get("max_existing_factor_correlation")
        verify("factor_correlation_contents", lambda: require(corr is not None and reported is not None
               and math.isclose(corr, reported, rel_tol=1e-12, abs_tol=1e-12), "factor correlation not reproduced from frozen reference features"))
        # Regimes are preregistered causal input evidence. Future portfolio
        # returns are computed here, so the plan has no circular result hash.
        def regime_evidence():
            from src.research.regime_experiments import (
                CausalRegimeArtifact,
                recompute_regime_artifact,
            )
            key = result.manifest["provenance"].get("regime_evaluation_artifact")
            if hasattr(plan,'regime_artifact_hash'):
                require(key == plan.regime_artifact_hash, 'executed causal regime differs from the registered recipe')
            artifact = CausalRegimeArtifact.model_validate_json(self.registry.blob(key))
            require(artifact.inputs_hash == inputs.content_hash(), "regime artifact input identity differs")
            if candidate.kind == 'regime':
                require(artifact.variant == plan.variant, 'evaluated HMM parameters differ from the candidate')
            import tempfile
            with tempfile.TemporaryDirectory(prefix='quantagent-regime-proof-') as work:
                recompute_regime_artifact(self.registry,self.evidence_store,artifact,inputs,work_dir=Path(work))
            periods = []
            for start,end,value in zip(result.timestamps[:-1],result.timestamps[1:],result.daily_returns,strict=True):
                eligible = [o for o in artifact.observations if o.context.as_of <= pd.Timestamp(start)
                            and o.status == 'available']
                if not eligible:
                    start_snapshot = result.snapshots[result.timestamps.index(start)]
                    require(not start_snapshot.positions and value == 0,
                            'exposed return interval has no eligible starting regime')
                    periods.append({'started_at':start,'ended_at':end,'return':value,
                                    'regime':None,'observation_hash':None,'reason':'unexposed_initial_cash'})
                    continue
                observation = max(eligible,key=lambda o:o.context.as_of)
                periods.append({'started_at':start,'ended_at':end,'return':value,
                                'regime':observation.regime,'observation_hash':observation.content_hash()})
            return self.registry.archive(canonical({'input_artifact':key,
                         'replay_manifest_hash':canonical_hash(result.manifest),'periods':periods}))
        regime_hash = verify("regime_artifact_contents", regime_evidence)

        report = ValidationSuite().validate(result, ledger_path=ledger_path, cost_results=cost_results, cost_artifacts=cost_artifacts, benchmark=benchmark)
        all_checks = list(report.checks)+checks
        report = report.model_copy(update={"checks": all_checks,
            "overall": "RELIABLE" if all(c.verdict == CheckVerdict.PASS for c in all_checks) else "NOT RELIABLE",
            "summary": "Ledger validation plus independent registered-input, model and source-content verification."})
        metrics = performance_metrics(result.equity_curve, inputs.samples_per_year)
        ics = [v for v in result.cross_sectional_ic.values() if v is not None]
        mean_ic = float(np.mean(ics)) if ics else None
        std_ic = float(np.std(ics, ddof=1)) if len(ics) > 1 else None
        outcome = Evaluation(
            trial_id=trial_id, candidate_hash=candidate.content_hash(), inputs_hash=trial.inputs_hash,
            replay_hash=self.registry.archive(canonical(result.model_dump(mode="json"))),
            manifest_hash=self.registry.archive(canonical(result.manifest)),
            validation_hash=self.registry.archive(canonical(report.model_dump(mode="json"))),
            registry_snapshot_hash=self.registry.trial_snapshot(trial_id), completed_at=completed_at,
            status="complete" if result.status == "complete" else "failed",
            validation_reliable=report.overall == "RELIABLE", actual_model_hashes=model_hashes,
            regime_evaluation_hash=regime_hash,
            execution_verification_hash=execution_hash, paired_evaluation_hash=paired_hash,
            observations=len(result.daily_returns), sharpe=metrics.sharpe_ratio, ic=mean_ic,
            icir=mean_ic/std_ic if std_ic and std_ic > 0 else None, max_factor_correlation=corr,
            unadjusted_p_value=block_bootstrap_p_value(paired_returns) if candidate.kind == 'regime' and paired_returns is not None
                else None if candidate.kind == 'regime' else block_bootstrap_p_value(result.daily_returns),
            synthetic=inputs.synthetic or candidate.synthetic,
            diagnostics=tuple(f"{c.name}: {c.explanation}" for c in all_checks if c.verdict != CheckVerdict.PASS))
        self.registry.record_evaluation(outcome)
        return outcome
