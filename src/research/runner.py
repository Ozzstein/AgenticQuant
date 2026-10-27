"""Registered custom factor/model experiments using actual fitted OOS portfolios."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.backtester import ReplayArtifact, split_folds, write_immutable
from src.core.benchmark import BenchmarkRunner
from src.core.walk_forward import WalkForwardRunner, reconstruct_oos_inputs
from src.data.evidence.store import EvidenceStore
from src.research.evaluator import ExperimentEvaluator, ExperimentPlan
from src.research.registry import (
    Candidate,
    Evaluation,
    PromotionPolicy,
    RegistryError,
    ResearchRegistry,
    Trial,
    canonical,
    content_hash,
)


class ResearchRunner:
    def __init__(self, registry: ResearchRegistry, evidence: EvidenceStore, config=None):
        self.registry, self.evidence, self.config = registry, evidence, config

    def register(self, *, plan: ExperimentPlan, candidate_id: str, trial_id: str, family_id: str,
                 kind: str, origin: str, code: bytes, environment: bytes, at: datetime,
                 evaluation_start: datetime, evaluation_end: datetime, holdout_id: str | None = None,
                 policy: PromotionPolicy | None = None) -> Trial:
        if plan.model_name not in ('Linear','LightGBM','CatBoost','XGBoost','Ensemble'):
            raise ValueError('custom model must use an actual supported fitted model family')
        if kind not in ('factor','model','semantic_question'):
            raise ValueError('strategy/regime candidates require their owned executable experiment adapter')
        policy = policy or PromotionPolicy()
        payload = canonical(plan.candidate_payload())
        snapshot_hash = self.registry.archive_contract(plan.inputs.snapshot)
        candidate = Candidate(candidate_id=candidate_id,kind=kind,origin=origin,
            payload_hash=content_hash(payload),code_hash=self.registry.archive(code),
            environment_hash=self.registry.archive(environment),source_evidence_hashes=(snapshot_hash,),
            created_at=at,synthetic=plan.inputs.synthetic)
        self.registry.quarantine(candidate,payload)
        trial = Trial(trial_id=trial_id,candidate_id=candidate_id,family_id=family_id,
            dataset_hash=snapshot_hash,split_hash=self.registry.archive(canonical(plan.inputs.split_manifest())),
            inputs_hash=self.registry.archive_contract(plan),policy_hash=self.registry.archive_contract(policy),
            registered_at=at,purpose='holdout' if holdout_id else 'development',holdout_id=holdout_id,
            evaluation_start=evaluation_start,evaluation_end=evaluation_end)
        self.registry.register_trial(trial)
        return trial

    def run(self, trial_id: str, *, output_dir: str | Path, simulations: int = 1000) -> Evaluation:
        trial = self.registry.trial(trial_id)
        plan = ExperimentPlan.model_validate_json(self.registry.blob(trial.inputs_hash))
        inputs = plan.inputs
        output = Path(output_dir)
        output.mkdir(parents=True,exist_ok=True)
        # A run may be examined after failure; it cannot silently restart the
        # same consumed holdout with new models or mutable output bytes.
        admission = output/'admission.json'
        if admission.exists():
            raise RegistryError('experiment output already exists; inspect/recover the registered run instead of retraining')
        write_immutable(admission,canonical(trial.model_dump(mode='json')).decode())
        try:
            data = inputs.materialize(plan.features)
            refs = dict(plan.provenance)
            folds = split_folds(data.features.index.get_level_values(0),
                                data.features.index.get_level_values(0).min().isoformat(),
                                inputs.walk_months,inputs.embargo_days,inputs.validation_months,inputs.initial_train_months)
            if not folds:
                raise ValueError('no actual OOS folds')
            oos_start = pd.Timestamp(folds[0]['test_start'])
            oos_end = pd.Timestamp(folds[-1]['test_end'])
            if oos_start < trial.evaluation_start or oos_end > trial.evaluation_end:
                raise RegistryError('actual OOS fold range exceeds registered evaluation interval')
            corr = None
            if plan.reference_features:
                reference = inputs.materialize(plan.reference_features)
                mask = data.features.index.get_level_values(0) >= oos_start
                values = [data.features.loc[mask,col].corr(reference.features[ref],method='spearman')
                          for col in data.features for ref in reference.features]
                values = [abs(float(v)) for v in values if np.isfinite(v)]
                corr = max(values) if values else None
            provenance = {**refs,'max_existing_factor_correlation':corr,
                'trial_registry':{'artifact_id':self.registry.trial_snapshot(trial_id),
                                  'trials':self.registry.trial_count_at_admission(trial_id),
                                  'untouched_holdout_id':trial.holdout_id}}
            result = WalkForwardRunner(self.config).run(
                data.features,data.labels,model_name=plan.model_name,
                model_factory=plan.model_factory,
                label_end_times=data.label_available_at,feature_available_at=data.feature_available_at,
                walk_months=inputs.walk_months,embargo_days=inputs.embargo_days,
                validation_months=inputs.validation_months,initial_train_months=inputs.initial_train_months,
                universe_version=inputs.universe_version,ledger_path=output/'ledger.sqlite3',
                artifact_dir=output/'models',provenance=provenance,**inputs.replay_kwargs())
            write_immutable(output/'replay.json',result.model_dump_json(indent=2))
            replay_inputs = reconstruct_oos_inputs(result,inputs.replay_kwargs())
            benchmark = BenchmarkRunner(self.config)
            benchmark.cost_sensitivity(inputs=replay_inputs,output_dir=output/'costs')
            costs = [ReplayArtifact.model_validate(r) for r in json.loads((output/'costs/cost-artifacts.json').read_text())]
            comparison = benchmark.compare_replay(result,inputs=replay_inputs,output_dir=output/'benchmarks',n_simulations=simulations)
            write_immutable(output/'comparison.json',comparison.model_dump_json(indent=2))
            evaluation = ExperimentEvaluator(self.registry,self.evidence).evaluate(
                trial_id,result,ledger_path=output/'ledger.sqlite3',artifact_dir=output/'models',
                completed_at=datetime.now(timezone.utc),cost_artifacts=costs,benchmark=comparison)
            write_immutable(output/'evaluation.json',evaluation.model_dump_json(indent=2))
            return evaluation
        except Exception as exc:
            failure = {'trial_id':trial_id,'inputs_hash':trial.inputs_hash,'status':'failed',
                       'error_type':type(exc).__name__,'error':str(exc),'finished_at':datetime.now(timezone.utc).isoformat()}
            write_immutable(output/'failure.json',canonical(failure).decode())
            # Admission was committed before execution, so every failed attempt
            # remains in the multiplicity count and consumes its holdout.
            raise
