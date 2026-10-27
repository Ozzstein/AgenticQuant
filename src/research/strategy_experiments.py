"""Preregistered strategy trials over real models and holdings-aware ledger replay.

This module executes admitted recipes. It cannot record a passing evaluation,
activate a candidate, manufacture returns, or place a broker order.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pydantic import model_validator
from src.research.datasets import ResearchInputs
from src.research.evidence import verify_provenance
from src.research.registry import RegistryError, ResearchRegistry

from src.core.backtester import (
    FoldEvidence,
    PortfolioReplay,
    ReplayResult,
    performance_metrics,
    split_folds,
    write_immutable,
)
from src.core.model_zoo import ModelWrapper
from src.core.walk_forward import cross_sectional_ic, frame_hash
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.data.features.contextual import NumericEvidenceBinding, materialize_evidence_fields
from src.data.features.dataset import FeatureDataset, feature_dataset
from src.domain.common import Sha256, VersionedContract
from src.domain.market import FeatureSpec
from src.research.regime_experiments import (
    CausalRegimeArtifact,
    FrozenEvidence,
    recompute_regime_artifact,
    verify_frozen_evidence,
)
from src.strategies.runner import evaluate_strategy, rebalance_due
from src.strategies.specification import StrategySpec


class StrategyExperimentPlan(VersionedContract):
    inputs: ResearchInputs
    strategy: StrategySpec
    reference_features: tuple[FeatureSpec, ...] = ()
    numeric_bindings: tuple[NumericEvidenceBinding, ...] = ()
    numeric_sources: tuple[FrozenEvidence, ...] = ()
    model_parameters: tuple[tuple[str, int | float | str | bool | None], ...] = ()
    regime_artifact_hash: Sha256
    provenance: tuple[tuple[str, str], ...]

    @model_validator(mode='after')
    def coherent(self):
        if bool(self.numeric_bindings) != bool(self.numeric_sources):
            raise ValueError('contextual bindings require exact frozen evidence envelopes')
        if len(dict(self.model_parameters)) != len(self.model_parameters) or len(dict(self.provenance)) != len(self.provenance):
            raise ValueError('duplicate model parameter/provenance key')
        if self.strategy.model_name is None:
            raise ValueError('model-free research requires archived agent decision evaluation; numeric trial cannot invent votes')
        return self

    @property
    def features(self):
        return self.strategy.features

    def candidate_payload(self):
        return self.strategy.model_dump(mode='json')

    def model_factory(self):
        from src.utils.config import AppConfig, QlibConfig
        config = AppConfig(_env_file=None).model_copy(update={
            'qlib':QlibConfig(model_name=self.strategy.model_name,model_params=dict(self.model_parameters))})
        return ModelWrapper(self.strategy.model_name, config)


def materialize_strategy_plan(plan: StrategyExperimentPlan, store: EvidenceStore, *,
                              specs: tuple[FeatureSpec, ...] | None = None) -> FeatureDataset:
    inputs = plan.inputs
    contextual = None
    if plan.numeric_bindings:
        records = verify_frozen_evidence(store, plan.numeric_sources, as_of=inputs.context.as_of)
        index = pd.MultiIndex.from_tuples(sorted((b.closed_at,b.instrument.instrument_id) for b in inputs.snapshot.bars),
                                          names=['datetime','instrument'])
        contextual = materialize_evidence_fields(store,index,inputs.context,plan.numeric_bindings,
                                                  allowed_evidence_ids=frozenset(records))
    data = feature_dataset(inputs.snapshot, specs or plan.features, horizon=inputs.label_horizon,
                           evidence_fields=contextual)
    dates = data.features.index.get_level_values(0)
    members = {}
    for at in dates.unique():
        choices = [u for u in inputs.universes if u.universe_version == inputs.universe_version
                   and u.effective_at <= at and u.available_at <= at]
        if not choices:
            raise RegistryError('documented historical universe missing')
        selected = max(choices,key=lambda u:(u.effective_at,u.available_at))
        members[at] = set(selected.instrument_ids)
    mask = np.asarray([key in members[at] for at,key in data.features.index])
    # Inference eligibility depends on inputs only. Missing future outcomes must
    # not retrospectively remove decisions; training masks purge unknown labels.
    mask &= np.isfinite(data.features.to_numpy()).all(axis=1) & (data.feature_available_at <= dates)
    return FeatureDataset(data.features.loc[mask],data.labels.loc[mask],data.feature_available_at.loc[mask],
                          data.label_available_at.loc[mask],data.snapshot_hash,data.feature_spec_hash)


@dataclass(frozen=True)
class ExecutedExperiment:
    result: ReplayResult
    ledger_path: Path
    artifact_dir: Path
    artifacts: tuple[tuple[str, bytes], ...]
    inputs_hash: str
    candidate_hash: str
    decision_provider: Callable
    replay_kwargs: dict
    comparisons: tuple[tuple[str, ExecutedExperiment], ...] = ()


def _fit_folds(plan, data, trial, artifact_dir):
    inputs = plan.inputs
    dates = data.features.index.get_level_values(0)
    if dates.empty:
        raise RegistryError('no eligible training observations')
    folds = split_folds(dates, dates.min().isoformat(), inputs.walk_months,inputs.embargo_days,
                        inputs.validation_months,inputs.initial_train_months)
    if not folds:
        raise RegistryError('no actual OOS folds')
    models, records, all_predictions = {}, [], []
    for number, fold in enumerate(folds):
        bounds = {key:pd.Timestamp(value) for key,value in fold.items() if key != 'complete_window'}
        if bounds['test_start'] < trial.evaluation_start or bounds['test_end'] > trial.evaluation_end:
            raise RegistryError('actual OOS fold exceeds registered evaluation interval')
        train_base = (dates >= bounds['train_start']) & (dates < bounds['train_end'])
        valid_base = (dates >= bounds['validation_start']) & (dates < bounds['validation_end'])
        train = train_base & (data.label_available_at < bounds['train_end'])
        valid = valid_base & (data.label_available_at < bounds['validation_end'])
        # The terminal close has no subsequent executable open in this frozen
        # replay. Truncate on execution coverage, never on future label values.
        test = (dates >= bounds['test_start']) & (dates < bounds['test_end']) & (dates < max(b.opened_at for b in inputs.bars))
        if not train.any() or not valid.any() or not test.any():
            raise RegistryError('empty train/validation/OOS window after causal purging')
        model = plan.model_factory()
        fit_context = inputs.context.model_copy(update={'as_of':bounds['train_end'].to_pydatetime()})
        model.train(data.features.loc[train],data.labels.loc[train],context=fit_context,
                    feature_available_at=data.feature_available_at.loc[train],
                    label_available_at=data.label_available_at.loc[train],
                    feature_spec_hash=data.feature_spec_hash,data_snapshot_hash=data.snapshot_hash)
        path = artifact_dir/f'fold-{number}.joblib'
        model.save(str(path))
        predictions = model.predict(data.features.loc[test],context=inputs.context,
                                    feature_available_at=data.feature_available_at.loc[test])
        all_predictions.append(predictions)
        for date in dates[test].unique():
            models[date] = model
        records.append(FoldEvidence(fold_id=number,**{key:value for key,value in fold.items() if key != 'complete_window'},
            train_rows=int(train.sum()),validation_rows=int(valid.sum()),test_rows=int(test.sum()),
            purged_rows=int(train_base.sum()+valid_base.sum()-train.sum()-valid.sum()),embargo_days=inputs.embargo_days,
            model_version=digest(path.read_bytes()),train_hash=frame_hash(data.features.loc[train]),
            validation_hash=frame_hash(data.features.loc[valid]),predictions_hash=frame_hash(predictions),
            label_end_max=data.label_available_at.loc[train].max().isoformat(),
            validation_label_end_max=data.label_available_at.loc[valid].max().isoformat(),
            feature_importance={key:float(value) for key,value in model.get_feature_importance().items()}))
    return models, records, pd.concat(all_predictions).sort_index(), any(not f["complete_window"] for f in folds)


def _execute_registered(registry, trial_id, *, plan_type, kind, store, ledger_path, artifact_dir, comparison=False):
    trial = registry.trial(trial_id)
    candidate = registry.candidate(trial.candidate_id)
    plan = plan_type.model_validate_json(registry.blob(trial.inputs_hash))
    inputs = plan.inputs
    if candidate.kind != kind or registry.blob(candidate.payload_hash) != canonical_bytes(plan.candidate_payload()):
        raise RegistryError('registered candidate differs from executable recipe')
    if trial.dataset_hash != inputs.snapshot.content_hash() or registry.blob(trial.split_hash) != canonical_bytes(inputs.split_manifest()):
        raise RegistryError('registered dataset/split differs')
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    admission = artifact_dir/'admission.json'
    if admission.exists():
        raise RegistryError('experiment already admitted in this output namespace')
    write_immutable(admission,registry.blob(trial.inputs_hash).decode())
    verify_provenance(registry,store,inputs,dict(plan.provenance))
    if comparison and kind != 'regime':
        raise RegistryError('only a preregistered paired regime baseline may use comparison execution')
    regime_key = plan.baseline_regime_artifact_hash if comparison else plan.regime_artifact_hash
    regime = CausalRegimeArtifact.model_validate_json(registry.blob(regime_key))
    expected_variant = plan.baseline_variant if comparison else getattr(plan,'variant',None)
    if kind == 'regime' and regime.variant != expected_variant:
        raise RegistryError('HMM candidate differs from actual causal regime proof')
    observations = recompute_regime_artifact(registry,store,regime,inputs,work_dir=artifact_dir/'regime-proof')
    data = materialize_strategy_plan(plan,store)
    if data.snapshot_hash not in inputs.context.data_snapshot_ids:
        raise RegistryError('research context must bind the exact materialized feature data snapshot')
    models, folds, predictions, incomplete = _fit_folds(plan,data,trial,artifact_dir)
    schedule = tuple(sorted(models))
    regimes = {pd.Timestamp(o.context.as_of):o for o in observations}
    if set(schedule)-regimes.keys():
        raise RegistryError('actual causal regime observation missing at strategy decision cutoff')
    identities = {b.instrument.instrument_id:b.instrument for b in inputs.bars}
    previous_rebalance = {}
    last = None
    for at in schedule:
        previous_rebalance[at] = last
        if rebalance_due(plan.strategy,at,last):
            last = at
    opportunity_rows = []
    for at in schedule:
        universe = max((u for u in inputs.universes if u.universe_version == inputs.universe_version
                        and u.effective_at <= at and u.available_at <= at),
                       key=lambda u:(u.effective_at,u.available_at))
        frame = data.features.xs(at,level='datetime')
        eligible = sorted(key for key in universe.instrument_ids if key in frame.index
            and identities[key].asset_class in plan.strategy.asset_classes and frame.loc[key,'adv_20'] >= plan.strategy.min_adv)
        opportunity_rows.append({'at':at.isoformat(),'universe_hash':universe.content_hash(),
            'universe_version':universe.universe_version,'model_payload_hash':models[at].artifact.payload_hash,
            'model_artifact_hash':models[at].artifact.content_hash(),'regime_observation_hash':regimes[at].content_hash(),
            'rebalance_due':rebalance_due(plan.strategy,at,previous_rebalance[at]),
            'eligible':[{'instrument_id':key,'prediction':float(predictions.loc[(at,key)])} for key in eligible],
            'trace_link':{'account_id':inputs.account.account_id,'run_id':inputs.context.run_id,'at':at.isoformat()}})
    opportunities = {'schema_version':'1.0','implementation':'registered-strategy-opportunities-1',
        'inputs_hash':inputs.content_hash(),'plan_hash':plan.content_hash(),'feature_data_hash':frame_hash(data.features),
        'feature_spec_hash':data.feature_spec_hash,'cadence':plan.strategy.cadence,
        'decision_times':[at.isoformat() for at in schedule],'rows':opportunity_rows}
    opportunity_path = (artifact_dir/'opportunities.json').resolve()
    opportunity_bytes = canonical_bytes(opportunities)
    write_immutable(opportunity_path,opportunity_bytes.decode())
    recipe = digest(canonical_bytes({'trial':trial.content_hash(),'plan':plan.content_hash(),
        'models':[fold.model_version for fold in folds], 'opportunities_hash':digest(opportunity_bytes), 'comparison':comparison, 'implementation':'registered-strategy-ledger-1'}))

    def decision_provider(at, snapshot):
        date = pd.Timestamp(at)
        if date not in models:
            raise RegistryError('callback date outside frozen decision schedule')
        if snapshot.as_of != at or snapshot.nav <= 0:
            raise RegistryError('actual positive NAV at callback cutoff required')
        holdings = pd.Series({p.instrument.instrument_id:float(p.market_value/snapshot.nav)
                              for p in snapshot.positions if p.quantity > 0},dtype=float)
        ages = {}
        for position in snapshot.positions:
            if position.quantity <= 0:
                continue
            if position.opened_at is None:
                if plan.strategy.max_holding_sessions is not None:
                    raise RegistryError('actual position opening time is unknown')
                continue
            sessions = {bar.session for bar in inputs.bars if bar.instrument.instrument_id == position.instrument.instrument_id
                        and position.opened_at <= bar.opened_at and bar.closed_at <= at}
            ages[position.instrument.instrument_id] = len(sessions)
        eligible = [u for u in inputs.universes if u.universe_version == inputs.universe_version
                    and u.effective_at <= at and u.available_at <= at]
        if not eligible:
            raise RegistryError('actual historical universe missing at decision cutoff')
        universe = max(eligible,key=lambda u:(u.effective_at,u.available_at))
        observation = regimes[date]
        if observation.status != 'available':
            return None  # Missing regime blocks new decisions, never implies liquidation.
        callback_context = inputs.context.model_copy(update={'as_of':at,'account_id':snapshot.account_id})
        output = evaluate_strategy(plan.strategy,data,callback_context,
            universe=universe,instruments=tuple(identities.values()),holdings=holdings,signal_at=at,
            regime=observation,model=models[date],holding_sessions=pd.Series(ages,dtype=float),
            last_rebalanced_at=previous_rebalance[date],research_registry=registry if kind == 'strategy' else None,
            trial_id=trial_id if kind == 'strategy' else None)
        # Pure callback: actual fill-time ages/holdings come from the supplied ledger snapshot.
        # Holding-only output means no orders; converting drifting weights could otherwise trade.
        if output.action.isin(['HOLD','ABSTAIN','VETO']).all():
            return None
        from src.execution.routing import target_from_strategy_frame
        return target_from_strategy_frame(output,callback_context,
            tuple(identities.values()),current_snapshot=snapshot,
            target_id=digest(canonical_bytes({'recipe':recipe,'at':date.isoformat(),'snapshot':snapshot.content_hash()})))


    refs = dict(plan.provenance)
    correlation = None
    if plan.reference_features:
        references = materialize_strategy_plan(plan,store,specs=plan.reference_features)
        values = [data.features.loc[predictions.index,col].corr(references.features[ref],method='spearman')
                  for col in data.features for ref in references.features]
        values = [abs(float(value)) for value in values if np.isfinite(value)]
        correlation = max(values) if values else None
    provenance = {**refs,'regime_evaluation_artifact':regime_key,
        'opportunities_artifact':{'path':str(opportunity_path),'sha256':digest(opportunity_bytes)},
        'features_hash':frame_hash(data.features),'labels_hash':frame_hash(data.labels),
        'feature_availability_hash':frame_hash(data.feature_available_at),'label_end_hash':frame_hash(data.label_available_at),
        'strategy_hash':plan.strategy.content_hash(),'max_existing_factor_correlation':correlation,
        'folds':[fold.model_dump(mode='json') for fold in folds],'walk_months':inputs.walk_months,
        'universe_version':inputs.universe_version,'mode':'expanding',
        'trial_registry':{'artifact_id':registry.trial_snapshot(trial_id),
            'trials':registry.trial_count_at_admission(trial_id),'untouched_holdout_id':trial.holdout_id}}
    bars = [b for b in inputs.bars if b.closed_at >= schedule[0]]
    start = min(b.opened_at for b in bars)
    kwargs = {**inputs.replay_kwargs(),'bars':bars,
        'account':inputs.account.model_copy(update={'opened_at':start-pd.Timedelta(microseconds=1)}),
        'actions':[a for a in inputs.actions if a.effective_at >= start],
        'decision_provider':decision_provider,'decision_times':[t.to_pydatetime() for t in schedule],
        'decision_recipe_hash':recipe,'provenance':provenance}
    result = PortfolioReplay(ledger_path).run(**kwargs)
    for fold in folds:
        observed = [s for s in result.snapshots if pd.Timestamp(fold.test_start) <= s.as_of < pd.Timestamp(fold.test_end)]
        preceding = [s for s in result.snapshots if s.as_of < pd.Timestamp(fold.test_start)]
        if observed and preceding:
            fold.oos_sharpe = performance_metrics([float(preceding[-1].nav),*[float(s.nav) for s in observed]],inputs.samples_per_year).sharpe_ratio
    result = result.model_copy(update={'walk_forward_results':folds,
        'cross_sectional_ic':cross_sectional_ic(predictions,data.labels.loc[predictions.index]),
        'status':'incomplete' if incomplete else result.status,
        'diagnostics':[*result.diagnostics,*(['final OOS window is incomplete; promotion prohibited'] if incomplete else [])]})
    from src.execution.ledger import PortfolioLedger
    ledger = PortfolioLedger(ledger_path)
    try:
        row = ledger.db.execute('SELECT payload FROM runs WHERE account=? AND id=?',
                                (result.account_id,result.run_id)).fetchone()
        payload = json.loads(row[0])
        payload.update(status=result.status,result=result.model_dump(mode='json'))
        ledger.save_run(result.account_id,result.run_id,payload)
    finally:
        ledger.close()
    write_immutable(artifact_dir/'replay.json',result.model_dump_json(indent=2))
    artifacts = tuple((str(path.relative_to(artifact_dir)),path.read_bytes()) for path in sorted(artifact_dir.glob('fold-*')))
    artifacts += (('opportunities.json',opportunity_bytes),)
    return ExecutedExperiment(result,Path(ledger_path),artifact_dir,artifacts,trial.inputs_hash,
                              candidate.content_hash(),decision_provider,kwargs)


def execute_registered_strategy(registry: ResearchRegistry, trial_id: str, *, store: EvidenceStore,
                                ledger_path: str | Path, artifact_dir: str | Path) -> ExecutedExperiment:
    return _execute_registered(registry,trial_id,plan_type=StrategyExperimentPlan,kind='strategy',store=store,
                               ledger_path=ledger_path,artifact_dir=artifact_dir)
