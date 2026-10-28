"""Actual registered recipe reconstruction and same-trial paired ledger proof."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.core.backtester import persist_replay_artifact, split_folds
from src.research.evaluator import ExperimentEvaluator
from src.research.execution_proof import (
    require_reproduced,
    verified_paired_returns,
    verify_execution,
)
from src.research.regime_experiments import (
    RegimeExperimentPlan,
    build_regime_artifact,
    execute_registered_regime,
)
from src.research.registry import Candidate, PromotionPolicy, RegistryError, Trial, canonical
from src.research.strategy_experiments import execute_registered_strategy, materialize_strategy_plan
from src.strategies.specification import baseline_specs
from tests.test_registered_quant_experiments import experiment as experiment


def test_actual_recipe_reproduces_fills_folds_and_source_bound_regime(experiment, tmp_path):
    _, store, registry, _, _, trial = experiment
    execution = execute_registered_strategy(registry,trial.trial_id,store=store,
        ledger_path=tmp_path/'ledger.sqlite3',artifact_dir=tmp_path/'models')
    evaluation = ExperimentEvaluator(registry,store).evaluate(trial.trial_id,execution.result,
        ledger_path=execution.ledger_path,artifact_dir=execution.artifact_dir,
        completed_at=datetime.now(timezone.utc))
    assert evaluation.execution_verification_hash
    assert evaluation.regime_evaluation_hash
    assert not evaluation.validation_reliable  # No holdout/1000 matched ledgers supplied.
    assert not any(c.startswith('registered_inputs:') for c in evaluation.diagnostics)
    forged = execution.result.model_copy(deep=True)
    forged.walk_forward_results[0].predictions_hash = 'f'*64
    with pytest.raises(RegistryError,match='does not reproduce'):
        require_reproduced(forged,execution.result)


def test_paired_baseline_is_reexecuted_and_excess_returns_are_actual(experiment, tmp_path):
    _, store, registry, strategy, artifact, _ = experiment
    baseline_variant = artifact.variant.model_copy(update={'min_regime_days':6})
    baseline_proof = build_regime_artifact(registry,store,strategy.inputs,dataset=artifact.dataset,
        variant=baseline_variant,fitted_at=artifact.fitted_at,inference_times=artifact.inference_times,
        work_dir=tmp_path/'macro-baseline')
    plan = RegimeExperimentPlan(**{**strategy.model_dump(),'strategy':baseline_specs()['mean_reversion'],
        'variant':artifact.variant,'baseline_variant':baseline_variant,
        'baseline_regime_artifact_hash':registry.archive_contract(baseline_proof)})
    name = 'root-paired-regime'
    candidate = Candidate(candidate_id=name,kind='regime',origin='custom',payload_hash=artifact.variant.content_hash(),
        code_hash=registry.archive(b'test implementation'),environment_hash=registry.archive(b'test environment'),
        source_evidence_hashes=(plan.inputs.snapshot.content_hash(),),created_at=plan.inputs.context.as_of,synthetic=True)
    registry.quarantine(candidate,canonical(plan.candidate_payload()))
    data = materialize_strategy_plan(plan,store)
    dates = data.features.index.get_level_values(0)
    folds = split_folds(dates,dates.min().isoformat(),1,0,1,1)
    trial = Trial(trial_id=name,candidate_id=name,family_id='paired-regime',dataset_hash=plan.inputs.snapshot.content_hash(),
        split_hash=registry.archive(canonical(plan.inputs.split_manifest())),inputs_hash=registry.archive_contract(plan),
        policy_hash=registry.archive_contract(PromotionPolicy()),registered_at=plan.inputs.context.as_of,purpose='development',
        evaluation_start=pd.Timestamp(folds[0]['test_start']),evaluation_end=pd.Timestamp(folds[-1]['test_end']))
    registry.register_trial(trial)
    count = registry.family_count('paired-regime')
    actual = execute_registered_regime(registry,name,store=store,ledger_path=tmp_path/'candidate.sqlite3',
                                      artifact_dir=tmp_path/'artifacts')
    baseline = dict(actual.comparisons)['baseline']
    reference = persist_replay_artifact(baseline.result,baseline.ledger_path,tmp_path/'baseline.json')
    proof = verify_execution(registry,store,name,actual.result,artifact_dir=actual.artifact_dir,baseline=reference)
    key = registry.archive(canonical(proof['paired']))
    assert verified_paired_returns(registry,key) == [a-b for a,b in zip(actual.result.daily_returns,baseline.result.daily_returns,strict=True)]
    assert registry.family_count('paired-regime') == count
    tampered = {**proof['paired'],'excess_returns':[1.]*len(actual.result.daily_returns)}
    with pytest.raises(RegistryError,match='contents differ'):
        verified_paired_returns(registry,registry.archive(canonical(tampered)))
    with pytest.raises(RegistryError,match='requires exactly'):
        verify_execution(registry,store,name,actual.result,artifact_dir=actual.artifact_dir)

    outcome = ExperimentEvaluator(registry,store).evaluate(name,actual.result,
        ledger_path=actual.ledger_path,artifact_dir=actual.artifact_dir,
        completed_at=datetime.now(timezone.utc),paired_baseline=reference)
    from src.execution.ledger import PortfolioLedger
    from src.research.execution_authority import EvaluationExecutionBindings, require_current
    with registry._connection() as db:
        binding_hash = db.execute('SELECT hash FROM evaluation_execution_bindings WHERE trial=?',
                                 (name,)).fetchone()[0]
    binding = EvaluationExecutionBindings.model_validate_json(registry.blob(binding_hash))
    assert {v.role for v in binding.ledgers} == {'replay','paired'}
    require_current(registry,outcome,binding)
    original = next(v for v in binding.ledgers if v.role == 'paired')
    ledger = PortfolioLedger(original.path)
    try:
        ledger.record_event(original.account_id,'alpaca-submit-attempt:later-unproved-paired-history','{}')
        assert ledger.reconcile(original.account_id)['reconciled']
    finally:
        ledger.close()
    with pytest.raises(RegistryError,match='execution'):
        require_current(registry,outcome,binding)
