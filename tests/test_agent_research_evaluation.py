import json
from datetime import datetime, timezone

import pytest

from src.research.authorities import AgentAuthorityPaths
from src.research.evaluator import ExperimentEvaluator
from src.research.execution_proof import verify_execution
from src.research.registry import RegistryError
from src.research.strategy_experiments import execute_registered_strategy
from tests.test_registered_agent_strategy import agent_experiment as agent_experiment
from tests.test_registered_quant_experiments import experiment as experiment


def test_model_free_evaluation_reproduces_real_advice_receipts_and_no_numeric_metrics(agent_experiment,tmp_path):
    store,registry,plan,_,trial,authorities,_=agent_experiment
    run=execute_registered_strategy(registry,trial.trial_id,store=store,ledger_path=tmp_path/'ledger.db',
                                   artifact_dir=tmp_path/'artifacts',authorities=authorities)
    with pytest.raises(RegistryError,match='authorities required'):
        verify_execution(registry,store,trial.trial_id,run.result,artifact_dir=run.artifact_dir)
    from src.core.backtester import ReplayArtifact
    from src.core.benchmark import BenchmarkRunner
    benchmark=BenchmarkRunner()
    benchmark.cost_sensitivity(inputs=run.replay_kwargs,output_dir=tmp_path/'costs')
    costs=[ReplayArtifact.model_validate(ref) for ref in json.loads((tmp_path/'costs/cost-artifacts.json').read_text())]
    comparison=benchmark.compare_replay(run.result,inputs=run.replay_kwargs,ledger_path=run.ledger_path,
        output_dir=tmp_path/'benchmarks',n_simulations=1000,seed=42)
    outcome=ExperimentEvaluator(registry,store,authorities=authorities).evaluate(trial.trial_id,run.result,
        ledger_path=run.ledger_path,artifact_dir=run.artifact_dir,completed_at=datetime.now(timezone.utc),
        benchmark=comparison,cost_artifacts=costs)
    assert outcome.actual_decision_hashes==run.actual_decision_hashes
    assert outcome.execution_verification_hash and outcome.regime_evaluation_hash
    assert not outcome.actual_model_hashes and outcome.ic is outcome.icir is outcome.max_factor_correlation is None
    assert outcome.synthetic and not outcome.validation_reliable
    assert not any(item.startswith(('registered_inputs:','registered_recipe_reproduction:','authoritative_advice_contents:'))
                   for item in outcome.diagnostics),outcome.diagnostics
    proof=json.loads(registry.blob(outcome.execution_verification_hash))
    assert proof['actual_decision_hashes']==list(outcome.actual_decision_hashes)
    assert len([name for name in proof['artifacts'] if name.startswith('receipts/')])==9
    report=json.loads(registry.blob(outcome.validation_hash))
    checks={c['name']:c for c in report['checks']}
    assert checks['authoritative_advice_history']['verdict']=='PASS'
    assert checks['cost_sensitivity']['verdict']=='PASS'
    assert not {'purged_oos','factor_ic','factor_correlation'} & checks.keys()
    assert registry.active('strategy:'+plan.strategy.name) is None

    from src.execution.ledger import PortfolioLedger
    from src.research.execution_authority import EvaluationExecutionBindings, require_current
    with registry._connection() as db:
        binding_hash = db.execute('SELECT hash FROM evaluation_execution_bindings WHERE trial=?',
                                 (trial.trial_id,)).fetchone()[0]
    binding = EvaluationExecutionBindings.model_validate_json(registry.blob(binding_hash))
    assert len(binding.ledgers) == 1 + len(outcome.actual_decision_hashes)
    assert {v.role for v in binding.ledgers} == {'replay', *('advice:'+key for key in outcome.actual_decision_hashes)}
    require_current(registry,outcome,binding)
    original = next(v for v in binding.ledgers if v.role.startswith('advice:'))
    ledger = PortfolioLedger(original.path)
    try:
        ledger.record_event(original.account_id,'alpaca-submit-attempt:later-unproved-native-history','{}')
        assert ledger.reconcile(original.account_id)['reconciled']
    finally:
        ledger.close()
    with pytest.raises(RegistryError,match='execution'):
        require_current(registry,outcome,binding)
    assert checks['current_execution_authority_binding']['verdict'] == 'PASS'
    assert registry.blob(outcome.replay_hash)  # Frozen historical result remains intact.


def test_operator_authority_config_never_creates_missing_sources(tmp_path):
    cfg=AgentAuthorityPaths(history_store=tmp_path/'history.db',decision_store=tmp_path/'decision.db',
                            budget_store=tmp_path/'budget.db',portfolio_ledgers={'paper:test':tmp_path/'ledger.db'})
    with pytest.raises(ValueError,match='authority file is absent'):
        cfg.open()
    assert not list(tmp_path.iterdir())
