"""Real model/replay proof binding. Fixture results must remain in quarantine."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.core.factor_backtester import FactorBacktester
from src.data.evidence.store import EvidenceStore
from src.domain.market import FeatureSpec
from src.research.evaluator import ExperimentEvaluator, ExperimentPlan, block_bootstrap_p_value
from src.research.registry import RegistryError, ResearchRegistry
from src.research.runner import ResearchRunner
from src.utils.exceptions import BacktestError
from src.utils.schemas import FactorDefinition
from tests.test_research_inputs import research_inputs


def admitted(tmp_path):
    registry = ResearchRegistry(tmp_path/'registry.db')
    store = EvidenceStore(tmp_path/'evidence.db')
    plan = ExperimentPlan(inputs=research_inputs(),features=(FeatureSpec(name='m',expression='$close/Ref($close,3)-1'),))
    runner = ResearchRunner(registry,store)
    now = datetime.now(timezone.utc)
    trial = runner.register(plan=plan,candidate_id='candidate',trial_id='trial',family_id='all-factor-research',
                            kind='factor',origin='custom',code=b'bounded-expression-v1',environment=b'lock-fixture',at=now,
                            evaluation_start=plan.inputs.account.opened_at,evaluation_end=plan.inputs.context.as_of+timedelta(days=1))
    return runner,plan,trial


def test_actual_outcome_cannot_promote_without_source_cost_benchmark_holdout(tmp_path):
    runner,plan,trial = admitted(tmp_path)
    result = FactorBacktester().run(FactorDefinition(name='m',expression=plan.features[0].expression),
                                    inputs=plan.inputs,ledger_path=tmp_path/'ledger.db',artifact_dir=tmp_path/'models')
    evaluation = ExperimentEvaluator(runner.registry,runner.evidence).evaluate(
        trial.trial_id,result,ledger_path=tmp_path/'ledger.db',artifact_dir=tmp_path/'models',
        completed_at=datetime.now(timezone.utc))
    assert not evaluation.validation_reliable and evaluation.synthetic
    assert evaluation.actual_model_hashes
    assert evaluation.sharpe == result.metrics.sharpe_ratio
    assert any('source_contents' in item for item in evaluation.diagnostics)
    assert any('trial_registry_contents' in item for item in evaluation.diagnostics)
    archived = json.loads(runner.registry.blob(evaluation.validation_hash))
    assert archived['overall'] == 'NOT RELIABLE'
    assert next(c for c in archived['checks'] if c['name']=='registered_inputs')['verdict']=='PASS'
    assert runner.registry.resolve('factor:m',payload_hash=runner.registry.candidate('candidate').payload_hash,
                                   as_of=datetime.now(timezone.utc)) is None


def test_actual_model_tamper_blocks_artifact_evidence(tmp_path):
    runner,plan,trial = admitted(tmp_path)
    result = FactorBacktester().run(FactorDefinition(name='m',expression=plan.features[0].expression),
                                    inputs=plan.inputs,ledger_path=tmp_path/'ledger.db',artifact_dir=tmp_path/'models')
    model = next((tmp_path/'models').glob('fold-*.joblib'))
    model.write_bytes(model.read_bytes()+b'tampered')
    evaluation = ExperimentEvaluator(runner.registry,runner.evidence).evaluate(
        trial.trial_id,result,ledger_path=tmp_path/'ledger.db',artifact_dir=tmp_path/'models',
        completed_at=datetime.now(timezone.utc))
    assert not evaluation.validation_reliable and not evaluation.actual_model_hashes
    assert any('fitted model bytes differ' in item for item in evaluation.diagnostics)


def test_registered_failure_counts_and_prevents_silent_retraining(tmp_path):
    runner,_,trial = admitted(tmp_path)
    # Invalid bounded experiment size fails after durable admission; it cannot
    # erase the attempted trial or silently re-enter the same output directory.
    with pytest.raises(BacktestError):
        runner.run(trial.trial_id,output_dir=tmp_path/'run',simulations=0)
    assert (tmp_path/'run/failure.json').exists()
    assert runner.registry.family_count(trial.family_id)==1
    with pytest.raises(RegistryError,match='already exists'):
        runner.run(trial.trial_id,output_dir=tmp_path/'run')


def test_centered_temporal_bootstrap_preserves_undefined_and_null():
    assert block_bootstrap_p_value([.01]*10) is None
    assert block_bootstrap_p_value([.01]*100) is None
    assert block_bootstrap_p_value([-.01]*100)==1
    rng=np.random.default_rng(123)
    assert block_bootstrap_p_value(list(rng.normal(.01,.01,300))) < .01
    assert block_bootstrap_p_value(list(rng.normal(0,.01,300))) > .01
