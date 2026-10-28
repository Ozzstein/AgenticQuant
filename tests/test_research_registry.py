"""Research lifecycle invariants; synthetic fixtures do not establish performance."""

from datetime import datetime, timedelta, timezone

import pytest

from src.research.registry import (
    Candidate,
    Evaluation,
    Holdout,
    PromotionPolicy,
    RegistryError,
    ResearchRegistry,
    Trial,
    content_hash,
)

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def registry(tmp_path):
    return ResearchRegistry(tmp_path / "research.sqlite3")


def admitted(registry, key="one", *, synthetic=False, family="family", p_value=0.001,
             origin="custom", parent_ids=()):
    code, environment, dataset, split, inputs = [registry.archive(v.encode()) for v in
                                                ("code", "environment", "dataset", "split", key)]
    payload = ("candidate " + key).encode()
    candidate = Candidate(candidate_id=key, kind="factor", origin=origin, payload_hash=content_hash(payload),
                          code_hash=code, environment_hash=environment, source_evidence_hashes=(dataset,),
                          created_at=T, synthetic=synthetic, parent_ids=parent_ids)
    registry.quarantine(candidate, payload)
    policy = PromotionPolicy()
    registry.archive_contract(policy)
    # Different data manifests represent independent untouched evaluation sets.
    held_data = registry.archive(("heldout " + key).encode())
    registry.register_holdout(Holdout(holdout_id=key, dataset_hash=held_data, starts_at=T-timedelta(days=400),
                                     ends_at=T-timedelta(days=1), registered_at=T))
    trial = Trial(trial_id=key, candidate_id=key, family_id=family, dataset_hash=held_data,
                  split_hash=split, inputs_hash=inputs, policy_hash=policy.content_hash(),
                  registered_at=T, purpose="holdout", holdout_id=key,
                  evaluation_start=T-timedelta(days=400), evaluation_end=T-timedelta(days=1))
    snapshot = registry.register_trial(trial)
    from src.core.backtester import PortfolioReplay
    from src.research.registry import canonical
    from tests.test_core_backtest import replay_inputs

    replay_path = registry.path.parent / ('registry-'+key+'.sqlite3')
    replay = PortfolioReplay(replay_path).run(**replay_inputs())
    artifacts = [registry.archive(canonical(replay.model_dump(mode='json'))),
                 registry.archive(canonical(replay.manifest)),
                 registry.archive((key+'validation').encode()), registry.archive((key+'model').encode())]
    evaluation = Evaluation(trial_id=key, candidate_hash=candidate.content_hash(), inputs_hash=inputs,
                            replay_hash=artifacts[0], manifest_hash=artifacts[1], validation_hash=artifacts[2],
                            registry_snapshot_hash=snapshot, completed_at=T+timedelta(hours=1), status="complete",
                            validation_reliable=True, actual_model_hashes=(artifacts[3],), observations=300,
                            sharpe=1.2, ic=0.04, icir=0.6, max_factor_correlation=0.3,
                            unadjusted_p_value=p_value)
    return candidate, trial, evaluation, policy


def record_bound_evaluation(registry, evaluation):
    """Protocol metrics remain synthetic; account binding uses the actual replay ledger."""
    from src.research.execution_authority import freeze

    candidate = registry.trial(evaluation.trial_id).candidate_id
    binding = freeze(registry,evaluation,ledger_path=registry.path.parent/('registry-'+candidate+'.sqlite3'))
    registry.record_evaluation(evaluation,execution_binding=binding)


def test_immutable_payload_lineage_and_idempotent_trial(registry):
    candidate, trial, outcome, _ = admitted(registry)
    with pytest.raises(RegistryError, match="payload hash"):
        registry.quarantine(candidate, b"wrong")
    assert registry.register_trial(trial) == outcome.registry_snapshot_hash
    with pytest.raises(RegistryError, match="trial ID"):
        registry.register_trial(trial.model_copy(update={"family_id": "hide-prior-trials"}))
    with pytest.raises(RegistryError, match="lineage"):
        registry.record_evaluation(outcome.model_copy(update={"inputs_hash": "f"*64}))
    registry.record_evaluation(outcome)
    registry.record_evaluation(outcome)
    with pytest.raises(RegistryError, match="immutable"):
        registry.record_evaluation(outcome.model_copy(update={"sharpe": 9.0}))


@pytest.mark.parametrize("changes", [
    {"synthetic": True}, {"status": "failed"}, {"validation_reliable": False},
    {"actual_model_hashes": ()}, {"observations": 251}, {"sharpe": None}, {"sharpe": 0.5},
    {"ic": 0.03}, {"icir": 0.5}, {"max_factor_correlation": -0.7}, {"unadjusted_p_value": None},
])
def test_incomplete_simulated_or_failed_research_cannot_activate(registry, changes):
    _, _, outcome, policy = admitted(registry)
    registry.record_evaluation(outcome.model_copy(update=changes))
    with pytest.raises(RegistryError):
        registry.activate("active-factor", "one", policy, expected_generation=0, at=T+timedelta(hours=2))
    assert registry.active("active-factor") is None


def test_synthetic_candidate_never_activates_even_with_passing_claim(registry):
    _, _, outcome, policy = admitted(registry, synthetic=True)
    registry.record_evaluation(outcome)
    with pytest.raises(RegistryError, match="synthetic"):
        registry.activate("factor", "one", policy, expected_generation=0, at=T+timedelta(hours=2))


def test_missing_artifacts_do_not_become_evidence(registry):
    _, _, outcome, _ = admitted(registry)
    with pytest.raises(RegistryError, match="artifact is missing"):
        registry.record_evaluation(outcome.model_copy(update={"validation_hash": "f"*64}))


def test_holdout_consumed_on_attempt_and_cannot_be_renamed(registry):
    _, trial, _, _ = admitted(registry)
    with pytest.raises(RegistryError, match="consumed"):
        registry.register_trial(trial.model_copy(update={"trial_id": "retry-as-new"}))
    with pytest.raises(RegistryError, match="overlaps"):
        registry.register_holdout(Holdout(holdout_id="rename", dataset_hash=trial.dataset_hash,
                                         starts_at=trial.evaluation_start, ends_at=trial.evaluation_end, registered_at=T))
    with pytest.raises(RegistryError, match="touches"):
        registry.register_trial(trial.model_copy(update={"trial_id": "peek", "purpose": "development", "holdout_id": None}))


def test_failed_and_unfinished_trials_increase_multiplicity(registry):
    _, _, first, policy = admitted(registry, p_value=0.03)
    registry.record_evaluation(first)
    admitted(registry, "second")  # Registered attempt without completed outcome still counts.
    with pytest.raises(RegistryError, match="multiple-trial"):
        registry.activate("factor", "one", policy, expected_generation=0, at=T+timedelta(hours=2))


def test_atomic_generation_restart_asof_and_rollback(registry):
    _, _, first, policy = admitted(registry)
    record_bound_evaluation(registry,first)
    a = registry.activate("factor", "one", policy, expected_generation=0, at=T+timedelta(hours=2))
    _, _, second, _ = admitted(registry, "two")
    record_bound_evaluation(registry,second)
    with pytest.raises(RegistryError, match="manifest changed"):
        registry.activate("factor", "two", policy, expected_generation=0, at=T+timedelta(hours=3))
    registry.activate("factor", "two", policy, expected_generation=1, at=T+timedelta(hours=3))
    reopened = ResearchRegistry(registry.path)
    assert reopened.active("factor").candidate_id == "two"
    assert reopened.active("factor", as_of=T+timedelta(hours=2)) == a
    rollback = reopened.rollback("factor", 1, expected_generation=2, at=T+timedelta(hours=4))
    assert rollback.generation == 3 and rollback.candidate_id == "one"
    assert rollback.action == "rollback" and rollback.payload_hash == a.payload_hash
    assert reopened.active("factor", as_of=T) is None


def test_activation_cannot_precede_result(registry):
    _, _, outcome, policy = admitted(registry)
    record_bound_evaluation(registry,outcome)
    with pytest.raises(RegistryError, match="precedes"):
        registry.activate("factor", "one", policy, expected_generation=0, at=T)


def test_development_reader_cannot_expose_holdout_or_unrecorded_outcomes(registry):
    _,trial,outcome,_ = admitted(registry)
    # Refusal occurs even without an outcome: the reader cannot probe whether a
    # holdout succeeded or failed by changing the requested evaluation shape.
    with pytest.raises(RegistryError,match='holdout'):
        registry.development_evaluation(trial.trial_id)
    registry.record_evaluation(outcome)
    with pytest.raises(RegistryError,match='holdout'):
        registry.development_evaluation(trial.trial_id)
    development = trial.model_copy(update={'trial_id':'dev','dataset_hash':registry.archive(b'development-data'),
                                          'purpose':'development','holdout_id':None})
    snapshot = registry.register_trial(development)
    with pytest.raises(RegistryError,match='no recorded outcome'):
        registry.development_evaluation('dev')
    result = outcome.model_copy(update={'trial_id':'dev','registry_snapshot_hash':snapshot,'status':'failed',
                                        'validation_reliable':False})
    registry.record_evaluation(result)
    assert registry.development_evaluation('dev') == result


def test_family_renaming_does_not_reset_multiplicity_or_mutate_admission(registry):
    _,_,first,policy = admitted(registry,p_value=.03)
    assert registry.trial_count_at_admission('one') == 1
    registry.record_evaluation(first)
    admitted(registry,'second',family='new-name')
    assert registry.trial_count_at_admission('one') == 1
    assert registry.family_count('family') == 2
    with pytest.raises(RegistryError,match='multiple-trial'):
        registry.activate('factor','one',policy,expected_generation=0,at=T+timedelta(hours=2))


def test_relabeling_dataset_context_cannot_reopen_a_reserved_security_interval(registry):
    from tests.data_quant_fixtures import archived_snapshot
    snapshot = archived_snapshot(crypto=True,periods=5)
    first = registry.archive_contract(snapshot)
    changed = snapshot.model_copy(update={'context':snapshot.context.model_copy(update={'run_id':'renamed'})})
    second = registry.archive_contract(changed)
    assert first != second
    start,end = snapshot.bars[0].opened_at,snapshot.context.as_of
    registry.register_holdout(Holdout(holdout_id='market',dataset_hash=first,starts_at=start,ends_at=end,registered_at=T))
    with pytest.raises(RegistryError,match='overlaps'):
        registry.register_holdout(Holdout(holdout_id='rename',dataset_hash=second,starts_at=start,ends_at=end,registered_at=T))
