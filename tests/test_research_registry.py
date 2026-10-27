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


def admitted(registry, key="one", *, synthetic=False, family="family", p_value=0.001):
    code, environment, dataset, split, inputs = [registry.archive(v.encode()) for v in
                                                ("code", "environment", "dataset", "split", key)]
    payload = ("candidate " + key).encode()
    candidate = Candidate(candidate_id=key, kind="factor", origin="custom", payload_hash=content_hash(payload),
                          code_hash=code, environment_hash=environment, source_evidence_hashes=(dataset,),
                          created_at=T, synthetic=synthetic)
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
    artifacts = [registry.archive((key + suffix).encode()) for suffix in ("replay", "manifest", "validation", "model")]
    evaluation = Evaluation(trial_id=key, candidate_hash=candidate.content_hash(), inputs_hash=inputs,
                            replay_hash=artifacts[0], manifest_hash=artifacts[1], validation_hash=artifacts[2],
                            registry_snapshot_hash=snapshot, completed_at=T+timedelta(hours=1), status="complete",
                            validation_reliable=True, actual_model_hashes=(artifacts[3],), observations=300,
                            sharpe=1.2, ic=0.04, icir=0.6, max_factor_correlation=0.3,
                            unadjusted_p_value=p_value)
    return candidate, trial, evaluation, policy


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
    registry.record_evaluation(first)
    a = registry.activate("factor", "one", policy, expected_generation=0, at=T+timedelta(hours=2))
    _, _, second, _ = admitted(registry, "two")
    registry.record_evaluation(second)
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
    registry.record_evaluation(outcome)
    with pytest.raises(RegistryError, match="precedes"):
        registry.activate("factor", "one", policy, expected_generation=0, at=T)
