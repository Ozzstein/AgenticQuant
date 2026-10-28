"""Actual registry projections preserve authority, privacy and read-only state."""

import json
import sqlite3
from datetime import timedelta

import pytest

from src.operations.reporting import research_view
from src.research.monitoring import research_snapshot
from src.research.native_recovery import NativeRecoveryEvent
from src.research.registry import Activation, RegistryError, ResearchRegistry, content_hash
from tests.test_native_lifecycle_registry import bound
from tests.test_research_registry import T, admitted, record_bound_evaluation

pytest_plugins = ('tests.test_native_lifecycle_registry',)


def development(registry):
    candidate, trial, evaluation, policy = admitted(registry)
    registry.record_evaluation(evaluation)
    trial = trial.model_copy(update={
        'trial_id': 'development', 'purpose': 'development', 'holdout_id': None,
        'dataset_hash': registry.archive(b'development-only-data'),
    })
    snapshot = registry.register_trial(trial)
    evaluation = evaluation.model_copy(update={
        'trial_id': trial.trial_id, 'registry_snapshot_hash': snapshot,
        'sharpe': -.75, 'status': 'failed', 'validation_reliable': False,
    })
    registry.record_evaluation(evaluation)
    return candidate, trial, evaluation, policy


def test_read_only_monitoring_never_initializes_or_mutates_authority(tmp_path, monkeypatch):
    from src.research.native_mounts import NativeMountAuthority

    missing = tmp_path/'missing'/'registry.db'
    assert research_view(missing)['status'] == 'missing'
    with pytest.raises(FileNotFoundError):
        ResearchRegistry(missing, read_only=True)
    assert not missing.parent.exists()
    registry = ResearchRegistry(tmp_path/'registry.db')
    _, trial, expected, _ = development(registry)
    before = registry.path.read_bytes()
    monkeypatch.setattr(NativeMountAuthority, 'protect',
                        lambda *a, **kw: pytest.fail('monitor registered new host controls'))
    reader = ResearchRegistry(registry.path, read_only=True)
    with reader.read_snapshot(), reader._connection() as db:
        # SQL permissions also deny accidental writes from a future reader.
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            db.execute('CREATE TABLE monitor_accident(id)')
        with pytest.raises(RegistryError, match='read-only'):
            reader.archive(b'forbidden')
        with pytest.raises(RegistryError, match='read-only'):
            reader.register_trial(trial)
        result = research_snapshot(reader)
    assert result['development_outcomes'] == [expected.model_dump(mode='json')]
    assert result['holdouts'] == [{'holdout_id': 'one', 'consumed': True}]
    assert registry.path.read_bytes() == before


def test_snapshot_readers_do_not_mix_concurrent_admissions(tmp_path):
    writer = ResearchRegistry(tmp_path/'registry.db')
    first = writer.archive(b'original')
    reader = ResearchRegistry(writer.path, read_only=True)
    with reader.read_snapshot():
        assert reader.blob(first) == b'original'
        later = writer.archive(b'later')
        with pytest.raises(RegistryError, match='missing'):
            reader.blob(later)
    assert reader.blob(later) == b'later'


def test_relative_read_only_path_remains_fixed_after_chdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    writer = ResearchRegistry('registry.db')
    key = writer.archive(b'exact-source')
    reader = ResearchRegistry('registry.db', read_only=True)
    elsewhere = tmp_path/'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert reader.blob(key) == b'exact-source'
    assert not (elsewhere/'registry.db').exists()


def test_monitor_does_not_read_or_export_holdout_outcome(tmp_path):
    registry = ResearchRegistry(tmp_path/'registry.db')
    _, _, expected, _ = development(registry)
    # A non-active holdout outcome is not a monitoring input at all.
    with registry._connection() as db:
        db.execute('DROP TRIGGER immutable_outcomes_UPDATE')
        db.execute("UPDATE outcomes SET payload=? WHERE trial='one'",
                   ('{"private_holdout_measurement":"do-not-read"}',))
    reader = ResearchRegistry(registry.path, read_only=True)
    statements = []
    with reader.read_snapshot(), reader._connection() as db:
        db.set_trace_callback(statements.append)
        result = research_snapshot(reader)
    assert result['development_outcomes'] == [expected.model_dump(mode='json')]
    assert 'do-not-read' not in json.dumps(result)
    assert not any('FROM outcomes' in sql and "'one'" in sql for sql in statements)


def test_changed_trial_purpose_cannot_expose_holdout_as_development(tmp_path):
    registry = ResearchRegistry(tmp_path/'registry.db')
    _, trial, evaluation, _ = admitted(registry)
    registry.record_evaluation(evaluation)
    changed = trial.model_copy(update={'purpose': 'development', 'holdout_id': None})
    with registry._connection() as db:
        db.execute('DROP TRIGGER immutable_trials_UPDATE')
        db.execute('UPDATE trials SET holdout=NULL,payload=? WHERE id=?',
                   (changed.model_dump_json(), trial.trial_id))
    with pytest.raises(RegistryError, match='immutable admission'):
        research_view(registry.path)


def test_monitor_rejects_tampered_development_and_candidate_records(tmp_path):
    registry = ResearchRegistry(tmp_path/'registry.db')
    candidate, _, evaluation, _ = development(registry)
    with registry._connection() as db:
        db.execute('DROP TRIGGER immutable_outcomes_UPDATE')
        db.execute('UPDATE outcomes SET payload=? WHERE trial=?', (
            evaluation.model_copy(update={'sharpe': 99.0}).model_dump_json(), evaluation.trial_id))
    with pytest.raises(RegistryError, match='identity'):
        research_view(registry.path)
    with registry._connection() as db:
        db.execute('DROP TRIGGER immutable_candidates_UPDATE')
        db.execute('UPDATE candidates SET payload=? WHERE id=?', (
            candidate.model_copy(update={'origin': 'native_rd_agent'}).model_dump_json(), candidate.candidate_id))
    with pytest.raises(RegistryError, match='candidate identity'):
        research_view(registry.path)


def test_current_eligibility_is_separate_from_legacy_native_activation_history(tmp_path):
    registry = ResearchRegistry(tmp_path/'registry.db')
    candidate, trial, evaluation, policy = admitted(registry, origin='native_rd_agent')
    registry.record_evaluation(evaluation)
    # Exact persisted old-generation shape; no current activation authority.
    legacy = Activation(slot='factor', generation=1, candidate_id=candidate.candidate_id,
        candidate_hash=candidate.content_hash(), payload_hash=candidate.payload_hash,
        evaluation_hash=evaluation.content_hash(), policy_hash=policy.content_hash(),
        activated_at=T+timedelta(hours=2), action='promote', previous_generation=0)
    with registry._connection() as db:
        db.execute('INSERT INTO activations VALUES (?,?,?)', ('factor', 1, legacy.model_dump_json()))
        db.execute('INSERT INTO active VALUES (?,?)', ('factor', 1))
    view = research_view(registry.path)
    assert view['activations'] == [legacy.model_dump(mode='json')]
    assert view['active'] == [] and view['blocked_activations'][0]['candidate_id'] == candidate.candidate_id
    assert view['development_outcomes'] == []
    assert not view['live_capital_authorized']


def test_verified_active_generation_retains_resolver_checks(tmp_path):
    registry = ResearchRegistry(tmp_path/'registry.db')
    _, trial, evaluation, policy = admitted(registry)
    record_bound_evaluation(registry,evaluation)
    activation = registry.activate('factor', trial.trial_id, policy,
                                   expected_generation=0, at=T+timedelta(hours=2))
    view = research_view(registry.path)
    assert view['active'] == [activation.model_dump(mode='json')]
    assert view['blocked_activations'] == []


def test_native_absence_keeps_uncertainty_and_never_exports_private_paths(authority):
    registry, launch = authority
    container = bound(registry, launch)
    registry.record_native_container(container.model_dump(mode='json'))
    event = NativeRecoveryEvent(run_id=launch.run_id, request_hash=launch.request_hash,
        discovery_exposure_hash=launch.discovery_exposure_hash, sequence=1,
        observed_at=container.bound_at, launch_hash=launch.content_hash(),
        container_binding_hash=container.content_hash(), observation='absent',
        reason='auto_removed_exit_unknown', cost_state='uncertain',
        inspection_hash=registry.archive(b'private inspection fixture'))
    registry.append_native_recovery(event.model_dump(mode='json'))
    before = content_hash(registry.path.read_bytes())
    view = research_view(registry.path)
    native = view['native_runs'][0]
    assert native['launch_recorded'] and native['container_bound']
    assert not native['result_recorded'] and native['result_status'] is None
    assert native['latest_recovery']['observation'] == 'absent'
    assert native['latest_recovery']['cost_state'] == 'uncertain'
    assert native['exposure_retained'] and native['activation'] == 'quarantined'
    assert native['internal_search_trials'] is None
    rendered = json.dumps(view)
    for private in (launch.gateway_token_path, launch.supervisor_root, 'private inspection fixture',
                    'mounts', 'gateway_receipt_hash', 'files', container.container_id):
        assert private not in rendered
    assert content_hash(registry.path.read_bytes()) == before


def test_corrupt_native_authority_is_explicitly_unverifiable(authority):
    registry, launch = authority
    container = bound(registry, launch)
    registry.record_native_container(container.model_dump(mode='json'))
    with registry._connection() as db:
        db.execute('DROP TRIGGER immutable_blobs_UPDATE')
        db.execute('UPDATE blobs SET content=? WHERE hash=?',
                   (b'private-corrupt-path', container.identity_hash))
    native = research_view(registry.path)['native_runs'][0]
    assert native['status'] == 'unverifiable'
    assert 'private-corrupt-path' not in json.dumps(native)
    assert native['activation'] == 'quarantined'
