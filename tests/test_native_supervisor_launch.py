"""Original native launch uses the same durable proof as orphan recovery."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from scripts.run_rd_agent import app
from src.research.native import NativeSupervisor
from src.research.native_recovery import NativeRecoveryService
from src.research.registry import ResearchRegistry
from tests.native_launch_fixtures import install_native_boundary
from tests.test_native_discovery_exposure import mounted_request


@pytest.fixture
def launch(tmp_path, monkeypatch):
    request, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    boundary = install_native_boundary(monkeypatch, supervisor, request, registry)
    boundary.token = tmp_path / "token"
    boundary.token.write_text("synthetic-scoped-token")
    boundary.token.chmod(0o600)
    return SimpleNamespace(request=request, registry=registry, supervisor=supervisor,
                           boundary=boundary, work=supervisor.root / request.run_id)


def execute(launch):
    return launch.supervisor.run(launch.request, registry=launch.registry,
                                gateway_token_path=launch.boundary.token)


def test_original_container_is_bound_before_start_and_success_is_audited(launch):
    result = execute(launch)
    calls = launch.boundary.calls
    assert next(i for i, c in enumerate(calls) if c[0] == "create") < next(
        i for i, c in enumerate(calls) if c[:3] == ("docker", "start", "--attach"))
    assert result['status'] == 'discovery_complete'
    assert result['files']['watchdog.json']
    assert result['container_id'] == launch.registry.native_container(launch.request.run_id)['container_id']
    assert json.loads((launch.work/'result.json').read_text()) == launch.registry.native_result(launch.request.run_id)
    assert 'synthetic-scoped-token' not in json.dumps(result)


@pytest.mark.parametrize('problem', ['uncertain', 'missing-watchdog', 'bad-inputs'])
def test_attached_exit_zero_cannot_override_missing_or_uncertain_proof(launch, problem):
    def started():
        if problem == 'uncertain':
            budget = launch.boundary.budget
            from tests.native_budget_fixtures import pending
            pending(budget)
        elif problem == 'bad-inputs':
            (launch.work/'request.json').write_text('{}')
    launch.boundary.start_hook = started
    launch.boundary.missing_watchdog = problem == 'missing-watchdog'
    result = execute(launch)
    assert result['status'] == ('audit_unavailable' if problem == 'uncertain' else 'artifacts_unavailable')
    assert result['activation'] == 'quarantined'
    if problem == 'uncertain':
        assert result['gateway_final_receipt']['uncertain']['requests'] == 1


def test_cleanup_and_result_failure_do_not_replace_primary_interruption(launch, monkeypatch):
    launch.boundary.interrupted = True
    launch.boundary.remove_error = RuntimeError('fixture cleanup unavailable')
    monkeypatch.setattr(launch.registry, 'record_native_result',
                        lambda result: (_ for _ in ()).throw(OSError('fixture storage failure')))
    with pytest.raises(KeyboardInterrupt, match='original interruption') as exc:
        execute(launch)
    assert 'Native finalization also failed: OSError' in exc.value.__notes__
    assert launch.registry.native_container(launch.request.run_id) is not None
    assert any(c[:2] == ('rm', '--force') and c[-1] == 'a'*64 for c in launch.boundary.calls)


def test_projection_failure_keeps_authoritative_result_for_recovery(launch, monkeypatch):
    from src.research import native_recovery
    original = native_recovery._publish_result
    monkeypatch.setattr(native_recovery, '_publish_result',
                        lambda *a: (_ for _ in ()).throw(OSError('fixture projection failure')))
    with pytest.raises(OSError, match='projection failure'):
        execute(launch)
    retained = launch.registry.native_result(launch.request.run_id)
    assert retained['status'] == 'discovery_complete' and not (launch.work/'result.json').exists()
    monkeypatch.setattr(native_recovery, '_publish_result', original)
    launch.boundary.docker.container = None
    event = NativeRecoveryService(launch.supervisor, launch.registry,
                                  docker=launch.boundary.docker).reconcile(launch.request.run_id)
    assert event.reason == 'authoritative_result_retained'
    assert json.loads((launch.work/'result.json').read_text()) == retained


def test_recovery_winning_publication_is_retained_by_original_runner(launch, monkeypatch):
    calls = []
    fresh_receipt = launch.supervisor.gateway_receipt
    def budget(*args):
        calls.append(1)
        if len(calls) == 2:
            bound = launch.registry.native_container(launch.request.run_id)
            launch.boundary.docker.container['State'].update(
                Status='exited', Running=False, ExitCode=0,
                StartedAt=bound['bound_at'], FinishedAt=datetime.now(timezone.utc).isoformat())
            event = NativeRecoveryService(launch.supervisor, launch.registry,
                docker=launch.boundary.docker,
                gateway_receipt=fresh_receipt).reconcile(launch.request.run_id)
            assert event.reason == 'completion_recovered'
        return fresh_receipt(*args)
    monkeypatch.setattr(launch.supervisor, 'gateway_receipt', budget)
    result = execute(launch)
    assert result['recovered'] is True
    assert result == launch.registry.native_result(launch.request.run_id)
    assert json.loads((launch.work/'result.json').read_text()) == result


@pytest.mark.parametrize('stop', [False, True])
def test_canonical_recovery_cli_uses_real_authority_and_only_public_state(launch, monkeypatch, stop):
    from src.utils.cli import app as canonical_app
    execute(launch)
    # Inspect the original absent --rm identity; no gateway request is needed.
    launch.boundary.docker.container = None
    monkeypatch.setattr(NativeSupervisor, '_docker', staticmethod(launch.boundary.docker))
    command = ['native-recover', '--run-id', launch.request.run_id, '--registry',
               str(launch.registry.path), '--workdir', str(launch.supervisor.root)]
    if stop:
        command.append('--stop')
    result = CliRunner().invoke(canonical_app if stop else app,
                                ['rd-agent', *command] if stop else command)
    assert result.exit_code == 0, result.output
    public = json.loads(result.output)
    assert set(public) == {'run_id', 'sequence', 'observed_at', 'observation', 'action',
                           'reason', 'cost_state', 'exposure_retained', 'activation',
                           'internal_search_trials', 'event_hash'}
    assert public['reason'] == 'authoritative_result_retained'
    assert str(launch.registry.path) not in result.output
    assert 'gateway-token' not in result.output


def test_canonical_lean_delegates_to_the_owned_runtime_command():
    from src.utils.cli import app as canonical_app
    result = CliRunner().invoke(canonical_app, ['lean', '--help'])
    assert result.exit_code == 0 and 'check-runtime' in result.output and 'prepare' in result.output
