"""Offline readiness tests: synthetic process replies are not runtime acceptance."""

import json
import sys
from pathlib import Path

import pytest

from scripts import check_runtime_readiness as readiness


def test_import_is_from_checkout():
    assert Path(readiness.__file__).resolve() == Path(__file__).resolve().parents[1] / 'scripts/check_runtime_readiness.py'


@pytest.mark.parametrize(('code', 'expected'), [
    ("print('27.1.0')", ('ok', '27.1.0')),
    ("import sys; print('private diagnostic'); sys.exit(3)", ('nonzero_exit', '')),
    ("import time; time.sleep(10)", ('timeout', '')),
    ("print('x' * 20000)", ('output_limit', '')),
])
def test_bounded_capture(code, expected):
    assert readiness.probe([sys.executable, '-I', '-S', '-c', code], timeout=0.2) == expected


def test_missing_executable(tmp_path):
    assert readiness.probe([str(tmp_path / 'absent')], timeout=0.2) == ('missing_executable', '')


def test_probe_environment_does_not_forward_secrets(monkeypatch):
    monkeypatch.setenv('READINESS_TEST_SECRET', 'do-not-forward')
    code = "import os; print(os.environ.get('READINESS_TEST_SECRET', 'absent'))"
    assert readiness.probe([sys.executable, '-I', '-S', '-c', code], timeout=1) == ('ok', 'absent')


def test_missing_prerequisites(monkeypatch, tmp_path):
    monkeypatch.setattr(readiness.shutil, 'which', lambda _: None)
    monkeypatch.setattr(readiness, 'probe', lambda *a, **k: pytest.fail('must not execute'))
    result = readiness.check_readiness(rd_python=tmp_path / 'absent')
    assert result['status'] == 'blocked'
    assert result['runtime_verified'] is False
    assert result['checks']['runtime_acceptance']['status'] == 'unverified'
    assert result['checks']['rd_metadata']['reason'] == 'missing_interpreter'


def isolated_python(tmp_path):
    path = tmp_path / 'bin/python'
    path.parent.mkdir()
    path.touch()
    (tmp_path / 'pyvenv.cfg').touch()
    return path


def test_fixed_probes_and_positive_inventory(monkeypatch, tmp_path):
    python = isolated_python(tmp_path)
    commands = []

    def probe(argv, *, timeout):
        commands.append(argv)
        assert timeout == 0.5
        if argv[0] == 'docker':
            return 'ok', '27.1.0'
        if argv[0] == 'dotnet':
            return 'ok', 'Microsoft.NETCore.App 8.0.1 [/redacted/path]'
        assert argv[:4] == [str(python), '-I', '-S', '-c']
        assert argv[4] == readiness.METADATA_PROBE
        return 'ok', json.dumps({'python': [3, 11, 9],
                                 'packages': {n: '1.0' for n in json.loads(argv[5])}})

    monkeypatch.setattr(readiness.shutil, 'which', lambda name: '/bin/' + name)
    monkeypatch.setattr(readiness, 'probe', probe)
    result = readiness.check_readiness(rd_python=python, timeout=0.5)
    assert result['status'] == 'prerequisites_available'
    assert result['runtime_verified'] is False
    assert commands[0] == ['docker', '--host', 'unix:///var/run/docker.sock',
                           'info', '--format', '{{.ServerVersion}}']
    assert commands[1] == ['dotnet', '--list-runtimes']
    assert len(commands) == 3  # Never execute LEAN or a workload/import check.


@pytest.mark.parametrize('reply', ['credentials=private', '', '27.1.0\nprivate'])
def test_unexpected_output_is_not_echoed(monkeypatch, reply):
    monkeypatch.setattr(readiness, 'probe', lambda *a, **k: ('ok', reply))
    assert readiness._version_probe(['docker'], r'\d+\.\d+\.\d+', 1) == {
        'status': 'unverified', 'reason': 'unexpected_output',
    }


@pytest.mark.parametrize('reason', ['timeout', 'nonzero_exit', 'output_limit', 'launch_failed'])
def test_probe_failure_states(monkeypatch, reason):
    monkeypatch.setattr(readiness, 'probe', lambda *a, **k: (reason, 'private'))
    assert readiness._version_probe(['docker'], '.*', 1) == {
        'status': 'unavailable', 'reason': reason,
    }


@pytest.mark.parametrize('data', [None, [], {}, {'python': 'private', 'packages': {}},
    {'python': [3, 11, 0], 'packages': {'rdagent': 'token=private'}},
    {'python': [3, 11, 0], 'packages': {'unknown': '1.0'}},
])
def test_malformed_metadata_is_unverified(data):
    assert readiness._metadata_result('ok', json.dumps(data), ['rdagent']) == {
        'status': 'unverified', 'reason': 'unexpected_output',
    }


def test_missing_metadata_and_python_mismatch():
    missing = readiness._metadata_result('ok', '{"python":[3,11,0],"packages":{}}', ['rdagent'])
    assert missing['reason'] == 'missing_packages'
    assert missing['missing'] == ['rdagent']
    wrong = readiness._metadata_result('ok', '{"python":[3,12,0],"packages":{}}', [])
    assert wrong['status'] == 'unavailable'
    assert wrong['reason'] == 'python_version_mismatch'


@pytest.mark.parametrize('timeout', [0, -1, 11, float('inf'), float('nan')])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError, match='timeout'):
        readiness.check_readiness(timeout=timeout)


def test_rejects_nonisolated_interpreter(monkeypatch, tmp_path):
    python = tmp_path / 'bin/python'
    python.parent.mkdir()
    python.touch()
    monkeypatch.setattr(readiness.shutil, 'which', lambda _: None)
    result = readiness.check_readiness(rd_python=python)
    assert result['checks']['rd_metadata']['reason'] == 'isolated_venv_required'


def test_cli_exit_and_json(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(readiness.shutil, 'which', lambda _: None)
    assert readiness.main(['--rd-python', str(tmp_path / 'absent')]) == 2
    assert json.loads(capsys.readouterr().out)['runtime_verified'] is False


def test_metadata_probe_reads_venv_without_site(tmp_path):
    # Fake venv metadata, no package modules and no site startup execution.
    python = tmp_path / 'bin/python'
    python.parent.mkdir()
    python.symlink_to(sys.executable)
    package = tmp_path / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    dist = package / 'rdagent-1.0.dist-info'
    dist.mkdir(parents=True)
    (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: rdagent\nVersion: 1.0\n')
    (package / 'danger.pth').write_text("import sys; sys.exit(99)\n")
    reason, output = readiness.probe(
        [str(python), '-I', '-S', '-c', readiness.METADATA_PROBE, '["rdagent"]'], timeout=2,
    )
    assert reason == 'ok'
    assert json.loads(output)['packages'] == {'rdagent': '1.0'}
