"""Offline smoke boundaries without Docker, provider access, or installed RD packages."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'environments/rd-agent'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_own_checkout_imports():
    import src
    assert Path(src.__file__).resolve() == ROOT/'src/__init__.py'


def test_container_is_offline_bounded_and_has_no_mounts():
    args = load('run_smoke').container_command('sha256:owned', ['-m', 'rdagent.app.cli', '--help'])
    for flag, value in [('--network', 'none'), ('--memory', '3g'), ('--memory-swap', '3g'),
                        ('--cpus', '2'), ('--pids-limit', '256'), ('--cap-drop', 'ALL')]:
        assert args[args.index(flag)+1] == value
    assert not {'--volume', '--mount', '-v', '--privileged', '--publish', '-p'} & set(args)
    assert args[-5:] == ['python', 'sha256:owned', '-m', 'rdagent.app.cli', '--help']


def test_archive_excludes_host_environment_and_configuration():
    paths = load('run_smoke').ARCHIVE_PATHS
    assert 'src' in paths
    assert not {'config', 'data', '.', '.venv', 'environments/rd-agent'} & set(paths)
    assert all('.venv' not in path for path in paths)


def test_bounded_runner_records_failure_and_deadline(tmp_path):
    runner = load('run_smoke')
    failure = runner.bounded([sys.executable, '-c', 'raise SystemExit(7)'],
                             tmp_path/'failure.log', {}, 2)
    assert failure['exit_code'] == 7 and not failure['timed_out']
    timeout = runner.bounded([sys.executable, '-c', 'import time; time.sleep(30)'],
                             tmp_path/'timeout.log', {}, .05)
    assert timeout['timed_out'] and timeout['exit_code'] != 0


def test_template_check_rejects_unpinned_source(tmp_path):
    pins = json.loads((ROOT/'src/research/native_runtime/upstream_files.json').read_text())
    for name in pins:
        target = tmp_path/'rdagent'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('changed upstream')
    with pytest.raises(ValueError, match='differs from pinned version'):
        load('offline_smoke').check_templates(tmp_path/'rdagent', tmp_path/'overlay')
    assert not (tmp_path/'overlay').exists()


def test_dockerfile_reuses_locked_environment():
    text = (ROOT/'environments/rd-agent/Dockerfile').read_text()
    assert 'uv sync --locked --no-dev' in text
    assert 'COPY .venv' not in text
    assert 'USER 10001:10001' in text
    assert 'offline_smoke.py' in text


def test_unavailable_docker_stops_before_archive_or_build(tmp_path, monkeypatch):
    runner = load('run_smoke')
    evidence = tmp_path/'evidence'
    evidence.mkdir()
    monkeypatch.setattr(runner.tempfile, 'mkdtemp', lambda **kwargs: str(evidence))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: 'a'*40)
    commands = []

    def unavailable(command, log, env, seconds):
        commands.append(command)
        assert seconds == 10
        assert set(env) <= {'PATH', 'HOME', 'DOCKER_CONFIG'}
        assert list((evidence/'docker-config').iterdir()) == []
        return dict(exit_code=1, timed_out=False)

    monkeypatch.setattr(runner, 'bounded', unavailable)
    previous_umask = os.umask(0o022)
    try:
        with pytest.raises(RuntimeError, match='local Docker unavailable'):
            runner.main()
        retained_umask = os.umask(0o022)
        assert retained_umask == 0o022
    finally:
        os.umask(previous_umask)
    assert len(commands) == 1
    assert commands[0][1:3] == ['--host', 'unix:///var/run/docker.sock']
    assert commands[0][-3:] == ['info', '--format', '{{.ServerVersion}}']
    result = json.loads((evidence/'result.json').read_text())
    assert result['packaging_smoke'] == 'blocked'
    assert result['runtime_acceptance'] == 'unverified'
    assert not (evidence/'source.tar').exists()
    assert 'build' not in result
