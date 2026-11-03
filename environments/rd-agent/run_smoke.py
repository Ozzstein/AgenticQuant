"""Build committed RD packaging and run bounded offline probes on the local Docker socket."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PATHS = (
    'environments/rd-agent/Dockerfile', 'environments/rd-agent/pyproject.toml',
    'environments/rd-agent/uv.lock', 'environments/rd-agent/offline_smoke.py', 'src',
)


def bounded(command: list[str], log: Path, env: dict, seconds: int) -> dict:
    """Run a process group with a wall deadline and retain its exact command and log."""
    started = time.monotonic()
    with log.open('wb') as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    return dict(command=command, exit_code=code, timed_out=timed_out,
                seconds=round(time.monotonic()-started, 2),
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())


def container_command(image: str, probe: list[str]) -> list[str]:
    """Construct a mount-free, network-free unprivileged probe container."""
    return ['create', '--network', 'none', '--cap-drop', 'ALL', '--security-opt',
            'no-new-privileges', '--memory', '3g', '--memory-swap', '3g', '--cpus', '2',
            '--pids-limit', '256', '--env', 'OPENBLAS_NUM_THREADS=1',
            '--env', 'OMP_NUM_THREADS=1', '--env', 'NUMEXPR_NUM_THREADS=1',
            '--env', 'LITELLM_LOCAL_MODEL_COST_MAP=True', '--env', 'HOME=/tmp',
            '--entrypoint', 'python', image, *probe]


def main() -> None:
    """Build only the committed allowlist; leave logs and the unique local image for review."""
    os.umask(0o077)
    evidence = Path(tempfile.mkdtemp(prefix='rd-runtime-smoke-'))
    config = evidence/'docker-config'
    config.mkdir()
    env = {key: os.environ[key] for key in ('PATH',) if key in os.environ}
    env.update(DOCKER_CONFIG=str(config), HOME=str(evidence))
    docker = ['docker', '--host', 'unix:///var/run/docker.sock', '--config', str(config)]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tag = f'quantagentlab-rd:offline-{revision[:12]}-{uuid.uuid4().hex[:12]}'
    result = dict(source_revision=revision, image_tag=tag, runtime_acceptance='unverified')
    print(f'Evidence: {evidence}', flush=True)
    try:
        result['docker_probe'] = bounded(
            docker + ['info', '--format', '{{.ServerVersion}}'],
            evidence/'docker-info.log', env, 10)
        if result['docker_probe']['exit_code'] or result['docker_probe']['timed_out']:
            result['packaging_smoke'] = 'blocked'
            raise RuntimeError('local Docker unavailable; inspect docker-info.log')
        archive = evidence/'source.tar'
        with archive.open('wb') as output:
            subprocess.run(['git', 'archive', '--format=tar', revision, *ARCHIVE_PATHS],
                           cwd=ROOT, stdout=output, check=True)
        result['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        context = evidence/'context'
        context.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(context, filter='data')
        lock = context/'environments/rd-agent/uv.lock'
        result['lock_sha256'] = hashlib.sha256(lock.read_bytes()).hexdigest()
        result['build'] = bounded(docker + ['build', '--progress=plain', '--tag', tag,
            '--build-arg', f'SOURCE_REVISION={revision}', '--iidfile', str(evidence/'image.id'),
            '--file', str(context/'environments/rd-agent/Dockerfile'), str(context)],
            evidence/'build.log', env, 880)
        if result['build']['exit_code'] or result['build']['timed_out']:
            raise RuntimeError('build failed or exceeded bounded deadline; inspect build.log')
        image = (evidence/'image.id').read_text().strip()
        result['image'] = json.loads(subprocess.check_output(
            docker + ['image', 'inspect', image], env=env, timeout=15))[0]
        probes = dict(import_templates=['/opt/quantagent/offline_smoke.py'],
                      upstream_help=['-m', 'rdagent.app.cli', '--help'],
                      factor_help=['-m', 'rdagent.app.cli', 'fin_factor', '--help'])
        result['probes'] = {}
        for name, probe in probes.items():
            container = subprocess.check_output(docker + container_command(image, probe),
                                                env=env, text=True, timeout=30).strip()
            try:
                status = bounded(docker + ['start', '--attach', container],
                                 evidence/f'{name}.log', env, 120)
                subprocess.run(docker + ['kill', container], env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=15, check=False)
                inspection = json.loads(subprocess.check_output(
                    docker + ['inspect', container], env=env, timeout=15))[0]
                status.update(container_id=container, state=inspection['State'],
                              host_config=inspection['HostConfig'], mounts=inspection['Mounts'],
                              user=inspection['Config']['User'])
                result['probes'][name] = status
                if status['timed_out'] or inspection['State']['ExitCode'] or status['exit_code']:
                    raise RuntimeError(f'{name} failed; inspect its log')
            finally:
                subprocess.run(docker + ['rm', '--force', container], env=env, check=True,
                               stdout=subprocess.DEVNULL, timeout=15)
        result['packaging_smoke'] = 'passed'
    except Exception as error:
        result.setdefault('packaging_smoke', 'failed')
        result['error'] = str(error)
        raise
    finally:
        (evidence/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'Offline packaging smoke passed: {tag}; native workflow remains unverified.')


if __name__ == '__main__':
    main()
