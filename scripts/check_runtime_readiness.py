"""Bounded local prerequisite inventory; never runtime acceptance or launch authority."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import selectors
import shutil
import signal
import subprocess
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_OUTPUT = 16384
# -S prevents site/.pth execution; explicit paths expose venv metadata without imports.
METADATA_PROBE = """
import importlib.metadata as m, json, pathlib, sys, sysconfig
base = str(pathlib.Path(sys.executable).parent.parent)
paths = sysconfig.get_paths(vars={'base': base, 'platbase': base})
found = {}
names = json.loads(sys.argv[1])
for d in m.distributions(path=list({paths['purelib'], paths['platlib']})):
    name = d.metadata.get('Name', '').lower().replace('_', '-')
    if name in names:
        found[name] = d.version
print(json.dumps({'python': list(sys.version_info[:3]), 'packages': found}))
"""


def probe(argv: list[str], *, timeout: float) -> tuple[str, str]:
    """Capture at most MAX_OUTPUT bytes; discard all diagnostics, including errors.

    Only fixed probes below call this helper. No shell, inherited secrets, stdin,
    remote Docker context, or user-supplied command string is used.
    """
    env = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT') if key in os.environ}
    env.update(DOTNET_CLI_TELEMETRY_OPTOUT='1', DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',
               DOTNET_NOLOGO='1')
    try:
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
    except FileNotFoundError:
        return 'missing_executable', ''
    except OSError:
        return 'launch_failed', ''
    output = bytearray()
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return 'timeout', ''
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if len(output) + len(chunk) > MAX_OUTPUT:
                        return 'output_limit', ''
                    output.extend(chunk)
            try:
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                return 'timeout', ''
        if code:
            return 'nonzero_exit', ''
        return 'ok', output.decode('utf-8', errors='replace').strip()
    finally:
        # Also terminate descendants holding pipes after a timeout/output flood.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()


def _check(status: str, reason: str, **details) -> dict:
    return {'status': status, 'reason': reason, **details}


def _version_probe(argv: list[str], pattern: str, timeout: float) -> dict:
    reason, output = probe(argv, timeout=timeout)
    if reason != 'ok':
        return _check('unavailable', reason)
    if re.fullmatch(pattern, output) is None:
        return _check('unverified', 'unexpected_output')
    # Do not echo raw output: even successful CLI diagnostics can contain secrets.
    return _check('available', 'version_response')


def check_readiness(*, rd_python: Path | None = None, timeout: float = 3.0) -> dict:
    """Inventory fixed local probes. Availability never verifies compatibility."""
    if not math.isfinite(timeout) or not 0 < timeout <= 10:
        raise ValueError('timeout must be finite and in (0, 10] seconds')
    checks = {}
    for name in ('docker', 'lean', 'dotnet'):
        checks[name + '_executable'] = _check(
            'available' if shutil.which(name) else 'unavailable', 'path_lookup',
        )
    if checks['docker_executable']['status'] == 'available':
        # Match native preflight's read-only info probe, but bound/redact it here.
        # Explicit socket bypasses remote contexts and DOCKER_HOST/TLS settings.
        checks['docker_daemon'] = _version_probe(
            ['docker', '--host', 'unix:///var/run/docker.sock', 'info',
             '--format', '{{.ServerVersion}}'], r'\d+\.\d+\.\d+(?:[-+][A-Za-z0-9_.-]+)?', timeout,
        )
    else:
        checks['docker_daemon'] = _check('unavailable', 'missing_executable')
    if checks['dotnet_executable']['status'] == 'available':
        checks['dotnet_runtime'] = _version_probe(
            ['dotnet', '--list-runtimes'],
            r'(?:Microsoft\.[A-Za-z0-9.]+ \d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)? \[[^\r\n]+\](?:\n|$))+',
            timeout,
        )
    else:
        checks['dotnet_runtime'] = _check('unavailable', 'missing_executable')
    python = rd_python if rd_python is not None else ROOT / 'environments/rd-agent/.venv/bin/python'
    if not python.is_absolute():
        raise ValueError('RD interpreter must be an explicit absolute path')
    if not python.is_file():
        checks['rd_metadata'] = _check('unavailable', 'missing_interpreter')
    elif not (python.parent.parent / 'pyvenv.cfg').is_file():
        checks['rd_metadata'] = _check('unavailable', 'isolated_venv_required')
    else:
        manifest = tomllib.loads((ROOT / 'environments/rd-agent/pyproject.toml').read_text())
        names = [re.match(r'[A-Za-z0-9-]+', dep)[0].lower()
                 for dep in manifest['project']['dependencies']]
        reason, output = probe([str(python), '-I', '-S', '-c', METADATA_PROBE,
                                json.dumps(names)], timeout=timeout)
        checks['rd_metadata'] = _metadata_result(reason, output, names)
    checks['runtime_acceptance'] = _check('unverified', 'not_exercised')
    return {'schema_version': '1.0', 'runtime_verified': False,
            'status': ('prerequisites_available' if all(
                item['status'] == 'available' for key, item in checks.items()
                if key != 'runtime_acceptance') else 'blocked'), 'checks': checks}


def _metadata_result(reason: str, output: str, names: list[str]) -> dict:
    if reason != 'ok':
        return _check('unavailable', reason)
    try:
        data = json.loads(output)
        python = data['python']
        packages = data['packages']
        if (not isinstance(python, list) or len(python) != 3
                or any(type(n) is not int or not 0 <= n <= 999 for n in python)
                or not isinstance(packages, dict) or set(packages) - set(names)
                or any(not isinstance(v, str) or re.fullmatch(r'\d[A-Za-z0-9.+!-]{0,79}', v) is None
                       for v in packages.values())):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        return _check('unverified', 'unexpected_output')
    missing = sorted(set(names) - set(packages))
    reason = ('python_version_mismatch' if python[:2] != [3, 11] else
              'missing_packages' if missing else 'metadata_only')
    return _check('available' if reason == 'metadata_only' else 'unavailable', reason,
                  python=python, packages=packages, missing=missing)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rd-python', type=Path, help='Trusted isolated venv interpreter (absolute path)')
    parser.add_argument('--timeout', type=float, default=3.0, help='Per-probe deadline, >0 and <=10 seconds')
    args = parser.parse_args(argv)
    try:
        result = check_readiness(rd_python=args.rd_python, timeout=args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'prerequisites_available' else 2


if __name__ == '__main__':
    raise SystemExit(main())
