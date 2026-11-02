"""Synthetic independent review probes; no installed runtimes or private inputs."""

import json
import sys
import time
from pathlib import Path

import pytest

from scripts import check_runtime_readiness as readiness


def test_review_import_origin():
    assert Path(readiness.__file__).resolve() == Path(__file__).resolve().parents[1] / 'scripts/check_runtime_readiness.py'


@pytest.mark.parametrize('size, expected', [
    (readiness.MAX_OUTPUT, 'ok'), (readiness.MAX_OUTPUT + 1, 'output_limit'),
])
def test_exact_output_boundary(size, expected):
    reason, output = readiness.probe(
        [sys.executable, '-I', '-S', '-c', f'import os; os.write(2, b"x" * {size})'],
        timeout=2,
    )
    assert reason == expected
    assert len(output) == (size if expected == 'ok' else 0)


@pytest.mark.parametrize('parent, expected', [
    ('time.sleep(2)', 'timeout'),
    ('os.write(1, b"x" * 20000); time.sleep(2)', 'output_limit'),
    ('os._exit(0)', 'timeout'),  # Descendant retains the stdout pipe.
    ('os.close(1); os.close(2); os._exit(0)', 'ok'),
])
def test_group_cleanup(tmp_path, parent, expected):
    marker = tmp_path / 'descendant-survived'
    # The success case closes the descendant's pipes as well as the parent's.
    close = 'os.close(1); os.close(2);' if expected == 'ok' else 'pass;'
    code = (
        'import os, time\n'
        'if os.fork() == 0:\n'
        f'    {close} time.sleep(0.6)\n'
        f'    open({str(marker)!r}, "w").close()\n'
        '    os._exit(0)\n'
        f'{parent}\n'
    )
    start = time.monotonic()
    assert readiness.probe([sys.executable, '-I', '-S', '-c', code], timeout=0.2)[0] == expected
    assert time.monotonic() - start < 1.0
    time.sleep(0.7)
    assert not marker.exists()


@pytest.mark.parametrize('python', [[True, 11, 0], [3, 11, -1], [3, 11, 1000]])
def test_invalid_python_versions(python):
    assert readiness._metadata_result('ok', json.dumps({'python': python, 'packages': {}}), []) == {
        'status': 'unverified', 'reason': 'unexpected_output',
    }


@pytest.mark.xfail(strict=True, raises=RecursionError, reason='R1: deeply nested JSON escapes metadata parser')
def test_nested_metadata_is_rejected():
    payload = '[' * 2000 + '0' + ']' * 2000
    assert len(payload) < readiness.MAX_OUTPUT
    assert readiness._metadata_result('ok', payload, []) == {
        'status': 'unverified', 'reason': 'unexpected_output',
    }
