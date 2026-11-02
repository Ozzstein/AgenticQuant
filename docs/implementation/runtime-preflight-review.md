# Independent runtime preflight review — round 1

Reviewed parent `a233bb5`, limited to `scripts/check_runtime_readiness.py`, its
focused tests, and `docs/implementation/runtime-preflight.md`. Live orchestration
confirmed provider `codex`, model `gpt-6-astra`, `reasoningEffort=high`,
`serviceTier=default` (Standard). Production source was not modified.

## Reproduced finding R1 — low / P3

At `scripts/check_runtime_readiness.py:152`, `json.loads` can raise
`RecursionError`, which the handler at line 161 does not catch. A synthetic reply
of `'[' * 2000 + '0' + ']' * 2000` is 4,001 bytes, below the 16 KiB subprocess
capture cap, but escapes `_metadata_result` rather than producing
`unverified/unexpected_output`. The CLI's handler at line 177 also does not catch
this exception, so such a reply would abort JSON reporting.

Remedy: include `RecursionError` in the metadata parser's exception handler,
then remove the strict expected-failure marker from
`test_nested_metadata_is_rejected` in the review tests. Severity is low because
the fixed metadata script does not normally generate nested arrays; this is a
malformed-response robustness gap, not a demonstrated normal-runtime failure or
credential disclosure.

## Evidence and boundaries

Using the root interpreter with this checkout on `PYTHONPATH`:

```bash
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python -m pytest tests/test_runtime_readiness.py tests/test_runtime_readiness_review.py -q --disable-socket
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python -m ruff check tests/test_runtime_readiness_review.py
git diff --check
```

Result: **41 passed, 1 xfailed** (R1). Both suites verify the runtime module's
import origin is this checkout. Original 31 tests also passed independently.
Additional probes cover exact 16 KiB stderr capture, one-byte overflow, malformed
Python version fields, and ordinary POSIX descendants after timeout, overflow,
parent exit with inherited pipes, and successful parent exit with closed pipes.
Descendants are synthetic, short lived, and checked via temporary marker files.
The initial descendant test construction had an indentation error; the corrected
synthetic test passed all four cases. Ruff and diff checks passed.

Source review and existing synthetic tests confirm fixed Docker socket arguments,
no LEAN execution, `-I -S` metadata invocation, environment allowlisting,
suppression of raw CLI diagnostics, and no affirmative runtime acceptance flag.
The metadata fixture exercises synthetic dist-info and a failing `.pth` file.
No real private input, installed RD environment, Docker CLI/daemon, provider,
network request, installation, workload, or full suite was exercised.

Limits: process-group cleanup does not establish containment of descendants that
escape their session; the tests do not simulate that behavior or uninterruptible
kernel waits. `--disable-socket` constrains pytest, not arbitrary child processes;
the child snippets here perform only local synthetic operations. Fixed Docker
arguments and environment filtering do not independently prove that the real
Docker client never loads a default credential/configuration file; that behavior
was not exercised or established within this source-only scope. No runtime
acceptance or real-world readiness conclusion follows from these tests.
