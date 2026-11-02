# Bounded runtime prerequisite inventory

`scripts/check_runtime_readiness.py` inventories local prerequisites without
launching a workload. It does not grant admission, establish runtime acceptance,
verify a broker/provider, or establish income-source completeness. Every report
includes `runtime_verified: false` and `runtime_acceptance: unverified`.

Run from the checkout whose code you intend to inspect:

```bash
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python scripts/check_runtime_readiness.py
```

The default RD interpreter is this checkout's
`environments/rd-agent/.venv/bin/python`. There is no automatic fallback to a
sibling checkout or the invoking interpreter. To inventory an existing trusted
isolated environment explicitly:

```bash
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python scripts/check_runtime_readiness.py \
  --rd-python /home/ozzstein/AgenticQuant/environments/rd-agent/.venv/bin/python
```

`--rd-python` must be an absolute interpreter path in a venv with `pyvenv.cfg`;
use only a trusted interpreter. It is executed with fixed `-I -S -c` metadata
inspection code, without site startup, `.pth` execution or package imports. The
package names come from this checkout's isolated RD `pyproject.toml` dependency
list. Python must be 3.11; versions are reported as metadata only. Dependency
constraint compatibility, upstream revision, binary loading and the installed
code's integrity remain unverified. No lock or dependency files are changed.

## Probes and interpretation

- Docker, LEAN and .NET executable presence uses PATH lookup. LEAN is not executed;
  executable presence says nothing about bindings or engine compatibility.
- Docker daemon reachability uses only
  `docker --host unix:///var/run/docker.sock info --format '{{.ServerVersion}}'`,
  matching the native preflight's read-only server-version question. Remote
  contexts and `DOCKER_HOST` are not used. This deliberately targets the standard
  local Linux socket; a rootless/custom socket can therefore remain unavailable
  here even when another Docker context works. No image inspection, image pulls,
  container runs or gateway/provider requests occur.
- .NET uses `dotnet --list-runtimes`; an expected runtime listing is availability,
  not proof that LEAN can load those runtimes.
- Each subprocess has a default three-second timeout (`--timeout` must be finite,
  greater than zero and at most ten seconds). Combined stdout/stderr capture is
  capped at 16 KiB. Timeout and overflow terminate the process group. This runner
  targets POSIX hosts. Only PATH and SYSTEMROOT are forwarded, plus fixed .NET
  telemetry/first-run opt-outs. No environment or configuration secrets are read
  or dumped. Raw child output and exception diagnostics are never printed.

Each check is `available`, `unavailable` (missing prerequisite or failed bounded
probe), or `unverified` (unexpected output or acceptance not exercised). Exit
**0** means only `prerequisites_available`; exit **2** means blocked or invalid
arguments. Even a zero exit never means runtime acceptance.

## Relationship to existing checks

[check_environment.py](../../scripts/check_environment.py) remains the canonical
native-import and Parquet round-trip check. It imports packages and writes a
temporary file, so this read-only inventory does not invoke it.
[LEAN check-runtime](../../src/execution/lean_bridge.py) tests actual QuantConnect
and System bindings and retains its own `runtime_verified: false` semantics.
[Native RD preflight](../../src/research/native.py) additionally binds images,
networks, admitted sources and authenticated gateway receipts. Invoking that
preflight would exceed this inventory's scope. Its subprocess helper cannot
provide this CLI's bounded capture/redaction guarantees, so it is not called.
These existing runtime/admission checks remain authoritative; no equivalent
bounded metadata-only inventory existed at base `4695f78`.

## Validation and local blockers (2026-11-02)

Live orchestrator: provider `codex`, model `gpt-6.1-sol`, reasoning effort `high`,
service tier `default`, runtime mode `full-access`. Started from committed
`4695f786127f5a5c0fd339ddad74058396e457df`; no applicable `AGENTS.md` was found.

```bash
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python -m pytest tests/test_runtime_readiness.py -q --disable-socket
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python -m ruff check scripts/check_runtime_readiness.py tests/test_runtime_readiness.py
git diff --check
```

Offline tests cover checkout import origin, missing executables/interpreters,
nonzero exit, deadlines, output overflow, malformed/unexpected responses,
environment isolation, fixed probe arguments, missing metadata, Python mismatch,
and metadata inspection without site startup. Synthetic successes are not runtime
acceptance. The focused suite passed all **31 tests**; Ruff and diff checks passed.

Both local CLI invocations above report `blocked` (exit 2). Docker resolves but
the local daemon probe exits nonzero; LEAN and .NET are missing on PATH. This
worktree has no isolated RD interpreter. The explicitly selected root RD venv
reports Python 3.11.16 and metadata for rdagent 1.0.0, pyqlib 0.9.7,
torch 2.14.1+cpu, numpy 1.26.4, pandas 2.3.3, pyarrow 19.0.1 and tables 3.11.1.
That external venv inventory does not establish this checkout's readiness.
Runtime acceptance remains unverified. Existing source/admission and income
blockers in [next-cycle-readiness.md](next-cycle-readiness.md) remain unchanged.
