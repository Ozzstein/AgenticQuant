# RD runtime offline packaging smoke

## Cleanup review fixes — 2026-11-03

Independent review of `c404b79` reproduced two P2 defects: a TERM-ignoring
descendant survived when its group leader exited, and a lost Docker-create
acknowledgement bypassed container cleanup. The runner now sends SIGKILL to the
original process group in `finally`, independently of leader exit, before
hashing its log. This also applies to normal and exceptional leader exits;
processes that deliberately escape the session remain outside this guarantee.

Each container now has a random name and ownership label recorded in a private
per-probe JSON file **before** creation. Creation is inside cleanup protection.
Cleanup inspects that name, requires the exact ownership label, then removes
the returned original ID. Cleanup outcomes and recovered IDs are retained;
unavailable inspection, ownership mismatch or failed removal remains explicitly
unresolved. Cleanup does not replace the primary create/probe exception and
unresolved cleanup cannot yield a passed smoke result. A daemon unavailable or
still processing creation can require later operator inspection using the
retained locator; the runner does not claim guaranteed removal in that case.

Six new synthetic regressions cover leader success/timeout with an ignoring
descendant, lost create ACK, cleanup inspection timeout, ownership mismatch and
removal failure. Focused smoke tests: **13 passed in 2.80s**. Combined smoke,
readiness, independent readiness review, IBKR scope and capture tests:
**112 passed in 6.78s**, sockets disabled. Ruff and diff checks passed. No real
Docker, private reports or providers were used for this fix verification.

## Coordinator integration — 2026-11-03

Integrated the smoke assets with the IBKR query-scope diagnostic and runtime
readiness checks. The first combined run exposed process-global umask leakage
from the smoke runner's failure path (223 passed, one IBKR permissions fixture
failed because its intended public directory was created privately). `main`
now restores the original umask in `finally`; the unavailable-Docker regression
also verifies that restoration. The subsequent combined run passed **224 tests
in 17.42s**, with sockets disabled; Ruff and diff checks passed.

The coordinator independently retried the fixed local Docker socket on this
date and received the Docker Desktop WSL forwarding error: `The command 'docker'
could not be found in this WSL 2 distro.` No new container execution is claimed.
The successful prior project-image smoke remains separate historical evidence.

This path checks the existing pinned upstream installation, imports, CLI help,
and reviewed US template rendering. It does not run native research, a provider
health check, data generation, backtesting, LEAN, or admission. No native
acceptance receipt is produced. Full workflow acceptance remains unverified;
missing sufficient income-source evidence remains blocked.

Start from the intended committed checkout, using the root test interpreter:

```bash
PYTHONPATH=$PWD /home/ozzstein/AgenticQuant/.venv/bin/python -m pytest tests/test_rd_runtime_smoke.py -q --disable-socket
PYTHONPATH=$PWD /home/ozzstein/AgenticQuant/.venv/bin/python -m ruff check environments/rd-agent/offline_smoke.py environments/rd-agent/run_smoke.py tests/test_rd_runtime_smoke.py
PYTHONPATH=$PWD /home/ozzstein/AgenticQuant/.venv/bin/python environments/rd-agent/run_smoke.py
```

Commit packaging changes before building: the runner intentionally uses **HEAD**,
not working files. It creates a private temporary evidence directory, archives
only committed `src/` and the four required RD environment files, and extracts
that archive into a dedicated build context. Configuration, data, host venvs,
and untracked files are excluded. It reuses the isolated `uv.lock` with
`uv sync --locked --no-dev`; it does not copy or modify the host environment.
Dependencies and locks remain unchanged. The build receives a unique tag and a
source-revision label. Python and uv base tags remain as originally pinned;
resolved base digests are available in the build log, rather than guaranteed
identical by these mutable tags.

All Docker commands explicitly select `unix:///var/run/docker.sock` and a fresh
empty private config. Only PATH is inherited. No Docker contexts, registry
credentials, host API variables or private files are used. Build public network
access is permitted; container probes use `--network none`, no mounts or ports,
UID/GID 10001, all capabilities dropped, no new privileges, 3 GiB memory/swap,
2 CPUs and 256 PIDs. Each probe has a 120-second deadline. Build deadline is
880 seconds plus at most 10 seconds termination grace. Exact containers are
removed in cleanup; the uniquely tagged image and private evidence are retained.
There is no prune, push, or provider credential access.

The import/template probe asserts `/opt/quantagent/src` and isolated site-package
origins, RD's VCS revision from installed metadata, Qlib/Torch versions, and
imports native binary dependencies. It calls the existing overlay function,
which checks eight exact upstream file hashes. Five YAML templates are rendered
and parsed with inert nonexistent data paths, then checked for the US universe,
benchmark and costs. The fixed data-generator source is compiled but never
executed. These are packaging fixtures, not research artifacts. Separate CLI
probes invoke only `rdagent.app.cli --help` and `fin_factor --help`.

## Inspected requirements and download decision

Baseline integration: `ab068b03e4c6b440de3b76b106f93f8d3ba784d2`, branch
`build/rd-offline-runtime`, existing app-owned worktree. Read `CLAUDE.md`; no
applicable `AGENTS.md` exists. No other agents or threads were started.

The installed pinned upstream source was inspected explicitly under
`/home/ozzstein/AgenticQuant/environments/rd-agent/.venv/lib/python3.11/site-packages/rdagent`.
This was inspection only; the build installs from the committed lock:

- RD-Agent 1.0.0, commit `484776c211e4fbbeef03e0ec00d6bbee7362a4f4`.
- Qlib 0.9.7, CPU Torch 2.14.1+cpu, Python 3.11.16, uv 0.12.18.
- Upstream `QlibDockerConf` defaults to `local_qlib:latest`, builds its own
  Dockerfile, enables GPU and requests 16g shared memory. That Dockerfile starts
  from `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime`, clones Qlib at
  `2fb9380b342556ddb50a4b24e4fe8655d548b2b8`, and installs further unpinned packages.
  It is not built by this smoke.
- The application native entrypoint already substitutes contained local helpers
  for the upstream Docker/Conda Qlib helpers. This probe does not call that
  entrypoint, patch its behavior, or establish that its workflow succeeds.

Conservative sum of matching Linux x86_64/Python 3.11 or universal lock wheels
with size metadata: **583,795,796 bytes** (includes alternate ABI wheels and is
not an exact resolver transfer estimate). CPU Torch lacks size metadata in the
lock; a public HEAD request returned **196,227,172 bytes**. Together these are
about 744 MiB. Allowing base images, apt, upstream Git and build dependencies,
expected download is below 3 GiB and runtime within 15 minutes based on the prior
411-second project-image build. This supports attempting the bounded CPU build;
it does not justify pulling the upstream CUDA image. These are estimates, not
an enforced network byte quota.

## Observed results

Focused offline boundary tests: **7 passed**. Ruff and `git diff --check` passed.
Tests assert own checkout imports, mount-free offline resource limits, archive
allowlist, subprocess failure/deadline handling, rejection of modified upstream
templates, and locked unprivileged Docker packaging. No full-suite claim.

Implementation commit: `db6dc8a14f550540a390b1d9837f7afe82c44555`.
The initial build used that committed source and a dedicated `git archive`
context in `/tmp/rd-runtime-smoke-x786eq0v`. Observed Docker output showed
3.28 MB context, successful `uv sync` (57.6 seconds), source/probe COPY steps,
unprivileged user creation and then `exporting layers`. **No final image ID,
build exit, or container probe result was observed.** The task was interrupted;
on resumption on 2026-11-03 that temporary directory no longer existed.
It cannot serve as retained acceptance evidence. In particular, do not infer
build completion or successful imports/help from dependency installation.

On resumption `/var/run/docker.sock` was absent. The Docker executable resolved
to the Docker Desktop WSL forwarding script, which returned exit **1** and:

```text
The command 'docker' could not be found in this WSL 2 distro.
```

No host installation or setup change was attempted. No repeat of completed
setup was requested. The runner now checks the same explicit local Docker
socket first with a 10-second bounded `info` call and records a blocked result
before archiving or building when Docker is unavailable. An added offline test
verifies this boundary and that no build command is issued.

Exact smoke command, run with that preflight fix in the working tree:

```bash
PYTHONPATH=$PWD /home/ozzstein/AgenticQuant/.venv/bin/python environments/rd-agent/run_smoke.py
```

Exit **1**; `packaging_smoke: blocked`, `runtime_acceptance: unverified`.
Docker info exited **1** in **0.17 seconds**, without timeout. Fresh config:
`/tmp/rd-runtime-smoke-7_lhfmg3/docker-config`. Result and diagnostic log:
`/tmp/rd-runtime-smoke-7_lhfmg3/result.json` and `docker-info.log`.
The reported source revision is `db6dc8a14f550540a390b1d9837f7afe82c44555`;
no image was built from uncommitted files. Diagnostic log SHA-256:
`663ca6c96d74b96ae6cdccb23c6be8294ad56f20da13a8da31b6551d66968f0b`.
These are temporary local artifacts, subject to deletion across restarts.

The installed pinned template files remain available. The following separate
host-side check passed: eight pinned file hashes checked, five actual YAML
templates rendered/parsed, and generator source compiled without execution.
It used the required root interpreter and asserted checkout source paths;
it did not import or run the native package:

```bash
PYTHONPATH=$PWD /home/ozzstein/AgenticQuant/.venv/bin/python - <<'PYCODE'
import importlib.util
import json
import tempfile
from pathlib import Path
import src
import src.research.native_runtime.overlay as overlay
assert Path(src.__file__).resolve() == Path.cwd()/'src/__init__.py'
assert Path(overlay.__file__).resolve() == Path.cwd()/'src/research/native_runtime/overlay.py'
spec=importlib.util.spec_from_file_location('offline_smoke', 'environments/rd-agent/offline_smoke.py')
smoke=importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
source=Path('/home/ozzstein/AgenticQuant/environments/rd-agent/.venv/lib/python3.11/site-packages/rdagent')
with tempfile.TemporaryDirectory(prefix='rd-template-check-') as temporary:
    result=smoke.check_templates(source, Path(temporary)/'rdagent')
    print(json.dumps(result, indent=2))
PYCODE
```

Unchanged isolated lock SHA-256:
`f0d6469ab96470d95f819e17446d061f673d0bc4b816a92781f77f0914735837`.
Unchanged isolated pyproject SHA-256:
`455a06718ae90246a32b75ef4910ae04efcecd0105e2ff94a68b232aa3bb73e9`.

Current status: tested packaging preparation and real pinned-template rendering
pass. Image completion/import/help smoke remains blocked by the currently
unavailable local Docker integration and missing prior completion evidence.
Native RD/LEAN workflow acceptance and income-source completeness remain
unestablished. Supervisor, gateway, registry, locks and dependency manifests
were not modified. No research, provider calls, secrets, trading, host installs,
push, merge, PR, or additional threads/agents were used.
