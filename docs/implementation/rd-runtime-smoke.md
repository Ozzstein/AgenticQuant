# RD runtime offline packaging smoke

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

Focused offline boundary tests: **6 passed**. Ruff and `git diff --check` passed.
Tests assert own checkout imports, mount-free offline resource limits, archive
allowlist, subprocess failure/deadline handling, rejection of modified upstream
templates, and locked unprivileged Docker packaging. No full-suite claim.

Actual build and container results are recorded below after execution.
