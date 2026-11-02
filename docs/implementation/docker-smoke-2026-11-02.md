# Project Docker smoke — 2026-11-02

The project image builds and passes the two bounded, network-disabled smoke
commands after removing one stale Dockerfile COPY instruction. This is image
packaging/import/CLI evidence, not full system, RD-Agent, LEAN, provider, or
trading acceptance.

## Source and reproduced fix

- Separate clean checkout: branch `t3/docker-smoke-report`.
- Initial source: `a233bb50ebfa86594027560ebeef19f66d630c94`, incorporating
  `ca8972d`, `a81ee1e`, and `a233bb5` over `4695f78`.
- Read `CLAUDE.md`, Dockerfile, `.dockerignore`, `pyproject.toml`, `uv.lock`,
  `scripts/check_environment.py`, and relevant CLI/LEAN code. No tracked
  `AGENTS.md` or additional nested `CLAUDE.md` was found.
- Initial actual build failed in **11.51 seconds**, exit **1**:
  `COPY --chown=appuser:appuser lean/ lean/` referenced an absent tracked
  directory. BuildKit reported `"/lean": not found` while computing its checksum.
- Fix commit: `da180f65566911bf3ae8ff7aa058d3f13d7bfc55`. Only that COPY
  instruction was removed. Current LEAN implementation is under
  `src/execution/`, already copied by the image; no replacement directory or
  dependency change was needed.
- Successful build source: committed `da180f65566911bf3ae8ff7aa058d3f13d7bfc55`.
  The requested tag retains `a233bb5` as the baseline identifier; it includes
  the Dockerfile fix.

## Isolation and bounds

Every Docker invocation used `--host unix:///var/run/docker.sock`. The daemon
reported Docker Desktop 4.85.0, Engine 29.6.2/API 1.55, Linux amd64 on WSL2.
No host packages, daemon settings, credentials, or provider configuration were
changed. Docker used a fresh empty private `DOCKER_CONFIG` directory, with no
registry credentials supplied.

Both build contexts were extracted from `git archive --format=tar HEAD` into
fresh private temporary directories, never from the whole working tree.
Tracked configuration filenames were inspected first: `config/.env.example`,
`config/agents_config.yaml`, `config/ibkr-flex-paper.op.env`,
`config/ibkr-flex.op.env`, `config/lean_config.json`, `config/qlib_config.yaml`,
and `config/settings.yaml`. The environment template contained placeholders;
the two `.op.env` files contained secret references. No secret was resolved or
printed. No untracked/private files, host `.venv`, or actual `config/.env`
entered the contexts. BuildKit transferred approximately 3.54 MB of context.

Each build used `timeout --signal=TERM --kill-after=15s 885s docker --host
unix:///var/run/docker.sock build --progress=plain --tag
quantagentlab:smoke-a233bb5 --iidfile <private-directory>/image.id
<archive-context>`, with output redirected to a private log. This allows at
most 15 minutes including the termination grace. Public base-image and
dependency downloads were allowed. The selected Linux lock includes XGBoost
3.2.0 and its roughly 335 MiB NCCL wheel, but no multi-GB Torch/CUDA dependency
stack was observed. Dependencies and lockfile were left unchanged.

Runtime checks used separate `docker create` / `start --attach` containers,
explicitly overriding the scheduler default with these commands:

```text
--entrypoint python <image-id> scripts/check_environment.py
--entrypoint aiquant <image-id> --help
```

Each container had `--network none --cap-drop ALL --security-opt
no-new-privileges --memory 2g --memory-swap 2g --cpus 2 --pids-limit 256`, no
mounts, no published ports, no credentials, and a 120-second attached-command
timeout. `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`
were set to `1`. Inspection confirmed user `appuser`, network `none`, and an
empty mount list. Timeout handling was prepared to kill only the exact
container ID created for that check; no timeout occurred. Both exact containers
were removed after recording their exit state; no broad cleanup/prune ran.

## Observed results (UTC)

- Corrected build started **16:14:07**, completed in **411.22 seconds**, exit
  **0**. Local image: `quantagentlab:smoke-a233bb5`, retained for inspection.
  Docker image inspection reported Linux amd64 and size **1,086,948,682 bytes**.
- Environment check: **16:21:37–16:21:47**, measured **10.80 seconds** including
  inspection, exit **0**, no timeout or OOM kill. All 15 imports succeeded:
  numpy 1.26.4, pandas 2.3.3, pyarrow 19.0.1, lightgbm 4.7.0,
  catboost 1.2.10, xgboost 3.2.0, cvxpy 1.7.5, riskfolio-lib 7.4.0,
  hmmlearn 0.3.3, httpx 0.28.1, mcp 2.3.0, exchange-calendars 4.13.2,
  pyqlib 0.9.7, streamlit 1.66.1, alpaca-py 0.44.0. The temporary Parquet
  write/read and DataFrame equality assertion also passed.
- CLI help: **16:21:49–16:21:55**, measured **7.23 seconds** including
  inspection, exit **0**, no timeout or OOM kill. Printed usage, options,
  and registered commands. No subcommand was executed.
- Environment container:
  `4a6d43d462e175ca1ffe87b23a03c68c1c3947e7f8bdf4c3b768b960a06f70c1`.
- CLI container:
  `b2eb7530a2e099fd0a4599d1e894c6e98b28dc86e4906e9306c8fb01298fc5b2`.
  Removal of each returned exit **0**.

## Retained evidence

Private directories have mode `0700`; build/runtime logs have mode `0600`.
These are local temporary artifacts, not committed logs or durable remote
storage. They contain source archives, contexts, result JSON, and log files:

- Initial attempt: `/tmp/quantagent-docker-smoke-a233bb5-x6hm1m6_`.
- Corrected attempt: `/tmp/quantagent-docker-smoke-fixed-e_06bp4y`.

SHA-256 evidence:

```text
initial HEAD.tar
3e5f4e1b0d132339a978c6cd6ea37ef8a921343500abc07b92d1ea661b1d44b4
initial build.log
cc4413c08c0c105226335ca565c91cad2ef17f0b05026444e1651f5ee1dac878
corrected HEAD.tar
c170888d8289766cca2f458b68797a653bfd558b5d42f87526537f5177d4622e
corrected build.log
c61706e5cdd1cdf480fe7f604a0118b684bb5e0c4ffd32557e1f0e1a557d0d23
environment.log
cd4944e979d3dd5d0a1285a5b4979ce5971418d978734b51c3e31483aece498c
cli-help.log
77462043bb8a79eff7b7c8e1ddd683c25d8d32580e3d055ea7070d620fa570e4
image ID / local repository digest
sha256:4bfcf144b1023809887f7b4a54b6fcf16e18d42f83e1fd1acbbdc23d59b277f5
resolved Python base
sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b
resolved uv base
sha256:3adc3706091ce7c2fe595e669628caedd6d951551b92b258b7e7dbe06d9440bc
uv.lock
bc2d780698e320eae672b8fd2ee4261e95f485ab9f0716fd485f4c08641c4a4d
```

The image remains local; it was not pushed. No scheduler, provider request,
credential workflow, trade, RD-Agent runtime, or LEAN engine was launched.
These checks do not verify Compose startup, persistent volumes, dashboard
serving, strategies, native external runtimes, or operating readiness. They
establish only that this corrected project image builds and its offline
environment/CLI smoke checks pass on the observed Docker host.
