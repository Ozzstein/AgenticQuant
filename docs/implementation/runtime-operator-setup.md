# Runtime operator setup

This is an operator procedure, not evidence of acceptance. On 2026-11-02 this checkout was at `4695f786127f5a5c0fd339ddad74058396e457df`; `WSL_DISTRO_NAME` was `Ubuntu`, and `wsl.exe -l -v` showed Ubuntu running as version 2 and `docker-desktop` stopped. Earlier Docker diagnostics reported WSL integration unavailable. No settings, installations or containers were changed for this guide.

## Windows GUI prerequisite

1. Start Docker Desktop on Windows and wait for it to start. In Settings > General, enable **Use WSL 2 based engine** if that option is shown. In Settings > Resources > WSL Integration, enable **Ubuntu**, then Apply. If WSL Integration is missing, switch Docker Desktop to Linux containers. See [Docker's WSL backend instructions](https://docs.docker.com/desktop/features/wsl/).
2. In Windows PowerShell, `wsl --list --verbose` confirms Ubuntu is version 2. From the Ubuntu shell, `printf '%s\n' "$WSL_DISTRO_NAME"` identifies the distribution whose toggle must be enabled. See [Microsoft's WSL commands](https://learn.microsoft.com/en-us/windows/wsl/basic-commands). If Ubuntu is not version 2 or Desktop cannot start, resolve that prerequisite with the platform instructions before continuing.

## Diagnostics from the checkout

These commands do not start containers or call providers. Run in Ubuntu at the checkout root; `docker info` and `docker version` contact the local daemon and may display host metadata.

```bash
pwd
git rev-parse HEAD
printf '%s\n' "$WSL_DISTRO_NAME"
wsl.exe -l -v
docker version
docker info
docker compose config --quiet
command -v lean || true
command -v dotnet || true
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python -c 'import src.execution.lean_bridge as m; print(m.__file__)'
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python scripts/check_environment.py
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python -m src.execution.lean_bridge check-runtime
```

Confirm the printed module path is inside this checkout. Docker diagnostics must show a reachable server; Compose configuration must exit zero. The environment check tests imports and a temporary Parquet round trip. The LEAN check may fail until its bindings are installed in the interpreter running it; even `bindings_available` explicitly reports `runtime_verified: false`. Record failures and stop at the relevant prerequisite.

## LEAN prerequisite route

Docker integration alone does not install LEAN or its .NET/Python bindings. Follow [QuantConnect's LEAN CLI installation guide](https://www.quantconnect.com/docs/v2/lean-cli/installation/installing-lean-cli) on the intended host; local engine execution requires Docker. Separately provision the actual LEAN runtime and repository dependencies in the intended runtime environment, with repository and ledger mounted at their declared paths. CLI availability or a host `dotnet` version alone does not prove that the Python process can import `QuantConnect.Algorithm` and `System`. Run `aiquant lean check-runtime` **inside that intended environment** and retain its output. This is a prerequisite check, not a native execution test. Installation and runtime provisioning require separate authorization.

## Acceptance sequence after explicit authorization

1. Record revision, lockfile and image identity, host/runtime versions and diagnostic results. Review [development](../DEVELOPMENT.md), [deployment](../DEPLOYMENT.md), [research evidence](research.md) and [requirements](../BUILD_REQUIREMENTS.md) for the scope and outstanding gates.
2. With permission to build and run local containers, use the existing offline image smoke sequence from the checkout:

   ```bash
   docker build -t quantagentlab:local .
   docker compose config --quiet
   docker run --rm quantagentlab:local python scripts/check_environment.py
   docker run --rm quantagentlab:local aiquant --help
   ```

   Record exit codes and image identity. This establishes only the image smoke result.
3. For LEAN, review `aiquant lean prepare --help`, then prepare from a reviewed `LeanLaunchConfig` using `aiquant lean prepare --config <reviewed-config.json> --output <new-directory>`. Preparation writes a project and manifest and does not launch it. Review the manifest, implementation hashes, mounts and runtime configuration; perform a separately authorized native runtime exercise and reconcile actual runtime health and execution history under the existing deployment gates. Do not treat preparation or imports as execution acceptance.
4. Native RD-Agent acceptance is separate: use its pinned environment and image, approved isolated network and private gateway lifecycle described in [deployment](../DEPLOYMENT.md) and [research evidence](research.md). Inspect `aiquant rd-agent native-recover --help`; recovery inspection is not a new search. Run an actual native job only under its own authorization and retain its evidence.
5. Record outcomes against the requirements register. Provider, source-completeness, ledger, operating and activation gates remain independent; the known missing positive income authority remains a blocker for the fresh cycle. No runtime smoke grants trading or provider access.
