# Deployment and runtime verification

The repository provides a locked application image and separate durable operating
and research workers. It is **not an accepted production release**. Current source
coverage cannot certify a fresh complete income-aware paper cycle, and actual
Docker, LEAN, authenticated vendor, statistical and prospective acceptance remain
outstanding. See [the requirements register](BUILD_REQUIREMENTS.md).

## Application image

The Dockerfile uses the repository Python/uv pins and locked dashboard, Alpaca and
Qlib extras. The runtime runs as UID 10001. The following are deployment commands
for an available Docker host; they have not been executed in this WSL environment:

```bash
docker build -t quantagentlab:local .
docker compose config --quiet
docker run --rm quantagentlab:local python scripts/check_environment.py
docker run --rm quantagentlab:local aiquant --help
```

The import/Parquet check verifies the installed stack, not provider access or
trading behavior. Size the host from measured workloads and stored evidence;
this repository does not establish a production memory or cost estimate.

Compose services are:

| Service | Command / role |
|---|---|
| `scheduler` | `python -m src.scheduler serve --lane operating` |
| `research-scheduler` | `python -m src.scheduler serve --lane research` |
| `dashboard` | Streamlit, published on host `127.0.0.1:8501` |
| `app` | Manual profile; defaults to `aiquant operations status` |

Both workers use the shared durable schedule store. They run only explicitly
registered jobs, with frozen input hashes, lane ownership and account locks.
Starting a worker does not create a strategy plan or authorize a broker order.
The research worker is a separate process so long research jobs do not occupy the
operating worker.

## Configuration and durable state

Configuration precedence is defaults, `config/settings.yaml`, `config/.env`, then
process environment. Nested `AIQUANT_SECTION__FIELD` variables override one field.
Use the exact models in `src/utils/config.py`; the extended research config uses
the same loader. Compose loads an optional private `config/.env` and mounts:

- `data` at `/app/data` for immutable input artifacts;
- `outputs` at `/app/outputs` for ledger, evidence, operations and research stores;
- `logs` at `/app/logs` for application logs.

The default paper account is `paper:default`, base currency USD. An existing
account cannot be reopened with changed initial economics. Use explicit separate
identities and stores for independent books. Input paths in JSON must refer to
paths inside the container. Stage approved manifests in the data volume before
registering them; do not substitute current data for frozen input references.

Configure only the provider credentials actually required by approved jobs. Keys
belong in private environment injection, never JSON plans, CLI arguments or
source control. Explicit pricing and bounded budgets are required for generative
work; an unset price is not measured zero cost. Native research uses its separate
scoped gateway credential boundary, described below.

## Start, inspect and register

On a prepared Docker host:

```bash
docker compose up -d scheduler research-scheduler dashboard
docker compose ps
docker compose run --rm app aiquant operations status
docker compose logs --tail=100 scheduler research-scheduler
```

The dashboard's local binding can be reached over an authenticated operator
tunnel. An accessible dashboard or running container alone is not a successful
cycle. Inspect durable run stages, reconciliation, source gaps and actual usage.

Schedules require explicit `ScheduleSpec` JSON, including the immutable job
payload and its execution lane. Examples below assume those files were staged:

```bash
docker compose run --rm app python -m src.scheduler register --input /app/data/schedule.json
docker compose run --rm app python -m src.scheduler tick --lane operating
docker compose run --rm app aiquant pipeline build --input /app/data/preparation.json --output /app/data/cycle.json
docker compose run --rm app aiquant pipeline run --input /app/data/cycle.json
docker compose run --rm app aiquant pipeline dispatch-pending --input /app/data/cycle.json
```

The builder reads authoritative completed refresh receipts, exact source
bindings, model/macro recipes and contextual/income authorities. A fresh plan
without sufficient source coverage fails before economic/provider work. The
current native Alpaca income capture reports gaps; it is not a complete positive
income feed. Do not shorten the required coverage interval or relabel receipt
times to obtain a completed run.

A prepared target can consume only a later actual opening event following actual
acceptance. Restarts use the same durable intent, original source identity and
receipts. Ambiguous submissions retain reservations and require reconciliation;
restarting the process is not permission to resubmit.

## Native research and LEAN

The application image does not supply a Docker daemon or a QuantConnect runtime.
Native RD-Agent needs its separately pinned image, an approved isolated network,
private gateway lifecycle and an accessible Docker host. Do not mount an
unrestricted Docker socket into generated research code. The managed native
supervisor reserves disjoint host paths, freezes input bytes/exposure, binds the
original created container before starting it, and preserves uncertain usage.
See [research implementation evidence](implementation/research.md).

```bash
aiquant rd-agent native-recover --help
aiquant lean prepare --help
aiquant lean check-runtime
```

Native recovery defaults to inspection. An absent auto-removed container is not
proof of success. Explicit stop operates only on the verified original container;
recovery does not restart a search. LEAN projects consume the canonical exclusive
outbox. Runtime health and actual native execution history must reconcile before
new risk; missing runtime/history remains blocked.

## Upgrades, backup and acceptance

Before changing software, retain exact source commits, image identity, lockfile,
configuration hashes, input manifests and authoritative receipts. Back up SQLite
stores through a consistent SQLite backup or coordinated quiescent copy, together
with their referenced artifacts and source blobs. A database-only copy can leave
unverifiable external artifacts. Verify restoration in isolated paths; never
point a restore test at the active account or source tree.

Do not delete durable volumes to clear a failed run. Keep original clocks,
ambiguous reservations, source receipts and account ownership. A new deployment
must preserve those restrictions and must not silently migrate to a competing
legacy broker implementation.

Acceptance still requires an actual image build/start/restart on a Docker host,
verified secrets/network isolation, authenticated entitled feeds and paper venue
receipts, complete income/protocol coverage, full statistical validation and the
prospective requirements in the build plan. None of these follows from the
offline protocol fixtures. This WSL host currently reports Docker Desktop WSL
integration unavailable; no container or native-runtime success is claimed.
