# Reproducible development

Use Python **3.11.16** from `.python-version` and uv **0.12.18**. `uv.lock` pins transitive dependencies and distribution hashes. Python metadata is bounded to 3.11–3.12 because the current numerical stack uses NumPy 1.x; only 3.11.16 is the execution baseline verified here.

```bash
uv sync --locked --extra dev --extra dashboard --extra alpaca --extra qlib
uv pip check
uv run --locked --no-sync python scripts/check_environment.py
uv run --locked --no-sync ruff check src scripts
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --locked --no-sync pytest tests/ --disable-socket --allow-unix-socket
uv build
```

`check_environment.py` imports native quantitative libraries, Qlib, MCP, calendars, dashboard and broker packages, and checks a Parquet round trip. It does not validate provider credentials, data entitlements, broker behavior, strategies, or RD-Agent. Runtime can omit `dev`; `dashboard`, `alpaca`, and `qlib` remain explicit environment extras. The historical `full` extra also retains experimental packages; it is not a declaration that every system requirement is implemented. Native RD-Agent needs its own pinned, Docker-capable environment and remains separately tracked.

The normal application loader resolves defaults, then `config/settings.yaml`, then `config/.env`, then process environment. Nested `AIQUANT_SECTION__FIELD` values override only that field. Explicit constructor arguments remain useful for isolated tests. The extended research loader uses the same precedence. Never commit actual credentials.

`paper_ledger_path` defaults to `<output_dir>/paper-ledger.sqlite3`; `paper_account_id` defaults to `paper:default`, with base currency `USD`. Reusing an account with different starting cash/currency is an error. Use distinct persistent paths or account IDs for independent desks. Tests use temporary paths and must not delete operating model artifacts. Walk-forward defaults to six months; explicit research segments must be chronological and non-overlapping.

CI installs the lock, checks imports/lint, runs the non-integration offline test selection, builds the Python distribution, and separately builds/smoke-checks the container. The full audit command above also includes integration-marked tests. Unix sockets allow local asyncio self-pipes and transport fixtures; external TCP remains disabled except explicitly scoped loopback MCP tests. Real-provider acceptance must be recorded separately. Known fresh-cycle income-source failures remain enabled; no expected-failure marker substitutes for the missing authority.

```bash
docker build -t quantagentlab:local .
docker compose config --quiet
docker run --rm quantagentlab:local python scripts/check_environment.py
```

Compose gives separate operating and research scheduler workers and the dashboard shared durable volumes. Workers consume explicitly registered immutable jobs through the same application services; an empty schedule does no economic work. The one-shot `app` service requires the `manual` profile and defaults to read-only status. Fresh successful operating cycles remain blocked by incomplete income authority. Do not infer release readiness from a successful image build or an idle scheduler.

On 2026-10-27, this WSL environment reported that Docker Desktop WSL integration was unavailable. Container execution and native RD-Agent acceptance cannot be verified here until a Docker host is available. The local Python environment remains independently usable.
