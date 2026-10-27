# Reproducible development

Use Python **3.11.16** from `.python-version` and uv **0.12.18**. `uv.lock` pins transitive dependencies and distribution hashes. Python metadata is bounded to 3.11–3.12 because the current numerical stack uses NumPy 1.x; only 3.11.16 is the execution baseline verified here.

```bash
uv sync --locked --extra dev --extra dashboard --extra alpaca --extra qlib
uv pip check
uv run --locked --no-sync python scripts/check_environment.py
uv run --locked --no-sync ruff check src scripts
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --locked --no-sync pytest tests/ -m 'not integration' --disable-socket
uv build
```

`check_environment.py` imports native quantitative libraries, Qlib, MCP, calendars, dashboard and broker packages, and checks a Parquet round trip. It does not validate provider credentials, data entitlements, broker behavior, strategies, or RD-Agent. Runtime can omit `dev`; `dashboard`, `alpaca`, and `qlib` remain explicit environment extras. The historical `full` extra also retains experimental packages; it is not a declaration that every system requirement is implemented. Native RD-Agent needs its own pinned, Docker-capable environment and remains separately tracked.

The normal application loader resolves defaults, then `config/settings.yaml`, then `config/.env`, then process environment. Nested `AIQUANT_SECTION__FIELD` values override only that field. Explicit constructor arguments remain useful for isolated tests. The extended research loader uses the same precedence. Never commit actual credentials.

`paper_ledger_path` defaults to `<output_dir>/paper-ledger.sqlite3`; `paper_account_id` defaults to `paper:default`, with base currency `USD`. Reusing an account with different starting cash/currency is an error. Use distinct persistent paths or account IDs for independent desks. Tests use temporary paths and must not delete operating model artifacts. Walk-forward defaults to six months; explicit research segments must be chronological and non-overlapping.

CI installs the lock, checks imports/lint, runs offline tests with sockets disabled, builds the Python distribution, and separately builds/smoke-checks the container. It is intentionally strict: integration regressions fail instead of being excluded as expected failures. Real-provider acceptance must be recorded separately.

```bash
docker build -t quantagentlab:local .
docker compose config --quiet
docker run --rm quantagentlab:local python scripts/check_environment.py
```

Compose gives scheduler and dashboard shared durable volumes. The one-shot `app` service requires the `manual` profile so it is not automatically started alongside a second pipeline owner. This fixes packaging topology; the canonical unattended cycle and operational acceptance are still under construction. Do not infer release readiness from a successful image build.

On 2026-10-27, this WSL environment reported that Docker Desktop WSL integration was unavailable. Container execution and native RD-Agent acceptance cannot be verified here until a Docker host is available. The local Python environment remains independently usable.
