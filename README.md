# QuantAgentLab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

QuantAgentLab combines archived market evidence, US Qlib features, fitted models,
a causal macro HMM, an evidence-grounded agent desk and a shared Decimal portfolio
ledger for US equities and crypto. Research, paper operation and remote execution
use explicit identities, source cutoffs and immutable receipts.

**The full system is still being built and is not an accepted release.** Offline
models and protocol fixtures are exercised; authenticated providers, native
Docker/LEAN execution, statistical acceptance and prospective trading remain
unverified. Fresh complete paper cycles currently fail closed on insufficient
income-source coverage. The [full build plan](docs/FULL_SYSTEM_BUILD_PLAN.md) and
[requirements/evidence register](docs/BUILD_REQUIREMENTS.md) retain the complete
scope and outstanding gates.

## System boundaries

- **Data:** raw bytes, normalized values, complete documents and exact spans are
  archived in `EvidenceStore`. Availability and observation time remain separate.
  US exchange sessions and UTC crypto sessions retain their actual timestamps.
- **Quantitative execution:** versioned features, four actual model families,
  train-only transformations, purged walk-forward folds and causal HMM state.
  Six strategy specifications feed four allocation modes and five constrained
  optimizers. A diagnostic result does not grant operational eligibility.
- **Agent desk:** source-grounded specialist reports, two-round debate, strategist
  derivation, qualitative risk veto and canonical BUY/HOLD/REDUCE/EXIT/ABSTAIN
  semantics. Generative usage has durable reservations, actual/unknown settlement
  and model/pricing identity. Missing evidence remains unavailable.
- **Accounting:** one canonical ledger for cash, quantities, basis, fees,
  receivables, targets and fills. Sleeve ownership derives from actual events.
  Opening prints never require fabricated completed OHLC bars; source aliases
  cannot consume the same liquidity twice.
- **Research:** immutable candidates/trials and independent result reconstruction,
  durable named and 1,000-random controls, cost reruns and untouched-holdout gates.
  Native search inputs are frozen and exposure is recorded before execution.
  Native outputs and descendants remain quarantined pending required proof.
- **Operations:** source-bound cycle plans, durable stage recovery and separate
  operating/research scheduler workers. The dashboard reads actual stores.
  Notifications use an explicit durable outbox.
- **Venues:** typed local paper, Alpaca, Binance, Kraken and LEAN adapters share the
  canonical intent/risk/reconciliation boundary. Offline wire fixtures do not
  establish authenticated venue or live-capital acceptance.

## Reproducible local environment

The verified Python baseline is **3.11.16**, with **uv 0.12.18** and the committed
`uv.lock`. Run from the repository root:

```bash
uv sync --locked --extra dev --extra dashboard --extra alpaca --extra qlib
uv pip check
uv run --locked --no-sync python scripts/check_environment.py
uv run --locked --no-sync aiquant --help
```

The environment check imports the native model/Qlib/calendar/provider packages
and performs a Parquet round trip. It makes no provider-entitlement or performance
claim. See [development instructions](docs/DEVELOPMENT.md) for environment details.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --locked --no-sync pytest tests/ --disable-socket --allow-unix-socket
uv run --locked --no-sync ruff check src scripts
```

The full suite includes retained fresh-cycle acceptance cases that require a
sufficient positive income/coverage source. These remain visible failures until
that source exists; no fabricated completeness flag makes them eligible. Unix
sockets support local asyncio/transport fixtures, while external TCP is disabled
apart from explicitly scoped loopback tests.

## Configuration

Defaults are loaded from the typed configuration, followed by
`config/settings.yaml`, private `config/.env`, then process environment. Nested
`AIQUANT_SECTION__FIELD` overrides one field. See `src/utils/config.py`,
`src/utils/config_loader.py` and `config/.env.example` for exact names.

Use only credentials for the providers required by an approved job; offline
inspection needs none. Store actual secrets privately, outside manifests and
source control. Explicit provider/model/pricing and caps bind generative work.
The native research gateway has a separate scoped bearer and retains its provider
key exclusively in the gateway environment.

`AIQUANT_OUTPUT_DIR` determines default durable store locations.
`AIQUANT_PAPER_LEDGER_PATH`, `AIQUANT_PAPER_ACCOUNT_ID` and
`AIQUANT_PAPER_BASE_CURRENCY` identify the paper book. Existing account economics
cannot be changed by editing configuration. Historical input/source IDs and
approved training identities remain immutable on reuse.

## Canonical commands

Commands consume versioned manifests, not a ticker or model label that silently
selects replacement data. Use each command's `--help` for required paths:

```bash
aiquant operations status
aiquant analyze --help
aiquant data --help
aiquant decision --help

# Explicit chronological replay bundle
aiquant backtest run --input replay.json --ledger replay.sqlite3 --output result.json
aiquant backtest compare --input replay.json --result result.json --output comparisons
aiquant backtest validate --result result.json --ledger replay.sqlite3 \
  --costs comparisons/costs --benchmarks comparisons/benchmarks.json --output validation.json

# Source preparation and the shared paper cycle
aiquant operations refresh --input refresh.json --store operations.sqlite3 --evidence evidence.sqlite3
aiquant pipeline build --input preparation.json --output cycle.json
aiquant pipeline run --input cycle.json
aiquant pipeline dispatch-pending --input cycle.json

# Explicit durable schedules; separate worker processes
python -m src.scheduler register --input schedule.json
python -m src.scheduler serve --lane operating
python -m src.scheduler serve --lane research

# Existing research lifecycle, original-container recovery and LEAN bridge
aiquant rd-agent --help
aiquant rd-agent native-recover --help
aiquant lean prepare --help
aiquant lean check-runtime
```

The installed commands and `scripts/run_backtest.py`, `run_agents.py`,
`run_pipeline.py`, `run_rd_agent.py` and `run_operations.py` share their service
implementations. `trade status` reads existing stores; `trade run --input ...`
uses the same complete paper cycle. Retired `--broker`, `trade reset`, ticker-only
pipeline and label-only backtest routes do not authorize execution.

## Deployment and evidence

[Deployment instructions](docs/DEPLOYMENT.md) describe the application image,
shared durable volumes, separate scheduler processes, recovery and native runtime
requirements. Compose binds the dashboard only to host loopback. As of
2026-11-03, Docker WSL integration is working: project and RD offline container
smoke tests passed. Selected actual LEAN synthetic-paper scenarios also passed,
including clean shutdown, market fill, limit cancellation and completed-order
restart. Broader runtime, authenticated broker and deployment acceptance remain
open.

Use the [full build plan](docs/FULL_SYSTEM_BUILD_PLAN.md) for scope and the
[group/subgroup progress register](docs/BUILD_REQUIREMENTS.md) for current
milestones, evidence and remaining work. Neither container smoke nor synthetic
paper execution establishes strategy release readiness.

Implementation records include exact source pins, offline test commands and
remaining limitations:

- [Data and quantitative models](docs/implementation/data-quant.md)
- [Accounting, venues and LEAN](docs/implementation/accounting-backtesting.md)
- [Agents, decision service and operating cycle](docs/implementation/jev-agents.md)
- [Root operations](docs/implementation/operations.md)
- [Research and native execution](docs/implementation/research.md)
- [Income-source sufficiency assessment](docs/implementation/income-source-assessment.md)

A passed local test does not substitute for real source coverage, calibrated
agent quality, held-out statistics or the prospective release requirements. No
live-capital authorization is implied by this repository.

## License

MIT License — see [LICENSE](LICENSE).
