# TASK-07: Production — Docker, Scheduler, Alerts, Dashboard, Audit Trail

## Goal
Make the system run unattended on a cloud VPS with monitoring, alerting, and a web dashboard.

## Depends On
Task 06 (working pipeline)

## What to Build

### 1. Docker (`Dockerfile`, `docker-compose.yml`, `.dockerignore`)

Dockerfile: multi-stage build, python:3.11-slim base. Qlib data as a separate Docker volume (not baked into image). All API keys via environment variables.

docker-compose.yml services:
- `app`: main pipeline runner
- `scheduler`: cron-based trigger for daily + weekly jobs
- `dashboard`: Streamlit app on port 8501

### 2. Scheduler (`src/scheduler.py`)

Daily schedule (all UTC):
- 06:00 — Update equity data
- 06:15 — Update crypto data
- 06:30 — Retrain model (if scheduled day: weekly, biweekly, or monthly per config)
- 07:00 — Generate predictions for full universe
- 07:15 — Run agent analysis on top-N candidates
- 07:30 — Portfolio optimization
- 07:45 — Pre-trade risk controls
- 08:00 — Execute rebalance
- 08:15 — Send daily report

Weekly schedule:
- Run RD-Agent factor mining (if configured)

Monthly schedule:
- Run RD-Agent model optimization (if configured)

Requirements: idempotent (running twice same day does not double-trade), graceful failure (one step fails → log, alert, continue to next), state file tracking what ran today, lock file preventing concurrent runs.

### 3. Telegram Alerts (`src/monitoring.py`)

Telegram bot notifications:
- Daily portfolio summary: total value, daily P&L, top/bottom positions, cash balance
- Trade execution confirmations: ticker, side, quantity, price, reasoning summary
- Drawdown warnings: alert at -5%, -10%, -15% thresholds
- Agent disagreement alerts: when agents strongly conflict (e.g., fundamental says strong buy, sentiment says strong sell)
- System health: data pipeline failures, API errors, missed scheduled runs, model retrain results

Also support email via SMTP as a backup channel.

### 4. Streamlit Dashboard (`src/dashboard.py`)

Pages:
- **Portfolio**: current holdings table, allocation pie chart, daily P&L line chart, cumulative returns vs benchmarks
- **Agent Analysis**: for each recently analyzed ticker, show agent reports, decision, confidence, risk flags. Searchable by ticker.
- **Backtests**: walk-forward results, validation suite results, model comparison table
- **Factors**: IC/ICIR of active factors over time, factor library contents, RD-Agent discovery history
- **System**: last run time, pipeline step status (pass/fail), data freshness per source, cumulative API costs

### 5. Audit Trail (extend `src/utils/logger.py` + all modules)

Structured JSON logging of every decision with full context. Stored in a persistent directory (data/audit/).

Each daily run produces a JSON file containing: date, pipeline steps executed (with status), model predictions summary, agent analyses (full AnalysisResult per ticker), risk control decisions (per order: passed/blocked + reason), trades executed, portfolio state before/after, API costs incurred.

Must be queryable: "why did the system buy NVDA on March 3rd?" should be answerable from the audit log.

Track API costs: log every LLM call with model, token count, estimated cost. Daily/weekly/monthly summaries.

### 6. Documentation

**README.md**: what it is, who it's for, architecture diagram, project structure, getting started, configuration reference, CLI reference, expansion roadmap, how agents work, risk management framework, risk disclaimer, license.

**docs/DOCUMENTATION.md**: deep technical docs covering all three layers in detail, every data source with cost, backtest methodology (single-split vs walk-forward), worked example of agent analyzing a ticker (show what each agent does), full risk management framework, deployment guide (VPS setup step-by-step), troubleshooting section, glossary of quant terms.

**docs/DEPLOYMENT.md**: step-by-step guide for deploying to EU VPS (Hetzner/DigitalOcean/OVH). VPS specs (4GB RAM, 50GB SSD), Docker install, SSH key setup, .env configuration on server, cron/systemd setup, log rotation, cost estimate (< €15/month).

### 7. Success Criteria Documentation

Document these in the README as the checklist before going live:
1. Walk-forward Sharpe > 1.0 after realistic costs (10 bps equity, 20 bps crypto)
2. Validation suite: zero FAIL
3. Beats random picker at 95th percentile
4. Positive Sharpe in ≥ 2 of 4 regimes
5. 3+ months paper trading confirms backtest
6. Max drawdown < 15% in paper trading
7. Agent reasoning reviewed for 20+ tickers
8. 30+ consecutive daily pipeline runs without failure
9. All monitoring channels working
10. Daily cost < €2/day

## Inputs
- Working pipeline from Task 06
- All configs

## Outputs
- Docker images that build and run
- Scheduler that triggers daily pipeline
- Telegram bot that sends alerts
- Streamlit dashboard accessible on port 8501
- Audit logs in data/audit/
- Complete README.md and docs

## Done Criteria
- [ ] `docker-compose up` starts all three services without errors
- [ ] Scheduler runs a simulated daily cycle and logs all steps
- [ ] Telegram bot sends a test message to configured chat
- [ ] Dashboard loads on localhost:8501 and shows portfolio page
- [ ] Audit log JSON file written after a pipeline run, contains all expected fields
- [ ] API cost tracking reports estimated spend per run
- [ ] README.md covers all sections listed above
- [ ] DOCUMENTATION.md has all 14 sections
- [ ] DEPLOYMENT.md has step-by-step VPS instructions
- [ ] `pytest tests/` passes with > 80% coverage on src/execution/ and src/core/
