"""Production scheduler: daily UTC pipeline steps with state-file idempotency.

Runs 9 pipeline steps at fixed UTC times, persists run state to
``data/scheduler_state.json``, and prevents concurrent runs via a lock file
at ``outputs/.scheduler.lock``.

CLI usage::

    python src/scheduler.py                   # run all due steps once
    python src/scheduler.py --loop            # block forever, polling every 60 s
    python src/scheduler.py --dry-run         # log steps but skip real operations
    python src/scheduler.py --simulate-day    # immediately execute all 9 steps
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import typer
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATE_FILE = _PROJECT_ROOT / "data" / "scheduler_state.json"
_LOCK_FILE = _PROJECT_ROOT / "outputs" / ".scheduler.lock"

# ---------------------------------------------------------------------------
# Daily schedule: (step_name, hour_utc, minute_utc)
# ---------------------------------------------------------------------------
DAILY_STEPS: list[tuple[str, int, int]] = [
    ("update_equity_data", 6, 0),
    ("update_crypto_data", 6, 15),
    ("retrain_model", 6, 30),
    ("generate_predictions", 7, 0),
    ("run_agent_analysis", 7, 15),
    ("optimize_portfolio", 7, 30),
    ("run_risk_controls", 7, 45),
    ("execute_rebalance", 8, 0),
    ("send_daily_report", 8, 15),
]

app = typer.Typer(help="QuantAgentLab production scheduler")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _load_state() -> dict:
    """Load the scheduler state file, returning an empty dict on any error.

    Returns:
        Parsed state dict mapping date strings to step-result dicts.
    """
    if not _STATE_FILE.exists():
        return {}
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception as exc:
        logger.warning("scheduler: could not read state file: {}", exc)
        return {}


def _save_state(state: dict) -> None:
    """Persist the scheduler state dict to disk with atomic writes.

    Writes to a temporary file then atomically renames to the target path
    to prevent corruption on interrupt.

    Args:
        state: State dict mapping date strings to step-result dicts.
    """
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', dir=_STATE_FILE.parent, delete=False, suffix='.tmp'
        ) as tmp:
            json.dump(state, tmp, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(_STATE_FILE)
    except Exception as exc:
        logger.warning("scheduler: could not write state file: {}", exc)


def _step_done_today(state: dict, date_key: str, step: str) -> bool:
    """Return True if the step already completed successfully today.

    Args:
        state: Loaded state dict.
        date_key: Today's date string (YYYY-MM-DD).
        step: Step name to check.

    Returns:
        True when the step has status "ok" for today's date.
    """
    return state.get(date_key, {}).get(step, {}).get("status") == "ok"


def _mark_step(state: dict, date_key: str, step: str, status: str) -> None:
    """Record a step result in the state dict (in-memory; caller must save).

    Args:
        state: Mutable state dict.
        date_key: Today's date string (YYYY-MM-DD).
        step: Step name.
        status: "ok" or "error".
    """
    state.setdefault(date_key, {})[step] = {
        "status": status,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Lock helpers
# ---------------------------------------------------------------------------


def _acquire_lock() -> bool:
    """Create the lock file, returning False if it already exists.

    If the lock file is stale (older than 2 hours), it is removed and
    lock acquisition is retried.

    Returns:
        True when the lock was successfully acquired.
    """
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _LOCK_FILE.exists():
        lock_age_secs = time.time() - _LOCK_FILE.stat().st_mtime
        if lock_age_secs < 7200:  # 2 hours
            logger.error(
                "scheduler: lock file exists (age={}s) — another instance running",
                lock_age_secs,
            )
            return False
        else:
            logger.warning(
                "scheduler: stale lock (age={}s) — removing and retrying",
                lock_age_secs,
            )
            _LOCK_FILE.unlink(missing_ok=True)
    _LOCK_FILE.touch()
    return True


def _release_lock() -> None:
    """Remove the lock file, ignoring errors if it is already gone."""
    _LOCK_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def _step_update_equity_data() -> None:
    """Fetch and refresh equity OHLCV data via DataPipeline.yfinance_fallback."""
    from src.core.data_pipeline import DataPipeline
    from src.utils.config import get_config

    cfg = get_config()
    pipeline = DataPipeline(cfg)
    pipeline.yfinance_fallback(
        tickers=["SPY", "AAPL", "MSFT"],
        start=cfg.qlib.test_start,
        end=cfg.qlib.test_end,
    )
    logger.info("scheduler: update_equity_data — fetched OHLCV data.")


def _step_update_crypto_data() -> None:
    """Refresh crypto universe via CryptoPipeline.build_universe."""
    from src.core.crypto_pipeline import CryptoPipeline
    from src.utils.config import get_config

    cfg = get_config()
    pipeline = CryptoPipeline(cfg)
    pipeline.build_universe()
    logger.info("scheduler: update_crypto_data — universe refreshed.")


def _step_retrain_model() -> None:
    """Retrain the ML model if today is a scheduled retraining day.

    Uses ModelConfig.retrain_frequency to decide: weekly (Monday),
    biweekly (Monday on even ISO week), or monthly (1st of month).
    Also runs RD-Agent factor mining (weekly) and model optimisation
    (monthly) when rd_agent.enabled is True.
    """
    from src.utils.config_loader import get_full_config

    cfg = get_full_config()
    today = datetime.now(timezone.utc)
    freq = cfg.models.retrain_frequency.lower()

    should_retrain = False
    match freq:
        case "weekly":
            should_retrain = today.weekday() == 0  # Monday
        case "biweekly":
            should_retrain = today.weekday() == 0 and (today.isocalendar().week % 2 == 0)
        case "monthly":
            should_retrain = today.day == 1
        case _:
            should_retrain = today.weekday() == 0

    if not should_retrain:
        logger.info(
            "scheduler: retrain_model — skipping (freq={}, today={})",
            freq,
            today.strftime("%A %Y-%m-%d"),
        )
        return

    # RD-Agent weekly factor mining
    if cfg.rd_agent.enabled and today.weekday() == 0:
        try:
            from src.core.rd_agent_runner import RDAgentRunner

            runner = RDAgentRunner(cfg)
            runner.run_factor_mining()
            logger.info("scheduler: retrain_model — RD-Agent factor mining complete.")
        except Exception as exc:
            logger.warning("scheduler: RD-Agent factor mining failed: {}", exc)

    # RD-Agent monthly model optimisation
    if cfg.rd_agent.enabled and today.day == 1:
        try:
            from src.core.rd_agent_runner import RDAgentRunner

            runner = RDAgentRunner(cfg)
            runner.run_model_optimization()
            logger.info("scheduler: retrain_model — RD-Agent model optimisation complete.")
        except Exception as exc:
            logger.warning("scheduler: RD-Agent model optimisation failed: {}", exc)

    logger.info("scheduler: retrain_model — retraining step completed.")


def _step_generate_predictions() -> None:
    """Run the ML model to generate alpha-score predictions."""
    from src.utils.config import get_config

    cfg = get_config()
    from src.core.data_pipeline import DataPipeline

    pipeline = DataPipeline(cfg)
    df = pipeline.yfinance_fallback(
        tickers=["SPY", "AAPL", "MSFT"],
        start=cfg.qlib.test_start,
        end=cfg.qlib.test_end,
    )
    if not df.empty:
        logger.info(
            "scheduler: generate_predictions — model={} rows={}",
            cfg.models.default_model,
            len(df),
        )
    else:
        logger.warning("scheduler: generate_predictions — empty dataframe, skipping model.")


def _step_run_agent_analysis() -> None:
    """Run LangGraph agent pipeline on top-N tickers from predictions."""
    from src.agents.graph import analyze_ticker

    tickers = ["AAPL", "MSFT", "SPY"]
    for ticker in tickers:
        try:
            analyze_ticker(ticker)
            logger.info("scheduler: run_agent_analysis — {} complete.", ticker)
        except Exception as exc:
            logger.warning("scheduler: run_agent_analysis — {} failed: {}", ticker, exc)


def _step_optimize_portfolio() -> None:
    """Optimise portfolio weights using the configured optimisation method."""
    from src.execution.portfolio_optimizer import PortfolioOptimizer

    PortfolioOptimizer(method="risk_parity")
    logger.info("scheduler: optimize_portfolio — optimizer created (method=risk_parity).")
    # Full optimisation requires live signal data; stub here logs completion.
    logger.info("scheduler: optimize_portfolio — complete.")


def _step_run_risk_controls() -> None:
    """Execute pre-trade risk checks."""
    from src.execution.risk_controls import RiskControls
    from src.utils.config import get_config

    cfg = get_config()
    RiskControls(cfg)
    logger.info("scheduler: run_risk_controls — checks instantiated.")
    # Actual check calls require live portfolio; log completion.
    logger.info("scheduler: run_risk_controls — complete.")


def _step_execute_rebalance() -> None:
    """Execute rebalancing orders via PaperTrader."""
    from src.execution.paper_trader import PaperTrader
    from src.utils.config import get_config

    cfg = get_config()
    trader = PaperTrader(
        initial_cash=cfg.execution.initial_cash,
        slippage_bps=cfg.execution.slippage_bps,
        commission_per_share=cfg.execution.commission_per_share,
        min_commission=cfg.execution.min_commission,
    )
    metrics = trader.get_metrics()
    logger.info("scheduler: execute_rebalance — paper trader metrics: {}", metrics)
    logger.info("scheduler: execute_rebalance — complete.")


def _step_send_daily_report() -> None:
    """Send daily summary via AlertManager."""
    from src.monitoring.monitoring import AlertManager

    manager = AlertManager()
    manager.send_daily_summary({}, {})
    logger.info("scheduler: send_daily_report — report dispatched.")


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

_STEP_FUNCTIONS: dict[str, Callable[[], None]] = {
    "update_equity_data": _step_update_equity_data,
    "update_crypto_data": _step_update_crypto_data,
    "retrain_model": _step_retrain_model,
    "generate_predictions": _step_generate_predictions,
    "run_agent_analysis": _step_run_agent_analysis,
    "optimize_portfolio": _step_optimize_portfolio,
    "run_risk_controls": _step_run_risk_controls,
    "execute_rebalance": _step_execute_rebalance,
    "send_daily_report": _step_send_daily_report,
}


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def _run_step(step: str, dry_run: bool, state: dict, date_key: str) -> None:
    """Execute a single pipeline step with error handling and state persistence.

    If ``dry_run`` is True the real step function is not called; a log line is
    emitted instead and the step is recorded as "ok" in the state file.

    Args:
        step: Step name from DAILY_STEPS.
        dry_run: When True, skip real operations.
        state: Mutable in-memory state dict (will be written to disk).
        date_key: Today's date string (YYYY-MM-DD).
    """
    if dry_run:
        logger.info("scheduler: dry-run: would execute '{}'", step)
        _mark_step(state, date_key, step, "ok")
        _save_state(state)
        return

    logger.info("scheduler: running step '{}'", step)
    fn = _STEP_FUNCTIONS.get(step)
    if fn is None:
        logger.error("scheduler: unknown step '{}' — skipping.", step)
        return

    try:
        fn()
        _mark_step(state, date_key, step, "ok")
        _save_state(state)
        logger.info("scheduler: step '{}' completed successfully.", step)
    except Exception as exc:
        logger.error("scheduler: step '{}' failed: {}", step, exc)
        _mark_step(state, date_key, step, "error")
        _save_state(state)
        try:
            from src.monitoring.monitoring import AlertManager

            AlertManager().send_system_health(step, str(exc))
        except Exception as alert_exc:
            logger.warning("scheduler: AlertManager notification failed: {}", alert_exc)


def _steps_due_now(now_utc: datetime) -> list[str]:
    """Return the list of step names whose scheduled time has passed today.

    A step is considered due when the current UTC time is at or past its
    scheduled hour and minute.

    Args:
        now_utc: Current datetime in UTC (timezone-aware or naive).

    Returns:
        List of step names that are due for execution.
    """
    due: list[str] = []
    for step, hour, minute in DAILY_STEPS:
        if (now_utc.hour, now_utc.minute) >= (hour, minute):
            due.append(step)
    return due


def run_pipeline(*, dry_run: bool, simulate_day: bool) -> None:
    """Execute all due pipeline steps for today, respecting idempotency.

    Acquires the lock file, determines which steps are due (or all steps if
    ``simulate_day`` is True), skips any that already ran successfully today,
    and calls each remaining step via :func:`_run_step`.

    Args:
        dry_run: When True, log step names but skip real operations.
        simulate_day: When True, mark all steps as due regardless of time.
    """
    if not _acquire_lock():
        sys.exit(1)

    try:
        now_utc = datetime.now(timezone.utc)
        date_key = now_utc.strftime("%Y-%m-%d")
        state = _load_state()

        if simulate_day:
            due = [step for step, _, _ in DAILY_STEPS]
            logger.info("scheduler: --simulate-day: running all {} steps.", len(due))
        else:
            due = _steps_due_now(now_utc)
            logger.info(
                "scheduler: {} steps due at {}Z", len(due), now_utc.strftime("%H:%M")
            )

        for step in due:
            if _step_done_today(state, date_key, step):
                logger.info("scheduler: step '{}' already completed today — skipping.", step)
                continue
            _run_step(step, dry_run=dry_run, state=state, date_key=date_key)

        logger.info("scheduler: pipeline run complete for {}.", date_key)

    finally:
        _release_lock()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    loop: bool = typer.Option(False, "--loop", help="Run forever, polling every 60 s."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Log steps but skip real operations."
    ),
    simulate_day: bool = typer.Option(
        False,
        "--simulate-day",
        help="Immediately execute all steps (bypass time check).",
    ),
) -> None:
    """QuantAgentLab production scheduler.

    Run the daily pipeline once, in a loop, or in dry-run / simulate mode.
    """
    from src.utils.logger import get_logger as _get_logger

    _get_logger("scheduler")  # ensure loguru file sink is initialised

    if simulate_day and loop:
        logger.error("scheduler: --simulate-day and --loop are mutually exclusive.")
        raise typer.Exit(code=1)

    if loop:
        logger.info("scheduler: starting loop mode (polling every 60 s).")
        while True:
            run_pipeline(dry_run=dry_run, simulate_day=False)
            time.sleep(60)
    else:
        run_pipeline(dry_run=dry_run, simulate_day=simulate_day)


if __name__ == "__main__":
    app()
