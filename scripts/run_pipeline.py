#!/usr/bin/env python3
"""Canonical paper-cycle CLI; all economics live in shared operating services."""
from __future__ import annotations

import typer

from scripts.run_operations import (
    build_cycle,
    cycle,
    dispatch_pending,
    prepare_cycle,
    schedule_app,
    status,
)

app = typer.Typer(help="Source-bound paper operation and durable scheduling.", add_completion=False)
app.command("run")(cycle)
app.command("prepare")(prepare_cycle)
app.command("build")(build_cycle)
app.command("dispatch-pending")(dispatch_pending)
app.command("status")(status)
app.add_typer(schedule_app, name="schedule")


def _run_pipeline_once(plan, *, config=None):
    """Programmatic compatibility entry requires the same explicit typed plan."""
    from src.operations.application import execute_cycle
    from src.operations.contracts import PaperCyclePlan
    from src.utils.config import get_config
    if not isinstance(plan, (dict, PaperCyclePlan)):
        raise ValueError("explicit PaperCyclePlan required; backtests use scripts.run_backtest")
    return execute_cycle(plan, config or get_config())


if __name__ == "__main__":
    app()
