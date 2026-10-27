"""Typer CLI for QuantAgentLab."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
from rich.console import Console

from scripts.run_agents import analyze as analyze_command
from scripts.run_agents import app as agents_app
from scripts.run_backtest import app as backtest_app
from scripts.run_operations import app as operations_app
from scripts.run_pipeline import app as pipeline_app
from scripts.run_rd_agent import app as rdagent_app

app = typer.Typer(
    name="aiquant",
    help="AI Quant Stack — LangGraph multi-agent quantitative trading platform",
    no_args_is_help=True,
)
console = Console()


@app.command()
def setup(
    download_data: bool = typer.Option(False, "--download", "-d", help="Download Qlib data"),
):
    """Initialize Qlib data and project directories."""
    from src.utils.config import get_config

    config = get_config()
    console.print(f"[green]Project:[/green] {config.project_name}")
    console.print(f"[green]Output dir:[/green] {config.output_dir}")

    if download_data:
        console.print("[yellow]Downloading Qlib data...[/yellow]")
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config)
        pipeline.init_qlib()
        console.print("[green]Qlib data initialized.[/green]")
    else:
        console.print("[dim]Skipping data download. Use --download to fetch Qlib data.[/dim]")


# One replay/validation CLI for direct scripts and the installed application.
# Old label-only backtests are intentionally not retained as a parallel route.
app.add_typer(backtest_app, name="backtest")


app.add_typer(agents_app, name="agents")
app.add_typer(operations_app, name="operations")
app.command("analyze")(analyze_command)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, add_help_option=False)
def data(ctx: typer.Context):
    """Archive, query, train, infer and export US Qlib through the shared data CLI."""
    from src.data.__main__ import main
    raise typer.Exit(main(ctx.args))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, add_help_option=False)
def decision(ctx: typer.Context):
    """Bounded Jev service with persistent observations and exact replay."""
    from src.decision.__main__ import main
    raise typer.Exit(main(ctx.args))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, add_help_option=False)
def lean(ctx: typer.Context):
    """Prepare or verify the canonical LEAN outbox project and runtime."""
    from src.execution.lean_bridge import main
    raise typer.Exit(main(ctx.args))


@app.command()
def trade(
    action: Literal["status", "run", "dispatch-pending"] = typer.Argument("status"),
    input: Path | None = typer.Option(None, exists=True, dir_okay=False),
):
    """Use the configured durable paper account and complete operating cycle."""
    from scripts.run_operations import cycle, dispatch_pending, status
    if action == "status":
        status()
        return
    if input is None:
        raise typer.BadParameter("an immutable PaperCyclePlan is required via --input")
    (cycle if action == "run" else dispatch_pending)(input)


app.add_typer(pipeline_app, name="pipeline")


@app.command()
def monitor():
    """Read actual ledger, operating, research, data and budget health."""
    from scripts.run_operations import status
    status()


@app.command()
def optimize(run_id: str = typer.Option(...)):
    """Inspect the actual constrained optimizer receipt of a paper cycle."""
    from scripts.run_operations import emit, fail
    from src.operations.store import OperationsStore
    from src.utils.config import get_config
    try:
        store = OperationsStore(get_config().operations.store_path, read_only=True)
        if store.manifest(run_id).kind != "paper_cycle":
            raise ValueError("optimization requires an actual paper-cycle identity")
        stage = store.status(run_id)["stages"].get("optimization", {})
        if stage.get("state") != "completed":
            raise ValueError("no completed optimization receipt for this run")
        emit(store.artifact(stage["output_hash"]))
    except Exception as exc:
        fail(exc)


# Canonical and direct research commands share one implementation.

app.add_typer(rdagent_app, name="rd-agent")
