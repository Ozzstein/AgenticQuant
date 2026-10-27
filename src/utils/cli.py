"""Typer CLI for QuantAgentLab."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from scripts.run_agents import analyze as analyze_command
from scripts.run_agents import app as agents_app
from scripts.run_backtest import app as backtest_app
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


@app.command()
def trade(
    action: str = typer.Argument("status", help="Action: status, run, reset"),
    broker_type: str = typer.Option("auto", "--broker", "-b", help="Broker type: auto, paper, alpaca, ccxt"),
):
    """Paper/live trading operations."""
    from src.execution.broker import create_broker
    from src.execution.paper_trader import PaperTrader
    from src.utils.config import get_config

    config = get_config()

    if broker_type == "paper":
        broker = PaperTrader(
            initial_cash=config.lean.initial_cash if config.lean.initial_cash > 0 else 100_000.0,
            slippage_bps=5,
            commission_per_share=0.005,
        )
    elif broker_type == "alpaca":
        from src.execution.alpaca_trader import AlpacaTrader

        broker = AlpacaTrader(config.alpaca)
    elif broker_type == "ccxt":
        from src.execution.ccxt_trader import CcxtTrader

        broker = CcxtTrader(config.ccxt)
    else:  # auto
        broker = create_broker(config)

    if action == "status":
        portfolio = broker.portfolio
        console.print(f"[green]Broker:[/green] {type(broker).__name__}")
        console.print(f"[green]NAV:[/green] ${portfolio.nav:,.2f}")
        console.print(f"[green]Cash:[/green] ${portfolio.cash:,.2f}")
        console.print(f"[green]Positions:[/green] {len(portfolio.positions)}")
    else:
        console.print(f"[red]Unknown action: {action}[/red]")


@app.command()
def pipeline(
    action: str = typer.Argument("run", help="Action: run, paper-loop, setup"),
):
    """Run the daily trading pipeline."""
    from src.monitoring.scheduler import PipelineScheduler
    from src.utils.config import get_config

    config = get_config()

    if action == "run":
        scheduler = PipelineScheduler(config)
        scheduler.run_once()
    elif action == "paper-loop":
        scheduler = PipelineScheduler(config)
        scheduler.start()
    elif action == "setup":
        setup(download_data=True)
    else:
        console.print(f"[red]Unknown action: {action}[/red]")


@app.command()
def monitor():
    """Check portfolio health and risk status."""
    from src.monitoring.portfolio_monitor import PortfolioMonitor
    from src.utils.config import get_config

    config = get_config()
    monitor = PortfolioMonitor(config)
    report = monitor.generate_report()
    console.print(report)


@app.command()
def optimize(
    method: str = typer.Option("risk_parity", "--method", "-m", help="Optimization method"),
    tickers: Optional[str] = typer.Option(None, "--tickers", "-t", help="Comma-separated tickers"),
):
    """Run portfolio optimization."""
    console.print(f"[cyan]Optimizing with method: {method}[/cyan]")

    if tickers:
        ticker_list = [t.strip() for t in tickers.split(",")]
        console.print(f"[dim]Universe: {ticker_list}[/dim]")


def _print_backtest_result(result) -> None:
    """Pretty-print backtest results."""
    from rich.table import Table

    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    m = result.metrics
    table.add_row("Total Return", f"{m.total_return:.2%}")
    table.add_row("Annual Return", f"{m.annual_return:.2%}")
    table.add_row("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")
    table.add_row("Max Drawdown", f"{m.max_drawdown:.2%}")
    table.add_row("Calmar Ratio", f"{m.calmar_ratio:.2f}")
    table.add_row("Win Rate", f"{m.win_rate:.2%}")
    table.add_row("Total Trades", str(m.total_trades))
    table.add_row("Volatility", f"{m.volatility:.2%}")
    table.add_row("Sortino Ratio", f"{m.sortino_ratio:.2f}")
    console.print(table)


def _print_analysis_result(result) -> None:
    """Pretty-print analysis results."""
    from rich.panel import Panel

    panel_content = (
        f"Decision: {result.decision.value} ({result.confidence:.0f}% confidence)\n"
        f"Target: ${result.target_price:.2f}\n"
        f"Stop Loss: ${result.stop_loss:.2f}\n"
        f"Position Size: {result.position_size_pct:.1f}%\n"
        f"Time Horizon: {result.time_horizon}\n\n"
        f"Reasoning: {result.reasoning}\n\n"
        f"Risk Flags: {', '.join(result.risk_flags) or 'None'}\n"
        f"Catalysts: {', '.join(result.catalysts) or 'None'}"
    )
    console.print(Panel(panel_content, title=f"Analysis: {result.ticker}"))


# Canonical and direct research commands share one implementation.

app.add_typer(rdagent_app, name="rd-agent")
