"""Typer CLI for QuantAgentLab."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

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


@app.command()
def backtest(
    model: str = typer.Option("LightGBM", "--model", "-m", help="Model name"),
    topk: int = typer.Option(30, "--topk", "-k", help="Top K stocks to hold"),
    start: str = typer.Option("2020-01-01", "--start", help="Backtest start date"),
    end: str = typer.Option("2023-12-31", "--end", help="Backtest end date"),
    output: str = typer.Option("rich", "--output", "-o", help="Output format: rich, json"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Run validation checks"),
    benchmark: bool = typer.Option(True, "--benchmark/--no-benchmark", help="Run benchmarks"),
):
    """Run walk-forward backtest with validation and benchmarks."""
    from src.core.backtester import WalkForwardBacktester
    from src.core.data_pipeline import DataPipeline
    from src.core.model_zoo import ModelWrapper
    from src.utils.config import get_config

    config = get_config()
    console.print(f"[cyan]Running backtest: {model} top-{topk} ({start} to {end})[/cyan]")

    pipeline = DataPipeline(config)
    backtester = WalkForwardBacktester(config)
    model_wrapper = ModelWrapper(model, config)

    features, labels = pipeline.get_features(start, end)
    result = backtester.run(features, labels, model_wrapper, topk=topk)

    if output == "json":
        console.print(result.model_dump_json(indent=2))
    else:
        _print_backtest_result(result)

    if validate:
        from src.core.backtester import BacktestValidator

        validator = BacktestValidator()
        validation = validator.validate(result)
        console.print(f"\n[bold]Validation: {validation.verdict.value}[/bold]")

    if benchmark:
        from src.core.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(config)
        comparison = runner.compare(result, start, end)
        console.print(f"[dim]Monte Carlo percentile: {comparison.monte_carlo_percentile:.1f}%[/dim]")


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Ticker symbol to analyze"),
    provider: str = typer.Option("anthropic", "--provider", "-p", help="LLM provider"),
    output: str = typer.Option("rich", "--output", "-o", help="Output format: rich, json"),
):
    """Run multi-agent analysis on a ticker."""
    from src.agents.graph import build_trading_desk_graph
    from src.utils.config import get_config

    config = get_config()
    console.print(f"[cyan]Analyzing {ticker} with {provider}...[/cyan]")

    graph = build_trading_desk_graph(config)
    result = graph.invoke({"ticker": ticker, "asset_class": "EQUITY"})

    if result.get("final_decision"):
        decision = result["final_decision"]
        if output == "json":
            console.print(decision.model_dump_json(indent=2))
        else:
            _print_analysis_result(decision)
    else:
        console.print("[red]Analysis failed. Check errors.[/red]")
        for err in result.get("errors", []):
            console.print(f"  [red]{err}[/red]")


@app.command()
def trade(
    action: str = typer.Argument("status", help="Action: status, run, reset"),
):
    """Paper trading operations."""
    from src.execution.paper_trader import PaperTrader
    from src.utils.config import get_config

    config = get_config()
    trader = PaperTrader(config)

    if action == "status":
        portfolio = trader.get_portfolio()
        console.print(f"[green]NAV:[/green] ${portfolio.nav:,.2f}")
        console.print(f"[green]Cash:[/green] ${portfolio.cash:,.2f}")
        console.print(f"[green]Positions:[/green] {len(portfolio.positions)}")
    elif action == "reset":
        trader.reset()
        console.print("[yellow]Paper portfolio reset.[/yellow]")
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


# ---------------------------------------------------------------------------
# RD-Agent sub-app
# ---------------------------------------------------------------------------

rdagent_app = typer.Typer(name="rd-agent", help="RD-Agent factor and model discovery")
app.add_typer(rdagent_app)


@rdagent_app.command("mine-factors")
def rdagent_mine_factors(
    iterations: int = typer.Option(50, "--iterations", "-n", help="Number of factor iterations"),
    min_ic: float = typer.Option(0.02, "--min-ic", help="Minimum IC threshold to accept a factor"),
) -> None:
    """Discover new alpha factors using RD-Agent."""
    from src.utils.config_loader import get_full_config

    cfg = get_full_config()
    rd = cfg.rd_agent
    console.print("[cyan]RD-Agent: mine-factors[/cyan]")
    console.print(f"  enabled           : {rd.enabled}")
    console.print(f"  factor_library_dir: {rd.factor_library_dir}")
    console.print(f"  iterations        : {iterations}")
    console.print(f"  min_ic            : {min_ic}")
    console.print("[dim](Stub — no real RD-Agent implementation yet.)[/dim]")


@rdagent_app.command("optimize-model")
def rdagent_optimize_model(
    iterations: int = typer.Option(20, "--iterations", "-n", help="Number of model iterations"),
    budget: int = typer.Option(10, "--budget", "-b", help="Compute budget"),
) -> None:
    """Optimize ML model configuration using RD-Agent."""
    from src.utils.config_loader import get_full_config

    cfg = get_full_config()
    rd = cfg.rd_agent
    console.print("[cyan]RD-Agent: optimize-model[/cyan]")
    console.print(f"  enabled                 : {rd.enabled}")
    console.print(f"  best_model_config_path  : {rd.best_model_config_path}")
    console.print(f"  iterations              : {iterations}")
    console.print(f"  budget                  : {budget}")
    console.print("[dim](Stub — no real RD-Agent implementation yet.)[/dim]")


@rdagent_app.command("library-status")
def rdagent_library_status() -> None:
    """Show factor library contents and statistics."""
    from pathlib import Path

    from src.utils.config_loader import get_full_config

    cfg = get_full_config()
    lib_dir = Path(cfg.rd_agent.factor_library_dir)
    console.print("[cyan]RD-Agent: library-status[/cyan]")
    console.print(f"  factor_library_dir: {lib_dir}")
    console.print(f"  exists            : {lib_dir.exists()}")
    if lib_dir.exists():
        entries = list(lib_dir.iterdir())
        console.print(f"  entries           : {len(entries)}")
    console.print("[dim](Stub — no real RD-Agent implementation yet.)[/dim]")


@rdagent_app.command("validate-library")
def rdagent_validate_library(
    min_ic: float = typer.Option(0.02, "--min-ic", help="Minimum IC threshold for validation"),
    min_icir: float = typer.Option(0.3, "--min-icir", help="Minimum ICIR threshold for validation"),
) -> None:
    """Validate factor library entries against IC threshold."""
    from src.utils.config_loader import get_full_config

    cfg = get_full_config()
    rd = cfg.rd_agent
    console.print("[cyan]RD-Agent: validate-library[/cyan]")
    console.print(f"  factor_library_dir: {rd.factor_library_dir}")
    console.print(f"  min_ic            : {min_ic}")
    console.print(f"  min_icir          : {min_icir}")
    console.print("[dim](Stub — no real RD-Agent implementation yet.)[/dim]")
