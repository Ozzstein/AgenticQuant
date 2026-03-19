"""CLI for RD-Agent factor/model discovery.

Usage::

    conda run -n aiquant python scripts/run_rd_agent.py --help
    conda run -n aiquant python scripts/run_rd_agent.py library-status
    conda run -n aiquant python scripts/run_rd_agent.py co-optimize --iterations 10
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Ensure the project root is on the Python path when executed directly.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.pretty import Pretty  # noqa: E402
from rich.table import Table  # noqa: E402

from src.core.rd_agent_runner import RDAgentRunner  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

app = typer.Typer(
    name="run_rd_agent",
    help="RD-Agent factor/model discovery CLI",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _runner() -> RDAgentRunner:
    """Instantiate a fresh :class:`~src.core.rd_agent_runner.RDAgentRunner`."""
    return RDAgentRunner()


def _print_dict(title: str, data: dict) -> None:
    """Pretty-print a dict inside a Rich panel."""
    console.print(Panel(Pretty(data), title=title, border_style="blue"))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("co-optimize")
def co_optimize(
    iterations: int = typer.Option(10, "--iterations", "-i", help="Evaluation iterations."),
    budget: float = typer.Option(1.0, "--budget", "-b", help="Budget multiplier (0-1)."),
    traces: int = typer.Option(1, "--traces", "-t", help="Number of independent traces."),
) -> None:
    """Joint factor-model optimisation loop.

    Runs ``iterations`` factor evaluations and a quick model hyper-
    parameter search, then saves results to the factor library and KB.
    """
    console.print(
        f"[bold cyan]co-optimize[/bold cyan]: iterations={iterations}, "
        f"budget={budget}, traces={traces}"
    )
    runner = _runner()
    result = runner.co_optimize(iterations=iterations, budget=budget, traces=traces)
    _print_dict("co-optimize result", result)


@app.command("mine-factors")
def mine_factors(
    iterations: int = typer.Option(10, "--iterations", "-i", help="Evaluation iterations."),
    min_ic: float = typer.Option(0.02, "--min-ic", help="Minimum absolute IC threshold."),
) -> None:
    """Factor-only evolution — no model optimisation.

    Generates factor proposals and evaluates their IC against historical
    data; accepted factors are saved to the factor library.
    """
    console.print(
        f"[bold cyan]mine-factors[/bold cyan]: iterations={iterations}, min_ic={min_ic}"
    )
    runner = _runner()
    factors = runner.mine_factors(iterations=iterations, min_ic=min_ic)

    table = Table(title=f"Accepted Factors ({len(factors)})", show_lines=True)
    table.add_column("Name", style="green")
    table.add_column("Category")
    table.add_column("IC Mean", justify="right")
    table.add_column("ICIR", justify="right")
    for f in factors:
        table.add_row(f.name, f.category, f"{f.ic_mean:.4f}", f"{f.icir:.4f}")

    console.print(table)
    console.print(f"[green]Total accepted: {len(factors)}[/green]")


@app.command("optimize-model")
def optimize_model(
    iterations: int = typer.Option(10, "--iterations", "-i", help="Config evaluations."),
    budget: float = typer.Option(1.0, "--budget", "-b", help="Budget multiplier (0-1)."),
) -> None:
    """LightGBM hyperparameter search using random search.

    Evaluates ``iterations`` random configs (scaled by ``budget``) and
    saves the best config to ``outputs/best_model_config.yaml``.
    """
    console.print(
        f"[bold cyan]optimize-model[/bold cyan]: iterations={iterations}, budget={budget}"
    )
    runner = _runner()
    result = runner.optimize_model(iterations=iterations, budget=budget)
    _print_dict("optimize-model result", result)


@app.command("multi-trace")
def multi_trace(
    traces: int = typer.Option(3, "--traces", "-t", help="Number of parallel traces."),
    iterations: int = typer.Option(5, "--iterations", "-i", help="Iterations per trace."),
) -> None:
    """Run multiple independent co_optimize traces in parallel and merge results.

    Uses a :class:`~concurrent.futures.ThreadPoolExecutor` to run ``traces``
    independent optimisation runs, then deduplicates the factor libraries.
    """
    console.print(
        f"[bold cyan]multi-trace[/bold cyan]: traces={traces}, iterations={iterations}"
    )
    runner = _runner()
    result = runner.multi_trace_co_optimize(traces=traces, iterations=iterations)

    summary = {
        "total_accepted": result["total_accepted"],
        "unique_factors_in_library": len(result["all_factors"]),
        "trace_results": result["trace_results"],
    }
    _print_dict("multi-trace result", summary)


@app.command("implement-paper")
def implement_paper(
    source: str = typer.Argument(..., help="URL, arXiv ID, or local file path."),
) -> None:
    """Derive a trading factor from a research paper or local text file.

    For remote URLs / arXiv IDs a stub message is printed (no keys needed
    for local files).
    """
    console.print(f"[bold cyan]implement-paper[/bold cyan]: source='{source}'")
    runner = _runner()
    factor = runner.implement_paper(source=source)

    if factor is None:
        console.print("[yellow]No factor derived from source.[/yellow]")
    else:
        _print_dict("Derived factor", factor.model_dump())


@app.command("library-status")
def library_status() -> None:
    """Report the current state of the factor library and knowledge base."""
    runner = _runner()
    status = runner.library_status()

    table = Table(title="Factor Library Status", show_header=False, show_lines=True)
    table.add_column("Key", style="bold")
    table.add_column("Value", justify="right")
    for key, value in status.items():
        table.add_row(str(key), str(value))

    console.print(table)


@app.command("validate-library")
def validate_library() -> None:
    """3-way comparison: Alpha158-only vs library-only vs combined."""
    runner = _runner()
    result = runner.validate_library()

    table = Table(title="Library Validation", show_lines=True)
    table.add_column("Scenario", style="bold")
    table.add_column("Sharpe Ratio", justify="right")
    table.add_row("Alpha158-only", f"{result['alpha158_sharpe']:.4f}")
    table.add_row("Library-only", f"{result['library_sharpe']:.4f}")
    table.add_row("Combined", f"{result['combined_sharpe']:.4f}")

    console.print(table)
    winner = result["winner"]
    console.print(f"[green bold]Winner:[/green bold] {winner}")


@app.command("copilot-factor")
def copilot_factor_cmd(
    description: str = typer.Argument(..., help="Natural-language description of the factor."),
) -> None:
    """Implement and evaluate a factor from a plain-English description.

    Matches keywords in the description to known factor templates, evaluates
    each, and saves passing factors to the library.

    Example::

        python scripts/run_rd_agent.py copilot-factor "momentum with rising earnings"
    """
    console.print(f"[bold cyan]copilot-factor[/bold cyan]: '{description}'")
    runner = _runner()
    result = runner.copilot_factor(description)

    _print_dict("copilot-factor result", {k: v for k, v in result.items() if k != "accepted"})

    accepted = result.get("accepted", [])
    if accepted:
        table = Table(title=f"Accepted Factors ({len(accepted)})", show_lines=True)
        table.add_column("Name", style="green")
        table.add_column("Category")
        table.add_column("IC Mean", justify="right")
        table.add_column("ICIR", justify="right")
        for f in accepted:
            table.add_row(
                f["name"],
                f["category"],
                f"{f['ic_mean']:.4f}",
                f"{f['icir']:.4f}",
            )
        console.print(table)
    else:
        console.print("[yellow]No factors accepted from this description.[/yellow]")


@app.command("copilot-model")
def copilot_model_cmd(
    source: str = typer.Option(..., "--source", "-s", help="URL, arXiv ID, or local file path."),
) -> None:
    """Build a model config from a paper or local text file.

    Reads the source, detects model architecture keywords, and produces a
    hyperparameter config. Saves to best_model_config.yaml if Sharpe > 0.5.

    Example::

        python scripts/run_rd_agent.py copilot-model --source ./papers/model.txt
    """
    console.print(f"[bold cyan]copilot-model[/bold cyan]: source='{source}'")
    runner = _runner()
    result = runner.copilot_model(source=source)
    _print_dict("copilot-model result", result)


@app.command("ui")
def ui_cmd(
    port: int = typer.Option(8080, "--port", "-p", help="Port to run the dashboard on."),
) -> None:
    """Launch the RD-Agent Streamlit dashboard.

    Shows knowledge base status, factor library, and R&D loop results.

    Example::

        python scripts/run_rd_agent.py ui --port 8080
    """
    dashboard_path = _ROOT / "src" / "rd_agent_dashboard.py"
    console.print(
        f"[bold cyan]ui[/bold cyan]: launching dashboard at http://localhost:{port}"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        check=False,
    )


@app.command("reset-knowledge")
def reset_knowledge() -> None:
    """Clear the knowledge base (irreversible).

    Prompts for confirmation before wiping all prior discoveries.
    """
    confirmed = typer.confirm(
        "This will permanently clear the knowledge base. Are you sure?",
        default=False,
    )
    if not confirmed:
        console.print("[yellow]Aborted — knowledge base unchanged.[/yellow]")
        raise typer.Abort()

    runner = _runner()
    runner.reset_knowledge()
    console.print("[green]Knowledge base has been reset.[/green]")


@app.command("evolve-strategies")
def evolve_strategies_cmd(
    iterations: int = typer.Option(10, "--iterations", "-i", help="Number of evolution iterations"),
    budget: float = typer.Option(15.0, "--budget", "-b", help="API budget in USD"),
    traces: int = typer.Option(1, "--traces", "-t", help="Parallel research threads"),
) -> None:
    """Propose, backtest, and save evolved strategies to the strategy library."""
    console.print(
        f"[bold cyan]evolve-strategies[/bold cyan]: iterations={iterations}, "
        f"budget={budget}, traces={traces}"
    )
    runner = _runner()
    results = runner.evolve_strategies(iterations=iterations, budget=budget, traces=traces)

    table = Table(title=f"Evolved Strategies ({len(results)})", show_lines=True)
    table.add_column("Name", style="green")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Validated")
    for s in results:
        table.add_row(
            s.get("name", ""),
            f"{s.get('backtest_sharpe', 0.0):.4f}",
            f"{s.get('backtest_max_drawdown', 0.0):.4f}",
            str(s.get("validated", False)),
        )

    console.print(table)
    typer.echo(f"Evolved {len(results)} strategies")


@app.command("evolve-regime")
def evolve_regime_cmd(
    iterations: int = typer.Option(10, "--iterations", "-i", help="Number of evolution iterations"),
    budget: float = typer.Option(10.0, "--budget", "-b", help="API budget in USD"),
) -> None:
    """Evolve the HMM regime detector via simulated backtest comparison loop.

    Cycles through 6 change archetypes (add/remove signal, change states,
    adjust threshold/window) and accepts changes that improve simulated Sharpe.
    Results are saved to the knowledge base.
    """
    console.print(
        f"[bold cyan]evolve-regime[/bold cyan]: iterations={iterations}, budget={budget}"
    )
    runner = _runner()
    results = runner.evolve_regime(iterations=iterations, budget=budget)

    table = Table(title=f"Regime Evolution Results ({len(results)} accepted)", show_lines=True)
    table.add_column("Iteration", justify="right")
    table.add_column("Change", style="green")
    table.add_column("Baseline Sharpe", justify="right")
    table.add_column("Modified Sharpe", justify="right")
    table.add_column("Improvement", justify="right")
    for r in results:
        table.add_row(
            str(r.get("iteration", "")),
            r.get("change_description", ""),
            f"{r.get('baseline_sharpe', 0.0):.4f}",
            f"{r.get('modified_sharpe', 0.0):.4f}",
            f"{r.get('improvement', 0.0):+.4f}",
        )

    console.print(table)
    typer.echo(f"Accepted {len(results)} regime changes out of {iterations} iterations")


@app.command("copilot-regime")
def copilot_regime_cmd(
    description: str = typer.Argument(..., help="Natural language regime change description"),
) -> None:
    """Apply a user-described change to the regime detector and evaluate it.

    Parses keywords to determine the change type (add/remove signal, try N
    states, adjust threshold), simulates a backtest, and reports results.

    Examples::

        python scripts/run_rd_agent.py copilot-regime "add put/call ratio as a signal"
        python scripts/run_rd_agent.py copilot-regime "try 5 states"
        python scripts/run_rd_agent.py copilot-regime "increase confidence threshold to 0.8"
    """
    console.print(f"[bold cyan]copilot-regime[/bold cyan]: description='{description}'")
    runner = _runner()
    result = runner.copilot_regime(description=description)

    if "error" in result:
        typer.echo(f"Error: {result['error']}", err=True)
        raise typer.Exit(1)

    _print_dict("copilot-regime result", result)
    typer.echo(f"Change: {result.get('change_description', '')}")
    typer.echo(f"Accepted: {result.get('accepted', False)}")
    typer.echo(f"Improvement: {result.get('improvement', 0.0):+.4f}")


@app.command("copilot-strategy")
def copilot_strategy_cmd(
    description: str = typer.Argument(..., help="Natural language strategy description"),
) -> None:
    """Convert an English strategy description to a validated Strategy definition."""
    console.print(
        f"[bold cyan]copilot-strategy[/bold cyan]: description='{description}'"
    )
    runner = _runner()
    result = runner.copilot_strategy(description=description)

    if "error" in result:
        typer.echo(f"Error: {result['error']}", err=True)
        raise typer.Exit(1)

    _print_dict("copilot-strategy result", result)
    typer.echo(f"Strategy: {result.get('name', 'unknown')}")
    typer.echo(f"Backtest Sharpe: {result.get('backtest_sharpe', 0.0):.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
