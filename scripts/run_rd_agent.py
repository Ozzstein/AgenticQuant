"""CLI for RD-Agent factor/model discovery.

Usage::

    conda run -n aiquant python scripts/run_rd_agent.py --help
    conda run -n aiquant python scripts/run_rd_agent.py library-status
    conda run -n aiquant python scripts/run_rd_agent.py co-optimize --iterations 10
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the Python path when executed directly.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from src.core.rd_agent_runner import RDAgentRunner
from src.utils.logger import get_logger

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
