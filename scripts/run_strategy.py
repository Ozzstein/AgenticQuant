#!/usr/bin/env python3
"""Strategy library management CLI."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from repo root: `python scripts/run_strategy.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.core.strategy_selector import StrategySelector
from src.core.strategy_tracker import StrategyTracker
from src.utils.schemas import Strategy

app = typer.Typer(
    name="run-strategy",
    help="Strategy library management — list, compare, inspect, and validate strategies.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# `list` command
# ---------------------------------------------------------------------------


@app.command("list")
def list_strategies() -> None:
    """Show all strategies in the library with key metrics.

    Loads every strategy from data/strategy_library/ and cross-references
    live performance data from StrategyTracker.  Columns displayed:

    Name, Model, Universe, Rebalance, Regimes, Backtest Sharpe,
    Live Sharpe (60d), Drift, Validated.
    """
    selector = StrategySelector()
    tracker = StrategyTracker()
    performances = tracker.get_performances()

    if not selector.strategies:
        console.print("[yellow]No strategies found in library.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(
        title=f"Strategy Library ({len(selector.strategies)} strategies)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="bold", min_width=20)
    table.add_column("Model", min_width=10)
    table.add_column("Universe", min_width=10)
    table.add_column("Rebalance", justify="center")
    table.add_column("Regimes", min_width=20)
    table.add_column("BT Sharpe", justify="right")
    table.add_column("Live Sharpe", justify="right")
    table.add_column("Drift", justify="center")
    table.add_column("Validated", justify="center")

    for strat in selector.strategies:
        perf = performances.get(strat.name)

        regimes_str = ", ".join(r.value for r in strat.regime_applicability) or "-"
        bt_sharpe = f"{strat.backtest_sharpe:.3f}" if strat.backtest_sharpe != 0.0 else "-"

        if perf is not None:
            live_sharpe = f"{perf.rolling_sharpe_60d:.3f}"
            drift_text = Text("YES", style="red bold") if perf.drift_detected else Text("no", style="green")
        else:
            live_sharpe = "-"
            drift_text = Text("-")

        validated_text = Text("✓", style="green bold") if strat.validated else Text("✗", style="dim")

        table.add_row(
            strat.name,
            strat.model,
            strat.universe,
            strat.rebalance_frequency.value,
            regimes_str,
            bt_sharpe,
            live_sharpe,
            drift_text,
            validated_text,
        )

    console.print(table)
    logger.info("Listed {} strategies", len(selector.strategies))


# ---------------------------------------------------------------------------
# `compare` command
# ---------------------------------------------------------------------------


@app.command("compare")
def compare_strategies(
    regime: str = typer.Option(
        "risk_on",
        "--regime",
        "-r",
        help="Macro regime to filter strategies by (risk_on | neutral | risk_off | crisis).",
    ),
) -> None:
    """Compare strategies side-by-side for a given regime.

    Filters the strategy library to those applicable to *regime*, then
    shows a comparison table with factor sets, sizing methods, entry rules,
    backtest metrics, and validation status.  The strategy that would be
    selected (highest backtest Sharpe among matching) is highlighted.

    Args:
        regime: Macro regime string used to filter strategies.
    """
    from src.utils.schemas import MacroRegime

    # Normalise and validate the regime string.
    try:
        macro = MacroRegime(regime)
    except ValueError:
        valid = [m.value for m in MacroRegime]
        console.print(
            f"[red]Unknown regime '{regime}'. Valid values: {valid}[/red]"
        )
        raise typer.Exit(code=1)

    selector = StrategySelector()

    matching = [s for s in selector.strategies if macro in s.regime_applicability]

    if not matching:
        console.print(
            f"[yellow]No strategies applicable to regime '{regime}'.[/yellow]"
        )
        raise typer.Exit(code=0)

    # Determine which strategy would be selected (highest backtest Sharpe).
    best = max(matching, key=lambda s: s.backtest_sharpe)

    table = Table(
        title=f"Strategy Comparison — Regime: {regime} ({len(matching)} matching)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="bold", min_width=20)
    table.add_column("Factor Set", min_width=20)
    table.add_column("Sizing Method", min_width=18)
    table.add_column("Entry Rules", min_width=22)
    table.add_column("BT Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Validated", justify="center")
    table.add_column("Would Select", justify="center")

    for strat in sorted(matching, key=lambda s: s.backtest_sharpe, reverse=True):
        factors_str = ", ".join(strat.factor_set[:3])
        if len(strat.factor_set) > 3:
            factors_str += f" (+{len(strat.factor_set) - 3})"

        entry_str = "; ".join(f"{k}={v}" for k, v in list(strat.entry_rules.items())[:2])
        if len(strat.entry_rules) > 2:
            entry_str += " ..."

        bt_sharpe = f"{strat.backtest_sharpe:.3f}" if strat.backtest_sharpe != 0.0 else "-"
        max_dd = f"{strat.backtest_max_drawdown:.2%}" if strat.backtest_max_drawdown != 0.0 else "-"
        validated_text = Text("✓", style="green bold") if strat.validated else Text("✗", style="dim")
        select_text = Text("★ YES", style="bold yellow") if strat.name == best.name else Text("")

        table.add_row(
            strat.name,
            factors_str or "-",
            strat.sizing_method.value,
            entry_str or "-",
            bt_sharpe,
            max_dd,
            validated_text,
            select_text,
        )

    console.print(table)
    if best.backtest_sharpe > 0:
        console.print(
            f"\n[bold]Would select:[/bold] [yellow]{best.name}[/yellow] "
            f"(backtest Sharpe: {best.backtest_sharpe:.3f})"
        )
    else:
        console.print(
            f"\n[bold]Would select:[/bold] [yellow]{best.name}[/yellow] "
            f"(no backtest data — first alphabetically)"
        )
    logger.info(
        "Compared {} strategies for regime '{}'; best={}", len(matching), regime, best.name
    )


# ---------------------------------------------------------------------------
# `detail` command
# ---------------------------------------------------------------------------


@app.command("detail")
def detail_strategy(
    name: str = typer.Argument(..., help="Strategy name to inspect."),
) -> None:
    """Show full details for a strategy.

    Prints all fields of the Strategy definition in a Rich Panel, followed
    by live performance metrics if any have been recorded.

    Args:
        name: Exact strategy name as stored in data/strategy_library/.
    """
    selector = StrategySelector()
    tracker = StrategyTracker()

    strat = next((s for s in selector.strategies if s.name == name), None)

    if strat is None:
        available = [s.name for s in selector.strategies]
        console.print(f"[red]Strategy '{name}' not found.[/red]")
        if available:
            console.print(f"Available strategies: {available}")
        raise typer.Exit(code=1)

    # Build detail lines.
    lines: list[str] = [
        f"[bold]Name:[/bold]                {strat.name}",
        f"[bold]Description:[/bold]         {strat.description or '(none)'}",
        f"[bold]Model:[/bold]               {strat.model}",
        f"[bold]Universe:[/bold]            {strat.universe}",
        f"[bold]Factor Set:[/bold]          {', '.join(strat.factor_set) or '(empty)'}",
        f"[bold]Sizing Method:[/bold]       {strat.sizing_method.value}",
        f"[bold]Rebalance Freq.:[/bold]     {strat.rebalance_frequency.value}",
        f"[bold]Regime Applicability:[/bold] {', '.join(r.value for r in strat.regime_applicability)}",
        f"[bold]Backtest Sharpe:[/bold]     {strat.backtest_sharpe:.4f}",
        f"[bold]Backtest Max DD:[/bold]     {strat.backtest_max_drawdown:.4f}",
        f"[bold]Validated:[/bold]           {'Yes' if strat.validated else 'No'}",
        f"[bold]Source:[/bold]              {strat.source}",
    ]

    if strat.entry_rules:
        lines.append("[bold]Entry Rules:[/bold]")
        for k, v in strat.entry_rules.items():
            lines.append(f"  {k}: {v}")

    if strat.exit_rules:
        lines.append("[bold]Exit Rules:[/bold]")
        for k, v in strat.exit_rules.items():
            lines.append(f"  {k}: {v}")

    if strat.risk_overrides:
        lines.append("[bold]Risk Overrides:[/bold]")
        for k, v in strat.risk_overrides.items():
            lines.append(f"  {k}: {v}")

    detail_text = "\n".join(lines)
    console.print(Panel(detail_text, title=f"Strategy: {strat.name}", border_style="cyan"))

    # Show live performance if available.
    perf = tracker.performances.get(strat.name)
    if perf is not None:
        perf_lines: list[str] = [
            f"[bold]Rolling Sharpe (30d):[/bold]  {perf.rolling_sharpe_30d:.4f}",
            f"[bold]Rolling Sharpe (60d):[/bold]  {perf.rolling_sharpe_60d:.4f}",
            f"[bold]Rolling Sharpe (90d):[/bold]  {perf.rolling_sharpe_90d:.4f}",
            f"[bold]Max Drawdown (live):[/bold]   {perf.max_drawdown:.4f}",
            f"[bold]Current Drawdown:[/bold]      {perf.current_drawdown:.4f}",
            f"[bold]Win Rate:[/bold]              {perf.win_rate:.2%}",
            f"[bold]Total P&L:[/bold]             {perf.total_pnl:.2f}",
            f"[bold]Days Tracked:[/bold]          {len(perf.daily_pnl)}",
            f"[bold]Drift Detected:[/bold]        {'YES' if perf.drift_detected else 'No'}",
            f"[bold]Last Updated:[/bold]          {perf.last_updated}",
        ]
        perf_text = "\n".join(perf_lines)
        console.print(Panel(perf_text, title="Live Performance", border_style="green"))
    else:
        console.print("[dim]No live performance data recorded yet for this strategy.[/dim]")

    logger.info("Displayed detail for strategy '{}'", name)


# ---------------------------------------------------------------------------
# `validate` command
# ---------------------------------------------------------------------------


@app.command("validate")
def validate_strategy(
    name: str = typer.Argument(..., help="Strategy name to validate."),
) -> None:
    """Run validation checks on a strategy.

    Checks performed:

    1. Strategy name exists and is unique in the library.
    2. factor_set is non-empty.
    3. regime_applicability is non-empty.
    4. If validated=True, backtest_sharpe must be > 0.
    5. entry_rules is non-empty.
    6. model field is non-empty.

    Exits with code 1 if any check fails.

    Args:
        name: Exact strategy name to validate.
    """
    selector = StrategySelector()

    # --- Check 1: strategy exists and is unique ---
    matching = [s for s in selector.strategies if s.name == name]
    exists = len(matching) >= 1
    unique = len(matching) == 1

    if not exists:
        console.print(f"[red]Strategy '{name}' not found in library.[/red]")
        available = [s.name for s in selector.strategies]
        if available:
            console.print(f"Available: {available}")
        raise typer.Exit(code=1)

    strat: Strategy = matching[0]

    checks: list[tuple[str, bool, str]] = []

    # Existence + uniqueness
    checks.append((
        "Strategy exists in library",
        exists,
        f"Found {len(matching)} definition(s) with name '{name}'",
    ))
    checks.append((
        "Name is unique in library",
        unique,
        f"{len(matching)} definition(s) found — expected exactly 1",
    ))

    # factor_set non-empty
    has_factors = bool(strat.factor_set)
    checks.append((
        "factor_set is non-empty",
        has_factors,
        f"factor_set has {len(strat.factor_set)} entries",
    ))

    # regime_applicability non-empty
    has_regimes = bool(strat.regime_applicability)
    checks.append((
        "regime_applicability is non-empty",
        has_regimes,
        f"regime_applicability has {len(strat.regime_applicability)} entries",
    ))

    # If validated=True, backtest_sharpe must be > 0
    if strat.validated:
        sharpe_ok = strat.backtest_sharpe > 0.0
        checks.append((
            "backtest_sharpe > 0 (required when validated=True)",
            sharpe_ok,
            f"backtest_sharpe = {strat.backtest_sharpe:.4f}",
        ))
    else:
        checks.append((
            "backtest_sharpe check (skipped — validated=False)",
            True,
            "Strategy is not yet validated; Sharpe check waived",
        ))

    # entry_rules non-empty
    has_entry = bool(strat.entry_rules)
    checks.append((
        "entry_rules is non-empty",
        has_entry,
        f"entry_rules has {len(strat.entry_rules)} entries",
    ))

    # model non-empty
    has_model = bool(strat.model.strip())
    checks.append((
        "model field is non-empty",
        has_model,
        f"model = '{strat.model}'",
    ))

    # --- Print results ---
    console.print(f"\n[bold]Validation report for:[/bold] [cyan]{name}[/cyan]\n")
    all_passed = True
    for check_name, passed, detail in checks:
        symbol = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"  {symbol}  {check_name}")
        console.print(f"      [dim]{detail}[/dim]")
        if not passed:
            all_passed = False

    console.print()
    if all_passed:
        console.print("[bold green]All checks passed.[/bold green]")
        logger.info("Strategy '{}' passed all validation checks", name)
    else:
        console.print("[bold red]One or more checks FAILED.[/bold red]")
        logger.warning("Strategy '{}' failed validation", name)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
