#!/usr/bin/env python3
"""Multi-agent trading desk analysis CLI."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Allow running directly from repo root: `python scripts/run_agents.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="run_agents",
    help="Multi-agent trading desk analysis CLI.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_qlib_context(ticker: str) -> str:
    """Fetch last 60 days of price data and compute 20-day momentum score.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Formatted qlib_context string with 20-day momentum, or empty string on failure.
    """
    try:
        import yfinance as yf

        hist = yf.download(ticker, period="90d", progress=False, auto_adjust=True)
        if hist.empty or len(hist) < 22:
            logger.warning("Insufficient yfinance data for {} (got {} rows)", ticker, len(hist))
            return ""

        close = hist["Close"].squeeze()
        mom20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100
        ctx = f"Qlib alpha score: {mom20:.2f}% 20d momentum"
        logger.info("Computed qlib_context for {}: {}", ticker, ctx)
        return ctx
    except Exception as exc:
        logger.warning("Could not compute qlib_context for {}: {}", ticker, exc)
        return ""


def _print_rich_result(result, ticker: str) -> None:
    """Print an AnalysisResult as a rich Panel.

    Args:
        result: AnalysisResult instance.
        ticker: Ticker symbol used as panel title.
    """
    decision_colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "HOLD": "yellow",
        "SELL": "red",
        "STRONG_SELL": "bold red",
    }
    decision_str = result.decision.value if hasattr(result.decision, "value") else str(result.decision)
    color = decision_colors.get(decision_str, "white")

    lines: list[str] = []
    lines.append(f"[{color}]Decision:[/{color}] [{color}]{decision_str}[/{color}]")
    lines.append(f"[bold]Confidence:[/bold] {result.confidence:.1f}%")
    if result.target_price is not None:
        lines.append(f"[bold]Target Price:[/bold] ${result.target_price:.2f}")
    if result.stop_loss is not None:
        lines.append(f"[bold]Stop Loss:[/bold] ${result.stop_loss:.2f}")
    lines.append(f"[bold]Position Size:[/bold] {result.position_size_pct:.1f}%")
    lines.append(f"[bold]Time Horizon:[/bold] {result.time_horizon}")
    lines.append("")
    lines.append(f"[bold]Reasoning:[/bold] {result.reasoning}")

    if result.risk_flags:
        lines.append("")
        lines.append("[bold red]Risk Flags:[/bold red]")
        for flag in result.risk_flags:
            lines.append(f"  • {flag}")

    if result.catalysts:
        lines.append("")
        lines.append("[bold green]Catalysts:[/bold green]")
        for cat in result.catalysts:
            lines.append(f"  • {cat}")

    if result.agent_reports:
        lines.append("")
        lines.append("[bold]Agent Reports:[/bold]")
        for agent_name, report in result.agent_reports.items():
            agent_decision = (
                report.decision.value if hasattr(report.decision, "value") else str(report.decision)
            )
            lines.append(
                f"  [cyan]{agent_name}[/cyan]: {agent_decision} "
                f"(confidence={report.confidence:.0f}%)"
            )

    body = "\n".join(lines)
    panel = Panel(
        Text.from_markup(body),
        title=f"[bold cyan]Analysis: {ticker}[/bold cyan]",
        expand=False,
    )
    console.print(panel)


def _print_text_result(result, ticker: str) -> None:
    """Print a plain-text summary of an AnalysisResult.

    Args:
        result: AnalysisResult instance.
        ticker: Ticker symbol.
    """
    decision_str = result.decision.value if hasattr(result.decision, "value") else str(result.decision)
    lines = [
        f"=== Analysis: {ticker} ===",
        f"Decision:       {decision_str}",
        f"Confidence:     {result.confidence:.1f}%",
        f"Position Size:  {result.position_size_pct:.1f}%",
        f"Time Horizon:   {result.time_horizon}",
    ]
    if result.target_price is not None:
        lines.append(f"Target Price:   ${result.target_price:.2f}")
    if result.stop_loss is not None:
        lines.append(f"Stop Loss:      ${result.stop_loss:.2f}")
    lines.append(f"Reasoning:      {result.reasoning}")
    if result.risk_flags:
        lines.append("Risk Flags:     " + ", ".join(result.risk_flags))
    if result.catalysts:
        lines.append("Catalysts:      " + ", ".join(result.catalysts))
    for line in lines:
        console.print(line)


# ---------------------------------------------------------------------------
# `analyze` command
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Stock ticker (e.g. AAPL)"),
    with_qlib: bool = typer.Option(
        False, help="Fetch basic alpha score via yfinance and inject as qlib_context"
    ),
    output: str = typer.Option(
        "rich", help="Output format: rich | json | text"
    ),
) -> None:
    """Run multi-agent analysis for a single ticker.

    Example::

        python scripts/run_agents.py analyze AAPL --output rich
        python scripts/run_agents.py analyze AAPL --with-qlib --output json
    """
    from src.agents.graph import analyze_ticker

    ticker = ticker.upper().strip()
    logger.info("Starting analyze for ticker={}, with_qlib={}, output={}", ticker, with_qlib, output)

    qlib_context = ""
    if with_qlib:
        logger.info("Fetching yfinance-based qlib_context for {}", ticker)
        qlib_context = _fetch_qlib_context(ticker)

    result = analyze_ticker(ticker, qlib_context=qlib_context)
    logger.info("Analysis complete for {}: decision={}", ticker, result.decision)

    if output == "json":
        console.print(result.model_dump_json(indent=2))
    elif output == "text":
        _print_text_result(result, ticker)
    else:
        _print_rich_result(result, ticker)


# ---------------------------------------------------------------------------
# `debate` command
# ---------------------------------------------------------------------------


@app.command()
def debate(
    ticker: str = typer.Argument(..., help="Stock ticker"),
    rounds: int = typer.Option(2, help="Number of debate rounds"),
) -> None:
    """Run multi-agent debate analysis for a ticker and print the transcript.

    The graph itself handles debate rounds via config; ``--rounds`` is noted
    in the log for informational purposes.

    Example::

        python scripts/run_agents.py debate AAPL --rounds 3
    """
    from src.agents.graph import build_trading_desk_graph

    ticker = ticker.upper().strip()
    logger.info("Starting debate for ticker={}, requested_rounds={}", ticker, rounds)

    graph = build_trading_desk_graph()

    initial_state: dict = {
        "ticker": ticker,
        "asset_class": "EQUITY",
        "qlib_context": "",
        "market_context": "",
        "macro_regime": "neutral",
        "macro_confidence": 50.0,
        "agent_reports": {},
        "debate_round": 0,
        "debate_transcript": [],
        "risk_veto": False,
        "risk_veto_reason": "",
        "result": None,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        logger.error("Debate graph failed for {}: {}", ticker, exc)
        console.print(f"[red]Debate pipeline failed: {exc}[/red]")
        raise typer.Exit(code=1)

    # Print debate transcript
    transcript = final_state.get("debate_transcript", [])
    if transcript:
        console.print(
            Panel(
                "\n".join(str(entry) for entry in transcript),
                title=f"[bold yellow]Debate Transcript: {ticker}[/bold yellow]",
                expand=False,
            )
        )
    else:
        console.print("[dim]No debate transcript recorded.[/dim]")

    # Print final result
    result = final_state.get("result")
    if result is not None:
        logger.info("Debate complete for {}: decision={}", ticker, result.decision)
        _print_rich_result(result, ticker)
    else:
        console.print(f"[red]No result produced for {ticker}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# `batch` command
# ---------------------------------------------------------------------------


@app.command()
def batch(
    tickers: str = typer.Argument(..., help="Comma-separated tickers (e.g. AAPL,MSFT,GOOGL)"),
    output_dir: str = typer.Option(
        "outputs/batch_analysis", help="Directory to save JSON results"
    ),
) -> None:
    """Run multi-agent analysis for a batch of tickers and save results as JSON.

    Example::

        python scripts/run_agents.py batch AAPL,MSFT,GOOGL \\
            --output-dir outputs/batch_analysis
    """
    from src.agents.graph import analyze_ticker

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        logger.error("No tickers provided.")
        raise typer.Exit(code=1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting batch analysis: {} tickers → {}", len(ticker_list), out_path
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_table = Table(
        title="Batch Analysis Summary",
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Ticker", style="bold")
    summary_table.add_column("Decision", justify="center")
    summary_table.add_column("Confidence", justify="right")
    summary_table.add_column("Position %", justify="right")
    summary_table.add_column("Output File")

    for ticker in ticker_list:
        logger.info("Analyzing {} ...", ticker)
        try:
            result = analyze_ticker(ticker)
            out_file = out_path / f"{ticker}_{timestamp}.json"
            out_file.write_text(result.model_dump_json(indent=2))
            decision_str = (
                result.decision.value if hasattr(result.decision, "value") else str(result.decision)
            )
            logger.info(
                "Saved result for {} → {} (confidence={:.1f}%)",
                ticker,
                out_file,
                result.confidence,
            )
            summary_table.add_row(
                ticker,
                decision_str,
                f"{result.confidence:.1f}%",
                f"{result.position_size_pct:.1f}%",
                str(out_file),
            )
        except Exception as exc:
            logger.error("Failed to analyze {}: {}", ticker, exc)
            summary_table.add_row(ticker, "ERROR", "-", "-", str(exc)[:60])

    console.print(summary_table)
    logger.info("Batch analysis complete. Results saved to {}", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
