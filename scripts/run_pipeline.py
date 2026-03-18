#!/usr/bin/env python3
"""End-to-end daily trading pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running directly from repo root: `python scripts/run_pipeline.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="run_pipeline",
    help="End-to-end daily trading pipeline CLI.",
    add_completion=False,
)
console = Console()

_UNIVERSE: list[str] = [
    "SPY", "AAPL", "MSFT", "GOOGL", "AMZN",
    "NVDA", "TSLA", "JPM", "V", "UNH",
]

_TOP_N = 5


# ---------------------------------------------------------------------------
# Core pipeline logic
# ---------------------------------------------------------------------------


def _run_pipeline_once(mode: str) -> dict:
    """Execute one full pipeline iteration.

    Steps:
    1. Load config
    2. Fetch OHLCV data via yfinance fallback
    3. Compute simple features (returns, mom5, mom20) per ticker
    4. Score tickers by mean 20-day momentum
    5. Select top-5 by alpha score
    6. Analyze each top ticker via multi-agent graph
    7. Map decisions → signals
    8. Translate signals → orders
    9. Run pre-trade risk checks
    10. Execute passing orders through PaperTrader
    11. Take a portfolio snapshot
    12. Print portfolio summary table

    Args:
        mode: One of "backtest", "paper", or "paper-loop".

    Returns:
        Summary dict with tickers_analyzed, orders_placed, nav.
    """
    from src.agents.graph import analyze_ticker
    from src.core.data_pipeline import DataPipeline
    from src.execution.paper_trader import PaperTrader
    from src.execution.risk_controls import check_order
    from src.execution.signal_translator import signals_to_orders
    from src.utils.config import get_config
    from src.utils.schemas import Decision, Signal, SignalDirection

    config = get_config()
    start = config.qlib.test_start
    end = config.qlib.test_end

    # ------------------------------------------------------------------
    # Step 1: Fetch data
    # ------------------------------------------------------------------
    logger.info("[pipeline] Fetching OHLCV data: {} → {}", start, end)
    pipeline = DataPipeline(config)
    try:
        df = pipeline.yfinance_fallback(tickers=_UNIVERSE, start=start, end=end)
    except Exception as exc:
        logger.error("[pipeline] Data fetch failed: {}", exc)
        df = __import__("pandas").DataFrame()

    if df.empty:
        logger.warning("[pipeline] No data returned — using empty DataFrame.")

    # ------------------------------------------------------------------
    # Step 2: Compute simple features and alpha scores
    # ------------------------------------------------------------------
    alpha_scores: dict[str, float] = {}

    if not df.empty and "ticker" in df.columns:
        import pandas as pd

        for ticker in _UNIVERSE:
            ticker_df = df[df["ticker"] == ticker].sort_index()
            if ticker_df.empty or "close" not in ticker_df.columns:
                alpha_scores[ticker] = 0.0
                continue
            close = ticker_df["close"]
            mom20 = close.pct_change(20)
            score = float(mom20.mean()) if not mom20.isna().all() else 0.0
            alpha_scores[ticker] = score
    else:
        for ticker in _UNIVERSE:
            alpha_scores[ticker] = 0.0

    # ------------------------------------------------------------------
    # Step 3: Select top tickers
    # ------------------------------------------------------------------
    top_tickers = sorted(alpha_scores, key=lambda t: alpha_scores[t], reverse=True)[:_TOP_N]
    logger.info("[pipeline] Top {} tickers by alpha score: {}", _TOP_N, top_tickers)

    # ------------------------------------------------------------------
    # Step 4: Multi-agent analysis
    # ------------------------------------------------------------------
    results = {}
    for ticker in top_tickers:
        score = alpha_scores[ticker]
        qlib_context = f"Alpha score: {score:.4f}"
        logger.info("[pipeline] Analyzing {} (alpha={:.4f})", ticker, score)
        try:
            result = analyze_ticker(ticker, qlib_context=qlib_context)
            results[ticker] = result
        except Exception as exc:
            logger.error("[pipeline] analyze_ticker failed for {}: {}", ticker, exc)

    # ------------------------------------------------------------------
    # Step 5: Map decisions → signals
    # ------------------------------------------------------------------
    signals: list[Signal] = []
    for ticker, result in results.items():
        decision_val = result.decision.value if hasattr(result.decision, "value") else str(result.decision)
        if decision_val in ("STRONG_BUY", "BUY"):
            direction = SignalDirection.LONG
        else:
            direction = SignalDirection.FLAT
        strength = min(result.confidence / 100.0, 1.0)
        signals.append(
            Signal(
                ticker=ticker,
                direction=direction,
                strength=strength,
                source="pipeline",
            )
        )

    # ------------------------------------------------------------------
    # Step 6: Get current prices
    # ------------------------------------------------------------------
    market_prices: dict[str, float] = {}
    if not df.empty and "ticker" in df.columns and "close" in df.columns:
        for ticker in top_tickers:
            ticker_rows = df[df["ticker"] == ticker]
            if not ticker_rows.empty:
                market_prices[ticker] = float(ticker_rows["close"].iloc[-1])

    # Fallback prices for any missing tickers
    for ticker in top_tickers:
        if ticker not in market_prices:
            market_prices[ticker] = 100.0

    # ------------------------------------------------------------------
    # Step 7: Build portfolio and trader
    # ------------------------------------------------------------------
    initial_cash = config.lean.initial_cash if config.lean.initial_cash > 0 else 100_000.0
    paper_trader = PaperTrader(
        initial_cash=initial_cash,
        slippage_bps=5,
        commission_per_share=0.005,
    )

    # Convert to Portfolio for signal_translator / risk_controls
    portfolio = paper_trader.portfolio

    # ------------------------------------------------------------------
    # Step 8: Translate signals → orders
    # ------------------------------------------------------------------
    orders = signals_to_orders(signals, portfolio, market_prices)
    logger.info("[pipeline] Generated {} orders from {} signals", len(orders), len(signals))

    # ------------------------------------------------------------------
    # Step 9: Pre-trade risk checks + execution
    # ------------------------------------------------------------------
    orders_placed = 0
    for order in orders:
        risk_result = check_order(order, portfolio, market_prices, config.risk)
        if not risk_result.passed:
            logger.warning(
                "[pipeline] Order {} {} FAILED risk check: {}",
                order.side,
                order.ticker,
                risk_result.failed_checks,
            )
            continue
        try:
            filled = paper_trader.execute_order(order, market_prices)
            if filled is not None:
                orders_placed += 1
                logger.info(
                    "[pipeline] Order executed: {} {} qty={} fill={:.2f}",
                    order.side,
                    order.ticker,
                    order.quantity,
                    filled.fill_price or 0.0,
                )
        except Exception as exc:
            logger.error("[pipeline] Order execution failed for {}: {}", order.ticker, exc)

    # ------------------------------------------------------------------
    # Step 10: Snapshot
    # ------------------------------------------------------------------
    paper_trader.snapshot(market_prices)
    final_portfolio = paper_trader.portfolio

    # ------------------------------------------------------------------
    # Step 11: Print portfolio summary table
    # ------------------------------------------------------------------
    table = Table(
        title=f"Pipeline Summary — Mode: {mode}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Ticker", style="bold")
    table.add_column("Quantity", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Weight %", justify="right")

    for ticker, pos in final_portfolio.positions.items():
        price = market_prices.get(ticker, pos.current_price)
        table.add_row(
            ticker,
            f"{pos.quantity:.0f}",
            f"${price:.2f}",
            f"{pos.weight_pct:.1f}%",
        )

    table.add_row(
        "[bold]CASH[/bold]",
        "",
        "",
        f"${final_portfolio.cash:,.2f}",
    )
    table.add_row(
        "[bold]NAV[/bold]",
        "",
        "",
        f"${final_portfolio.nav:,.2f}",
    )

    console.print(table)
    logger.info(
        "[pipeline] Complete — tickers={}, orders={}, nav={:.2f}",
        len(results),
        orders_placed,
        final_portfolio.nav,
    )

    return {
        "tickers_analyzed": len(results),
        "orders_placed": orders_placed,
        "nav": final_portfolio.nav,
    }


# ---------------------------------------------------------------------------
# `run` command
# ---------------------------------------------------------------------------


@app.command()
def run(
    mode: str = typer.Option(
        "backtest",
        help="Pipeline mode: backtest | paper | paper-loop",
    ),
) -> None:
    """Run the end-to-end daily trading pipeline.

    Modes:

    - backtest: Run once with historical data, print BacktestResult-style summary.
    - paper:    Run once with latest data, print portfolio state.
    - paper-loop: Loop 3 times with 1-second sleep between iterations (demo).

    Example::

        python scripts/run_pipeline.py run --mode paper
        python scripts/run_pipeline.py run --mode paper-loop
    """
    valid_modes = {"backtest", "paper", "paper-loop"}
    if mode not in valid_modes:
        console.print(f"[red]Invalid mode: {mode}. Choose from: {valid_modes}[/red]")
        raise typer.Exit(code=1)

    logger.info("[pipeline] Starting pipeline in mode={}", mode)

    if mode == "paper-loop":
        num_iterations = 3
        for i in range(1, num_iterations + 1):
            console.print(f"\n[bold cyan]--- Iteration {i}/{num_iterations} ---[/bold cyan]")
            logger.info("[pipeline] paper-loop iteration {}/{}", i, num_iterations)
            try:
                summary = _run_pipeline_once(mode)
                console.print(
                    f"[green]Iteration {i} complete — "
                    f"tickers={summary['tickers_analyzed']}, "
                    f"orders={summary['orders_placed']}, "
                    f"nav=${summary['nav']:,.2f}[/green]"
                )
            except Exception as exc:
                logger.error("[pipeline] Iteration {} failed: {}", i, exc)
                console.print(f"[red]Iteration {i} failed: {exc}[/red]")
            if i < num_iterations:
                time.sleep(1)
        console.print("\n[bold green]Paper-loop complete.[/bold green]")

    else:
        # backtest or paper — run once
        try:
            summary = _run_pipeline_once(mode)
            if mode == "backtest":
                console.print(
                    f"\n[bold green]Backtest complete.[/bold green]\n"
                    f"  Tickers analyzed : {summary['tickers_analyzed']}\n"
                    f"  Orders placed    : {summary['orders_placed']}\n"
                    f"  Final NAV        : ${summary['nav']:,.2f}"
                )
            else:
                console.print(
                    f"\n[bold green]Paper run complete.[/bold green]\n"
                    f"  Tickers analyzed : {summary['tickers_analyzed']}\n"
                    f"  Orders placed    : {summary['orders_placed']}\n"
                    f"  Final NAV        : ${summary['nav']:,.2f}"
                )
        except Exception as exc:
            logger.error("[pipeline] Pipeline run failed: {}", exc)
            console.print(f"[red]Pipeline failed: {exc}[/red]")
            raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# `setup` command
# ---------------------------------------------------------------------------


@app.command()
def setup() -> None:
    """Create output directories and initialize portfolio state."""
    from src.utils.config import get_config

    config = get_config()
    dirs = [
        Path(config.output_dir),
        Path(config.output_dir) / "factor_library",
        Path(config.log_dir),
        Path("outputs/batch_analysis"),
        Path("data/rd_knowledge_base"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: {}", d)
    console.print("[green]Setup complete. Directories created.[/green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
