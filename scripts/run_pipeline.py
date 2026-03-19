#!/usr/bin/env python3
"""End-to-end daily trading pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running directly from repo root: `python scripts/run_pipeline.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import date

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

_PIPELINE_STATE_DIR = Path("data/pipeline_state")

_UNIVERSE: list[str] = [
    "SPY", "AAPL", "MSFT", "GOOGL", "AMZN",
    "NVDA", "TSLA", "JPM", "V", "UNH",
]

_TOP_N = 5


# ---------------------------------------------------------------------------
# Idempotency helpers
# ---------------------------------------------------------------------------


def _check_idempotency(mode: str) -> dict | None:
    """Return cached summary if pipeline already ran today for this mode.

    Args:
        mode: Pipeline mode string (backtest, paper, paper-loop).

    Returns:
        Cached summary dict if today's run exists, else None.
    """
    state_file = _PIPELINE_STATE_DIR / f"last_run_{mode}.json"
    if not state_file.exists():
        return None
    try:
        state = json.loads(state_file.read_text())
        if state.get("date") == date.today().isoformat():
            return state.get("summary")
    except Exception:
        pass
    return None


def _save_pipeline_state(mode: str, summary: dict) -> None:
    """Persist pipeline run state for idempotency checks.

    Args:
        mode: Pipeline mode string.
        summary: Summary dict returned by _run_pipeline_once.
    """
    _PIPELINE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = _PIPELINE_STATE_DIR / f"last_run_{mode}.json"
    state = {"date": date.today().isoformat(), "summary": summary}
    state_file.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Core pipeline logic
# ---------------------------------------------------------------------------


def _run_pipeline_once(mode: str, strategy: str = "auto") -> dict:
    """Execute one full pipeline iteration.

    Steps:
    1. Load config
    2. Strategy selection (regime-based, performance-weighted, or manual)
    3. Fetch OHLCV data via yfinance fallback
    4. Compute simple features (returns, mom5, mom20) per ticker
    5. Score tickers by mean 20-day momentum
    6. Select top-N by alpha score (using active strategy's rank_threshold)
    7. Analyze each top ticker via multi-agent graph
    8. Map decisions → signals
    9. Translate signals → orders
    10. Run pre-trade risk checks
    11. Execute passing orders through PaperTrader
    12. Take a portfolio snapshot
    13. Print portfolio summary table

    Args:
        mode: One of "backtest", "paper", or "paper-loop".
        strategy: Strategy selection mode — "auto" for regime-based, "blend" for
            performance-weighted, or an explicit strategy name for manual override.

    Returns:
        Summary dict with tickers_analyzed, orders_placed, nav, active_strategy.
    """
    # Idempotency: skip if already ran today (not applicable to paper-loop demo)
    if mode != "paper-loop":
        cached = _check_idempotency(mode)
        if cached is not None:
            logger.warning(
                "[pipeline] Already ran today in mode='{}'. Returning cached result.", mode
            )
            return cached

    from src.agents.graph import analyze_ticker
    from src.core.data_pipeline import DataPipeline
    from src.core.strategy_selector import StrategySelector
    from src.core.strategy_tracker import StrategyTracker
    from src.execution.paper_trader import PaperTrader
    from src.execution.risk_controls import check_order
    from src.execution.signal_translator import signals_to_orders
    from src.utils.audit import AuditLogger
    from src.utils.config import get_config
    from src.utils.schemas import Signal, SignalDirection

    config = get_config()
    start = config.qlib.test_start
    end = config.qlib.test_end

    audit = AuditLogger()
    audit.start_run()

    # Default: use module-level constants unless strategy overrides them.
    active_top_n = _TOP_N
    active_strategy_name = strategy

    # ------------------------------------------------------------------
    # Step 0: Strategy selection (graceful degradation on failure)
    # ------------------------------------------------------------------
    try:
        selector = StrategySelector()
        tracker = StrategyTracker()
        performances = tracker.get_performances()

        # Infer macro regime from pre-fitted HMM model (graceful fallback to neutral)
        regime = "neutral"
        try:
            from datetime import date as _date
            from datetime import timedelta as _timedelta

            from src.core.macro_regime import HMMRegimeDetector
            _detector = HMMRegimeDetector(config)
            _model_path = "data/regime_model/hmm_model.joblib"
            _detector.load(_model_path)
            _end = _date.today().isoformat()
            _start = (_date.today() - _timedelta(days=60)).isoformat()
            _raw = _detector.fetch_signals(_start, _end)
            _processed = _detector.preprocess(_raw)
            _regime_result = _detector.predict(_processed)
            regime = _regime_result.regime.value
            logger.info("[pipeline] HMM regime detected: {} (confidence={:.2%})", regime, _regime_result.confidence)
        except Exception as _exc:
            logger.warning("[pipeline] Regime detection unavailable ({}); defaulting to neutral.", _exc)

        if strategy == "blend":
            allocation = selector.select(regime, performances=performances, method="performance_weighted")
        elif strategy == "auto":
            allocation = selector.select(regime, performances=performances, method="regime_based")
        else:
            allocation = selector.select_manual(strategy)

        active_strategy_name = allocation.active_strategy
        logger.info(
            "[pipeline] Active strategy: {} (method={})",
            active_strategy_name,
            allocation.selection_method,
        )
        audit.log_step(
            "strategy_selection",
            "ok",
            {
                "active_strategy": active_strategy_name,
                "method": str(allocation.selection_method),
                "allocations": allocation.allocations,
            },
        )

        # Override top_n from strategy entry rules when available.
        active_strat = next(
            (s for s in selector.strategies if s.name == active_strategy_name), None
        )
        if active_strat is not None:
            rank_threshold = active_strat.entry_rules.get("rank_threshold", _TOP_N)
            active_top_n = int(rank_threshold) if isinstance(rank_threshold, (int, float)) else _TOP_N
    except Exception as exc:
        logger.warning("[pipeline] Strategy selection failed ({}); using defaults.", exc)
        audit.log_step("strategy_selection", "error", {"error": str(exc)})
        active_strategy_name = "default"
        active_top_n = _TOP_N

    try:
        # ------------------------------------------------------------------
        # Step 1: Fetch data
        # ------------------------------------------------------------------
        logger.info("[pipeline] Fetching OHLCV data: {} → {}", start, end)
        pipeline = DataPipeline(config)
        try:
            df = pipeline.yfinance_fallback(tickers=_UNIVERSE, start=start, end=end)
            audit.log_step("equity_data", "ok", {"tickers": len(_UNIVERSE), "rows": len(df)})
        except Exception as exc:
            logger.error("[pipeline] Data fetch failed: {}", exc)
            df = __import__("pandas").DataFrame()
            audit.log_step("equity_data", "error", {"error": str(exc)})

        if df.empty:
            logger.warning("[pipeline] No data returned — using empty DataFrame.")

        # ------------------------------------------------------------------
        # Step 2: Compute simple features and alpha scores
        # ------------------------------------------------------------------
        alpha_scores: dict[str, float] = {}

        if not df.empty and "ticker" in df.columns:
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

        audit.log_step("feature_engineering", "ok", {"scored_tickers": len(alpha_scores)})

        # ------------------------------------------------------------------
        # Step 3: Select top tickers
        # ------------------------------------------------------------------
        top_tickers = sorted(alpha_scores, key=lambda t: alpha_scores[t], reverse=True)[:active_top_n]
        logger.info("[pipeline] Top {} tickers by alpha score: {}", active_top_n, top_tickers)

        for ticker in _UNIVERSE:
            audit.log_prediction(ticker, alpha_scores.get(ticker, 0.0))

        audit.log_step("ticker_selection", "ok", {"top_tickers": top_tickers})

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
                audit.log_analysis(ticker, result)
            except Exception as exc:
                logger.error("[pipeline] analyze_ticker failed for {}: {}", ticker, exc)
                audit.log_step(f"analysis_{ticker}", "error", {"error": str(exc)})

        audit.log_step("agent_analysis", "ok", {"analyzed": len(results)})

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

        # Capture portfolio state before trading
        audit.log_portfolio_state("before", portfolio)

        # ------------------------------------------------------------------
        # Step 8: Translate signals → orders
        # ------------------------------------------------------------------
        orders = signals_to_orders(signals, portfolio, market_prices)
        logger.info("[pipeline] Generated {} orders from {} signals", len(orders), len(signals))
        audit.log_step("signal_translation", "ok", {"signals": len(signals), "orders": len(orders)})

        # ------------------------------------------------------------------
        # Step 9: Pre-trade risk checks + execution
        # ------------------------------------------------------------------
        orders_placed = 0
        for order in orders:
            risk_result = check_order(order, portfolio, market_prices, config.risk)
            audit.log_risk_decision(order, risk_result)
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
                    fill_price = filled.fill_price or market_prices.get(order.ticker, 0.0)
                    audit.log_trade(order, fill_price, 0.0)
                    logger.info(
                        "[pipeline] Order executed: {} {} qty={} fill={:.2f}",
                        order.side,
                        order.ticker,
                        order.quantity,
                        fill_price,
                    )
            except Exception as exc:
                logger.error("[pipeline] Order execution failed for {}: {}", order.ticker, exc)

        audit.log_step("order_execution", "ok", {"orders_placed": orders_placed})

        # ------------------------------------------------------------------
        # Step 9b: P&L attribution to active strategy (placeholder — daily_pnl = 0.0)
        # ------------------------------------------------------------------
        try:
            tracker.attribute_pnl(active_strategy_name, 0.0)
        except Exception as exc:
            logger.warning("[pipeline] P&L attribution failed: {}", exc)

        # ------------------------------------------------------------------
        # Step 10: Snapshot
        # ------------------------------------------------------------------
        paper_trader.snapshot(market_prices)
        final_portfolio = paper_trader.portfolio

        # Capture portfolio state after trading
        audit.log_portfolio_state("after", final_portfolio)
        audit.log_step("portfolio_snapshot", "ok", {"nav": final_portfolio.nav})

        # ------------------------------------------------------------------
        # Step 11: Print portfolio summary table
        # ------------------------------------------------------------------
        table = Table(
            title=f"Pipeline Summary — Mode: {mode} | Strategy: {active_strategy_name}",
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
            "[pipeline] Complete — tickers={}, orders={}, nav={:.2f}, strategy={}",
            len(results),
            orders_placed,
            final_portfolio.nav,
            active_strategy_name,
        )

        summary = {
            "tickers_analyzed": len(results),
            "orders_placed": orders_placed,
            "nav": final_portfolio.nav,
            "active_strategy": active_strategy_name,
        }

        # Save state for idempotency (skip paper-loop demo mode)
        if mode != "paper-loop":
            _save_pipeline_state(mode, summary)

        return summary

    finally:
        audit_path = audit.finalize_run()
        if audit_path:
            logger.info("[pipeline] Audit trail written to {}", audit_path)
        else:
            logger.warning("[pipeline] Audit trail write failed — check logs")


# ---------------------------------------------------------------------------
# `run` command
# ---------------------------------------------------------------------------


@app.command()
def run(
    mode: str = typer.Option(
        "backtest",
        help="Pipeline mode: backtest | paper | paper-loop",
    ),
    strategy: str = typer.Option(
        "auto",
        "--strategy",
        "-s",
        help=(
            "Strategy selection: 'auto' for regime-based, 'blend' for "
            "performance-weighted blend, or an explicit strategy name for manual override."
        ),
    ),
) -> None:
    """Run the end-to-end daily trading pipeline.

    Modes:

    - backtest: Run once with historical data, print BacktestResult-style summary.
    - paper:    Run once with latest data, print portfolio state.
    - paper-loop: Loop 3 times with 1-second sleep between iterations (demo).

    Strategy selection:

    - auto:  Regime-based selection (default).
    - blend: Performance-weighted blend across strategies.
    - <name>: Force a specific strategy by name (e.g. momentum_topk).

    Example::

        python scripts/run_pipeline.py run --mode paper
        python scripts/run_pipeline.py run --mode paper-loop
        python scripts/run_pipeline.py run --mode paper --strategy momentum_topk
        python scripts/run_pipeline.py run --mode paper --strategy blend
    """
    valid_modes = {"backtest", "paper", "paper-loop"}
    if mode not in valid_modes:
        console.print(f"[red]Invalid mode: {mode}. Choose from: {valid_modes}[/red]")
        raise typer.Exit(code=1)

    logger.info("[pipeline] Starting pipeline in mode={} strategy={}", mode, strategy)

    if mode == "paper-loop":
        num_iterations = 3
        for i in range(1, num_iterations + 1):
            console.print(f"\n[bold cyan]--- Iteration {i}/{num_iterations} ---[/bold cyan]")
            logger.info("[pipeline] paper-loop iteration {}/{}", i, num_iterations)
            try:
                summary = _run_pipeline_once(mode, strategy=strategy)
                console.print(
                    f"[green]Iteration {i} complete — "
                    f"tickers={summary['tickers_analyzed']}, "
                    f"orders={summary['orders_placed']}, "
                    f"nav=${summary['nav']:,.2f}, "
                    f"strategy={summary.get('active_strategy', 'unknown')}[/green]"
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
            summary = _run_pipeline_once(mode, strategy=strategy)
            active = summary.get("active_strategy", "unknown")
            if mode == "backtest":
                console.print(
                    f"\n[bold green]Backtest complete.[/bold green]\n"
                    f"  Active strategy  : {active}\n"
                    f"  Tickers analyzed : {summary['tickers_analyzed']}\n"
                    f"  Orders placed    : {summary['orders_placed']}\n"
                    f"  Final NAV        : ${summary['nav']:,.2f}"
                )
            else:
                console.print(
                    f"\n[bold green]Paper run complete.[/bold green]\n"
                    f"  Active strategy  : {active}\n"
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
        _PIPELINE_STATE_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: {}", d)
    console.print("[green]Setup complete. Directories created.[/green]")


# ---------------------------------------------------------------------------
# `regime-status` command
# ---------------------------------------------------------------------------


@app.command("regime-status")
def regime_status() -> None:
    """Show current macro regime, state probabilities, signal values, and 20-day history.

    Loads a pre-fitted HMM model from ``data/regime_model/hmm_model.joblib``.
    Falls back gracefully if the model is not found.

    Example::

        python scripts/run_pipeline.py regime-status
    """
    from src.core.macro_regime import HMMRegimeDetector
    from src.utils.config import get_config

    config = get_config()
    detector = HMMRegimeDetector(config)

    # Determine date range: last 2 years of data for inference
    from datetime import date, timedelta
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=730)).isoformat()

    model_path = "data/regime_model/hmm_model.joblib"

    try:
        detector.load(model_path)
        raw = detector.fetch_signals(start_date, end_date)
        processed = detector.preprocess(raw)
        result = detector.predict(processed)
    except Exception as exc:
        console.print(f"[yellow]Could not load pre-fitted model ({exc}). Fitting fresh model...[/yellow]")
        try:
            raw = detector.fetch_signals(start_date, end_date)
            processed = detector.preprocess(raw)
            detector.fit(processed)
            result = detector.predict(processed)
        except Exception as exc2:
            console.print(f"[red]Regime detection failed: {exc2}[/red]")
            raise typer.Exit(1)

    # Print regime table
    table = Table(title="Current Macro Regime", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Regime", f"[bold]{result.regime.value}[/bold]")
    table.add_row("Confidence", f"{result.confidence:.1%}")
    table.add_row("Timestamp", result.timestamp.strftime("%Y-%m-%d %H:%M"))

    console.print(table)

    # State probabilities
    prob_table = Table(title="State Probabilities", show_header=True, header_style="bold blue")
    prob_table.add_column("State")
    prob_table.add_column("Probability", justify="right")
    for state_name, prob in sorted(result.state_probabilities.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        prob_table.add_row(state_name, f"{prob:.2%}  {bar}")
    console.print(prob_table)

    # Signal values
    sig_table = Table(title="Signal Values (z-score)", show_header=True, header_style="bold green")
    sig_table.add_column("Signal")
    sig_table.add_column("Value", justify="right")
    for sig, val in result.signal_values.items():
        sig_table.add_row(sig, f"{val:.3f}")
    console.print(sig_table)

    # 20-day history
    console.print("\n[bold]20-Day Regime History[/bold] (oldest → newest):")
    console.print(" → ".join(result.regime_history_20d))


# ---------------------------------------------------------------------------
# `regime-backtest` command
# ---------------------------------------------------------------------------


@app.command("regime-backtest")
def regime_backtest() -> None:
    """Evaluate the HMM regime detector via regime-conditional portfolio performance.

    Fits a fresh HMM on the last 5 years of data, then uses
    :class:`~src.core.macro_regime.RegimeEvaluator` to compute regime-conditional
    Sharpe ratios and overall performance metrics.

    Example::

        python scripts/run_pipeline.py regime-backtest
    """
    from datetime import date, timedelta

    import numpy as np
    import pandas as pd

    from src.core.macro_regime import HMMRegimeDetector, RegimeEvaluator
    from src.utils.config import get_config

    config = get_config()
    detector = HMMRegimeDetector(config)
    evaluator = RegimeEvaluator(config)

    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=5 * 365)).isoformat()

    console.print(f"[bold cyan]regime-backtest[/bold cyan]: fitting HMM on {start_date} → {end_date}")

    try:
        raw = detector.fetch_signals(start_date, end_date)
        processed = detector.preprocess(raw)
        detector.fit(processed)
    except Exception as exc:
        console.print(f"[red]HMM fitting failed: {exc}[/red]")
        raise typer.Exit(1)

    # Generate synthetic daily returns aligned with the signal index (S&P 500 log-returns)
    try:
        import yfinance as yf
        sp500_raw = yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=True)
        sp500_close = sp500_raw["Close"]
        if isinstance(sp500_close, pd.DataFrame):
            sp500_close = sp500_close.iloc[:, 0]
        returns = np.log(sp500_close / sp500_close.shift(1)).dropna()
        returns.index = pd.to_datetime(returns.index)
    except Exception:
        # Fallback: random returns aligned with processed index
        returns = pd.Series(
            np.random.normal(0.0005, 0.01, len(processed)),
            index=processed.index,
        )

    try:
        eval_result = evaluator.evaluate(detector, processed, returns)
    except Exception as exc:
        console.print(f"[red]Evaluation failed: {exc}[/red]")
        raise typer.Exit(1)

    # Print overall metrics
    metric_table = Table(title="Regime Detector Evaluation", show_header=True, header_style="bold cyan")
    metric_table.add_column("Metric", style="bold")
    metric_table.add_column("Value", justify="right")
    metric_table.add_row("Overall Sharpe", f"{eval_result.get('overall_sharpe', 0.0):.4f}")
    metric_table.add_row("Max Drawdown", f"{eval_result.get('max_drawdown', 0.0):.2%}")
    metric_table.add_row("Regime Transitions", str(eval_result.get("regime_transitions", 0)))
    console.print(metric_table)

    # Print per-regime Sharpe
    rc_sharpe = eval_result.get("regime_conditional_sharpe", {})
    rc_table = Table(title="Regime-Conditional Sharpe", show_header=True, header_style="bold blue")
    rc_table.add_column("Regime")
    rc_table.add_column("Sharpe", justify="right")
    for regime, sharpe in rc_sharpe.items():
        sharpe_str = f"{sharpe:.4f}" if sharpe == sharpe else "N/A"  # NaN check
        rc_table.add_row(regime, sharpe_str)
    console.print(rc_table)

    # Print regime distribution
    dist = eval_result.get("regime_distribution", {})
    dist_table = Table(title="Regime Distribution", show_header=True, header_style="bold green")
    dist_table.add_column("Regime")
    dist_table.add_column("% of Days", justify="right")
    for regime, frac in dist.items():
        dist_table.add_row(regime, f"{frac:.1%}")
    console.print(dist_table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
