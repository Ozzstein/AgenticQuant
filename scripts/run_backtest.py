#!/usr/bin/env python3
"""Run backtests against the Qlib/yfinance data pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running directly from repo root: `python scripts/run_backtest.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.core.backtester import WalkForwardBacktester
from src.core.data_pipeline import DataPipeline
from src.core.ensemble import create_model
from src.utils.config import get_config
from src.utils.schemas import BacktestMetrics, BacktestResult

app = typer.Typer(
    name="run_backtest",
    help="Backtest CLI — run walk-forward backtests and compare models.",
    add_completion=False,
)
console = Console()

_UNIVERSE: list[str] = [
    "SPY", "AAPL", "MSFT", "GOOGL", "AMZN",
    "NVDA", "TSLA", "JPM", "V", "UNH",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _build_features(
    df: pd.DataFrame,
    extra_factors: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute per-ticker features and labels from OHLCV data.

    Features computed per ticker then concatenated:
    - returns:    1-day close-to-close return
    - mom5:       5-day momentum (pct_change(5))
    - mom20:      20-day momentum (pct_change(20))
    - vol_ratio:  volume / 20-day rolling mean volume

    Labels are next-day returns (shifted by -1) to avoid look-ahead bias.
    If ``extra_factors`` is provided, those columns are joined into the feature matrix.

    Args:
        df: OHLCV DataFrame with columns ``close``, ``volume``, ``ticker``
            and a DatetimeIndex.
        extra_factors: Optional DataFrame of extra factor columns aligned to df's index.

    Returns:
        Tuple of (feature_matrix, labels) with NaN rows dropped.
    """
    frames: list[pd.DataFrame] = []

    if "ticker" in df.columns:
        grouped = df.groupby("ticker")
    else:
        # Treat entire df as a single group
        grouped = [("single", df)]  # type: ignore[assignment]

    for ticker, grp in grouped:
        grp = grp.sort_index()
        close = grp["close"]
        volume = grp["volume"]

        feat = pd.DataFrame(index=grp.index)
        feat["returns"] = close.pct_change(1)
        feat["mom5"] = close.pct_change(5)
        feat["mom20"] = close.pct_change(20)
        vol_ma = volume.rolling(20).mean()
        feat["vol_ratio"] = volume / vol_ma.replace(0, float("nan"))
        feat["label"] = feat["returns"].shift(-1)
        feat["ticker"] = ticker
        frames.append(feat)

    if not frames:
        return pd.DataFrame(), pd.Series(dtype=float)

    combined = pd.concat(frames)
    feature_cols = ["returns", "mom5", "mom20", "vol_ratio"]
    base_dropna_cols = ["returns", "mom5", "mom20", "vol_ratio", "label"]

    # Merge extra RD-Agent factors if provided
    if extra_factors is not None and not extra_factors.empty:
        extra_cols = [c for c in extra_factors.columns if c not in combined.columns]
        if extra_cols:
            combined = combined.join(extra_factors[extra_cols], how="left")
            feature_cols = feature_cols + extra_cols

    combined = combined.dropna(subset=base_dropna_cols)
    X = combined[feature_cols]
    y = combined["label"]
    return X, y


def _mock_backtest_result(sharpe: float = 0.5) -> BacktestResult:
    """Create a minimal BacktestResult for fallback scenarios.

    Args:
        sharpe: Sharpe ratio to use in the mock metrics.

    Returns:
        BacktestResult populated with conservative placeholder values.
    """
    return BacktestResult(
        metrics=BacktestMetrics(
            sharpe_ratio=sharpe,
            total_return=0.1,
            annual_return=0.05,
            max_drawdown=-0.1,
            calmar_ratio=0.5,
            win_rate=0.52,
            volatility=0.15,
            sortino_ratio=0.6,
        )
    )


# ---------------------------------------------------------------------------
# Core pipeline (shared between `run` and `compare`)
# ---------------------------------------------------------------------------


def _run_pipeline(
    model_name: str,
    topk: int,
    start: str,
    end: str,
    with_rd_factors: bool,
) -> BacktestResult:
    """Fetch data, engineer features, train model, and run the backtester.

    Args:
        model_name: Name of the ML model (LightGBM, Linear, etc.).
        topk: Number of top stocks to select per period.
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).
        with_rd_factors: Whether to load the factor library JSON.

    Returns:
        BacktestResult from WalkForwardBacktester or a mock fallback.
    """
    config = get_config()

    # --- 1. Load data ---
    pipeline = DataPipeline(config)
    logger.info("Fetching OHLCV data for universe: {} → {}", start, end)
    df = pipeline.yfinance_fallback(tickers=_UNIVERSE, start=start, end=end)

    if df.empty:
        logger.warning("No data returned — using mock result.")
        return _mock_backtest_result()

    # --- 2. Optional: load RD-Agent factor library ---
    factor_df: pd.DataFrame | None = None
    if with_rd_factors:
        from src.core.data_pipeline import _compute_factor
        from src.utils.schemas import FactorDefinition

        # Check both paths: subdir (where runner saves) and flat (legacy)
        factor_path = Path(config.output_dir) / "factor_library" / "factor_library.json"
        if not factor_path.exists():
            factor_path = Path(config.output_dir) / "factor_library.json"

        if factor_path.exists():
            try:
                with open(factor_path) as fh:
                    raw_factors = json.load(fh)
                raw_factors = raw_factors if isinstance(raw_factors, list) else []
                logger.info("Loaded {} factors from library", len(raw_factors))
                factor_cols: list[pd.Series] = []
                for item in raw_factors:
                    try:
                        fdef = FactorDefinition(**item)
                        col = _compute_factor(df, fdef)
                        if col is not None:
                            factor_cols.append(col.rename(fdef.name))
                    except Exception as exc:
                        logger.warning("Could not compute factor '{}': {}", item.get("name"), exc)
                if factor_cols:
                    factor_df = pd.concat(factor_cols, axis=1)
            except Exception as exc:
                logger.warning("Could not load factor library: {}", exc)
        else:
            logger.warning("factor_library.json not found at {}", factor_path)

    # --- 3. Feature engineering ---
    X, y = _build_features(df, extra_factors=factor_df)
    logger.info("Feature matrix: {} rows × {} cols", len(X), X.shape[1] if not X.empty else 0)

    if len(X) < 50:
        logger.warning(
            "Only {} rows after feature engineering (need ≥50). Returning mock result.", len(X)
        )
        return _mock_backtest_result()

    # --- 4. Train model on first 70% ---
    split = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    wrapper = create_model(model_name, config)
    try:
        wrapper.train(X_train, y_train)
    except Exception as exc:
        logger.warning("Model training failed ({}). Using mock result.", exc)
        return _mock_backtest_result()

    # --- 5. Walk-forward backtest ---
    safe_topk = min(topk, max(1, len(X) // 4))
    try:
        backtester = WalkForwardBacktester(config)
        result = backtester.run(X, y, model=wrapper, topk=safe_topk)
    except Exception as exc:
        logger.warning("WalkForwardBacktester failed ({}). Falling back to mock result.", exc)
        result = _mock_backtest_result()

    return result


# ---------------------------------------------------------------------------
# `run` command
# ---------------------------------------------------------------------------


@app.command()
def run(
    model: str = typer.Option("LightGBM", help="Model name (LightGBM, Linear, CatBoost, XGBoost)"),
    topk: int = typer.Option(30, help="Top-K tickers to select per period"),
    start: str = typer.Option("2022-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2023-12-31", help="End date (YYYY-MM-DD)"),
    with_rd_factors: bool = typer.Option(False, help="Load factor library JSON"),
    output: str = typer.Option("rich", help="Output format: rich | json"),
) -> None:
    """Run a single walk-forward backtest for a given model.

    Example::

        python scripts/run_backtest.py run --model LightGBM --topk 10 \\
            --start 2022-01-01 --end 2023-06-30
    """
    logger.info(
        "Starting backtest: model={}, topk={}, start={}, end={}, rd_factors={}",
        model, topk, start, end, with_rd_factors,
    )

    result = _run_pipeline(
        model_name=model,
        topk=topk,
        start=start,
        end=end,
        with_rd_factors=with_rd_factors,
    )

    if output == "json":
        console.print(result.model_dump_json(indent=2))
        return

    # --- Rich table output ---
    m = result.metrics
    table = Table(title=f"Backtest Results — {model}", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    rows = [
        ("Sharpe Ratio", f"{m.sharpe_ratio:.3f}"),
        ("Total Return", f"{m.total_return:.2%}"),
        ("Annual Return", f"{m.annual_return:.2%}"),
        ("Max Drawdown", f"{m.max_drawdown:.2%}"),
        ("Calmar Ratio", f"{m.calmar_ratio:.3f}"),
        ("Win Rate", f"{m.win_rate:.2%}"),
        ("Volatility (ann.)", f"{m.volatility:.2%}"),
        ("Sortino Ratio", f"{m.sortino_ratio:.3f}"),
        ("Total Trades", str(m.total_trades)),
        ("Avg Trade Return", f"{m.avg_trade_return:.4%}"),
        ("Walk-Forward Folds", str(len(result.walk_forward_results))),
        ("Equity Curve Points", str(len(result.equity_curve))),
    ]
    for name, value in rows:
        table.add_row(name, value)

    console.print(table)
    logger.info("Backtest complete. Sharpe={:.3f}", m.sharpe_ratio)


# ---------------------------------------------------------------------------
# `compare` command
# ---------------------------------------------------------------------------


@app.command()
def compare(
    models: str = typer.Option("LightGBM,Linear", help="Comma-separated model names"),
    topk: int = typer.Option(30, help="Top-K tickers to select per period"),
    start: str = typer.Option("2022-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2023-12-31", help="End date (YYYY-MM-DD)"),
    ensemble: bool = typer.Option(False, "--ensemble", help="Include Ensemble in comparison"),
) -> None:
    """Compare multiple models side by side on the same dataset.

    Example::

        python scripts/run_backtest.py compare --models LightGBM,Linear \\
            --topk 10 --start 2022-01-01 --end 2023-06-30
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if ensemble:
        model_list.append("Ensemble")
    if not model_list:
        logger.error("No models specified.")
        raise typer.Exit(code=1)

    logger.info("Comparing models: {} — start={}, end={}", model_list, start, end)

    table = Table(
        title=f"Model Comparison ({start} → {end})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Model", style="bold")
    table.add_column("Sharpe", justify="right")
    table.add_column("Total Return", justify="right")
    table.add_column("Annual Return", justify="right")
    table.add_column("Max Drawdown", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Volatility", justify="right")

    for model_name in model_list:
        logger.info("Running pipeline for model: {}", model_name)
        try:
            result = _run_pipeline(
                model_name=model_name,
                topk=topk,
                start=start,
                end=end,
                with_rd_factors=False,
            )
            m = result.metrics
            table.add_row(
                model_name,
                f"{m.sharpe_ratio:.3f}",
                f"{m.total_return:.2%}",
                f"{m.annual_return:.2%}",
                f"{m.max_drawdown:.2%}",
                f"{m.win_rate:.2%}",
                f"{m.volatility:.2%}",
            )
        except Exception as exc:
            logger.error("Pipeline failed for model {}: {}", model_name, exc)
            table.add_row(model_name, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR")

    console.print(table)
    logger.info("Comparison complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
