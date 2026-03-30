"""Unit tests for IntradayRunner and supporting utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.intraday_runner import IntradayRunner, _minutes_to_timeframe
from src.utils.config import AppConfig
from src.utils.schemas import IntradayRunResult, SignalDirection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**intraday_kwargs) -> AppConfig:
    """Return an AppConfig with optional intraday overrides."""
    return AppConfig(intraday=intraday_kwargs)


# ---------------------------------------------------------------------------
# _minutes_to_timeframe
# ---------------------------------------------------------------------------


def test_timeframe_mapping_240_to_4h():
    assert _minutes_to_timeframe(240) == "4h"


def test_timeframe_mapping_60_to_1h():
    assert _minutes_to_timeframe(60) == "1h"


def test_timeframe_mapping_1440_to_1d():
    assert _minutes_to_timeframe(1440) == "1d"


# ---------------------------------------------------------------------------
# IntradayRunner.run — basic result type
# ---------------------------------------------------------------------------


def test_run_returns_intraday_run_result():
    config = AppConfig()
    runner = IntradayRunner(config)

    with (
        patch.object(runner, "_resolve_universe", return_value=["BTC", "ETH"]),
        patch.object(runner, "_fetch_prices", return_value={"BTC": 60000.0, "ETH": 3000.0}),
        patch.object(runner, "_score_tickers", return_value={"BTC": 1.0, "ETH": 0.0}),
    ):
        result = runner.run()

    assert isinstance(result, IntradayRunResult)


def test_run_completed_status_on_success():
    config = AppConfig()
    runner = IntradayRunner(config)

    with (
        patch.object(runner, "_resolve_universe", return_value=["BTC", "ETH"]),
        patch.object(runner, "_fetch_prices", return_value={"BTC": 60000.0, "ETH": 3000.0}),
        patch.object(runner, "_score_tickers", return_value={"BTC": 1.0, "ETH": 0.0}),
    ):
        result = runner.run()

    assert result.status == "completed"
    assert result.error is None


def test_run_failed_status_on_exception():
    config = AppConfig()
    runner = IntradayRunner(config)

    with patch.object(runner, "_resolve_universe", side_effect=RuntimeError("boom")):
        result = runner.run()

    assert result.status == "failed"
    assert result.error == "boom"


def test_empty_universe_returns_skipped():
    config = AppConfig()
    runner = IntradayRunner(config)

    with patch.object(runner, "_resolve_universe", return_value=[]):
        result = runner.run()

    assert result.status == "skipped"
    assert result.signals_generated == 0


# ---------------------------------------------------------------------------
# Broker / execution path
# ---------------------------------------------------------------------------


def test_no_broker_skips_execution():
    config = AppConfig()
    runner = IntradayRunner(config, broker=None)

    with (
        patch.object(runner, "_resolve_universe", return_value=["BTC"]),
        patch.object(runner, "_fetch_prices", return_value={"BTC": 50000.0}),
        patch.object(runner, "_score_tickers", return_value={"BTC": 0.9}),
    ):
        result = runner.run()

    assert result.orders_placed == 0
    assert result.status == "completed"


def test_risk_rejected_orders_not_executed():
    config = AppConfig()
    mock_portfolio = MagicMock()
    mock_portfolio.nav = 100_000.0
    mock_broker = MagicMock()
    mock_broker.portfolio = mock_portfolio

    runner = IntradayRunner(config, broker=mock_broker)

    with (
        patch.object(runner, "_resolve_universe", return_value=["BTC"]),
        patch.object(runner, "_fetch_prices", return_value={"BTC": 50000.0}),
        patch.object(runner, "_score_tickers", return_value={"BTC": 0.9}),
        patch("src.execution.signal_translator.signals_to_orders") as mock_s2o,
        patch("src.execution.risk_controls.check_order") as mock_check,
    ):
        mock_s2o.return_value = [MagicMock()]
        mock_check.return_value = MagicMock(passed=False, failed_checks=["position_size"])

        result = runner.run()

    mock_broker.execute_order.assert_not_called()
    assert result.orders_rejected == 1
    assert result.orders_placed == 0


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def test_top_n_respected():
    config = AppConfig()
    runner = IntradayRunner(config)

    tickers = {f"T{i}": float(i) for i in range(10)}
    scores = {f"T{i}": float(i) / 9 for i in range(10)}

    signals = runner._build_signals(scores, top_n=3)

    long_signals = [s for s in signals if s.direction == SignalDirection.LONG]
    flat_signals = [s for s in signals if s.direction == SignalDirection.FLAT]

    assert len(long_signals) == 3
    assert len(flat_signals) == 7


def test_score_tickers_higher_price_higher_score():
    config = AppConfig()
    runner = IntradayRunner(config)

    scores = runner._score_tickers({"A": 100.0, "B": 200.0})

    assert scores["B"] > scores["A"]


# ---------------------------------------------------------------------------
# yfinance_fallback interval param
# ---------------------------------------------------------------------------


def test_yfinance_fallback_accepts_interval_param():
    from src.core.data_pipeline import DataPipeline

    config = AppConfig()
    pipeline = DataPipeline(config)

    with patch("yfinance.download") as mock_dl:
        mock_dl.return_value = pd.DataFrame()
        pipeline.yfinance_fallback(["AAPL"], interval="1h", period="2d")

    call_kwargs = mock_dl.call_args.kwargs
    assert call_kwargs.get("interval") == "1h"
