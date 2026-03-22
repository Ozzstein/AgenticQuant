"""Tests for BaseBroker protocol and compute_broker_metrics utility."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.execution.broker import BaseBroker, compute_broker_metrics
from src.execution.paper_trader import PaperTrader
from src.utils.config import AlpacaConfig, AppConfig, reset_config
from src.utils.exceptions import (
    AlpacaConnectionError,
    AlpacaError,
    AlpacaOrderError,
    ExecutionError,
)


class TestBaseBrokerProtocol:
    """Verify that BaseBroker is a well-formed runtime-checkable Protocol."""

    def test_paper_trader_conforms_to_protocol(self) -> None:
        """PaperTrader satisfies BaseBroker via structural subtyping at runtime."""
        trader = PaperTrader(initial_cash=100_000.0)
        assert isinstance(trader, BaseBroker), (
            "PaperTrader must satisfy the BaseBroker Protocol at runtime"
        )

    def test_protocol_has_required_methods(self) -> None:
        """BaseBroker Protocol defines all four required interface members."""
        required = {"execute_order", "snapshot", "portfolio", "get_metrics"}
        # Protocol members are recorded in __protocol_attrs__ (Python 3.12+)
        # or can be read from the annotations / members of the class.
        # Use a concrete check against the Protocol's own annotations + callable attrs.
        protocol_members = set()
        for name in dir(BaseBroker):
            if not name.startswith("_"):
                protocol_members.add(name)
        assert required.issubset(protocol_members), (
            f"Missing protocol members: {required - protocol_members}"
        )

    def test_broker_type_hint_accepts_paper_trader(self) -> None:
        """A function typed BaseBroker can call get_metrics() on a PaperTrader."""

        def use_broker(b: BaseBroker) -> dict:
            return b.get_metrics()

        trader = PaperTrader(initial_cash=50_000.0)
        result = use_broker(trader)
        assert isinstance(result, dict), "get_metrics() must return a dict"
        assert "total_return" in result

    def test_compute_broker_metrics_empty(self) -> None:
        """compute_broker_metrics returns all-zero dict when inputs are empty."""
        result = compute_broker_metrics(nav_history=[], trade_log=[])

        expected_keys = {
            "total_return",
            "annualized_return",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "volatility",
            "win_rate",
            "avg_win",
            "avg_loss",
            "profit_factor",
        }
        assert set(result.keys()) == expected_keys, (
            f"Unexpected keys in metrics: {set(result.keys()) ^ expected_keys}"
        )
        for key, value in result.items():
            assert value == 0.0, f"Expected 0.0 for '{key}' but got {value}"

    def test_compute_broker_metrics_single_entry(self) -> None:
        """compute_broker_metrics returns zeros when only one NAV entry exists."""
        ts = datetime(2024, 1, 1)
        result = compute_broker_metrics(nav_history=[(ts, 100_000.0)], trade_log=[])
        for key, value in result.items():
            assert value == 0.0, f"Expected 0.0 for '{key}' with single entry but got {value}"

    def test_compute_broker_metrics_positive_return(self) -> None:
        """compute_broker_metrics correctly computes total_return for simple growth."""
        base = datetime(2024, 1, 1)
        nav_history = [(base + timedelta(days=i), 100_000.0 + i * 100.0) for i in range(10)]
        result = compute_broker_metrics(nav_history=nav_history, trade_log=[])

        assert result["total_return"] > 0.0, "Total return should be positive for growing NAV"
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-9), (
            "Max drawdown should be 0 for monotonically growing NAV"
        )

    def test_compute_broker_metrics_trade_stats(self) -> None:
        """compute_broker_metrics correctly computes win_rate, avg_win, avg_loss, profit_factor."""
        nav = [(datetime(2024, 1, i), 100_000.0 + i * 100) for i in range(1, 11)]
        trade_log = [
            {"side": "SELL", "pnl": 500.0},
            {"side": "SELL", "pnl": -200.0},
            {"side": "SELL", "pnl": 300.0},
        ]
        metrics = compute_broker_metrics(nav, trade_log)

        assert metrics["win_rate"] == pytest.approx(2 / 3), (
            "win_rate should be 2/3 with two wins and one loss"
        )
        assert metrics["avg_win"] > 0, "avg_win should be positive"
        assert metrics["avg_loss"] < 0, "avg_loss should be negative"
        assert metrics["profit_factor"] > 0, "profit_factor should be positive"

    def test_get_metrics_delegates_to_compute_broker_metrics(self) -> None:
        """PaperTrader.get_metrics() must return the same result as compute_broker_metrics."""
        from src.execution.broker import compute_broker_metrics

        trader = PaperTrader(initial_cash=100_000.0)
        base = datetime(2024, 1, 1)
        for i in range(5):
            trader._nav_history.append((base + timedelta(days=i), 100_000.0 + i * 200.0))

        expected = compute_broker_metrics(trader._nav_history, trader._trade_log)
        actual = trader.get_metrics()

        assert actual == expected, "PaperTrader.get_metrics() must delegate to compute_broker_metrics"


class TestAlpacaConfig:
    """Tests for AlpacaConfig model and its integration with AppConfig."""

    def test_alpaca_config_defaults(self) -> None:
        """AlpacaConfig() has the expected default values."""
        cfg = AlpacaConfig()
        assert cfg.enabled is False
        assert cfg.paper is True
        assert cfg.api_key == ""
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 30

    def test_alpaca_config_in_app_config(self) -> None:
        """AppConfig().alpaca is an AlpacaConfig instance."""
        app_cfg = AppConfig()
        assert isinstance(app_cfg.alpaca, AlpacaConfig)

    def test_alpaca_config_from_env(self, monkeypatch) -> None:
        """Env vars AIQUANT_ALPACA__API_KEY and AIQUANT_ALPACA__ENABLED are picked up."""
        monkeypatch.setenv("AIQUANT_ALPACA__API_KEY", "test123")
        monkeypatch.setenv("AIQUANT_ALPACA__ENABLED", "true")
        reset_config()
        try:
            app_cfg = AppConfig()
            assert app_cfg.alpaca.api_key == "test123"
            assert app_cfg.alpaca.enabled is True
        finally:
            reset_config()

    def test_alpaca_exceptions_hierarchy(self) -> None:
        """AlpacaError, AlpacaConnectionError, AlpacaOrderError all inherit from ExecutionError."""
        assert issubclass(AlpacaError, ExecutionError)
        assert issubclass(AlpacaConnectionError, AlpacaError)
        assert issubclass(AlpacaConnectionError, ExecutionError)
        assert issubclass(AlpacaOrderError, AlpacaError)
        assert issubclass(AlpacaOrderError, ExecutionError)

    def test_alpaca_config_from_yaml(self) -> None:
        """AppConfig loads alpaca section from settings.yaml with enabled=False."""
        # get_config() reads settings.yaml which sets alpaca.enabled: false
        from src.utils.config import get_config
        app_cfg = get_config()
        assert app_cfg.alpaca.enabled is False
        assert app_cfg.alpaca.paper is True
