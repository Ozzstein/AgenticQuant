"""Tests for BaseBroker protocol and compute_broker_metrics utility."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.execution.broker import BaseBroker, compute_broker_metrics
from src.execution.paper_trader import PaperTrader


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
