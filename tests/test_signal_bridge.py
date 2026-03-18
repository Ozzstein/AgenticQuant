"""Tests for signal_translator and lean_bridge."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.execution.lean_bridge import LEAN_ALGORITHM_TEMPLATE, write_signals_json
from src.execution.signal_translator import signals_to_orders, target_weights_to_orders
from src.utils.schemas import Order, OrderSide, Portfolio, Position, Signal, SignalDirection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PRICES = {"AAPL": 200.0, "MSFT": 300.0, "BTC/USDT": 40000.0}


@pytest.fixture
def empty_portfolio() -> Portfolio:
    """Empty portfolio with $100k cash."""
    return Portfolio(cash=100_000.0, nav=100_000.0)


@pytest.fixture
def portfolio_with_aapl() -> Portfolio:
    """Portfolio with 100 AAPL shares @ $150 avg cost."""
    return Portfolio(
        positions={
            "AAPL": Position(ticker="AAPL", quantity=100, avg_cost=150.0, current_price=200.0)
        },
        cash=80_000.0,
        nav=100_000.0,
    )


@pytest.fixture
def portfolio_with_25_aapl() -> Portfolio:
    """Portfolio already holding exactly the target 25 AAPL shares."""
    return Portfolio(
        positions={
            "AAPL": Position(ticker="AAPL", quantity=25, avg_cost=200.0, current_price=200.0)
        },
        cash=95_000.0,
        nav=100_000.0,
    )


@pytest.fixture
def portfolio_with_10_aapl() -> Portfolio:
    """Portfolio holding 10 AAPL shares (below target)."""
    return Portfolio(
        positions={
            "AAPL": Position(ticker="AAPL", quantity=10, avg_cost=200.0, current_price=200.0)
        },
        cash=98_000.0,
        nav=100_000.0,
    )


# ---------------------------------------------------------------------------
# signals_to_orders tests
# ---------------------------------------------------------------------------


def test_long_signal_creates_buy(empty_portfolio: Portfolio) -> None:
    """LONG signal on empty portfolio should create a BUY for floor(25) shares."""
    signals = [Signal(ticker="AAPL", direction=SignalDirection.LONG, strength=0.5)]
    orders = signals_to_orders(signals, empty_portfolio, PRICES)

    # target_shares = floor(0.5 * 0.10 * 100_000 / 200) = floor(25.0) = 25
    assert len(orders) == 1
    order = orders[0]
    assert order.side == OrderSide.BUY
    assert order.quantity == 25.0
    assert order.ticker == "AAPL"


def test_long_signal_delta_buy(portfolio_with_10_aapl: Portfolio) -> None:
    """LONG signal with partial existing position buys only the delta."""
    signals = [Signal(ticker="AAPL", direction=SignalDirection.LONG, strength=0.5)]
    orders = signals_to_orders(signals, portfolio_with_10_aapl, PRICES)

    # target=25, current=10 → delta=15 → BUY 15
    assert len(orders) == 1
    assert orders[0].side == OrderSide.BUY
    assert orders[0].quantity == 15.0


def test_flat_signal_closes_position(portfolio_with_aapl: Portfolio) -> None:
    """FLAT signal liquidates the entire existing position."""
    signals = [Signal(ticker="AAPL", direction=SignalDirection.FLAT, strength=0.0)]
    orders = signals_to_orders(signals, portfolio_with_aapl, PRICES)

    assert len(orders) == 1
    assert orders[0].side == OrderSide.SELL
    assert orders[0].quantity == 100.0


def test_idempotency(portfolio_with_25_aapl: Portfolio) -> None:
    """LONG signal when portfolio already has exact target produces no orders."""
    signals = [Signal(ticker="AAPL", direction=SignalDirection.LONG, strength=0.5)]
    orders = signals_to_orders(signals, portfolio_with_25_aapl, PRICES)

    assert len(orders) == 0


def test_missing_price_skipped(empty_portfolio: Portfolio) -> None:
    """Signal for a ticker absent from market_prices is silently skipped."""
    signals = [Signal(ticker="UNKNOWN", direction=SignalDirection.LONG, strength=0.5)]
    orders = signals_to_orders(signals, empty_portfolio, PRICES)

    assert len(orders) == 0


def test_multiple_signals(empty_portfolio: Portfolio) -> None:
    """Three LONG signals; AAPL and MSFT produce BUY orders.

    BTC/USDT at strength=1.0, price=40_000 yields
    floor(1.0 * 0.10 * 100_000 / 40_000) = floor(0.25) = 0 shares → no order.
    So only 2 orders are expected.
    """
    signals = [
        Signal(ticker="AAPL", direction=SignalDirection.LONG, strength=0.5),
        Signal(ticker="MSFT", direction=SignalDirection.LONG, strength=0.3),
        Signal(ticker="BTC/USDT", direction=SignalDirection.LONG, strength=1.0),
    ]
    orders = signals_to_orders(signals, empty_portfolio, PRICES)

    assert len(orders) == 2
    tickers = {o.ticker for o in orders}
    assert tickers == {"AAPL", "MSFT"}
    for order in orders:
        assert order.side == OrderSide.BUY


def test_long_signal_no_order_at_zero_strength(empty_portfolio: Portfolio) -> None:
    """LONG signal with strength=0 computes target=0 shares → no order."""
    signals = [Signal(ticker="AAPL", direction=SignalDirection.LONG, strength=0.0)]
    orders = signals_to_orders(signals, empty_portfolio, PRICES)

    assert len(orders) == 0


def test_short_signal_closes_position(portfolio_with_aapl: Portfolio) -> None:
    """SHORT signal (same as FLAT) liquidates the existing position."""
    signals = [Signal(ticker="AAPL", direction=SignalDirection.SHORT, strength=1.0)]
    orders = signals_to_orders(signals, portfolio_with_aapl, PRICES)

    assert len(orders) == 1
    assert orders[0].side == OrderSide.SELL
    assert orders[0].quantity == 100.0


def test_flat_signal_no_position(empty_portfolio: Portfolio) -> None:
    """FLAT signal with no existing position produces no orders."""
    signals = [Signal(ticker="AAPL", direction=SignalDirection.FLAT, strength=0.0)]
    orders = signals_to_orders(signals, empty_portfolio, PRICES)

    assert len(orders) == 0


# ---------------------------------------------------------------------------
# target_weights_to_orders tests
# ---------------------------------------------------------------------------


def test_target_weights_buy(empty_portfolio: Portfolio) -> None:
    """target_weights with AAPL=0.1 on empty portfolio creates BUY 50 shares."""
    orders = target_weights_to_orders({"AAPL": 0.1}, empty_portfolio, PRICES)

    # target_value = 0.1 * 100_000 = 10_000
    # target_shares = floor(10_000 / 200) = 50
    assert len(orders) == 1
    assert orders[0].side == OrderSide.BUY
    assert orders[0].quantity == 50.0


def test_target_weights_sell(portfolio_with_aapl: Portfolio) -> None:
    """Portfolio with 100 AAPL and weight=0 for AAPL should sell all 100 shares."""
    # AAPL in portfolio but NOT in target_weights → treated as weight=0
    orders = target_weights_to_orders({}, portfolio_with_aapl, PRICES)

    sell_orders = [o for o in orders if o.ticker == "AAPL" and o.side == OrderSide.SELL]
    assert len(sell_orders) == 1
    assert sell_orders[0].quantity == 100.0


def test_target_weights_rebalance(portfolio_with_aapl: Portfolio) -> None:
    """Rebalancing: target weight implies 50 shares but portfolio has 100 → SELL 50."""
    orders = target_weights_to_orders({"AAPL": 0.1}, portfolio_with_aapl, PRICES)

    # target = floor(0.1 * 100_000 / 200) = 50, current = 100 → SELL 50
    assert len(orders) == 1
    assert orders[0].side == OrderSide.SELL
    assert orders[0].quantity == 50.0


def test_target_weights_idempotency(empty_portfolio: Portfolio) -> None:
    """Already-matching position produces no orders."""
    portfolio = Portfolio(
        positions={
            "AAPL": Position(ticker="AAPL", quantity=50, avg_cost=200.0, current_price=200.0)
        },
        cash=90_000.0,
        nav=100_000.0,
    )
    orders = target_weights_to_orders({"AAPL": 0.1}, portfolio, PRICES)

    assert len(orders) == 0


def test_target_weights_missing_price_skipped(empty_portfolio: Portfolio) -> None:
    """Tickers absent from market_prices are silently skipped."""
    orders = target_weights_to_orders({"UNKNOWN": 0.5}, empty_portfolio, PRICES)

    assert len(orders) == 0


# ---------------------------------------------------------------------------
# lean_bridge tests
# ---------------------------------------------------------------------------


def test_write_signals_json(tmp_path: Path) -> None:
    """write_signals_json writes valid JSON with correct structure."""
    signals = [
        Signal(ticker="AAPL", direction=SignalDirection.LONG, strength=0.8, source="test"),
        Signal(ticker="MSFT", direction=SignalDirection.FLAT, strength=0.0, source="test"),
    ]
    out = tmp_path / "signals" / "signals.json"
    write_signals_json(signals, out)

    assert out.exists()
    data = json.loads(out.read_text())
    assert set(data.keys()) == {"AAPL", "MSFT"}
    assert data["AAPL"]["direction"] == "LONG"
    assert data["AAPL"]["strength"] == 0.8
    assert data["AAPL"]["source"] == "test"
    assert "timestamp" in data["AAPL"]
    assert data["MSFT"]["direction"] == "FLAT"


def test_write_signals_json_empty(tmp_path: Path) -> None:
    """write_signals_json with empty list writes valid empty JSON object."""
    out = tmp_path / "signals.json"
    write_signals_json([], out)

    assert out.exists()
    data = json.loads(out.read_text())
    assert data == {}


def test_lean_template_nonempty() -> None:
    """LEAN_ALGORITHM_TEMPLATE is a non-trivial string."""
    assert isinstance(LEAN_ALGORITHM_TEMPLATE, str)
    assert len(LEAN_ALGORITHM_TEMPLATE) > 100


def test_lean_template_contains_key_symbols() -> None:
    """LEAN template contains expected class and method identifiers."""
    assert "AIQuantStackAlgorithm" in LEAN_ALGORITHM_TEMPLATE
    assert "TradeOnSignals" in LEAN_ALGORITHM_TEMPLATE
    assert "SetHoldings" in LEAN_ALGORITHM_TEMPLATE
    assert "Liquidate" in LEAN_ALGORITHM_TEMPLATE
