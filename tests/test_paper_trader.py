"""Tests for PaperTrader paper trading engine."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.execution.paper_trader import PaperTrader
from src.utils.schemas import Order, OrderSide, OrderStatus, OrderType

INITIAL_CASH = 100_000.0
AAPL_PRICE = 200.0
PRICES = {"AAPL": AAPL_PRICE}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trader() -> PaperTrader:
    """Return a fresh PaperTrader with standard test parameters."""
    return PaperTrader(
        initial_cash=INITIAL_CASH,
        slippage_bps=5,
        commission_per_share=0.005,
        min_commission=1.0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_order(ticker: str, side: OrderSide, qty: float) -> Order:
    """Create a MARKET order for tests."""
    return Order(ticker=ticker, side=side, quantity=qty)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_buy_decreases_cash(trader: PaperTrader) -> None:
    """Buy 100 AAPL at $200 — cash must decrease by exact cost."""
    order = make_order("AAPL", OrderSide.BUY, 100)
    result = trader.execute_order(order, PRICES)

    assert result is not None
    assert result.status == OrderStatus.FILLED

    fill_price = AAPL_PRICE * 1.0005  # 200.10
    commission = max(100 * 0.005, 1.0)  # 0.50  →  1.00  (min applies)
    # commission: 100 * 0.005 = 0.50, but min_commission = 1.0, so commission = 1.0
    cost = fill_price * 100 + commission  # 200.10 * 100 + 1.0 = 20011.0

    assert abs(trader.portfolio.cash - (INITIAL_CASH - cost)) < 0.01


def test_avg_cost_two_buys(trader: PaperTrader) -> None:
    """Two buys at different prices — avg_cost must be the weighted average fill."""
    prices1 = {"AAPL": 100.0}
    prices2 = {"AAPL": 120.0}

    order1 = make_order("AAPL", OrderSide.BUY, 100)
    order2 = make_order("AAPL", OrderSide.BUY, 100)
    trader.execute_order(order1, prices1)
    trader.execute_order(order2, prices2)

    fill1 = 100.0 * 1.0005
    fill2 = 120.0 * 1.0005
    expected_avg = (100 * fill1 + 100 * fill2) / 200

    assert abs(trader.portfolio.positions["AAPL"].avg_cost - expected_avg) < 0.01


def test_sell_rejection_insufficient_shares(trader: PaperTrader) -> None:
    """Buy 100 shares then try to sell 200 — must be rejected, portfolio unchanged."""
    buy_order = make_order("AAPL", OrderSide.BUY, 100)
    trader.execute_order(buy_order, PRICES)

    cash_before = trader.portfolio.cash
    qty_before = trader.portfolio.positions["AAPL"].quantity

    sell_order = make_order("AAPL", OrderSide.SELL, 200)
    result = trader.execute_order(sell_order, PRICES)

    assert result is None
    assert trader.portfolio.positions["AAPL"].quantity == qty_before
    assert abs(trader.portfolio.cash - cash_before) < 0.01


def test_nav_conservation(trader: PaperTrader) -> None:
    """After buying and taking a snapshot, NAV must equal cash + position value."""
    buy_order = make_order("AAPL", OrderSide.BUY, 100)
    trader.execute_order(buy_order, PRICES)
    trader.snapshot(PRICES)

    portfolio = trader.portfolio
    pos = portfolio.positions["AAPL"]
    expected_nav = portfolio.cash + pos.quantity * pos.current_price

    assert abs(portfolio.nav - expected_nav) < 0.01


def test_metrics_after_multiple_trades(trader: PaperTrader) -> None:
    """11 snapshots with alternating prices produce valid finite metric values."""
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

    # Initial buy
    buy_order = make_order("AAPL", OrderSide.BUY, 100)
    trader.execute_order(buy_order, PRICES)
    trader.snapshot(PRICES)

    # Simulate 10 more daily snapshots with slightly varying prices
    for i in range(10):
        price = AAPL_PRICE + (i % 3) * 2.0  # 200, 202, 204, 200, 202, ...
        trader.snapshot({"AAPL": price})

    metrics = trader.get_metrics()

    assert set(metrics.keys()) == expected_keys
    for key, value in metrics.items():
        assert isinstance(value, float), f"{key} is not float: {value!r}"
        assert math.isfinite(value), f"{key} is not finite: {value}"


def test_insufficient_cash(trader: PaperTrader) -> None:
    """Attempt to buy 10,000 shares at $200 (cost ~$2M) — must be rejected."""
    order = make_order("AAPL", OrderSide.BUY, 10_000)
    result = trader.execute_order(order, PRICES)

    assert result is None
    # Cash should be unchanged
    assert abs(trader.portfolio.cash - INITIAL_CASH) < 0.01


def test_sell_reduces_position(trader: PaperTrader) -> None:
    """Buy 100 then sell 50 — position quantity must be 50."""
    trader.execute_order(make_order("AAPL", OrderSide.BUY, 100), PRICES)
    result = trader.execute_order(make_order("AAPL", OrderSide.SELL, 50), PRICES)

    assert result is not None
    assert result.status == OrderStatus.FILLED
    assert trader.portfolio.positions["AAPL"].quantity == 50.0


def test_sell_removes_position(trader: PaperTrader) -> None:
    """Buy 100 then sell all 100 — ticker must be removed from positions."""
    trader.execute_order(make_order("AAPL", OrderSide.BUY, 100), PRICES)
    result = trader.execute_order(make_order("AAPL", OrderSide.SELL, 100), PRICES)

    assert result is not None
    assert result.status == OrderStatus.FILLED
    assert "AAPL" not in trader.portfolio.positions


def test_trade_log_to_csv(trader: PaperTrader, tmp_path: Path) -> None:
    """Buy and sell — CSV file must exist and contain trade rows."""
    trader.execute_order(make_order("AAPL", OrderSide.BUY, 100), PRICES)
    trader.execute_order(make_order("AAPL", OrderSide.SELL, 100), PRICES)

    csv_path = tmp_path / "trades.csv"
    trader.trade_log_to_csv(csv_path)

    assert csv_path.exists()
    content = csv_path.read_text()
    assert len(content) > 0
    lines = [ln for ln in content.splitlines() if ln.strip()]
    # Header + 2 trade rows (buy + sell)
    assert len(lines) >= 3


def test_portfolio_property(trader: PaperTrader) -> None:
    """After buying 100 shares, portfolio.positions['AAPL'].quantity must be 100."""
    trader.execute_order(make_order("AAPL", OrderSide.BUY, 100), PRICES)

    portfolio = trader.portfolio
    assert "AAPL" in portfolio.positions
    assert portfolio.positions["AAPL"].quantity == 100.0
