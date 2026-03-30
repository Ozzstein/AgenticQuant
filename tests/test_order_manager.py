"""Unit tests for ContingentOrderSimulator (src/execution/order_manager.py)."""
from __future__ import annotations

import pytest

from src.execution.order_manager import ContingentOrderSimulator
from src.utils.schemas import OrderSide, OrderStatus, OrderType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim() -> ContingentOrderSimulator:
    return ContingentOrderSimulator()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_bracket() -> None:
    """Register sets TP and SL correctly; ticker appears in open_tickers."""
    sim = _make_sim()
    sim.register("AAPL", quantity=10, entry_price=150.0, take_profit_price=160.0, stop_loss_price=140.0)

    assert "AAPL" in sim.open_tickers
    entry = sim._entries["AAPL"]
    assert entry.take_profit_price == 160.0
    assert entry.stop_loss_price == 140.0
    assert entry.trail_percent is None
    assert entry.trail_peak_price == 150.0


def test_register_trailing_stop() -> None:
    """Register with trail_percent; trail_peak equals entry_price and stop_loss is cleared."""
    sim = _make_sim()
    sim.register("BTC", quantity=1, entry_price=30_000.0, trail_percent=5.0, stop_loss_price=28_000.0)

    assert "BTC" in sim.open_tickers
    entry = sim._entries["BTC"]
    assert entry.trail_percent == 5.0
    assert entry.trail_peak_price == 30_000.0
    assert entry.stop_loss_price is None


def test_register_overwrites_existing_ticker() -> None:
    """Second register for the same ticker replaces the first entry."""
    sim = _make_sim()
    sim.register("TSLA", quantity=5, entry_price=200.0, stop_loss_price=180.0)
    sim.register("TSLA", quantity=8, entry_price=210.0, stop_loss_price=190.0)

    entry = sim._entries["TSLA"]
    assert entry.quantity == 8
    assert entry.stop_loss_price == 190.0
    assert entry.trail_peak_price == 210.0


# ---------------------------------------------------------------------------
# Take-profit triggers
# ---------------------------------------------------------------------------


def test_tp_triggers_when_price_above() -> None:
    """evaluate() returns a SELL order with fill_price=take_profit_price when price >= TP."""
    sim = _make_sim()
    sim.register("AAPL", quantity=10, entry_price=150.0, take_profit_price=160.0, stop_loss_price=140.0)

    orders = sim.evaluate({"AAPL": 162.0})

    assert len(orders) == 1
    order = orders[0]
    assert order.ticker == "AAPL"
    assert order.side == OrderSide.SELL
    assert order.fill_price == 160.0
    assert order.status == OrderStatus.PENDING
    assert order.order_type == OrderType.MARKET


def test_tp_does_not_trigger_below() -> None:
    """No order is returned when price is below TP and above SL."""
    sim = _make_sim()
    sim.register("AAPL", quantity=10, entry_price=150.0, take_profit_price=160.0, stop_loss_price=140.0)

    orders = sim.evaluate({"AAPL": 155.0})
    assert orders == []
    assert "AAPL" in sim.open_tickers


# ---------------------------------------------------------------------------
# Stop-loss triggers
# ---------------------------------------------------------------------------


def test_sl_triggers_when_price_below() -> None:
    """evaluate() returns a SELL order with fill_price=stop_loss_price when price <= SL."""
    sim = _make_sim()
    sim.register("AAPL", quantity=10, entry_price=150.0, take_profit_price=160.0, stop_loss_price=140.0)

    orders = sim.evaluate({"AAPL": 138.0})

    assert len(orders) == 1
    assert orders[0].fill_price == 140.0
    assert "AAPL" not in sim.open_tickers


def test_sl_does_not_trigger_above() -> None:
    """No order when price is between SL and TP."""
    sim = _make_sim()
    sim.register("AAPL", quantity=10, entry_price=150.0, take_profit_price=160.0, stop_loss_price=140.0)

    orders = sim.evaluate({"AAPL": 145.0})
    assert orders == []


# ---------------------------------------------------------------------------
# Trailing stop mechanics
# ---------------------------------------------------------------------------


def test_trailing_stop_peak_ratchets_up() -> None:
    """update_trailing_peaks raises trail_peak_price when current price is higher."""
    sim = _make_sim()
    sim.register("ETH", quantity=2, entry_price=2000.0, trail_percent=10.0)

    sim.update_trailing_peaks({"ETH": 2500.0})

    assert sim._entries["ETH"].trail_peak_price == 2500.0


def test_trailing_stop_peak_does_not_fall() -> None:
    """trail_peak_price never decreases when price falls."""
    sim = _make_sim()
    sim.register("ETH", quantity=2, entry_price=2000.0, trail_percent=10.0)

    sim.update_trailing_peaks({"ETH": 2500.0})
    sim.update_trailing_peaks({"ETH": 1800.0})

    assert sim._entries["ETH"].trail_peak_price == 2500.0


def test_trailing_stop_triggers_from_peak() -> None:
    """Trailing stop fires when price falls trail_percent% from the recorded peak."""
    sim = _make_sim()
    sim.register("ETH", quantity=2, entry_price=2000.0, trail_percent=10.0)

    sim.update_trailing_peaks({"ETH": 2500.0})

    # effective stop = 2500 * (1 - 0.10) = 2250; price 2200 is below that
    orders = sim.evaluate({"ETH": 2200.0})

    assert len(orders) == 1
    assert orders[0].ticker == "ETH"
    assert orders[0].fill_price == pytest.approx(2250.0)
    assert "ETH" not in sim.open_tickers


# ---------------------------------------------------------------------------
# OCO (one-cancels-other) behaviour
# ---------------------------------------------------------------------------


def test_oco_both_legs_removed_on_tp_trigger() -> None:
    """After TP fires, the ticker is removed from open_tickers (both legs cancelled)."""
    sim = _make_sim()
    sim.register("MSFT", quantity=5, entry_price=300.0, take_profit_price=320.0, stop_loss_price=280.0)

    orders = sim.evaluate({"MSFT": 325.0})
    assert len(orders) == 1
    assert "MSFT" not in sim.open_tickers


# ---------------------------------------------------------------------------
# cancel / cleanup
# ---------------------------------------------------------------------------


def test_cancel_removes_entry() -> None:
    """cancel() removes the ticker from open_tickers."""
    sim = _make_sim()
    sim.register("NVDA", quantity=3, entry_price=400.0, stop_loss_price=360.0)

    sim.cancel("NVDA")
    assert "NVDA" not in sim.open_tickers


def test_evaluate_multiple_tickers() -> None:
    """evaluate() correctly handles multiple concurrent positions."""
    sim = _make_sim()
    sim.register("AAPL", quantity=10, entry_price=150.0, take_profit_price=160.0, stop_loss_price=140.0)
    sim.register("GOOGL", quantity=2, entry_price=2800.0, take_profit_price=3000.0, stop_loss_price=2600.0)
    sim.register("MSFT", quantity=5, entry_price=300.0, take_profit_price=320.0, stop_loss_price=280.0)

    orders = sim.evaluate({"AAPL": 162.0, "GOOGL": 2500.0, "MSFT": 305.0})

    tickers_triggered = {o.ticker for o in orders}
    assert tickers_triggered == {"AAPL", "GOOGL"}
    assert "MSFT" in sim.open_tickers
    assert "AAPL" not in sim.open_tickers
    assert "GOOGL" not in sim.open_tickers


def test_cleanup_stale_removes_unheld_tickers() -> None:
    """cleanup_stale() removes entries for positions no longer held in the portfolio."""
    sim = _make_sim()
    sim.register("AAPL", quantity=10, entry_price=150.0, stop_loss_price=140.0)
    sim.register("TSLA", quantity=5, entry_price=200.0, stop_loss_price=180.0)
    sim.register("NVDA", quantity=3, entry_price=400.0, stop_loss_price=360.0)

    sim.cleanup_stale({"AAPL"})

    assert "AAPL" in sim.open_tickers
    assert "TSLA" not in sim.open_tickers
    assert "NVDA" not in sim.open_tickers
