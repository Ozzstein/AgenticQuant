"""Tests for PaperTrader advanced order types: bracket, trailing stop, STOP orders."""
from __future__ import annotations

from src.execution.paper_trader import PaperTrader
from src.utils.schemas import Order, OrderSide, OrderStatus, OrderType, TimeInForce


def _trader(cash: float = 100_000.0) -> PaperTrader:
    """Create a PaperTrader with zero friction for predictable prices."""
    return PaperTrader(
        initial_cash=cash,
        slippage_bps=0,
        commission_per_share=0,
        min_commission=0,
    )


def _buy_order(
    ticker: str = "AAPL",
    qty: float = 10.0,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    trail_percent: float | None = None,
) -> Order:
    return Order(
        ticker=ticker,
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.MARKET,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price,
        trail_percent=trail_percent,
    )


# ---------------------------------------------------------------------------
# 1. Bracket — take-profit fires
# ---------------------------------------------------------------------------


def test_paper_bracket_tp_fills_on_snapshot() -> None:
    """Bracket order with TP=220: snapshot at 225 should close the position."""
    trader = _trader()
    order = _buy_order(ticker="AAPL", qty=10, take_profit_price=220.0)
    trader.execute_order(order, {"AAPL": 200.0})
    assert order.status == OrderStatus.FILLED
    assert "AAPL" in trader.portfolio.positions

    trader.snapshot({"AAPL": 225.0})
    assert "AAPL" not in trader.portfolio.positions


# ---------------------------------------------------------------------------
# 2. Bracket — stop-loss fires
# ---------------------------------------------------------------------------


def test_paper_bracket_sl_fills_on_snapshot() -> None:
    """Bracket order with SL=180: snapshot at 175 should close the position."""
    trader = _trader()
    order = _buy_order(ticker="AAPL", qty=10, stop_loss_price=180.0)
    trader.execute_order(order, {"AAPL": 200.0})
    assert order.status == OrderStatus.FILLED

    trader.snapshot({"AAPL": 175.0})
    assert "AAPL" not in trader.portfolio.positions


# ---------------------------------------------------------------------------
# 3. Trailing stop — triggers after rally then drop
# ---------------------------------------------------------------------------


def test_paper_trailing_stop_triggers_after_rally() -> None:
    """Trail 10%: buy at 200, rally to 220, drop to 195 — stop at 198 triggers."""
    trader = _trader()
    order = _buy_order(ticker="AAPL", qty=10, trail_percent=10.0)
    trader.execute_order(order, {"AAPL": 200.0})

    # Rally: peak becomes 220, effective stop = 220 * 0.9 = 198
    trader.snapshot({"AAPL": 220.0})
    assert "AAPL" in trader.portfolio.positions  # not triggered yet

    # Drop to 195 <= 198 — should trigger
    trader.snapshot({"AAPL": 195.0})
    assert "AAPL" not in trader.portfolio.positions


# ---------------------------------------------------------------------------
# 4. Trailing stop — no trigger during rally
# ---------------------------------------------------------------------------


def test_paper_trailing_stop_does_not_trigger_during_rally() -> None:
    """Trail 10%: buy at 200, rally to 220 — position still held."""
    trader = _trader()
    order = _buy_order(ticker="AAPL", qty=10, trail_percent=10.0)
    trader.execute_order(order, {"AAPL": 200.0})

    trader.snapshot({"AAPL": 220.0})
    assert "AAPL" in trader.portfolio.positions


# ---------------------------------------------------------------------------
# 5. STOP order rejected when price is above stop level
# ---------------------------------------------------------------------------


def test_paper_stop_order_rejects_if_price_above_stop() -> None:
    """STOP SELL at 180 while market price is 200 — order must be rejected."""
    trader = _trader()
    # First buy the position
    buy = _buy_order(ticker="AAPL", qty=10)
    trader.execute_order(buy, {"AAPL": 200.0})

    stop_order = Order(
        ticker="AAPL",
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.STOP,
        stop_loss_price=180.0,
    )
    result = trader.execute_order(stop_order, {"AAPL": 200.0})
    assert result is None
    assert stop_order.status == OrderStatus.REJECTED
    assert "AAPL" in trader.portfolio.positions  # position unchanged


# ---------------------------------------------------------------------------
# 6. STOP order fills when price is at or below stop
# ---------------------------------------------------------------------------


def test_paper_stop_order_fills_when_price_at_stop() -> None:
    """STOP SELL at 180 while market price is 175 — order must fill."""
    trader = _trader()
    buy = _buy_order(ticker="AAPL", qty=10)
    trader.execute_order(buy, {"AAPL": 200.0})

    stop_order = Order(
        ticker="AAPL",
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.STOP,
        stop_loss_price=180.0,
    )
    result = trader.execute_order(stop_order, {"AAPL": 175.0})
    assert result is not None
    assert stop_order.status == OrderStatus.FILLED
    assert "AAPL" not in trader.portfolio.positions


# ---------------------------------------------------------------------------
# 7. FOK / insufficient cash — existing cash rejection still works
# ---------------------------------------------------------------------------


def test_paper_buy_rejects_if_insufficient_cash() -> None:
    """Trader with $100 cash cannot buy 100 shares at $200 each; TimeInForce.FOK is preserved."""
    trader = _trader(cash=100.0)
    order = Order(
        ticker="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.FOK,
    )
    result = trader.execute_order(order, {"AAPL": 200.0})
    assert result is None
    assert order.status == OrderStatus.REJECTED


# ---------------------------------------------------------------------------
# 8. Manual sell cancels contingent entry
# ---------------------------------------------------------------------------


def test_paper_manual_sell_cancels_contingent() -> None:
    """After a bracket buy, a manual SELL removes the contingent entry."""
    trader = _trader()
    order = _buy_order(ticker="AAPL", qty=10, take_profit_price=220.0, stop_loss_price=180.0)
    trader.execute_order(order, {"AAPL": 200.0})
    assert "AAPL" in trader._contingent.open_tickers

    # Manual close
    sell = Order(
        ticker="AAPL",
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.MARKET,
    )
    trader.execute_order(sell, {"AAPL": 210.0})
    assert "AAPL" not in trader._contingent.open_tickers


# ---------------------------------------------------------------------------
# 9. No double-fill after contingent fires
# ---------------------------------------------------------------------------


def test_paper_no_double_fill_after_contingent_fires() -> None:
    """TP fires at 225; a second snapshot at 230 must not crash or double-fill."""
    trader = _trader()
    order = _buy_order(ticker="AAPL", qty=10, take_profit_price=220.0)
    trader.execute_order(order, {"AAPL": 200.0})

    # First snapshot: TP fires, position gone
    trader.snapshot({"AAPL": 225.0})
    assert "AAPL" not in trader.portfolio.positions

    # Second snapshot: no position, no entry — must not crash
    trader.snapshot({"AAPL": 230.0})
    assert "AAPL" not in trader.portfolio.positions
    assert "AAPL" not in trader._contingent.open_tickers
