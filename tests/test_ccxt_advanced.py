"""Tests for CcxtTrader advanced order features: bracket orders, trailing stops, TIF."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.utils.schemas import (
    AssetClass,
    Order,
    OrderSide,
    OrderType,
    Portfolio,
    Position,
    TimeInForce,
)
from tests.test_ccxt_trader import (
    _make_ccxt_order,
    _make_mock_exchange,
    make_config,
)


def _make_trader():
    """Return (trader, mock_exchange) with _exchange injected."""
    from src.execution.ccxt_trader import CcxtTrader

    trader = CcxtTrader(make_config())
    mock_ex = _make_mock_exchange()
    trader._exchange = mock_ex
    return trader, mock_ex


def _seed_position(trader, ticker: str, quantity: float = 0.01, price: float = 50_000.0) -> None:
    """Inject a Position into the trader's portfolio so cleanup_stale keeps it."""
    pos = Position(
        ticker=ticker,
        quantity=quantity,
        avg_cost=price,
        current_price=price,
        asset_class=AssetClass.CRYPTO,
    )
    trader._portfolio = Portfolio(
        positions={ticker: pos},
        cash=trader._portfolio.cash,
        nav=trader._portfolio.nav,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCcxtAdvancedOrders:

    # 1 — Bracket BUY registers contingent entry
    def test_ccxt_bracket_registers_contingent_after_fill(self) -> None:
        """After a BUY with TP set fills, the ticker is tracked by the contingent simulator."""
        trader, mock_ex = _make_trader()
        mock_ex.create_order.return_value = _make_ccxt_order(status="open", id="ord-1")
        mock_ex.fetch_order.return_value = _make_ccxt_order(
            status="closed", side="buy", average=50_000.0
        )

        order = Order(
            ticker="BTC",
            side=OrderSide.BUY,
            quantity=0.01,
            take_profit_price=55_000.0,
            stop_loss_price=45_000.0,
        )
        filled = trader.execute_order(order, {"BTC": 50_000.0})

        assert filled is not None
        assert "BTC" in trader._contingent.open_tickers

    # 2 — Take-profit triggers SELL in snapshot
    def test_ccxt_bracket_tp_triggers_in_snapshot(self) -> None:
        """When market price exceeds take_profit_price, a SELL is triggered via snapshot."""
        trader, _ = _make_trader()

        # Manually register the contingent entry (avoids full execute_order flow)
        trader._contingent.register("BTC", 0.01, 50_000.0, take_profit_price=55_000.0)
        # Seed portfolio so cleanup_stale does not evict the entry
        _seed_position(trader, "BTC")

        # Prevent _sync_balance from making exchange calls
        trader._sync_balance = MagicMock()
        # Capture calls to execute_order
        trader.execute_order = MagicMock()

        trader.snapshot({"BTC": 56_000.0})

        assert trader.execute_order.called
        triggered_order: Order = trader.execute_order.call_args[0][0]
        assert triggered_order.side == OrderSide.SELL
        assert triggered_order.ticker == "BTC"
        # Contingent entry should be gone after trigger
        assert "BTC" not in trader._contingent.open_tickers

    # 3 — Stop-loss triggers SELL in snapshot
    def test_ccxt_bracket_sl_triggers_in_snapshot(self) -> None:
        """When market price drops below stop_loss_price, a SELL is triggered via snapshot."""
        trader, _ = _make_trader()

        trader._contingent.register("BTC", 0.01, 50_000.0, stop_loss_price=45_000.0)
        # Seed portfolio so cleanup_stale does not evict the entry
        _seed_position(trader, "BTC")
        trader._sync_balance = MagicMock()
        trader.execute_order = MagicMock()

        trader.snapshot({"BTC": 44_000.0})

        assert trader.execute_order.called
        triggered_order: Order = trader.execute_order.call_args[0][0]
        assert triggered_order.side == OrderSide.SELL
        assert triggered_order.ticker == "BTC"
        assert "BTC" not in trader._contingent.open_tickers

    # 4 — Trailing stop ratchets up and triggers on pullback
    def test_ccxt_trailing_stop_triggers(self) -> None:
        """Trailing stop: peak ratchets to 55000, then triggers when price falls to 51000 (< 52250)."""
        trader, _ = _make_trader()

        # Entry at 50000, trail 5%
        trader._contingent.register("BTC", 0.01, 50_000.0, trail_percent=5.0)
        # Seed portfolio so cleanup_stale keeps the entry on both snapshot calls
        _seed_position(trader, "BTC")
        trader._sync_balance = MagicMock()
        trader.execute_order = MagicMock()

        # First snapshot at 55000 — peak ratchets to 55000; stop = 55000 * 0.95 = 52250
        # 55000 > 52250, so NOT triggered
        trader.snapshot({"BTC": 55_000.0})
        assert not trader.execute_order.called
        assert "BTC" in trader._contingent.open_tickers

        # Second snapshot at 51000 — 51000 <= 52250, so SELL triggered
        trader.snapshot({"BTC": 51_000.0})
        assert trader.execute_order.called
        triggered_order: Order = trader.execute_order.call_args[0][0]
        assert triggered_order.side == OrderSide.SELL
        assert triggered_order.ticker == "BTC"

    # 5 — TIF is passed only for LIMIT orders
    def test_ccxt_tif_passed_only_for_limit_orders(self) -> None:
        """LIMIT orders include timeInForce in the params dict passed to create_order."""
        trader, mock_ex = _make_trader()
        mock_ex.create_order.return_value = _make_ccxt_order(status="open", id="ord-lim")
        mock_ex.fetch_order.return_value = _make_ccxt_order(
            status="closed", side="buy", average=49_500.0
        )

        order = Order(
            ticker="ETH",
            side=OrderSide.BUY,
            quantity=0.1,
            order_type=OrderType.LIMIT,
            limit_price=49_500.0,
            time_in_force=TimeInForce.GTC,
        )
        trader.execute_order(order, {"ETH": 50_000.0})

        params = mock_ex.create_order.call_args[0][5]
        assert "timeInForce" in params
        assert params["timeInForce"] == "GTC"

    # 6 — MARKET orders do NOT include timeInForce
    def test_ccxt_market_order_no_tif_param(self) -> None:
        """MARKET orders must not include timeInForce — strict exchanges like Binance reject it."""
        trader, mock_ex = _make_trader()
        mock_ex.create_order.return_value = _make_ccxt_order(status="open", id="ord-mkt")
        mock_ex.fetch_order.return_value = _make_ccxt_order(status="closed", side="buy")

        order = Order(ticker="BTC", side=OrderSide.BUY, quantity=0.01)
        trader.execute_order(order, {"BTC": 50_000.0})

        params = mock_ex.create_order.call_args[0][5]
        assert "timeInForce" not in params
