"""Tests for AlpacaTrader advanced order types — bracket, trailing stop, and TIF mapping.

All Alpaca API calls are mocked; alpaca-py is NOT required to be installed.
"""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Inject alpaca stubs BEFORE importing AlpacaTrader so the lazy imports inside
# _get_client() and _submit_order() pick up the mocks, not real modules.
# ---------------------------------------------------------------------------

_alpaca_stub = MagicMock()
sys.modules.setdefault("alpaca", _alpaca_stub)
sys.modules.setdefault("alpaca.trading", _alpaca_stub.trading)
sys.modules.setdefault("alpaca.trading.client", _alpaca_stub.trading.client)
sys.modules.setdefault("alpaca.trading.requests", _alpaca_stub.trading.requests)
sys.modules.setdefault("alpaca.trading.enums", _alpaca_stub.trading.enums)

# ---------------------------------------------------------------------------
# Now it is safe to import classes under test
# ---------------------------------------------------------------------------

from src.execution.alpaca_trader import AlpacaTrader  # noqa: E402
from src.utils.config import AlpacaConfig  # noqa: E402
from src.utils.schemas import Order, OrderSide, OrderStatus, OrderType, TimeInForce  # noqa: E402

# ---------------------------------------------------------------------------
# Set up enum sentinel values on the mock so comparisons in AlpacaTrader work
# ---------------------------------------------------------------------------

from alpaca.trading.enums import OrderSide as _AS  # noqa: E402
from alpaca.trading.enums import OrderStatus as _AOS  # noqa: E402
from alpaca.trading.enums import TimeInForce as _TIF  # noqa: E402

_AS.BUY = "buy"
_AS.SELL = "sell"
_AOS.FILLED = "filled"
_AOS.CANCELED = "canceled"
_AOS.EXPIRED = "expired"
_AOS.REJECTED = "rejected"

# Configure TIF sentinel values
_TIF.GTC = "gtc"
_TIF.DAY = "day"
_TIF.IOC = "ioc"
_TIF.FOK = "fok"

from alpaca.trading.requests import (  # noqa: E402
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(**kwargs) -> AlpacaConfig:
    """Create an AlpacaConfig with test-friendly defaults."""
    defaults = {
        "enabled": True,
        "api_key": "test-key",
        "api_secret": "test-secret",
        "paper": True,
        "timeout_seconds": 2,
    }
    return AlpacaConfig(**(defaults | kwargs))


def _make_filled_order(ticker: str = "AAPL") -> Order:
    """Return a filled Order stub for patching _wait_for_fill."""
    return Order(
        ticker=ticker,
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        fill_price=200.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client():
    """A mock TradingClient with a pre-configured submit_order response."""
    client = MagicMock()
    alpaca_order = MagicMock()
    alpaca_order.id = "order-123"
    alpaca_order.status = "filled"
    alpaca_order.side = _AS.BUY
    alpaca_order.symbol = "AAPL"
    alpaca_order.qty = "10"
    alpaca_order.filled_qty = "10"
    alpaca_order.filled_avg_price = "200.0"
    alpaca_order.submitted_at = MagicMock()
    client.submit_order.return_value = alpaca_order
    return client


@pytest.fixture()
def trader(mock_client):
    """AlpacaTrader with the mock client already injected."""
    t = AlpacaTrader(make_config())
    t._client = mock_client
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlpacaAdvancedOrders:
    """7 tests covering advanced order paths in AlpacaTrader._submit_order."""

    # 1 -------------------------------------------------------------------
    def test_alpaca_both_legs_submits_bracket_request(self, trader, mock_client):
        """Bracket order (TP + SL) calls submit_order with order_class='bracket'."""
        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            take_profit_price=220.0,
            stop_loss_price=180.0,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )

        MarketOrderRequest.reset_mock()

        with patch.object(trader, "_wait_for_fill", return_value=_make_filled_order()):
            trader.execute_order(order, {"AAPL": 200.0})

        assert mock_client.submit_order.called
        call_kwargs = MarketOrderRequest.call_args[1] if MarketOrderRequest.call_args else {}
        assert call_kwargs.get("order_class") == "bracket"
        assert "take_profit" in call_kwargs
        assert "stop_loss" in call_kwargs

    # 2 -------------------------------------------------------------------
    def test_alpaca_sl_only_submits_oto_request(self, trader, mock_client):
        """SL-only order calls submit_order with order_class='oto' and stop_loss set."""
        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            stop_loss_price=180.0,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )

        MarketOrderRequest.reset_mock()

        with patch.object(trader, "_wait_for_fill", return_value=_make_filled_order()):
            trader.execute_order(order, {"AAPL": 200.0})

        assert mock_client.submit_order.called
        call_kwargs = MarketOrderRequest.call_args[1] if MarketOrderRequest.call_args else {}
        assert call_kwargs.get("order_class") == "oto"
        assert "stop_loss" in call_kwargs
        assert "take_profit" not in call_kwargs

    # 3 -------------------------------------------------------------------
    def test_alpaca_tp_only_submits_oto_request(self, trader, mock_client):
        """TP-only order calls submit_order with order_class='oto' and take_profit set."""
        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            take_profit_price=220.0,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )

        MarketOrderRequest.reset_mock()

        with patch.object(trader, "_wait_for_fill", return_value=_make_filled_order()):
            trader.execute_order(order, {"AAPL": 200.0})

        assert mock_client.submit_order.called
        call_kwargs = MarketOrderRequest.call_args[1] if MarketOrderRequest.call_args else {}
        assert call_kwargs.get("order_class") == "oto"
        assert "take_profit" in call_kwargs
        assert "stop_loss" not in call_kwargs

    # 4 -------------------------------------------------------------------
    def test_alpaca_trailing_stop_submits_trailing_stop_request(self, trader, mock_client):
        """TRAILING_STOP order uses TrailingStopOrderRequest with trail_percent."""
        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.TRAILING_STOP,
            trail_percent=5.0,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )

        TrailingStopOrderRequest.reset_mock()

        with patch.object(trader, "_wait_for_fill", return_value=_make_filled_order()):
            trader.execute_order(order, {"AAPL": 200.0})

        assert mock_client.submit_order.called
        assert TrailingStopOrderRequest.called
        call_kwargs = TrailingStopOrderRequest.call_args[1] if TrailingStopOrderRequest.call_args else {}
        assert call_kwargs.get("trail_percent") == 5.0

    # 5 -------------------------------------------------------------------
    def test_alpaca_tif_gtc_applied_to_non_bracket_order(self, trader, mock_client):
        """Plain MARKET order with time_in_force=GTC passes AlpacaTIF.GTC."""
        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )

        MarketOrderRequest.reset_mock()

        with patch.object(trader, "_wait_for_fill", return_value=_make_filled_order()):
            trader.execute_order(order, {"AAPL": 200.0})

        assert mock_client.submit_order.called
        call_kwargs = MarketOrderRequest.call_args[1] if MarketOrderRequest.call_args else {}
        assert call_kwargs.get("time_in_force") == _TIF.GTC

    # 6 -------------------------------------------------------------------
    def test_alpaca_tif_day_mapped_correctly(self, trader, mock_client):
        """Order with time_in_force=DAY passes AlpacaTIF.DAY to the request."""
        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )

        MarketOrderRequest.reset_mock()

        with patch.object(trader, "_wait_for_fill", return_value=_make_filled_order()):
            trader.execute_order(order, {"AAPL": 200.0})

        assert mock_client.submit_order.called
        call_kwargs = MarketOrderRequest.call_args[1] if MarketOrderRequest.call_args else {}
        assert call_kwargs.get("time_in_force") == _TIF.DAY

    # 7 -------------------------------------------------------------------
    def test_alpaca_tif_ioc_fok_mapped_correctly(self, trader, mock_client):
        """IOC and FOK values are correctly mapped through _map_tif."""
        assert trader._map_tif(TimeInForce.IOC) == _TIF.IOC
        assert trader._map_tif(TimeInForce.FOK) == _TIF.FOK
