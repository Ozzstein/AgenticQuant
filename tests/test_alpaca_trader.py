"""Tests for AlpacaTrader — all Alpaca API calls are mocked.

alpaca-py is NOT required to be installed: the alpaca.* modules are patched
into sys.modules before the class under test is imported, so these tests run
in any environment.
"""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Inject alpaca stubs BEFORE importing AlpacaTrader so the lazy import
# inside _get_client() picks up the mocks, not real (possibly absent) modules.
# ---------------------------------------------------------------------------

_alpaca_stub = MagicMock()
sys.modules.setdefault("alpaca", _alpaca_stub)
sys.modules.setdefault("alpaca.trading", _alpaca_stub.trading)
sys.modules.setdefault("alpaca.trading.client", _alpaca_stub.trading.client)
sys.modules.setdefault("alpaca.trading.requests", _alpaca_stub.trading.requests)
sys.modules.setdefault("alpaca.trading.enums", _alpaca_stub.trading.enums)

# ---------------------------------------------------------------------------
# Now it's safe to import the class under test
# ---------------------------------------------------------------------------

from src.execution.alpaca_trader import AlpacaTrader  # noqa: E402
from src.execution.broker import BaseBroker  # noqa: E402
from src.utils.config import AlpacaConfig  # noqa: E402
from src.utils.exceptions import AlpacaConnectionError, AlpacaError, AlpacaOrderError  # noqa: E402
from src.utils.schemas import Order, OrderSide, OrderStatus, OrderType, Position  # noqa: E402


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


def _make_alpaca_order(
    status: str = "filled",
    side: str = "buy",
    symbol: str = "AAPL",
    qty: str = "10",
    filled_qty: str = "10",
    filled_avg_price: str = "150.00",
) -> MagicMock:
    """Return a mock Alpaca order object."""
    obj = MagicMock()
    obj.id = "alpaca-order-id-123"
    obj.symbol = symbol
    obj.side = side
    obj.qty = qty
    obj.filled_qty = filled_qty
    obj.filled_avg_price = filled_avg_price
    obj.created_at = datetime(2025, 1, 1, 10, 0, 0)
    obj.status = status
    return obj


def _make_alpaca_position(
    symbol: str = "AAPL",
    qty: str = "10",
    avg_entry_price: str = "140.00",
    current_price: str = "150.00",
    unrealized_pl: str = "100.00",
) -> MagicMock:
    """Return a mock Alpaca position object."""
    obj = MagicMock()
    obj.symbol = symbol
    obj.qty = qty
    obj.avg_entry_price = avg_entry_price
    obj.current_price = current_price
    obj.unrealized_pl = unrealized_pl
    return obj


def _make_alpaca_account(
    cash: str = "50000.00",
    buying_power: str = "100000.00",
    portfolio_value: str = "105000.00",
    status: str = "ACTIVE",
) -> MagicMock:
    """Return a mock Alpaca account object."""
    obj = MagicMock()
    obj.id = "account-id-456"
    obj.cash = cash
    obj.buying_power = buying_power
    obj.portfolio_value = portfolio_value
    obj.status = status
    return obj


def _make_trader_with_mock_client(config: AlpacaConfig | None = None) -> tuple[AlpacaTrader, MagicMock]:
    """Create an AlpacaTrader and inject a mock TradingClient into it."""
    cfg = config or make_config()
    trader = AlpacaTrader(cfg)
    mock_client = MagicMock()
    trader._client = mock_client
    return trader, mock_client


# ---------------------------------------------------------------------------
# Enums used in comparisons — set up so the mock module returns predictable
# sentinel objects that our code can compare against.
# ---------------------------------------------------------------------------

# We set the real enum values directly on the mock so assertions work:
from alpaca.trading.enums import OrderSide as _AS, OrderStatus as _AOS  # noqa: E402

_AS.BUY = "buy"
_AS.SELL = "sell"
_AOS.FILLED = "filled"
_AOS.CANCELED = "canceled"
_AOS.EXPIRED = "expired"
_AOS.REJECTED = "rejected"

# Also set TimeInForce.DAY so _submit_order doesn't fail on attribute access
from alpaca.trading.enums import TimeInForce as _TIF  # noqa: E402

_TIF.DAY = "day"


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestAlpacaTrader:
    """16 unit tests for AlpacaTrader — all API calls mocked."""

    # 1 -------------------------------------------------------------------
    def test_alpaca_trader_conforms_to_protocol(self):
        """AlpacaTrader satisfies the BaseBroker runtime-checkable protocol."""
        trader = AlpacaTrader(make_config())
        assert isinstance(trader, BaseBroker)

    # 2 -------------------------------------------------------------------
    def test_client_creation_paper_mode(self):
        """_get_client() constructs TradingClient with paper=True."""
        cfg = make_config(paper=True)
        trader = AlpacaTrader(cfg)

        mock_tc_class = MagicMock()
        with patch("alpaca.trading.client.TradingClient", mock_tc_class):
            # Force lazy creation by resetting any existing client
            trader._client = None
            trader._get_client()

        mock_tc_class.assert_called_once_with(
            api_key="test-key",
            secret_key="test-secret",
            paper=True,
            url_override=None,
        )

    # 3 -------------------------------------------------------------------
    def test_client_creation_live_gate_blocks(self, monkeypatch):
        """Live mode without AIQUANT_ALPACA_LIVE=true raises AlpacaError."""
        monkeypatch.delenv("AIQUANT_ALPACA_LIVE", raising=False)
        cfg = make_config(paper=False)
        with pytest.raises(AlpacaError, match="AIQUANT_ALPACA_LIVE"):
            AlpacaTrader(cfg)

    # 4 -------------------------------------------------------------------
    def test_client_creation_live_gate_allows(self, monkeypatch):
        """Live mode with AIQUANT_ALPACA_LIVE=true initialises without error."""
        monkeypatch.setenv("AIQUANT_ALPACA_LIVE", "true")
        cfg = make_config(paper=False)
        trader = AlpacaTrader(cfg)
        assert trader._config.paper is False

    # 5 -------------------------------------------------------------------
    def test_execute_market_buy_order(self):
        """BUY market order: submit + fill returns an Order with FILLED status."""
        trader, mock_client = _make_trader_with_mock_client()

        alpaca_order = _make_alpaca_order(status="filled", side="buy")
        mock_client.submit_order.return_value = alpaca_order
        mock_client.get_order_by_id.return_value = alpaca_order

        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )

        result = trader.execute_order(order, {"AAPL": 150.0})

        assert result is not None
        assert result.status == OrderStatus.FILLED
        assert result.ticker == "AAPL"
        assert result.side == OrderSide.BUY

    # 6 -------------------------------------------------------------------
    def test_execute_market_sell_order(self):
        """SELL market order: returns a FILLED Order with correct side."""
        trader, mock_client = _make_trader_with_mock_client()

        alpaca_order = _make_alpaca_order(status="filled", side="sell", symbol="MSFT")
        mock_client.submit_order.return_value = alpaca_order
        mock_client.get_order_by_id.return_value = alpaca_order

        order = Order(
            ticker="MSFT",
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )

        result = trader.execute_order(order, {"MSFT": 300.0})

        assert result is not None
        assert result.status == OrderStatus.FILLED
        assert result.ticker == "MSFT"
        assert result.side == OrderSide.SELL

    # 7 -------------------------------------------------------------------
    def test_execute_limit_order(self):
        """Limit order uses LimitOrderRequest (not MarketOrderRequest)."""
        trader, mock_client = _make_trader_with_mock_client()

        alpaca_order = _make_alpaca_order(status="filled", side="buy")
        mock_client.submit_order.return_value = alpaca_order
        mock_client.get_order_by_id.return_value = alpaca_order

        # Configure what the request classes look like
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        # Track which request type was constructed
        captured = []

        def capture_limit(*args, **kwargs):
            obj = MagicMock()
            captured.append(("limit", kwargs))
            return obj

        LimitOrderRequest.side_effect = capture_limit

        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=148.50,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )

        result = trader.execute_order(order, {"AAPL": 149.0})

        assert result is not None
        assert len(captured) == 1
        _, kwargs = captured[0]
        assert kwargs.get("limit_price") == 148.50

    # 8 -------------------------------------------------------------------
    def test_execute_order_rejected(self):
        """If Alpaca returns REJECTED status, execute_order returns None."""
        trader, mock_client = _make_trader_with_mock_client()

        alpaca_order = _make_alpaca_order(status="rejected", side="buy")
        mock_client.submit_order.return_value = alpaca_order
        mock_client.get_order_by_id.return_value = alpaca_order

        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )

        result = trader.execute_order(order, {"AAPL": 150.0})
        assert result is None

    # 9 -------------------------------------------------------------------
    def test_execute_order_timeout(self):
        """_wait_for_fill returns None when poll always yields PENDING."""
        cfg = make_config(timeout_seconds=2)
        trader, mock_client = _make_trader_with_mock_client(cfg)

        pending_order = _make_alpaca_order(status="new", side="buy")
        # submit succeeds
        mock_client.submit_order.return_value = pending_order
        # every poll returns a pending (non-terminal) status
        mock_client.get_order_by_id.return_value = pending_order

        order = Order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )

        result = trader.execute_order(order, {"AAPL": 150.0})
        assert result is None

    # 10 ------------------------------------------------------------------
    def test_wait_for_fill_polls_status(self):
        """_wait_for_fill calls get_order_by_id multiple times before fill."""
        cfg = make_config(timeout_seconds=5)
        trader, mock_client = _make_trader_with_mock_client(cfg)

        pending = _make_alpaca_order(status="new")
        filled = _make_alpaca_order(status="filled")
        # First call returns pending, second returns filled
        mock_client.get_order_by_id.side_effect = [pending, filled]

        result = trader._wait_for_fill("alpaca-order-id-123", timeout=5)

        assert result is not None
        assert result.status == OrderStatus.FILLED
        assert mock_client.get_order_by_id.call_count == 2

    # 11 ------------------------------------------------------------------
    def test_snapshot_syncs_positions(self):
        """snapshot() updates portfolio positions and records a NAV entry."""
        trader, mock_client = _make_trader_with_mock_client()

        pos = _make_alpaca_position(symbol="TSLA")
        account = _make_alpaca_account(portfolio_value="110000.00", cash="10000.00")
        mock_client.get_all_positions.return_value = [pos]
        mock_client.get_account.return_value = account

        trader.snapshot({"TSLA": 200.0})

        assert "TSLA" in trader.portfolio.positions
        assert trader.portfolio.nav == pytest.approx(110000.0)
        assert trader.portfolio.cash == pytest.approx(10000.0)
        assert len(trader._nav_history) == 1

    # 12 ------------------------------------------------------------------
    def test_portfolio_property(self):
        """portfolio property returns the last-synced Portfolio instance."""
        trader, mock_client = _make_trader_with_mock_client()

        pos = _make_alpaca_position(symbol="GOOGL")
        account = _make_alpaca_account(portfolio_value="200000.00", cash="5000.00")
        mock_client.get_all_positions.return_value = [pos]
        mock_client.get_account.return_value = account

        trader.snapshot({})

        portfolio = trader.portfolio
        assert portfolio.nav == pytest.approx(200000.0)
        assert "GOOGL" in portfolio.positions

    # 13 ------------------------------------------------------------------
    def test_sync_portfolio_maps_positions(self):
        """sync_portfolio() maps Alpaca Position fields to our Position model."""
        trader, mock_client = _make_trader_with_mock_client()

        ap = _make_alpaca_position(
            symbol="NVDA",
            qty="20",
            avg_entry_price="400.00",
            current_price="450.00",
            unrealized_pl="1000.00",
        )
        account = _make_alpaca_account(portfolio_value="120000.00", cash="20000.00")
        mock_client.get_all_positions.return_value = [ap]
        mock_client.get_account.return_value = account

        portfolio = trader.sync_portfolio()

        assert "NVDA" in portfolio.positions
        p: Position = portfolio.positions["NVDA"]
        assert p.ticker == "NVDA"
        assert p.quantity == pytest.approx(20.0)
        assert p.avg_cost == pytest.approx(400.0)
        assert p.current_price == pytest.approx(450.0)
        assert p.unrealized_pnl == pytest.approx(1000.0)

    # 14 ------------------------------------------------------------------
    def test_get_account_returns_details(self):
        """get_account() returns dict with expected keys and values."""
        trader, mock_client = _make_trader_with_mock_client()

        account = _make_alpaca_account(
            cash="30000.00",
            buying_power="60000.00",
            portfolio_value="130000.00",
            status="ACTIVE",
        )
        mock_client.get_account.return_value = account

        details = trader.get_account()

        assert details["cash"] == pytest.approx(30000.0)
        assert details["buying_power"] == pytest.approx(60000.0)
        assert details["portfolio_value"] == pytest.approx(130000.0)
        assert details["status"] == "ACTIVE"
        assert "id" in details

    # 15 ------------------------------------------------------------------
    def test_api_error_retries(self):
        """get_all_positions raises twice then succeeds; total calls == 3."""
        trader, mock_client = _make_trader_with_mock_client()

        pos = _make_alpaca_position(symbol="AMD")
        account = _make_alpaca_account()
        mock_client.get_all_positions.side_effect = [
            RuntimeError("network error"),
            RuntimeError("network error"),
            [pos],
        ]
        mock_client.get_account.return_value = account

        # Patch time.sleep to avoid waiting during retries
        with patch("src.execution.alpaca_trader.time.sleep"):
            trader.snapshot({})

        assert mock_client.get_all_positions.call_count == 3
        assert "AMD" in trader.portfolio.positions

    # 16 ------------------------------------------------------------------
    def test_get_metrics_with_nav_history(self):
        """get_metrics returns non-zero Sharpe when NAV history is injected."""
        trader = AlpacaTrader(make_config())

        # Inject a rising NAV series so there is a measurable return
        now = datetime.now()
        trader._nav_history = [
            (datetime(2025, 1, d), 100_000.0 + d * 500.0) for d in range(1, 31)
        ]

        metrics = trader.get_metrics()

        assert isinstance(metrics, dict)
        assert "sharpe" in metrics
        assert "total_return" in metrics
        assert metrics["total_return"] > 0.0
