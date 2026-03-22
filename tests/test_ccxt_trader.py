"""Tests for CcxtTrader — all CCXT API calls mocked via ccxt exchange mock."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.utils.config import CcxtConfig
from src.utils.exceptions import CcxtError
from src.utils.schemas import (
    AssetClass,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(**kwargs) -> CcxtConfig:
    defaults = {
        "enabled": True,
        "exchange": "binance",
        "api_key": "test-key",
        "api_secret": "test-secret",
        "paper": True,
        "quote_currency": "USDT",
        "max_retries": 3,
        "timeout_seconds": 2,  # keep tests fast
    }
    return CcxtConfig(**(defaults | kwargs))


def _make_ccxt_order(
    id: str = "order-1",
    status: str = "closed",   # open | closed | canceled | rejected
    side: str = "buy",
    symbol: str = "BTC/USDT",
    amount: float = 0.01,
    filled: float = 0.01,
    average: float = 50_000.0,
) -> dict:
    return {
        "id": id,
        "status": status,
        "side": side,
        "symbol": symbol,
        "amount": amount,
        "filled": filled,
        "average": average,
        "timestamp": int(datetime.now().timestamp() * 1000),
    }


def _make_ccxt_balance(
    quote: float = 10_000.0,
    holdings: dict[str, float] | None = None,
) -> dict:
    """Return a ccxt-style balance dict."""
    holdings = holdings or {}
    total = {"USDT": quote}
    total.update(holdings)
    free = dict(total)
    return {"total": total, "free": free, "used": {k: 0.0 for k in total}}


def _make_mock_exchange() -> MagicMock:
    ex = MagicMock()
    ex.set_sandbox_mode = MagicMock()
    ex.create_order = MagicMock()
    ex.fetch_order = MagicMock()
    ex.fetch_balance = MagicMock(return_value=_make_ccxt_balance())
    ex.fetch_ticker = MagicMock(return_value={"last": 50_000.0})
    ex.cancel_order = MagicMock()
    ex.cancel_all_orders = MagicMock()
    return ex


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCcxtTrader:

    def _make_trader(self, config: CcxtConfig | None = None) -> tuple:
        """Return (trader, mock_exchange) with _exchange injected."""
        from src.execution.ccxt_trader import CcxtTrader
        cfg = config or make_config()
        trader = CcxtTrader(cfg)
        mock_ex = _make_mock_exchange()
        trader._exchange = mock_ex  # bypass _get_exchange()
        return trader, mock_ex

    # 1 — Protocol conformance
    def test_ccxt_trader_conforms_to_protocol(self) -> None:
        from src.execution.broker import BaseBroker
        from src.execution.ccxt_trader import CcxtTrader
        trader = CcxtTrader(make_config())
        assert isinstance(trader, BaseBroker)

    # 2 — Exchange created in paper mode (sandbox)
    def test_exchange_created_in_paper_mode(self) -> None:
        import ccxt

        from src.execution.ccxt_trader import CcxtTrader
        config = make_config(exchange="binance", paper=True)
        trader = CcxtTrader(config)
        with patch.object(ccxt, "binance", return_value=_make_mock_exchange()) as mock_cls:
            ex = trader._get_exchange()
            mock_cls.assert_called_once()
            ex.set_sandbox_mode.assert_called_once_with(True)

    # 3 — Live gate blocks without env var
    def test_live_gate_blocks_without_env_var(self) -> None:
        from src.execution.ccxt_trader import CcxtTrader
        config = make_config(paper=False)
        with pytest.raises(CcxtError, match="AIQUANT_CCXT_LIVE"):
            CcxtTrader(config)

    # 4 — Live gate allows with env var
    def test_live_gate_allows_with_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("AIQUANT_CCXT_LIVE", "true")
        from src.execution.ccxt_trader import CcxtTrader
        trader = CcxtTrader(make_config(paper=False))
        assert trader is not None

    # 5 — Execute market buy order
    def test_execute_market_buy_order(self) -> None:
        trader, mock_ex = self._make_trader()
        filled_order = _make_ccxt_order(status="closed", side="buy", filled=0.01, average=50_000.0)
        mock_ex.create_order.return_value = _make_ccxt_order(status="open", id="ord-1")
        mock_ex.fetch_order.return_value = filled_order

        order = Order(ticker="BTC", side=OrderSide.BUY, quantity=0.01)
        result = trader.execute_order(order, {"BTC": 50_000.0})

        assert result is not None
        assert result.status == OrderStatus.FILLED
        assert result.fill_price == pytest.approx(50_000.0)
        mock_ex.create_order.assert_called_once_with("BTC/USDT", "market", "buy", 0.01, None)

    # 6 — Execute market sell order
    def test_execute_market_sell_order(self) -> None:
        trader, mock_ex = self._make_trader()
        filled_order = _make_ccxt_order(status="closed", side="sell", filled=0.01, average=51_000.0)
        mock_ex.create_order.return_value = _make_ccxt_order(status="open", id="ord-2")
        mock_ex.fetch_order.return_value = filled_order

        order = Order(ticker="BTC", side=OrderSide.SELL, quantity=0.01)
        result = trader.execute_order(order, {"BTC": 51_000.0})

        assert result is not None
        assert result.status == OrderStatus.FILLED
        assert result.fill_price == pytest.approx(51_000.0)
        mock_ex.create_order.assert_called_once_with("BTC/USDT", "market", "sell", 0.01, None)

    # 7 — Execute limit order uses limit price
    def test_execute_limit_order(self) -> None:
        trader, mock_ex = self._make_trader()
        filled_order = _make_ccxt_order(status="closed", side="buy", average=49_500.0)
        mock_ex.create_order.return_value = _make_ccxt_order(status="open", id="ord-3")
        mock_ex.fetch_order.return_value = filled_order

        order = Order(ticker="ETH", side=OrderSide.BUY, quantity=0.1,
                      order_type=OrderType.LIMIT, limit_price=49_500.0)
        trader.execute_order(order, {"ETH": 50_000.0})

        mock_ex.create_order.assert_called_once_with("ETH/USDT", "limit", "buy", 0.1, 49_500.0)

    # 8 — Rejected order returns None
    def test_execute_order_rejected(self) -> None:
        trader, mock_ex = self._make_trader()
        mock_ex.create_order.return_value = _make_ccxt_order(status="open", id="ord-4")
        mock_ex.fetch_order.return_value = _make_ccxt_order(status="canceled")

        order = Order(ticker="BTC", side=OrderSide.BUY, quantity=0.01)
        result = trader.execute_order(order, {"BTC": 50_000.0})
        assert result is None

    # 9 — Order times out returns None
    def test_execute_order_timeout(self) -> None:
        trader, mock_ex = self._make_trader(make_config(timeout_seconds=2))
        mock_ex.create_order.return_value = _make_ccxt_order(status="open", id="ord-5")
        mock_ex.fetch_order.return_value = _make_ccxt_order(status="open")  # never fills

        order = Order(ticker="BTC", side=OrderSide.BUY, quantity=0.01)
        time_seq = [0.0] + [100.0] * 10  # start at 0, then immediately past timeout on each check
        with patch("src.execution.ccxt_trader.time.sleep"), \
             patch("src.execution.ccxt_trader.time.time", side_effect=time_seq):
            result = trader.execute_order(order, {"BTC": 50_000.0})
        assert result is None

    # 10 — _wait_for_fill polls until filled
    def test_wait_for_fill_polls_status(self) -> None:
        trader, mock_ex = self._make_trader()
        pending = _make_ccxt_order(status="open", id="ord-6")
        filled = _make_ccxt_order(status="closed", id="ord-6", average=50_000.0)
        mock_ex.fetch_order.side_effect = [pending, pending, filled]

        time_vals = [0.0, 0.5, 1.0, 1.5]
        with patch("src.execution.ccxt_trader.time.sleep"), \
             patch("src.execution.ccxt_trader.time.time", side_effect=time_vals):
            result = trader._wait_for_fill("ord-6", "BTC/USDT", timeout=10)

        assert result is not None
        assert result.fill_price == pytest.approx(50_000.0)
        assert mock_ex.fetch_order.call_count == 3

    # 11 — snapshot syncs balance and records NAV
    def test_snapshot_syncs_balance_and_nav(self) -> None:
        trader, mock_ex = self._make_trader()
        mock_ex.fetch_balance.return_value = _make_ccxt_balance(
            quote=5_000.0,
            holdings={"BTC": 0.1},
        )
        mock_ex.fetch_ticker.return_value = {"last": 50_000.0}

        trader.snapshot({"BTC": 50_000.0})

        assert trader.portfolio.cash == pytest.approx(5_000.0)
        # NAV = cash + positions market value = 5000 + 0.1*50000 = 10000
        assert trader.portfolio.nav == pytest.approx(10_000.0)
        assert len(trader._nav_history) == 1

    # 12 — portfolio property reflects synced data
    def test_portfolio_property(self) -> None:
        trader, mock_ex = self._make_trader()
        mock_ex.fetch_balance.return_value = _make_ccxt_balance(
            quote=8_000.0, holdings={"ETH": 2.0}
        )
        mock_ex.fetch_ticker.return_value = {"last": 2_000.0}
        trader.snapshot({"ETH": 2_000.0})

        p = trader.portfolio
        assert isinstance(p, Portfolio)
        assert p.cash == pytest.approx(8_000.0)
        assert "ETH" in p.positions

    # 13 — sync_portfolio maps spot balance to positions
    def test_sync_portfolio_maps_balance_to_positions(self) -> None:
        trader, mock_ex = self._make_trader()
        mock_ex.fetch_balance.return_value = _make_ccxt_balance(
            quote=1_000.0,
            holdings={"BTC": 0.05, "ETH": 1.0},
        )
        mock_ex.fetch_ticker.side_effect = [
            {"last": 50_000.0},  # BTC
            {"last": 2_000.0},   # ETH
        ]
        portfolio = trader.sync_portfolio({"BTC": 50_000.0, "ETH": 2_000.0})

        assert "BTC" in portfolio.positions
        assert "ETH" in portfolio.positions
        assert portfolio.positions["BTC"].quantity == pytest.approx(0.05)
        assert portfolio.positions["BTC"].asset_class == AssetClass.CRYPTO
        assert portfolio.positions["ETH"].quantity == pytest.approx(1.0)

    # 14 — get_account returns summary dict
    def test_get_account_returns_details(self) -> None:
        trader, mock_ex = self._make_trader()
        mock_ex.fetch_balance.return_value = _make_ccxt_balance(quote=9_500.0)
        details = trader.get_account()
        assert "cash" in details
        assert "exchange" in details
        assert details["exchange"] == "binance"

    # 15 — API error retries 3x
    def test_api_error_retries(self) -> None:
        import ccxt as ccxt_module
        trader, mock_ex = self._make_trader()
        mock_ex.fetch_balance.side_effect = [
            ccxt_module.NetworkError("timeout"),
            ccxt_module.NetworkError("timeout"),
            _make_ccxt_balance(quote=5_000.0),
        ]
        with patch("src.execution.ccxt_trader.time.sleep"):
            trader.snapshot({})
        assert mock_ex.fetch_balance.call_count == 3

    # 16 — get_metrics uses nav_history
    def test_get_metrics_with_nav_history(self) -> None:
        trader, _ = self._make_trader()
        trader._nav_history = [
            (datetime(2024, 1, i), 100_000.0 + i * 500) for i in range(1, 11)
        ]
        metrics = trader.get_metrics()
        assert metrics["total_return"] > 0
        assert "sharpe" in metrics
