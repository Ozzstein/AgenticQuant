"""Alpaca broker bridge via alpaca-py SDK.

Conforms to the BaseBroker protocol defined in src.execution.broker.
alpaca-py is imported lazily (only inside _get_client) so the rest of the
platform works even when the optional dependency is not installed.

Install:  pip install -e '.[alpaca]'
"""

from __future__ import annotations

import os
import time
from datetime import datetime

from loguru import logger

from src.utils.config import AlpacaConfig
from src.utils.exceptions import AlpacaConnectionError, AlpacaError, AlpacaOrderError
from src.utils.schemas import (
    AssetClass,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    TimeInForce,
)


class AlpacaTrader:
    """Alpaca broker bridge via alpaca-py SDK.

    Connects to Alpaca's REST API for order execution and portfolio management.
    Conforms to BaseBroker protocol.

    Args:
        config: AlpacaConfig instance with credentials and behaviour flags.

    Raises:
        AlpacaError: If live trading is requested without the required env var.
    """

    def __init__(self, config: AlpacaConfig) -> None:
        self._config = config
        self._client: object | None = None  # TradingClient, lazily created
        self._nav_history: list[tuple[datetime, float]] = []
        self._trade_log: list[dict] = []
        self._portfolio: Portfolio = Portfolio(timestamp=datetime.now())

        # Live gate — require explicit opt-in to prevent accidental live orders
        if not config.paper and os.getenv("AIQUANT_ALPACA_LIVE") != "true":
            raise AlpacaError("Live trading requires AIQUANT_ALPACA_LIVE=true env var")

        logger.info("AlpacaTrader initialized — paper={}", config.paper)

    # ------------------------------------------------------------------
    # BaseBroker protocol
    # ------------------------------------------------------------------

    def execute_order(self, order: Order, market_prices: dict[str, float]) -> Order | None:
        """Submit an order to Alpaca and wait for fill.

        Args:
            order: The order to submit.
            market_prices: Mapping of ticker → current price (accepted for protocol
                compatibility; Alpaca fill prices are used when available).

        Returns:
            The filled Order with status=FILLED, or None if rejected/cancelled.

        Raises:
            AlpacaOrderError: On unexpected submission failure.
        """
        try:
            alpaca_order = self._submit_order(order)
            if alpaca_order is None:
                order.status = OrderStatus.REJECTED
                return None
            filled = self._wait_for_fill(alpaca_order.id, self._config.timeout_seconds)
            if filled is not None:
                self._trade_log.append(
                    {
                        "timestamp": filled.timestamp.isoformat(),
                        "ticker": filled.ticker,
                        "side": filled.side.value,
                        "quantity": filled.quantity,
                        "fill_price": filled.fill_price,
                        "commission": 0.0,
                        "pnl": 0.0,
                    }
                )
            return filled
        except (AlpacaOrderError, AlpacaConnectionError):
            raise
        except Exception as exc:
            raise AlpacaOrderError(f"Order failed for {order.ticker}: {exc}") from exc

    def snapshot(self, market_prices: dict[str, float]) -> None:
        """Sync positions from Alpaca and record a NAV observation.

        Args:
            market_prices: Mapping of ticker → current price (accepted for protocol
                compatibility; Alpaca account values are used for NAV).
        """
        self._sync_positions()
        self._sync_account()
        nav = self._portfolio.nav
        self._nav_history.append((datetime.now(), nav))
        logger.debug("AlpacaTrader snapshot — NAV={:.2f}", nav)

    @property
    def portfolio(self) -> Portfolio:
        """Return the current portfolio snapshot.

        Returns:
            Portfolio instance reflecting last-synced positions and cash.
        """
        return self._portfolio

    def get_metrics(self) -> dict:
        """Compute performance metrics from NAV history and trade log.

        Returns:
            Dictionary with keys: total_return, annualized_return, sharpe,
            sortino, max_drawdown, calmar, volatility, win_rate, avg_win,
            avg_loss, profit_factor.
        """
        from src.execution.broker import compute_broker_metrics

        return compute_broker_metrics(self._nav_history, self._trade_log)

    # ------------------------------------------------------------------
    # Extended public API
    # ------------------------------------------------------------------

    def sync_portfolio(self) -> Portfolio:
        """Force-sync portfolio from the Alpaca API and return it.

        Returns:
            Up-to-date Portfolio snapshot.
        """
        self._sync_positions()
        self._sync_account()
        return self._portfolio

    def get_account(self) -> dict:
        """Return Alpaca account details as a plain dict.

        Returns:
            Dictionary with keys: id, cash, buying_power, portfolio_value, status.

        Raises:
            AlpacaConnectionError: If the API call fails after retries.
        """
        client = self._get_client()
        account = client.get_account()
        return {
            "id": account.id,
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "status": account.status,
        }

    def cancel_all_orders(self) -> None:
        """Cancel all open orders on Alpaca.

        Raises:
            AlpacaConnectionError: If the API call fails.
        """
        client = self._get_client()
        client.cancel_orders()
        logger.info("AlpacaTrader: cancelled all open orders")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return the lazily created TradingClient instance.

        Returns:
            alpaca.trading.client.TradingClient

        Raises:
            ImportError: If alpaca-py is not installed.
        """
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
            except ImportError as exc:
                raise ImportError(
                    "alpaca-py required. Install: pip install -e '.[alpaca]'"
                ) from exc

            base_url = self._config.base_url or None
            self._client = TradingClient(
                api_key=self._config.api_key,
                secret_key=self._config.api_secret,
                paper=self._config.paper,
                url_override=base_url,
            )
        return self._client

    def _map_tif(self, tif: TimeInForce):
        """Map our TimeInForce enum to Alpaca's TimeInForce enum.

        Args:
            tif: Our TimeInForce value.

        Returns:
            Corresponding alpaca.trading.enums.TimeInForce value.
        """
        from alpaca.trading.enums import TimeInForce as AlpacaTIF

        return {
            TimeInForce.GTC: AlpacaTIF.GTC,
            TimeInForce.DAY: AlpacaTIF.DAY,
            TimeInForce.IOC: AlpacaTIF.IOC,
            TimeInForce.FOK: AlpacaTIF.FOK,
        }[tif]

    def _submit_order(self, order: Order):
        """Map our Order to an Alpaca request and submit with retries.

        Args:
            order: Internal Order model.

        Returns:
            Alpaca order object returned by the API.

        Raises:
            AlpacaConnectionError: After exhausting all retry attempts.
        """
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
            TrailingStopOrderRequest,
        )

        client = self._get_client()
        side = AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL
        tif = self._map_tif(order.time_in_force)

        for attempt in range(self._config.max_retries):
            try:
                # 1. Trailing stop
                if order.order_type == OrderType.TRAILING_STOP:
                    req = TrailingStopOrderRequest(
                        symbol=order.ticker,
                        qty=order.quantity,
                        side=side,
                        time_in_force=tif,
                        trail_percent=order.trail_percent,
                    )
                # 2. Bracket (both TP and SL)
                elif order.take_profit_price is not None and order.stop_loss_price is not None:
                    req = MarketOrderRequest(
                        symbol=order.ticker,
                        qty=order.quantity,
                        side=side,
                        time_in_force=tif,
                        order_class="bracket",
                        take_profit=TakeProfitRequest(limit_price=order.take_profit_price),
                        stop_loss=StopLossRequest(stop_price=order.stop_loss_price),
                    )
                # 3. SL only
                elif order.stop_loss_price is not None:
                    req = MarketOrderRequest(
                        symbol=order.ticker,
                        qty=order.quantity,
                        side=side,
                        time_in_force=tif,
                        order_class="oto",
                        stop_loss=StopLossRequest(stop_price=order.stop_loss_price),
                    )
                # 4. TP only
                elif order.take_profit_price is not None:
                    req = MarketOrderRequest(
                        symbol=order.ticker,
                        qty=order.quantity,
                        side=side,
                        time_in_force=tif,
                        order_class="oto",
                        take_profit=TakeProfitRequest(limit_price=order.take_profit_price),
                    )
                # 5. Limit order
                elif order.order_type == OrderType.LIMIT and order.limit_price is not None:
                    req = LimitOrderRequest(
                        symbol=order.ticker,
                        qty=order.quantity,
                        side=side,
                        time_in_force=tif,
                        limit_price=order.limit_price,
                    )
                # 6. Plain market order
                else:
                    req = MarketOrderRequest(
                        symbol=order.ticker,
                        qty=order.quantity,
                        side=side,
                        time_in_force=tif,
                    )
                return client.submit_order(order_data=req)
            except Exception as exc:
                if attempt == self._config.max_retries - 1:
                    raise AlpacaConnectionError(
                        f"Submit order failed after {self._config.max_retries} retries: {exc}"
                    ) from exc
                time.sleep(2**attempt)
        return None

    def _wait_for_fill(self, alpaca_order_id: str, timeout: int) -> Order | None:
        """Poll Alpaca until the order is filled or the timeout expires.

        Args:
            alpaca_order_id: Alpaca order UUID string.
            timeout: Maximum seconds to wait.

        Returns:
            Mapped Order with status=FILLED, or None on cancellation/timeout.
        """
        from alpaca.trading.enums import OrderStatus as AlpacaOrderStatus

        client = self._get_client()
        start = time.time()
        terminal_statuses = (
            AlpacaOrderStatus.CANCELED,
            AlpacaOrderStatus.EXPIRED,
            AlpacaOrderStatus.REJECTED,
        )

        while time.time() - start < timeout:
            try:
                alpaca_order = client.get_order_by_id(alpaca_order_id)
                if alpaca_order.status == AlpacaOrderStatus.FILLED:
                    return self._map_alpaca_order(alpaca_order)
                if alpaca_order.status in terminal_statuses:
                    return None
            except Exception as exc:
                logger.warning(
                    "AlpacaTrader: poll order {} failed: {}", alpaca_order_id, exc
                )
            time.sleep(1.0)

        logger.warning(
            "AlpacaTrader: order {} timed out after {}s", alpaca_order_id, timeout
        )
        return None

    def _sync_positions(self) -> None:
        """Sync open positions from Alpaca into self._portfolio.

        Raises:
            AlpacaConnectionError: After exhausting all retry attempts.
        """
        client = self._get_client()
        for attempt in range(self._config.max_retries):
            try:
                positions = client.get_all_positions()
                self._portfolio.positions = {
                    p.symbol: self._map_alpaca_position(p) for p in positions
                }
                return
            except Exception as exc:
                if attempt == self._config.max_retries - 1:
                    raise AlpacaConnectionError(f"Get positions failed: {exc}") from exc
                time.sleep(2**attempt)

    def _sync_account(self) -> None:
        """Sync cash and NAV from the Alpaca account into self._portfolio.

        Raises:
            AlpacaConnectionError: After exhausting all retry attempts.
        """
        client = self._get_client()
        for attempt in range(self._config.max_retries):
            try:
                account = client.get_account()
                self._portfolio.cash = float(account.cash)
                self._portfolio.nav = float(account.portfolio_value)
                self._portfolio.timestamp = datetime.now()
                return
            except Exception as exc:
                if attempt == self._config.max_retries - 1:
                    raise AlpacaConnectionError(f"Get account failed: {exc}") from exc
                time.sleep(2**attempt)

    def _map_alpaca_position(self, pos) -> Position:
        """Map an Alpaca position object to our Position model.

        Args:
            pos: Alpaca position object with symbol, qty, avg_entry_price,
                 current_price, unrealized_pl attributes.

        Returns:
            Position model instance.
        """
        return Position(
            ticker=pos.symbol,
            quantity=float(pos.qty),
            avg_cost=float(pos.avg_entry_price),
            current_price=float(pos.current_price),
            unrealized_pnl=float(pos.unrealized_pl),
            weight_pct=0.0,  # computed during snapshot, not available per-position
            asset_class=AssetClass.EQUITY,
        )

    def _map_alpaca_order(self, alpaca_order) -> Order:
        """Map a filled Alpaca order to our Order model.

        Args:
            alpaca_order: Alpaca order object with symbol, side, qty,
                          filled_qty, filled_avg_price, created_at attributes.

        Returns:
            Order model instance with status=FILLED.
        """
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide

        side = (
            OrderSide.BUY if alpaca_order.side == AlpacaOrderSide.BUY else OrderSide.SELL
        )
        return Order(
            ticker=alpaca_order.symbol,
            side=side,
            quantity=float(alpaca_order.filled_qty or alpaca_order.qty),
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            fill_price=(
                float(alpaca_order.filled_avg_price)
                if alpaca_order.filled_avg_price
                else None
            ),
            timestamp=alpaca_order.created_at or datetime.now(),
        )
