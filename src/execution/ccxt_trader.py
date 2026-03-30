"""CCXT crypto exchange broker bridge.

Connects to any CCXT-compatible exchange for live or sandbox spot order execution.
Conforms to BaseBroker protocol (src/execution/broker.py).
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import ccxt
from loguru import logger

from src.execution.broker import compute_broker_metrics
from src.execution.order_manager import ContingentOrderSimulator
from src.utils.config import CcxtConfig
from src.utils.exceptions import CcxtConnectionError, CcxtError, CcxtOrderError
from src.utils.schemas import (
    AssetClass,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
)


class CcxtTrader:
    """CCXT exchange broker for live or sandbox spot crypto trading.

    Connects to any ccxt-compatible exchange (Binance, Kraken, Coinbase, etc.)
    for order execution and portfolio management. Conforms to BaseBroker protocol.

    Paper mode uses the exchange's sandbox/testnet when available.
    Live mode requires the AIQUANT_CCXT_LIVE=true environment variable (double opt-in).

    Args:
        config: CcxtConfig specifying exchange, credentials, and mode.

    Raises:
        CcxtError: If live mode requested without AIQUANT_CCXT_LIVE=true.
    """

    def __init__(self, config: CcxtConfig) -> None:
        self._config = config
        self._exchange: ccxt.Exchange | None = None  # lazily created in _get_exchange()
        self._nav_history: list[tuple[datetime, float]] = []
        self._trade_log: list[dict] = []
        self._portfolio: Portfolio = Portfolio(timestamp=datetime.now())

        # Live gate — prevent accidental live orders
        if not config.paper and os.getenv("AIQUANT_CCXT_LIVE") != "true":
            raise CcxtError(
                f"Live trading on {config.exchange} requires AIQUANT_CCXT_LIVE=true env var"
            )

        logger.info(
            "CcxtTrader initialised — exchange={} paper={}",
            config.exchange,
            config.paper,
        )
        self._contingent = ContingentOrderSimulator()

    # ------------------------------------------------------------------
    # BaseBroker protocol methods
    # ------------------------------------------------------------------

    def execute_order(self, order: Order, market_prices: dict[str, float]) -> Order | None:
        """Submit an order to the exchange and wait for fill.

        Maps our Order model to a ccxt create_order call, then polls for fill.

        Args:
            order: Order to execute (MARKET or LIMIT).
            market_prices: Ticker → price mapping (used as fallback price reference).

        Returns:
            Filled Order with status=FILLED and fill_price set, or None on rejection/timeout.

        Raises:
            CcxtOrderError: If order submission fails after all retries.
        """
        symbol = self._symbol(order.ticker)
        try:
            ccxt_order = self._submit_order(order, symbol)
            if ccxt_order is None:
                order.status = OrderStatus.REJECTED
                return None

            filled = self._wait_for_fill(ccxt_order["id"], symbol, self._config.timeout_seconds)
            if filled is not None:
                self._trade_log.append({
                    "timestamp": filled.timestamp.isoformat(),
                    "ticker": order.ticker,
                    "side": filled.side.value,
                    "quantity": filled.quantity,
                    "fill_price": filled.fill_price,
                    "commission": 0.0,
                    "pnl": 0.0,  # realized P&L computed post-trade
                })
                if filled.side == OrderSide.BUY and (
                    order.take_profit_price is not None
                    or order.stop_loss_price is not None
                    or order.trail_percent is not None
                ):
                    self._contingent.register(
                        ticker=order.ticker,
                        quantity=filled.quantity,
                        entry_price=filled.fill_price,
                        take_profit_price=order.take_profit_price,
                        stop_loss_price=order.stop_loss_price,
                        trail_percent=order.trail_percent,
                    )
                elif filled.side == OrderSide.SELL:
                    self._contingent.cancel(order.ticker)
            return filled
        except (CcxtOrderError, CcxtConnectionError):
            raise
        except Exception as exc:
            raise CcxtOrderError(f"Order failed for {order.ticker}: {exc}") from exc

    def snapshot(self, market_prices: dict[str, float]) -> None:
        """Sync positions from exchange balance and record NAV observation.

        Fetches balance, maps spot holdings to Position objects, computes NAV,
        and appends to internal NAV history for get_metrics().

        Args:
            market_prices: Ticker → price mapping used when exchange prices unavailable.
        """
        # Clean stale entries, ratchet trailing stops, execute triggers
        self._contingent.cleanup_stale(set(self._portfolio.positions.keys()))
        self._contingent.update_trailing_peaks(market_prices)
        triggered = self._contingent.evaluate(market_prices)
        for contingent_order in triggered:
            self.execute_order(contingent_order, market_prices)

        self._sync_balance(market_prices)
        nav = self._portfolio.nav
        self._nav_history.append((datetime.now(), nav))
        logger.debug("CcxtTrader snapshot — NAV={:.2f}", nav)

    @property
    def portfolio(self) -> Portfolio:
        """Return current portfolio snapshot.

        Returns:
            Portfolio with positions, cash, nav, and timestamp.
        """
        return self._portfolio

    def get_metrics(self) -> dict:
        """Compute performance metrics from NAV history.

        Returns:
            Dict with keys: total_return, annualized_return, sharpe, sortino,
            max_drawdown, calmar, volatility, win_rate, avg_win, avg_loss, profit_factor.
        """
        return compute_broker_metrics(self._nav_history, self._trade_log)

    # ------------------------------------------------------------------
    # Extended API (not in BaseBroker protocol)
    # ------------------------------------------------------------------

    def sync_portfolio(self, market_prices: dict[str, float] | None = None) -> Portfolio:
        """Force-sync portfolio from exchange and return it.

        Args:
            market_prices: Optional price overrides; fetched from exchange if absent.

        Returns:
            Updated Portfolio.
        """
        self._sync_balance(market_prices or {})
        return self._portfolio

    def get_account(self) -> dict:
        """Return exchange account summary.

        Returns:
            Dict with keys: exchange, cash, quote_currency.
        """
        ex = self._get_exchange()
        balance = self._fetch_balance_with_retry(ex)
        return {
            "exchange": self._config.exchange,
            "cash": balance["total"].get(self._config.quote_currency, 0.0),
            "quote_currency": self._config.quote_currency,
        }

    def cancel_all_orders(self, ticker: str | None = None) -> None:
        """Cancel all open orders, optionally filtered to one ticker.

        Args:
            ticker: If provided, cancel only orders for this ticker.
        """
        ex = self._get_exchange()
        if ticker:
            symbol = self._symbol(ticker)
            ex.cancel_all_orders(symbol)
            logger.info("CcxtTrader: cancelled all orders for {}", symbol)
        else:
            ex.cancel_all_orders()
            logger.info("CcxtTrader: cancelled all open orders")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_exchange(self) -> ccxt.Exchange:
        """Return (lazily created) ccxt exchange instance.

        Creates the exchange object with API credentials on first call.
        Sets sandbox mode if config.paper=True.

        Returns:
            Configured ccxt.Exchange instance.

        Raises:
            CcxtConnectionError: If exchange id is unknown.
        """
        if self._exchange is None:
            exchange_cls = getattr(ccxt, self._config.exchange, None)
            if exchange_cls is None:
                raise CcxtConnectionError(
                    f"Unknown ccxt exchange: {self._config.exchange!r}. "
                    f"Check ccxt.exchanges for valid ids."
                )
            self._exchange = exchange_cls({
                "apiKey": self._config.api_key,
                "secret": self._config.api_secret,
                "enableRateLimit": True,
                "timeout": self._config.timeout_seconds * 1000,
            })
            if self._config.paper:
                self._exchange.set_sandbox_mode(True)
            logger.debug(
                "CcxtTrader: exchange {} initialised (paper={})",
                self._config.exchange,
                self._config.paper,
            )
        return self._exchange

    def _symbol(self, ticker: str) -> str:
        """Format ticker as ccxt symbol (e.g. BTC → BTC/USDT).

        Args:
            ticker: Base currency ticker (e.g. BTC, ETH).

        Returns:
            ccxt trading pair symbol.
        """
        return f"{ticker}/{self._config.quote_currency}"

    def _submit_order(self, order: Order, symbol: str) -> dict | None:
        """Map our Order to ccxt create_order and submit.

        Retries up to max_retries times with exponential backoff.

        Args:
            order: Order to submit.
            symbol: ccxt symbol (e.g. BTC/USDT).

        Returns:
            ccxt order object, or None if unable to submit.

        Raises:
            CcxtConnectionError: After all retries exhausted.
        """
        ex = self._get_exchange()
        order_type = "limit" if order.order_type == OrderType.LIMIT else "market"
        side = "buy" if order.side == OrderSide.BUY else "sell"
        price = order.limit_price if order.order_type == OrderType.LIMIT else None

        params: dict = {}
        if order.order_type == OrderType.LIMIT:
            params["timeInForce"] = order.time_in_force.value

        for attempt in range(self._config.max_retries):
            try:
                return ex.create_order(symbol, order_type, side, order.quantity, price, params)
            except Exception as exc:
                if attempt == self._config.max_retries - 1:
                    raise CcxtConnectionError(
                        f"Order submit failed after {self._config.max_retries} retries: {exc}"
                    ) from exc
                time.sleep(2 ** attempt)
        return None

    def _wait_for_fill(self, order_id: str, symbol: str, timeout: int) -> Order | None:
        """Poll the exchange until the order fills or times out.

        Args:
            order_id: ccxt order id returned by create_order.
            symbol: ccxt trading pair symbol.
            timeout: Maximum seconds to wait.

        Returns:
            Filled Order, or None on cancellation/timeout.
        """
        ex = self._get_exchange()
        start = time.time()

        while time.time() - start < timeout:
            try:
                ccxt_order = ex.fetch_order(order_id, symbol)
                status = ccxt_order["status"]
                if status == "closed":
                    return self._map_ccxt_order(ccxt_order)
                if status in ("canceled", "cancelled", "expired", "rejected"):
                    return None
            except Exception as exc:
                logger.warning("CcxtTrader: poll order {} failed: {}", order_id, exc)
            time.sleep(1.0)

        logger.warning("CcxtTrader: order {} timed out after {}s", order_id, timeout)
        return None

    def _sync_balance(self, market_prices: dict[str, float]) -> None:
        """Sync portfolio from exchange balance.

        Fetches balance and maps each non-quote-currency holding to a Position.
        Computes NAV as cash + sum(qty * price) for all positions.

        Args:
            market_prices: Ticker → price overrides; fetches from exchange if missing.
        """
        ex = self._get_exchange()
        balance = self._fetch_balance_with_retry(ex)
        quote = self._config.quote_currency
        cash = float(balance["total"].get(quote, 0.0))

        positions: dict[str, Position] = {}
        total_pos_value = 0.0

        for currency, qty in balance["total"].items():
            if currency == quote or float(qty) <= 0.0:
                continue
            qty_f = float(qty)
            # Use provided price; fall back to exchange fetch
            price = market_prices.get(currency)
            if price is None:
                try:
                    ticker_data = ex.fetch_ticker(self._symbol(currency))
                    price = float(ticker_data["last"])
                except Exception as exc:
                    logger.warning(
                        "CcxtTrader: fetch_ticker failed for {} — using price=0.0: {}",
                        currency,
                        exc,
                    )
                    price = 0.0
            pos_value = qty_f * price
            total_pos_value += pos_value
            positions[currency] = Position(
                ticker=currency,
                quantity=qty_f,
                avg_cost=price,       # spot: no avg cost from exchange; use current price
                current_price=price,
                unrealized_pnl=0.0,   # no baseline cost tracked yet
                weight_pct=0.0,       # computed below
                asset_class=AssetClass.CRYPTO,
            )

        nav = cash + total_pos_value
        if nav > 0:
            for pos in positions.values():
                pos.weight_pct = (pos.current_price * pos.quantity) / nav * 100.0

        self._portfolio = Portfolio(
            positions=positions,
            cash=cash,
            nav=nav,
            timestamp=datetime.now(),
        )

    def _fetch_balance_with_retry(self, ex: ccxt.Exchange) -> dict:
        """Fetch exchange balance with exponential backoff retry.

        Args:
            ex: Configured ccxt exchange instance.

        Returns:
            ccxt balance dict.

        Raises:
            CcxtConnectionError: After all retries exhausted.
        """
        for attempt in range(self._config.max_retries):
            try:
                return ex.fetch_balance()
            except Exception as exc:
                if attempt == self._config.max_retries - 1:
                    raise CcxtConnectionError(
                        f"fetch_balance failed after {self._config.max_retries} retries: {exc}"
                    ) from exc
                time.sleep(2 ** attempt)
        return {}  # unreachable

    def _map_ccxt_order(self, ccxt_order: dict) -> Order:
        """Map a filled ccxt order dict to our Order model.

        Args:
            ccxt_order: ccxt order dict with status='closed'.

        Returns:
            Order with status=FILLED and fill_price set.
        """
        side = OrderSide.BUY if ccxt_order["side"] == "buy" else OrderSide.SELL
        # symbol is "BTC/USDT"; extract base currency
        ticker = ccxt_order["symbol"].split("/")[0]
        return Order(
            ticker=ticker,
            side=side,
            quantity=float(ccxt_order.get("filled") or ccxt_order["amount"]),
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            fill_price=float(ccxt_order.get("average")) if ccxt_order.get("average") else None,
            timestamp=datetime.fromtimestamp(ccxt_order.get("timestamp") / 1000)
            if ccxt_order.get("timestamp")
            else datetime.now(),
        )
