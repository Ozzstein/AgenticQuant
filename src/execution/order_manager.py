"""Contingent order simulation for bracket orders and trailing stops."""
from __future__ import annotations

from dataclasses import dataclass

from src.utils.schemas import Order, OrderSide, OrderStatus, OrderType


@dataclass
class ContingentEntry:
    """Tracks the contingent legs (take-profit / stop-loss / trailing stop) for one position."""

    ticker: str
    quantity: float
    take_profit_price: float | None
    stop_loss_price: float | None
    trail_percent: float | None
    trail_peak_price: float  # ratchets up, never down


class ContingentOrderSimulator:
    """Tracks open bracket and trailing stop entries; evaluates price triggers each snapshot cycle.

    Used by PaperTrader and CcxtTrader. AlpacaTrader uses native Alpaca API instead.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ContingentEntry] = {}

    def register(
        self,
        ticker: str,
        quantity: float,
        entry_price: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
        trail_percent: float | None = None,
    ) -> None:
        """Register contingent legs after a BUY fills.

        Args:
            ticker: Instrument symbol.
            quantity: Number of units held.
            entry_price: Fill price of the opening trade; sets the initial trail peak.
            take_profit_price: Price at which to take profit (optional).
            stop_loss_price: Hard stop price (optional, ignored when trail_percent is set).
            trail_percent: Trailing distance as a percentage of peak price (optional).
        """
        if trail_percent is not None:
            stop_loss_price = None
        if take_profit_price is not None and stop_loss_price is not None:
            if take_profit_price <= stop_loss_price:
                raise ValueError("take_profit_price must be greater than stop_loss_price")
        self._entries[ticker] = ContingentEntry(
            ticker=ticker,
            quantity=quantity,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            trail_percent=trail_percent,
            trail_peak_price=entry_price,
        )

    def update_trailing_peaks(self, market_prices: dict[str, float]) -> None:
        """Ratchet trail_peak_price upward as prices move higher. Call before evaluate().

        Args:
            market_prices: Latest market prices keyed by ticker.
        """
        for ticker, entry in self._entries.items():
            if entry.trail_percent is not None:
                price = market_prices.get(ticker, entry.trail_peak_price)
                if price > entry.trail_peak_price:
                    entry.trail_peak_price = price

    def evaluate(self, market_prices: dict[str, float]) -> list[Order]:
        """Check all entries against current prices. Return triggered SELL orders.

        Take-profit is checked first; if not triggered, the stop / trailing stop is checked.
        Triggered entries are removed from the tracker.

        Args:
            market_prices: Latest market prices keyed by ticker.

        Returns:
            List of SELL orders for every triggered entry.
        """
        triggered: list[Order] = []
        to_remove: list[str] = []

        for ticker, entry in self._entries.items():
            price = market_prices.get(ticker)
            if price is None:
                continue

            fill_price: float | None = None

            if entry.take_profit_price is not None and price >= entry.take_profit_price:
                fill_price = entry.take_profit_price
            else:
                effective_stop: float | None = None
                if entry.trail_percent is not None:
                    effective_stop = entry.trail_peak_price * (1 - entry.trail_percent / 100)
                elif entry.stop_loss_price is not None:
                    effective_stop = entry.stop_loss_price

                if effective_stop is not None and price <= effective_stop:
                    fill_price = effective_stop

            if fill_price is not None:
                order = Order(
                    ticker=ticker,
                    side=OrderSide.SELL,
                    quantity=entry.quantity,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.PENDING,
                    fill_price=fill_price,
                )
                triggered.append(order)
                to_remove.append(ticker)

        for ticker in to_remove:
            del self._entries[ticker]

        return triggered

    def cancel(self, ticker: str) -> None:
        """Remove contingent entry for ticker (called on manual close).

        Args:
            ticker: Instrument symbol to cancel.
        """
        self._entries.pop(ticker, None)

    def cleanup_stale(self, held_tickers: set[str]) -> None:
        """Remove entries for tickers no longer held in the portfolio.

        Args:
            held_tickers: Set of tickers currently in the portfolio.
        """
        stale = [t for t in self._entries if t not in held_tickers]
        for ticker in stale:
            del self._entries[ticker]

    @property
    def open_tickers(self) -> set[str]:
        """Set of tickers currently being monitored."""
        return set(self._entries.keys())
