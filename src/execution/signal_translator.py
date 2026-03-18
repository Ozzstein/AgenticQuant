"""Convert trading signals and target weights into Order objects."""
from __future__ import annotations

import math

from loguru import logger

from src.utils.schemas import (
    Order,
    OrderSide,
    OrderType,
    Portfolio,
    Signal,
    SignalDirection,
)

# Max position size as fraction of NAV per signal unit strength
_MAX_POSITION_FRACTION = 0.10


def signals_to_orders(
    signals: list[Signal],
    portfolio: Portfolio,
    market_prices: dict[str, float],
) -> list[Order]:
    """Convert signals to market orders based on delta from current portfolio.

    For LONG signals, computes a target share count as::

        target_shares = floor(signal.strength * 0.10 * portfolio.nav / price)

    then diffs against the current position:

    * delta > 0 → BUY order
    * delta < 0 → SELL order
    * delta == 0 → no order

    For FLAT or SHORT signals, if a position exists the entire position is
    liquidated.  No new short position is opened.

    Idempotency: This is a pure function.  Running it against a portfolio that
    already matches the target produces an empty list.

    Args:
        signals: List of trading signals.
        portfolio: Current portfolio state.
        market_prices: Ticker → current price mapping.

    Returns:
        List of Order objects (all ``status=PENDING``, ``order_type=MARKET``).
        Tickers absent from *market_prices* are silently skipped.
    """
    orders: list[Order] = []

    for signal in signals:
        ticker = signal.ticker
        price = market_prices.get(ticker)

        if price is None:
            logger.debug("signal_translator: no price for {}, skipping", ticker)
            continue

        current_qty = portfolio.positions[ticker].quantity if ticker in portfolio.positions else 0.0

        if signal.direction == SignalDirection.LONG:
            target_qty = float(math.floor(signal.strength * _MAX_POSITION_FRACTION * portfolio.nav / price))
            delta = target_qty - current_qty

            if delta > 0:
                orders.append(
                    Order(
                        ticker=ticker,
                        side=OrderSide.BUY,
                        quantity=delta,
                        order_type=OrderType.MARKET,
                    )
                )
                logger.debug("signal_translator: BUY {} qty={}", ticker, delta)
            elif delta < 0:
                orders.append(
                    Order(
                        ticker=ticker,
                        side=OrderSide.SELL,
                        quantity=abs(delta),
                        order_type=OrderType.MARKET,
                    )
                )
                logger.debug("signal_translator: SELL {} qty={}", ticker, abs(delta))
            else:
                logger.debug("signal_translator: {} already at target, no order", ticker)

        else:  # FLAT or SHORT — liquidate any existing position
            if current_qty > 0:
                orders.append(
                    Order(
                        ticker=ticker,
                        side=OrderSide.SELL,
                        quantity=current_qty,
                        order_type=OrderType.MARKET,
                    )
                )
                logger.debug("signal_translator: liquidate {} qty={}", ticker, current_qty)

    return orders


def target_weights_to_orders(
    target_weights: dict[str, float],
    portfolio: Portfolio,
    market_prices: dict[str, float],
) -> list[Order]:
    """Convert target portfolio weights to rebalancing orders.

    For every ticker in *target_weights*::

        target_value  = target_weights[ticker] * portfolio.nav
        target_shares = floor(target_value / price)
        delta         = target_shares - current_shares

    BUY if delta > 0, SELL if delta < 0, skip if delta == 0.

    Tickers present in *portfolio* but absent from *target_weights* are treated
    as having a target weight of 0 and are fully liquidated.

    Args:
        target_weights: Ticker → target weight (0–1).  Values should sum ≤ 1.
        portfolio: Current portfolio state.
        market_prices: Ticker → current price mapping.

    Returns:
        List of rebalancing Order objects (all ``status=PENDING``, ``order_type=MARKET``).
    """
    orders: list[Order] = []

    # Build complete universe: targets + any existing positions not in target
    all_tickers = set(target_weights.keys()) | set(portfolio.positions.keys())

    for ticker in all_tickers:
        price = market_prices.get(ticker)
        if price is None:
            logger.debug("target_weights_to_orders: no price for {}, skipping", ticker)
            continue

        weight = target_weights.get(ticker, 0.0)
        target_value = weight * portfolio.nav
        target_qty = float(math.floor(target_value / price))
        current_qty = portfolio.positions[ticker].quantity if ticker in portfolio.positions else 0.0
        delta = target_qty - current_qty

        if delta > 0:
            orders.append(
                Order(
                    ticker=ticker,
                    side=OrderSide.BUY,
                    quantity=delta,
                    order_type=OrderType.MARKET,
                )
            )
            logger.debug("target_weights_to_orders: BUY {} qty={}", ticker, delta)
        elif delta < 0:
            orders.append(
                Order(
                    ticker=ticker,
                    side=OrderSide.SELL,
                    quantity=abs(delta),
                    order_type=OrderType.MARKET,
                )
            )
            logger.debug("target_weights_to_orders: SELL {} qty={}", ticker, abs(delta))

    return orders
