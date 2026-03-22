"""Broker protocol and shared metric utilities for all broker implementations."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np

from src.utils.schemas import Order, OrderSide, Portfolio


@runtime_checkable
class BaseBroker(Protocol):
    """Structural protocol for all broker implementations.

    Any class that implements ``execute_order``, ``snapshot``, ``portfolio``,
    and ``get_metrics`` with matching signatures satisfies this protocol at
    runtime via ``isinstance`` checks (PEP 544).

    Concrete implementations:
        - ``PaperTrader`` — simulated trading engine (``src/execution/paper_trader.py``)
        - ``AlpacaTrader`` — live/paper Alpaca REST bridge (future)
    """

    def execute_order(self, order: Order, market_prices: dict[str, float]) -> Order | None:
        """Submit an order and return the filled order, or None if rejected.

        Args:
            order: The order to execute.
            market_prices: Mapping of ticker → current market price.

        Returns:
            The filled ``Order`` with ``status=FILLED``, or ``None`` on rejection.
        """
        ...

    def snapshot(self, market_prices: dict[str, float]) -> None:
        """Mark positions to market and record a NAV observation.

        Args:
            market_prices: Mapping of ticker → current market price.
        """
        ...

    @property
    def portfolio(self) -> Portfolio:
        """Return the current portfolio snapshot.

        Returns:
            A ``Portfolio`` instance reflecting current positions and cash.
        """
        ...

    def get_metrics(self) -> dict:
        """Compute and return portfolio performance metrics.

        Returns:
            Dictionary mapping metric name to float value.  At minimum the
            keys returned by ``compute_broker_metrics`` are expected.
        """
        ...


def compute_broker_metrics(
    nav_history: list[tuple[datetime, float]],
    trade_log: list[dict],
) -> dict:
    """Compute portfolio performance metrics from NAV history and trade log.

    This utility function encapsulates the shared metric logic so that both
    ``PaperTrader`` and future broker implementations (e.g. ``AlpacaTrader``)
    can compute identical statistics without duplicating code.

    Metrics are calculated using daily NAV observations.  A risk-free rate of
    4% per annum is assumed for Sharpe / Sortino calculations.

    Args:
        nav_history: Ordered list of ``(timestamp, nav)`` tuples.  Must contain
            at least two entries for any non-zero metric to be returned.
        trade_log: List of trade log dicts.  Each sell-side entry is expected to
            carry a ``"pnl"`` key with the realised P&L for that trade and a
            ``"side"`` key equal to ``OrderSide.SELL.value``.

    Returns:
        Dictionary with keys:
            ``total_return``, ``annualized_return``, ``sharpe``, ``sortino``,
            ``max_drawdown``, ``calmar``, ``volatility``, ``win_rate``,
            ``avg_win``, ``avg_loss``, ``profit_factor``.
        All values are floats.  Returns zeros for every key when fewer than
        two NAV snapshots are available.
    """
    zeros: dict = {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "calmar": 0.0,
        "volatility": 0.0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
    }
    if len(nav_history) < 2:
        return zeros

    navs = np.array([nav for _, nav in nav_history], dtype=float)
    returns = np.diff(navs) / navs[:-1]

    total_return = float((navs[-1] - navs[0]) / navs[0])
    n = max(len(returns), 1)
    annualized_return = float((1.0 + total_return) ** (252.0 / n) - 1.0)

    vol = float(np.std(returns, ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    rf = 0.04
    sharpe = float((annualized_return - rf) / vol) if vol != 0.0 else 0.0

    downside = returns[returns < 0]
    if len(downside) > 1:
        downside_vol = float(np.std(downside, ddof=1) * np.sqrt(252))
    elif len(downside) == 1:
        downside_vol = float(abs(downside[0]) * np.sqrt(252))
    else:
        downside_vol = 0.0
    sortino = float((annualized_return - rf) / downside_vol) if downside_vol != 0.0 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(navs)
    drawdowns = (navs - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))

    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown != 0.0 else 0.0

    # Trade-level stats (sell trades only)
    sell_pnls = [t["pnl"] for t in trade_log if t.get("side") == OrderSide.SELL.value]

    if sell_pnls:
        wins = [p for p in sell_pnls if p > 0]
        losses = [p for p in sell_pnls if p <= 0]
        win_rate = float(len(wins) / len(sell_pnls))
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = float(gross_profit / gross_loss) if gross_loss != 0.0 else 0.0
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "volatility": vol,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }
