"""Paper trading engine with FIFO avg-cost accounting, slippage, and performance metrics."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from src.utils.exceptions import OrderError
from src.utils.schemas import (
    AssetClass,
    Order,
    OrderSide,
    OrderStatus,
    Portfolio,
    Position,
)


class PaperTrader:
    """Simulated paper trading engine.

    Supports market orders with slippage and per-share commission. Tracks
    FIFO average-cost positions, realized PnL, NAV history, and trade logs.

    Args:
        initial_cash: Starting cash balance in dollars.
        slippage_bps: One-way slippage in basis points applied to fill price.
        commission_per_share: Per-share commission in dollars.
        min_commission: Minimum commission per order in dollars.
    """

    def __init__(
        self,
        initial_cash: float,
        slippage_bps: int = 5,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
    ) -> None:
        self._cash: float = initial_cash
        self._slippage_bps: int = slippage_bps
        self._commission_per_share: float = commission_per_share
        self._min_commission: float = min_commission
        self._positions: dict[str, Position] = {}
        self._nav_history: list[tuple[datetime, float]] = []
        self._trade_log: list[dict] = []
        logger.info(
            "PaperTrader initialised — cash={:.2f}, slippage={}bps, "
            "commission={}/share min={}",
            initial_cash,
            slippage_bps,
            commission_per_share,
            min_commission,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_order(self, order: Order, market_prices: dict[str, float]) -> Order | None:
        """Execute a market order against simulated prices.

        For BUY orders the fill price is marked up by slippage; for SELL orders
        it is marked down. Orders are rejected when there is insufficient cash
        (BUY) or insufficient position size (SELL).

        Args:
            order: The order to execute. Must be a MARKET order type.
            market_prices: Mapping of ticker → current market price.

        Returns:
            The filled ``Order`` with ``status=FILLED`` and ``fill_price`` set,
            or ``None`` when the order is rejected.

        Raises:
            OrderError: If the ticker is not found in ``market_prices``.
        """
        ticker = order.ticker
        if ticker not in market_prices:
            raise OrderError(f"No market price available for {ticker}")

        raw_price = market_prices[ticker]
        qty = order.quantity
        slip = self._slippage_bps / 10_000.0

        if order.side == OrderSide.BUY:
            return self._execute_buy(order, raw_price, qty, slip)
        return self._execute_sell(order, raw_price, qty, slip)

    def snapshot(self, market_prices: dict[str, float]) -> None:
        """Mark positions to market and record a NAV observation.

        Updates ``current_price``, ``unrealized_pnl``, and ``weight_pct`` for
        every open position, then recomputes NAV as cash plus total market
        value of all positions. Appends ``(datetime.now(), nav)`` to the
        internal NAV history used by ``get_metrics``.

        Args:
            market_prices: Mapping of ticker → current market price.
        """
        total_pos_value = 0.0
        for ticker, pos in self._positions.items():
            price = market_prices.get(ticker, pos.current_price)
            pos_value = price * pos.quantity
            pos.current_price = price
            pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity
            total_pos_value += pos_value

        nav = self._cash + total_pos_value

        # Weight % requires nav > 0
        if nav > 0:
            for pos in self._positions.values():
                pos.weight_pct = (pos.current_price * pos.quantity) / nav * 100.0

        ts = datetime.now()
        self._nav_history.append((ts, nav))
        logger.debug("Snapshot at {} — NAV={:.2f}", ts.isoformat(), nav)

    def get_metrics(self) -> dict:
        """Compute portfolio performance metrics from NAV history.

        Returns a dict with keys: ``total_return``, ``annualized_return``,
        ``sharpe``, ``sortino``, ``max_drawdown``, ``calmar``, ``volatility``,
        ``win_rate``, ``avg_win``, ``avg_loss``, ``profit_factor``.

        Returns zeros for all metrics when fewer than two NAV snapshots exist.

        Returns:
            Dictionary mapping metric name to float value.
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
        if len(self._nav_history) < 2:
            return zeros

        navs = np.array([nav for _, nav in self._nav_history], dtype=float)
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
        sell_pnls = [t["pnl"] for t in self._trade_log if t.get("side") == OrderSide.BUY.value
                     # pnl is stored on sell trades; skip buys
                     and False]
        sell_pnls = [t["pnl"] for t in self._trade_log if t.get("side") == OrderSide.SELL.value]

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

    def trade_log_to_csv(self, path: Path) -> None:
        """Write the trade log to a CSV file.

        Each row represents one filled order. Columns are determined by the
        keys of the first trade log entry.

        Args:
            path: Destination file path (parent directory must exist).
        """
        if not self._trade_log:
            logger.warning("trade_log_to_csv called but trade log is empty — writing header only")
            path.write_text("timestamp,ticker,side,quantity,fill_price,commission,pnl\n")
            return

        fieldnames = list(self._trade_log[0].keys())
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._trade_log)
        logger.info("Trade log written to {}", path)

    def portfolio_history_to_csv(self, path: Path) -> None:
        """Write NAV history to a CSV file with columns (timestamp, nav).

        Args:
            path: Destination file path (parent directory must exist).
        """
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "nav"])
            for ts, nav in self._nav_history:
                writer.writerow([ts.isoformat(), nav])
        logger.info("Portfolio history written to {}", path)

    @property
    def portfolio(self) -> Portfolio:
        """Return a ``Portfolio`` snapshot of the current state.

        Returns:
            A ``Portfolio`` instance built from internal positions and cash.
        """
        positions_copy = {k: v.model_copy() for k, v in self._positions.items()}
        nav = self._cash + sum(p.current_price * p.quantity for p in self._positions.values())
        return Portfolio(
            positions=positions_copy,
            cash=self._cash,
            nav=nav,
            timestamp=datetime.now(),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _commission(self, qty: float) -> float:
        """Calculate commission for a given order quantity.

        Args:
            qty: Number of shares/units traded.

        Returns:
            Commission amount in dollars.
        """
        return max(qty * self._commission_per_share, self._min_commission)

    def _execute_buy(
        self, order: Order, raw_price: float, qty: float, slip: float
    ) -> Order | None:
        """Internal handler for BUY orders."""
        fill_price = raw_price * (1.0 + slip)
        commission = self._commission(qty)
        cost = fill_price * qty + commission

        if self._cash < cost:
            logger.warning(
                "BUY REJECTED {} qty={} — insufficient cash ({:.2f} < {:.2f})",
                order.ticker,
                qty,
                self._cash,
                cost,
            )
            order.status = OrderStatus.REJECTED
            return None

        # Update cash
        self._cash -= cost

        # Update position with FIFO average-cost
        ticker = order.ticker
        if ticker in self._positions:
            pos = self._positions[ticker]
            old_qty = pos.quantity
            old_avg = pos.avg_cost
            new_qty = old_qty + qty
            pos.avg_cost = (old_qty * old_avg + qty * fill_price) / new_qty
            pos.quantity = new_qty
        else:
            self._positions[ticker] = Position(
                ticker=ticker,
                quantity=qty,
                avg_cost=fill_price,
                current_price=raw_price,
                asset_class=AssetClass.EQUITY,
            )

        # Finalise order
        order.fill_price = fill_price
        order.status = OrderStatus.FILLED

        # Trade log
        self._trade_log.append(
            {
                "timestamp": order.timestamp.isoformat(),
                "ticker": ticker,
                "side": OrderSide.BUY.value,
                "quantity": qty,
                "fill_price": fill_price,
                "commission": commission,
                "pnl": 0.0,  # PnL realised on sell
            }
        )
        logger.info(
            "BUY FILLED {} qty={} fill={:.4f} commission={:.4f} cash_remaining={:.2f}",
            ticker,
            qty,
            fill_price,
            commission,
            self._cash,
        )
        return order

    def _execute_sell(
        self, order: Order, raw_price: float, qty: float, slip: float
    ) -> Order | None:
        """Internal handler for SELL orders."""
        ticker = order.ticker
        pos = self._positions.get(ticker)

        if pos is None or pos.quantity < qty:
            logger.warning(
                "SELL REJECTED {} qty={} — position={}", ticker, qty, pos.quantity if pos else 0
            )
            order.status = OrderStatus.REJECTED
            return None

        fill_price = raw_price * (1.0 - slip)
        commission = self._commission(qty)
        proceeds = fill_price * qty - commission

        # Realized PnL
        avg_cost_at_time = pos.avg_cost
        realized_pnl = (fill_price - avg_cost_at_time) * qty - commission

        # Update cash
        self._cash += proceeds

        # Update / remove position
        pos.quantity -= qty
        if pos.quantity == 0.0:
            del self._positions[ticker]
        else:
            # avg_cost unchanged for partial sells
            pos.quantity = pos.quantity  # no-op, clarity

        # Finalise order
        order.fill_price = fill_price
        order.status = OrderStatus.FILLED

        # Trade log
        self._trade_log.append(
            {
                "timestamp": order.timestamp.isoformat(),
                "ticker": ticker,
                "side": OrderSide.SELL.value,
                "quantity": qty,
                "fill_price": fill_price,
                "commission": commission,
                "pnl": realized_pnl,
            }
        )
        logger.info(
            "SELL FILLED {} qty={} fill={:.4f} commission={:.4f} pnl={:.4f} cash={:.2f}",
            ticker,
            qty,
            fill_price,
            commission,
            realized_pnl,
            self._cash,
        )
        return order
