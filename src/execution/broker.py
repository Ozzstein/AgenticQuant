"""Broker protocol and shared metric utilities for all broker implementations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from src.utils.schemas import Order, OrderSide, Portfolio

if TYPE_CHECKING:
    from src.utils.config import AppConfig


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
    *,
    samples_per_year: int = 252,
) -> dict:
    """Use the canonical replay metrics with zero cash return.

    Keep initial capital and the final observed NAV per UTC date. Repeated
    intraday monitoring calls cannot increase the annualization sample count.
    Missing/undefined metrics are None, including profit factor without losses.
    Legacy naive timestamps retain their recorded date; typed ledger clocks are UTC.
    """
    from datetime import timezone

    from src.core.backtester import performance_metrics

    names = (
        "total_return",
        "annualized_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "volatility",
        "win_rate",
        "avg_win",
        "avg_loss",
        "profit_factor",
    )
    result = dict.fromkeys(names)
    if samples_per_year <= 0:
        raise ValueError("positive declared annualization required")
    if nav_history:
        initial_at, initial_nav = nav_history[0]
        sampled = {}
        last_at = initial_at
        for at, nav in nav_history[1:]:
            if at < last_at:
                raise ValueError("NAV observations must be chronological")
            if at == initial_at and nav == initial_nav:
                continue
            if at == last_at:
                raise ValueError("conflicting duplicate NAV timestamp")
            last_at = at
            day = at.astimezone(timezone.utc).date() if at.tzinfo is not None else at.date()
            sampled[day] = nav
        if sampled:
            metrics = performance_metrics([initial_nav, *sampled.values()], samples_per_year)
            result.update(
                total_return=metrics.total_return,
                annualized_return=metrics.annual_return,
                sharpe=metrics.sharpe_ratio,
                sortino=metrics.sortino_ratio,
                max_drawdown=metrics.max_drawdown,
                calmar=metrics.calmar_ratio,
                volatility=metrics.volatility,
            )
    pnls = [float(t["pnl"]) for t in trade_log if t.get("side") == OrderSide.SELL.value]
    if pnls:
        if not np.isfinite(pnls).all():
            raise ValueError("finite realized P&L required")
        wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p < 0]
        result.update(
            win_rate=len(wins) / len(pnls),
            avg_win=sum(wins) / len(wins) if wins else None,
            avg_loss=sum(losses) / len(losses) if losses else None,
            profit_factor=sum(wins) / abs(sum(losses)) if losses else None,
        )
    return result


def create_broker(config: AppConfig | None = None) -> BaseBroker:
    """Create a broker based on config.

    Priority: ccxt (if enabled) > alpaca (if enabled) > PaperTrader (default).
    Only one broker should be enabled at a time.

    Args:
        config: Application config. Uses get_config() singleton if None.

    Returns:
        A broker conforming to BaseBroker protocol.

    Raises:
        ImportError: If alpaca.enabled=True but alpaca-py is not installed.
        CcxtError: If ccxt live gate fails.
        AlpacaError: If alpaca live gate fails.
    """
    from src.execution.paper_trader import PaperTrader
    from src.utils.config import get_config

    if config is None:
        config = get_config()

    if config.ccxt.enabled:
        from src.execution.ccxt_trader import CcxtTrader

        return CcxtTrader(config.ccxt)

    if config.alpaca.enabled:
        try:
            from src.execution.alpaca_trader import AlpacaTrader
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is required for Alpaca trading. Install with: pip install -e '.[alpaca]'"
            ) from exc
        return AlpacaTrader(config.alpaca)

    initial_cash = config.lean.initial_cash if config.lean.initial_cash > 0 else 100_000.0
    return PaperTrader(
        initial_cash=initial_cash,
        slippage_bps=5,
        commission_per_share=0.005,
        ledger_path=getattr(config, "paper_ledger_path", None)
        or str(Path(config.output_dir) / "paper-ledger.sqlite3"),
        account_id=getattr(config, "paper_account_id", "paper:default"),
        base_currency=getattr(config, "paper_base_currency", "USD"),
    )


def create_typed_paper_execution(
    *, ledger_path, account, rules, fill_policy=None, risk_policy=None
):
    """Bind the operational paper owner to explicit account and venue metadata.

    The caller supplies the persisted AccountSpec, typed VenueRules by instrument,
    and observed market events. Reopening requires identical account/policy state.
    Ownership of the returned service's ledger passes to the caller.
    """
    from src.execution.ledger import PortfolioLedger
    from src.execution.venue import PaperVenue, VenueExecution

    ledger = PortfolioLedger(ledger_path)
    try:
        ledger.create_account(account)
        adapter = PaperVenue(
            ledger, account.account_id, rules, fill_policy=fill_policy, risk_policy=risk_policy
        )
        return VenueExecution(
            ledger,
            account.account_id,
            {adapter.venue: adapter},
            {key: adapter.venue for key in rules},
            owner="direct",
            fill_policy=fill_policy,
            risk_policy=risk_policy,
        )
    except Exception:
        ledger.close()
        raise


def create_typed_alpaca_execution(
    *,
    ledger_path,
    account,
    broker_account_id,
    api_key,
    api_secret,
    rules,
    fill_policy=None,
    risk_policy=None,
    allow_live=False,
    max_balance_age_seconds=30,
    client=None,
    clock=None,
):
    """Bind the actual Alpaca REST adapter to the canonical durable book.

    Construction makes no network requests. Call adapter.capture() before the
    decision cutoff, reconcile receipts and posted fees, then verify balances.
    The caller owns adapter.close() and execution.ledger.close(). No legacy
    float/average-price order facade participates in this path.
    """
    from src.execution.alpaca_venue import AlpacaVenue
    from src.execution.ledger import PortfolioLedger
    from src.execution.venue import VenueExecution

    ledger = PortfolioLedger(ledger_path)
    adapter = None
    try:
        ledger.create_account(account)
        adapter = AlpacaVenue(
            ledger,
            account.account_id,
            broker_account_id=broker_account_id,
            api_key=api_key,
            api_secret=api_secret,
            rules=rules,
            allow_live=allow_live,
            client=client,
            clock=clock,
        )
        return VenueExecution(
            ledger,
            account.account_id,
            {adapter.venue: adapter},
            {key: adapter.venue for key in rules},
            owner="direct",
            fill_policy=fill_policy,
            risk_policy=risk_policy,
            allow_live=allow_live,
            max_balance_age_seconds=max_balance_age_seconds,
        )
    except Exception:
        if adapter is not None:
            adapter.close()
        ledger.close()
        raise
