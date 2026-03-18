"""Pre-trade risk controls for order validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from src.utils.config import RiskConfig
from src.utils.schemas import Order, OrderSide, Portfolio, Position


@dataclass
class RiskCheckResult:
    """Result of running all pre-trade risk checks on a single order."""

    passed: bool
    failed_checks: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)


def check_order(
    order: Order,
    portfolio: Portfolio,
    market_prices: dict[str, float],
    config: RiskConfig,
    asset_class: str = "EQUITY",
    daily_turnover_so_far: float = 0.0,
    peak_nav: float | None = None,
) -> RiskCheckResult:
    """Run all 7 pre-trade risk checks on an order.

    Any failed check sets passed=False and appends the check name to failed_checks.

    Args:
        order: Order to validate.
        portfolio: Current portfolio state.
        market_prices: Dict of ticker → price. Include ``{ticker}_adv`` for
            liquidity and market-impact checks.
        config: RiskConfig with risk limits.
        asset_class: ``"EQUITY"`` or ``"CRYPTO"``; crypto gets a tighter 5 %
            max position weight instead of the equity default.
        daily_turnover_so_far: USD notional value of orders already placed today.
        peak_nav: Highest NAV ever (used for drawdown circuit-breaker). If
            ``None`` the drawdown check is skipped entirely.

    Returns:
        RiskCheckResult with ``passed=True`` only if every applicable check
        passes.
    """
    failed_checks: list[str] = []
    details: dict[str, str] = {}

    price = market_prices.get(order.ticker, 0.0)
    order_value = price * order.quantity

    # ------------------------------------------------------------------
    # 1. Position-size check
    # ------------------------------------------------------------------
    existing_pos: Position | None = portfolio.positions.get(order.ticker)
    current_pos_value = (existing_pos.quantity * price) if existing_pos else 0.0
    max_weight = 5.0 if asset_class == "CRYPTO" else config.max_position_weight
    post_trade_weight_pct = (current_pos_value + order_value) / portfolio.nav * 100

    if post_trade_weight_pct > max_weight:
        failed_checks.append("position_size")
        details["position_size"] = (
            f"{post_trade_weight_pct:.1f}% > {max_weight}%"
        )
        logger.warning(
            "position_size check failed for {}: {:.1f}% > {:.1f}%",
            order.ticker,
            post_trade_weight_pct,
            max_weight,
        )

    # ------------------------------------------------------------------
    # 2. Cash-buffer check (BUY only)
    # ------------------------------------------------------------------
    if order.side == OrderSide.BUY:
        order_cost = price * order.quantity
        available_cash = portfolio.cash * 0.99
        if order_cost > available_cash:
            failed_checks.append("cash_buffer")
            details["cash_buffer"] = (
                f"order cost ${order_cost:.0f} > available ${available_cash:.0f}"
            )
            logger.warning(
                "cash_buffer check failed for {}: ${:.0f} > ${:.0f}",
                order.ticker,
                order_cost,
                available_cash,
            )

    # ------------------------------------------------------------------
    # 3. Sufficient-shares check (SELL only)
    # ------------------------------------------------------------------
    if order.side == OrderSide.SELL:
        current_qty = existing_pos.quantity if existing_pos else 0.0
        if existing_pos is None or existing_pos.quantity < order.quantity:
            failed_checks.append("sufficient_shares")
            details["sufficient_shares"] = (
                f"need {order.quantity} but have {current_qty}"
            )
            logger.warning(
                "sufficient_shares check failed for {}: need {} but have {}",
                order.ticker,
                order.quantity,
                current_qty,
            )

    # ------------------------------------------------------------------
    # 4. Liquidity check
    # ------------------------------------------------------------------
    adv_key = f"{order.ticker}_adv"
    if adv_key in market_prices:
        adv = market_prices[adv_key]
        if adv < config.min_liquidity_adv:
            failed_checks.append("liquidity")
            details["liquidity"] = (
                f"ADV ${adv:.0f} < min ${config.min_liquidity_adv:.0f}"
            )
            logger.warning(
                "liquidity check failed for {}: ADV ${:.0f} < min ${:.0f}",
                order.ticker,
                adv,
                config.min_liquidity_adv,
            )

    # ------------------------------------------------------------------
    # 5. Market-impact check
    # ------------------------------------------------------------------
    if adv_key in market_prices and price > 0:
        adv = market_prices[adv_key]
        adv_shares = adv / price
        max_shares = config.max_market_impact_pct / 100 * adv_shares
        if order.quantity > max_shares:
            failed_checks.append("market_impact")
            details["market_impact"] = (
                f"order {order.quantity:.0f} shares > 1% ADV ({adv_shares * 0.01:.0f} shares)"
            )
            logger.warning(
                "market_impact check failed for {}: {} shares > 1% ADV ({:.0f} shares)",
                order.ticker,
                order.quantity,
                adv_shares * 0.01,
            )

    # ------------------------------------------------------------------
    # 6. Drawdown circuit-breaker (BUY only, only when peak_nav provided)
    # ------------------------------------------------------------------
    if order.side == OrderSide.BUY and peak_nav is not None:
        drawdown = (portfolio.nav - peak_nav) / peak_nav * 100
        if drawdown < -config.max_drawdown_halt:
            failed_checks.append("drawdown_cb")
            details["drawdown_cb"] = (
                f"drawdown {drawdown:.1f}% exceeds -{config.max_drawdown_halt}% halt"
            )
            logger.warning(
                "drawdown_cb check failed: drawdown {:.1f}% exceeds -{:.1f}% halt",
                drawdown,
                config.max_drawdown_halt,
            )

    # ------------------------------------------------------------------
    # 7. Daily-turnover check
    # ------------------------------------------------------------------
    projected_turnover_pct = (daily_turnover_so_far + order_value) / portfolio.nav * 100
    if projected_turnover_pct > config.max_daily_turnover:
        failed_checks.append("daily_turnover")
        details["daily_turnover"] = (
            f"turnover {projected_turnover_pct:.1f}% > {config.max_daily_turnover}%"
        )
        logger.warning(
            "daily_turnover check failed: {:.1f}% > {:.1f}%",
            projected_turnover_pct,
            config.max_daily_turnover,
        )

    passed = len(failed_checks) == 0
    result = RiskCheckResult(passed=passed, failed_checks=failed_checks, details=details)
    logger.debug(
        "Risk check for {} {}: passed={}, failed={}",
        order.side,
        order.ticker,
        passed,
        failed_checks,
    )
    return result
