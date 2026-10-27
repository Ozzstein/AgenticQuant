"""Deterministic projected exposure checks for the shared execution owner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal

from src.domain.portfolio import InstrumentRisk, OrderIntent, PortfolioSnapshot, RiskPolicy
from src.utils.config import RiskConfig
from src.utils.schemas import Order, OrderSide, Portfolio

D = Decimal


@dataclass
class RiskCheckResult:
    passed: bool
    failed_checks: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)


def evaluate_order(
    intent: OrderIntent,
    portfolio: PortfolioSnapshot,
    pending: list[dict],
    instruments: dict[str, InstrumentRisk],
    policy: RiskPolicy,
) -> RiskCheckResult:
    """Check complete projected exposures, without crediting unfilled sale proceeds.

    Include all resting buys in concentration/cash/turnover. A candidate sale
    subtracts holdings and is allowed to reduce an existing concentration breach.
    Existing sell reservations prohibit double selling. ADV is base-currency
    notional; the order limit is strictly less than the configured participation.
    """
    failures = {}

    def fail(name, detail):
        failures[name] = detail

    nav = portfolio.nav
    if (
        nav <= 0
        or portfolio.account_id != intent.account_id
        or portfolio.as_of < intent.decision_at
    ):
        return RiskCheckResult(False, ["state"], {"state": "invalid NAV/account/decision clock"})
    marks = {p.instrument.instrument_id: p for p in portfolio.positions}
    exposures = {k: p.market_value for k, p in marks.items()}
    identities = {k: p.instrument for k, p in marks.items()}
    key = intent.instrument.instrument_id
    identities[key] = intent.instrument
    value = intent.quantity * intent.reference_price * intent.fx_rate
    pending_value, reserved_buys, reserved_sells = D(0), D(0), D(0)
    for o in pending:
        other = o["intent"]
        if other.intent_id == intent.intent_id:
            continue
        other_key = other.instrument.instrument_id
        identities[other_key] = other.instrument
        amount = o["remaining"] * other.reference_price * other.fx_rate
        pending_value += amount
        if other.side == "buy":
            exposures[other_key] = exposures.get(other_key, D(0)) + amount
            reserved_buys += o["reserved"]
        elif other_key == key:
            reserved_sells += o["remaining"]
    held_qty = marks[key].quantity if key in marks else D(0)
    current_value = exposures.get(key, D(0))
    projected_value = current_value + (value if intent.side == "buy" else -value)
    if intent.side == "sell":
        if intent.quantity > held_qty - reserved_sells:
            fail("sufficient_shares", "sale exceeds unreserved holdings")
        # Use held mark for projected exposure; execution proceeds may differ on gaps.
        projected_value = current_value - (
            marks[key].market_value * intent.quantity / held_qty if held_qty else value
        )
    exposures[key] = max(D(0), projected_value)
    meta = instruments.get(key)
    if meta is None:
        fail("liquidity", "missing instrument sector/ADV/lot metadata")
    else:
        age = (portfolio.as_of - meta.as_of).total_seconds()
        if not 0 <= age <= policy.max_liquidity_age_seconds:
            fail("liquidity", "stale/future liquidity metadata")
        if meta.instrument != intent.instrument:
            fail("identity", "instrument metadata mismatch")
        if meta.adv_base < policy.min_adv:
            fail("liquidity", "ADV below minimum")
        if value >= meta.adv_base * policy.max_adv_participation:
            fail("market_impact", "order participation must be strictly below limit")
        if intent.quantity % meta.lot_size:
            fail("lot_size", "quantity violates lot size")
        if value < meta.min_notional:
            fail("min_notional", "notional below venue minimum")
    # Costs reduce NAV; never treat them as invested capital.
    projected_nav = nav - intent.estimated_fee
    if projected_nav <= 0:
        fail("cash_buffer", "fees exhaust NAV")
        projected_nav = nav
    if intent.side == "buy":
        if (
            portfolio.cash - reserved_buys - value - intent.estimated_fee
            < policy.min_cash * projected_nav
        ):
            fail("cash_buffer", "cash after fees/reservations is below NAV reserve")
        if nav <= portfolio.high_water_mark * (1 - policy.max_drawdown):
            fail("drawdown_cb", "drawdown buy halt")
        for k, exposure in exposures.items():
            instrument = identities[k]
            cap = (
                policy.crypto_weight if instrument.asset_class == "crypto" else policy.equity_weight
            )
            if exposure > projected_nav * cap:
                fail("position_size", f"projected position exceeds cap: {k}")
        sectors = {}
        crypto = D(0)
        for k, exposure in exposures.items():
            if not exposure:
                continue
            if k not in instruments:
                fail("sector", f"missing sector for {k}")
                continue
            other_meta = instruments[k]
            if (
                other_meta.instrument != identities[k]
                or not 0
                <= (portfolio.as_of - other_meta.as_of).total_seconds()
                <= policy.max_liquidity_age_seconds
            ):
                fail("sector", f"stale/future or mismatched metadata for {k}")
            sector = other_meta.sector
            sectors[sector] = sectors.get(sector, D(0)) + exposure
            if identities[k].asset_class == "crypto":
                crypto += exposure
        if any(v > policy.sector_weight * projected_nav for v in sectors.values()):
            fail("sector", "projected sector exposure exceeds cap")
        if crypto > policy.crypto_total * projected_nav:
            fail("crypto_total", "projected aggregate crypto exceeds regime limit")
    if portfolio.turnover_today + pending_value + value > policy.max_daily_turnover * nav:
        fail("daily_turnover", "filled plus pending plus new turnover exceeds cap")
    return RiskCheckResult(not failures, list(failures), failures)


def check_order(
    order: Order,
    portfolio: Portfolio,
    market_prices: dict[str, float],
    config: RiskConfig,
    asset_class: str = "EQUITY",
    daily_turnover_so_far: float = 0.0,
    peak_nav: float | None = None,
) -> RiskCheckResult:
    """Legacy schema adapter; operating execution uses evaluate_order instead.

    This adapter fixes sell arithmetic and NAV cash reserves. It cannot certify
    sector/pending-order exposure because the legacy schema lacks those inputs.
    """
    failures = {}
    price = market_prices.get(order.ticker, 0)
    if not all(math.isfinite(v) and v > 0 for v in (price, order.quantity, portfolio.nav)):
        return RiskCheckResult(
            False,
            ["invalid_input"],
            {"invalid_input": "positive finite price, quantity, NAV required"},
        )
    value = price * order.quantity
    existing = portfolio.positions.get(order.ticker)
    qty = existing.quantity if existing else 0
    buy = order.side == OrderSide.BUY
    projected = (qty + order.quantity if buy else qty - order.quantity) * price
    if buy and projected > portfolio.nav * (
        0.05 if asset_class.upper() == "CRYPTO" else config.max_position_weight / 100
    ):
        failures["position_size"] = (
            f"{projected / portfolio.nav * 100:.2f}% > {5.0 if asset_class.upper() == 'CRYPTO' else config.max_position_weight}%"
        )
    if buy and portfolio.cash - value < portfolio.nav * config.min_cash_pct / 100:
        failures["cash_buffer"] = (
            f"order cost ${value:.2f} > available ${portfolio.cash - portfolio.nav * config.min_cash_pct / 100:.2f} after NAV reserve"
        )
    if not buy and order.quantity > qty:
        failures["sufficient_shares"] = "sale exceeds holdings"
    adv = market_prices.get(f"{order.ticker}_adv")
    if adv is not None:
        if not math.isfinite(adv) or adv < config.min_liquidity_adv:
            failures["liquidity"] = "invalid/insufficient ADV"
        if value >= adv * config.max_market_impact_pct / 100:
            failures["market_impact"] = "participation must be below limit"
    if (
        buy
        and peak_nav is not None
        and (peak_nav <= 0 or portfolio.nav <= peak_nav * (1 - config.max_drawdown_halt / 100))
    ):
        failures["drawdown_cb"] = "drawdown buy halt"
    if daily_turnover_so_far + value > portfolio.nav * config.max_daily_turnover / 100:
        failures["daily_turnover"] = "turnover exceeds limit"
    return RiskCheckResult(not failures, list(failures), failures)
