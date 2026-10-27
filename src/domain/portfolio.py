"""Versioned long-only accounting contracts shared by paper and historical replay.

Money/quantities are Decimal; fees and balances use the account base currency.
Foreign instruments require an explicit quote-to-base FX rate on every mark/fill.
Calendar sessions are supplied by a versioned data artifact, never inferred from
weekdays. The simulator executes only at observed session/bar opens.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, model_validator

from src.domain.common import Instrument, NonEmptyString, UtcTimestamp, VersionedContract

Positive = Annotated[Decimal, Field(gt=0)]
NonNegative = Annotated[Decimal, Field(ge=0)]
Fraction = Annotated[Decimal, Field(ge=0, le=1)]


class AccountSpec(VersionedContract):
    account_id: NonEmptyString
    mode: Literal["paper", "backtest"]
    base_currency: NonEmptyString
    initial_cash: Positive
    opened_at: UtcTimestamp

    @model_validator(mode="after")
    def namespace(self):
        if not self.account_id.startswith(self.mode + ":"):
            raise ValueError("account_id must be namespaced paper: or backtest:")
        return self


class OrderIntent(VersionedContract):
    intent_id: NonEmptyString
    account_id: NonEmptyString
    run_id: NonEmptyString
    instrument: Instrument
    side: Literal["buy", "sell"]
    quantity: Positive
    decision_at: UtcTimestamp
    eligible_at: UtcTimestamp
    reference_price: Positive
    fx_rate: Positive = Decimal(1)
    estimated_fee: NonNegative = Decimal(0)
    strategy_id: NonEmptyString
    order_type: Literal["market", "limit", "stop"] = "market"
    limit_price: Positive | None = None
    stop_price: Positive | None = None
    tif: Literal["GTC", "DAY", "IOC", "FOK"] = "GTC"
    expires_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def valid_order(self):
        if self.eligible_at <= self.decision_at:
            raise ValueError("execution eligibility must be strictly after decision availability")
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit order needs limit_price")
        if self.order_type == "stop" and self.stop_price is None:
            raise ValueError("stop order needs stop_price")
        if self.tif == "DAY" and self.expires_at is None:
            raise ValueError("DAY order needs explicit calendar session expiry")
        if self.expires_at is not None and self.expires_at < self.eligible_at:
            raise ValueError("expiry precedes eligibility")
        return self


class Fill(VersionedContract):
    fill_id: NonEmptyString
    intent_id: NonEmptyString
    account_id: NonEmptyString
    quantity: Positive
    price: Positive
    fee: NonNegative
    fx_rate: Positive = Decimal(1)
    executed_at: UtcTimestamp
    event_id: NonEmptyString


class PriceMark(VersionedContract):
    instrument: Instrument
    price: Positive
    fx_rate: Positive = Decimal(1)
    as_of: UtcTimestamp


class ExecutionBar(VersionedContract):
    event_id: NonEmptyString
    instrument: Instrument
    opened_at: UtcTimestamp
    closed_at: UtcTimestamp
    open: Positive
    high: Positive
    low: Positive
    close: Positive
    volume: NonNegative
    # Known opening liquidity, e.g. opening auction print; NOT full future bar volume.
    opening_volume: NonNegative
    fx_rate: Positive = Decimal(1)
    calendar: NonEmptyString
    calendar_version: NonEmptyString
    session: date
    session_open: UtcTimestamp
    session_close: UtcTimestamp
    tradable: bool = True
    price_convention: Literal["raw"] = "raw"

    @model_validator(mode="after")
    def valid_bar(self):
        if not self.session_open <= self.opened_at < self.closed_at <= self.session_close:
            raise ValueError("bar must be contained within the declared calendar session")
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("invalid OHLC bounds")
        if self.opening_volume > self.volume:
            raise ValueError("opening volume exceeds bar volume")
        if self.instrument.asset_class == "crypto" and self.calendar != "24/7":
            raise ValueError("spot crypto requires the 24/7 calendar")
        if self.instrument.asset_class != "crypto" and self.calendar == "24/7":
            raise ValueError("equities require an explicit exchange session calendar")
        return self


class CorporateAction(VersionedContract):
    action_id: NonEmptyString
    instrument: Instrument
    effective_at: UtcTimestamp
    kind: Literal["split", "dividend", "delist"]
    # split ratio, base cash per share for dividend/delist (zero delist is allowed).
    value: NonNegative
    source_id: NonEmptyString

    @model_validator(mode="after")
    def split_positive(self):
        if self.kind == "split" and self.value <= 0:
            raise ValueError("split ratio must be positive")
        return self


class Holding(VersionedContract):
    instrument: Instrument
    quantity: NonNegative
    cost_basis: NonNegative
    market_value: NonNegative
    price: Positive
    fx_rate: Positive
    marked_at: UtcTimestamp


class PortfolioSnapshot(VersionedContract):
    account_id: NonEmptyString
    as_of: UtcTimestamp
    cash: NonNegative
    reserved_cash: NonNegative
    nav: NonNegative
    high_water_mark: NonNegative
    realized_pnl: Decimal
    fees: NonNegative
    turnover_today: NonNegative
    positions: tuple[Holding, ...] = ()


class InstrumentRisk(VersionedContract):
    instrument: Instrument
    sector: NonEmptyString
    adv_base: Positive
    as_of: UtcTimestamp
    lot_size: Positive = Decimal("0.00000001")
    min_notional: NonNegative = Decimal(0)


class RiskPolicy(VersionedContract):
    equity_weight: Fraction = Decimal("0.10")
    crypto_weight: Fraction = Decimal("0.05")
    sector_weight: Fraction = Decimal("0.30")
    crypto_total: Fraction = Decimal("0.20")
    min_cash: Fraction = Decimal("0.05")
    min_adv: NonNegative = Decimal("1000000")
    max_adv_participation: Fraction = Decimal("0.01")
    max_drawdown: Fraction = Decimal("0.15")
    max_daily_turnover: Fraction = Decimal("0.25")
    max_price_age_seconds: int = Field(default=86400 * 4, ge=0)
    max_liquidity_age_seconds: int = Field(default=86400 * 4, ge=0)


class FillPolicy(VersionedContract):
    equity_cost_bps: NonNegative = Decimal(10)
    crypto_cost_bps: NonNegative = Decimal(20)
    slippage_bps: NonNegative = Decimal(0)
    commission_per_unit: NonNegative = Decimal(0)
    minimum_commission: NonNegative = Decimal(0)
    opening_participation: Fraction = Decimal("0.01")
    convention: Literal["next_observed_open_only"] = "next_observed_open_only"

    @model_validator(mode="after")
    def bounded_slippage(self):
        if self.slippage_bps >= 10000:
            raise ValueError("slippage must be below 10000 bps")
        return self

    def fee(self, instrument: Instrument, quantity: Decimal, price: Decimal, fx: Decimal):
        bps = self.crypto_cost_bps if instrument.asset_class == "crypto" else self.equity_cost_bps
        return quantity * price * fx * bps / 10000 + max(
            quantity * self.commission_per_unit, self.minimum_commission
        )


class RankedSignal(VersionedContract):
    signal_id: NonEmptyString
    instrument: Instrument
    decision_at: UtcTimestamp
    available_at: UtcTimestamp
    score: float
    model_version: NonEmptyString
    universe_version: NonEmptyString
    data_snapshot_id: NonEmptyString
    strategy_id: NonEmptyString

    @model_validator(mode="after")
    def available(self):
        if self.available_at > self.decision_at:
            raise ValueError("signal uses information unavailable at decision time")
        return self


class TargetWeight(VersionedContract):
    instrument: Instrument
    weight: Fraction


class PortfolioTarget(VersionedContract):
    universe_version: NonEmptyString
    target_id: NonEmptyString
    decision_at: UtcTimestamp
    strategy_id: NonEmptyString
    artifact_version: NonEmptyString
    data_snapshot_id: NonEmptyString
    weights: tuple[TargetWeight, ...]

    @model_validator(mode="after")
    def coherent(self):
        if len({w.instrument.instrument_id for w in self.weights}) != len(self.weights):
            raise ValueError("duplicate target instrument")
        if sum((w.weight for w in self.weights), Decimal(0)) > 1:
            raise ValueError("target is leveraged")
        return self


class UniverseEligibility(VersionedContract):
    """Documented eligible members, effective and known by a decision cutoff."""

    universe_version: NonEmptyString
    data_snapshot_id: NonEmptyString
    effective_at: UtcTimestamp
    available_at: UtcTimestamp
    instrument_ids: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def unique_members(self):
        if len(set(self.instrument_ids)) != len(self.instrument_ids):
            raise ValueError("duplicate universe member")
        return self
