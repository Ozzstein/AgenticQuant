"""Versioned long-only accounting contracts shared by paper and historical replay.

Money/quantities are Decimal; fees and balances use the account base currency.
Foreign instruments require an explicit quote-to-base FX rate on every mark/fill.
Calendar sessions are supplied by a versioned data artifact, never inferred from
weekdays. The simulator executes only at observed session/bar opens.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from typing import Annotated, Literal

from pydantic import Field, model_serializer, model_validator

from src.domain.common import Instrument, NonEmptyString, Sha256, UtcTimestamp, VersionedContract

Positive = Annotated[Decimal, Field(gt=0)]
NonNegative = Annotated[Decimal, Field(ge=0)]
Fraction = Annotated[Decimal, Field(ge=0, le=1)]


def exact_decimal_sum(values):
    values = list(values)
    if not values:
        return Decimal(0)
    precision = (
        max(v.adjusted() for v in values)
        - min(v.as_tuple().exponent for v in values)
        + len(str(len(values)))
        + 3
    )
    with localcontext() as ctx:
        ctx.prec = max(28, precision)
        return sum(values, Decimal(0))


def exact_decimal_product(left: Decimal, right: Decimal):
    with localcontext() as ctx:
        ctx.prec = max(28, len(left.as_tuple().digits) + len(right.as_tuple().digits) + 2)
        return left * right


class _FrozenDict(dict):
    """JSON-compatible immutable nested contract mapping."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("contract mappings are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class AccountSpec(VersionedContract):
    account_id: NonEmptyString
    mode: Literal["paper", "backtest", "live"]
    base_currency: NonEmptyString
    initial_cash: Positive
    opened_at: UtcTimestamp

    @model_validator(mode="after")
    def namespace(self):
        if not self.account_id.startswith(self.mode + ":"):
            raise ValueError("account_id must be namespaced for its paper/backtest/live mode")
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


class AccountFee(VersionedContract):
    """Actual separately posted cash fee; no rate estimate or inferred fill fee.

    Amount is account base currency. The provider's effective date is retained
    separately from first observation/booking, since some feeds omit exact
    posting instants. Expense belongs to the observed accounting period.
    """

    fee_id: NonEmptyString
    account_id: NonEmptyString
    amount: Positive
    currency: NonEmptyString
    effective_date: date
    observed_at: UtcTimestamp
    source_hash: Sha256
    intent_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def causal(self):
        if self.effective_date > self.observed_at.date():
            raise ValueError("future fee effective date")
        return self


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
    # Explicit identity of the observed opening print, if different from this
    # completed bar. Its opening price/size/time must match that event exactly.
    opening_event_id: NonEmptyString | None = None

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

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        result = handler(self)
        if self.opening_event_id is None:
            result.pop("opening_event_id", None)
        return result


class CorporateAction(VersionedContract):
    """Legacy immediate-effective-cash-1 action, not provider ex/pay terms.

    Provider cash dividends must use CashDividend and an actual payment event.
    This closed schema rejects the additional entitlement/payment fields.
    """

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


class CashDividend(VersionedContract):
    """Immutable ordinary cash-dividend terms with explicit ownership evidence."""

    action_id: NonEmptyString
    revision_id: NonEmptyString
    supersedes_hash: Sha256 | None = None
    instrument: Instrument
    ex_at: UtcTimestamp
    entitlement_at: UtcTimestamp
    record_date: date | None
    payable_date: date | None
    published_at: UtcTimestamp | None
    available_at: UtcTimestamp
    cash_per_unit: Positive
    currency: NonEmptyString
    convention: Literal["pre-ex-beneficial-ownership-1"]
    entitlement_source_id: NonEmptyString
    entitlement_source_hash: Sha256
    source_id: NonEmptyString
    source_hash: Sha256

    @model_validator(mode="after")
    def causal_terms(self):
        if self.instrument.asset_class == "crypto":
            raise ValueError("cash-dividend convention does not establish protocol entitlements")
        if self.ex_at != self.entitlement_at:
            raise ValueError("supported ordinary dividend requires explicit pre-ex ownership")
        if self.published_at is not None and self.published_at > self.available_at:
            raise ValueError("availability cannot precede known publication")
        return self


class DividendEntitlement(VersionedContract):
    account_id: NonEmptyString
    action_id: NonEmptyString
    terms_hash: Sha256
    instrument: Instrument
    at: UtcTimestamp
    quantity: NonNegative
    gross_amount: NonNegative
    currency: NonEmptyString
    ownership_hash: Sha256
    # Only sleeve projections set this; the account entitlement is authoritative.
    parent_entitlement_hash: Sha256 | None = None

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        result = handler(self)
        if self.parent_entitlement_hash is None:
            result.pop("parent_entitlement_hash", None)
        return result


class DividendReceivable(VersionedContract):
    entitlement: DividendEntitlement
    outstanding: NonNegative

    @model_validator(mode="after")
    def no_invented_income(self):
        if self.outstanding > self.entitlement.gross_amount:
            raise ValueError("receivable exceeds original entitlement")
        return self


class DividendPayment(VersionedContract):
    payment_id: NonEmptyString
    account_id: NonEmptyString
    action_id: NonEmptyString
    terms_hash: Sha256
    entitlement_hash: Sha256
    origin: Literal["posted-account", "replay-distribution", "paper-distribution"]
    sizing_policy_hash: Sha256 | None = None
    effective_date: date
    observed_at: UtcTimestamp
    available_at: UtcTimestamp
    currency: NonEmptyString
    gross_amount: NonNegative
    net_amount: NonNegative
    fee: NonNegative
    withholding_tax: NonNegative
    fee_source_hash: Sha256 | None
    tax_source_hash: Sha256 | None
    source_id: NonEmptyString
    source_hash: Sha256

    @model_validator(mode="after")
    def reconciled_payment(self):
        if (self.origin == "paper-distribution") != (self.sizing_policy_hash is not None):
            raise ValueError("only local paper sizing requires an explicit policy hash")
        if self.gross_amount != exact_decimal_sum(
            [self.net_amount, self.fee, self.withholding_tax]
        ):
            raise ValueError("actual gross payment must equal net cash plus explicit fee/tax")
        if self.observed_at > self.available_at or self.effective_date > self.observed_at.date():
            raise ValueError("payment requires causal occurrence/receipt/availability")
        if (self.fee and self.fee_source_hash is None) or (
            self.withholding_tax and self.tax_source_hash is None
        ):
            raise ValueError("actual deductions require explicit source bindings")
        return self

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        result = handler(self)
        if self.sizing_policy_hash is None:
            result.pop("sizing_policy_hash", None)
        return result


class PaperDividendPolicy(VersionedContract):
    """Explicit local simulation policy; never proof of a broker account credit."""

    policy_id: NonEmptyString
    account_id: NonEmptyString
    distribution_hash: Sha256
    convention: Literal["source-confirmed-paper-distribution-1"]
    fee_tax_convention: Literal["source-confirmed-per-unit-deductions-1"]
    available_at: UtcTimestamp
    source_id: NonEmptyString
    source_hash: Sha256


class DividendDistribution(VersionedContract):
    """Actual paid-per-unit notice, sized only under an explicit simulation policy."""

    distribution_id: NonEmptyString
    action_id: NonEmptyString
    terms_hash: Sha256
    effective_date: date
    observed_at: UtcTimestamp
    available_at: UtcTimestamp
    currency: NonEmptyString
    gross_per_unit: Positive
    net_per_unit: NonNegative
    fee_per_unit: NonNegative
    withholding_per_unit: NonNegative
    fee_source_hash: Sha256 | None
    tax_source_hash: Sha256 | None
    source_id: NonEmptyString
    source_hash: Sha256

    def payment(
        self, entitlement: DividendEntitlement, *, paper_policy: PaperDividendPolicy | None = None
    ) -> DividendPayment:
        if paper_policy is not None and (
            paper_policy.account_id != entitlement.account_id
            or paper_policy.distribution_hash != self.content_hash()
            or paper_policy.available_at > self.available_at
        ):
            raise ValueError(
                "paper dividend policy differs from account, distribution or availability"
            )
        return DividendPayment(
            payment_id="distribution:" + self.distribution_id,
            account_id=entitlement.account_id,
            action_id=self.action_id,
            terms_hash=self.terms_hash,
            entitlement_hash=entitlement.content_hash(),
            origin="replay-distribution" if paper_policy is None else "paper-distribution",
            sizing_policy_hash=paper_policy.content_hash() if paper_policy else None,
            effective_date=self.effective_date,
            observed_at=self.observed_at,
            available_at=self.available_at,
            currency=self.currency,
            gross_amount=exact_decimal_product(entitlement.quantity, self.gross_per_unit),
            net_amount=exact_decimal_product(entitlement.quantity, self.net_per_unit),
            fee=exact_decimal_product(entitlement.quantity, self.fee_per_unit),
            withholding_tax=exact_decimal_product(entitlement.quantity, self.withholding_per_unit),
            fee_source_hash=self.fee_source_hash,
            tax_source_hash=self.tax_source_hash,
            source_id=self.source_id,
            source_hash=self.source_hash,
        )

    @model_validator(mode="after")
    def exact_distribution(self):
        if self.gross_per_unit != exact_decimal_sum(
            [self.net_per_unit, self.fee_per_unit, self.withholding_per_unit]
        ):
            raise ValueError("distribution rates require actual reconciled gross/net/deductions")
        if self.observed_at > self.available_at or self.effective_date > self.observed_at.date():
            raise ValueError("distribution notice has invalid availability")
        if (self.fee_per_unit and self.fee_source_hash is None) or (
            self.withholding_per_unit and self.tax_source_hash is None
        ):
            raise ValueError("distribution deductions require explicit source bindings")
        return self


class ActionCoverage(VersionedContract):
    """Documented source scope; a process-date query is not economic coverage."""

    coverage_id: NonEmptyString
    instrument_ids: tuple[NonEmptyString, ...]
    venue: NonEmptyString
    scope: Literal[
        "effective-event-stream",
        "provider-process-date-query",
        "custodial-credit-ledger",
        "protocol-state-events",
    ]
    started_at: UtcTimestamp
    ended_at: UtcTimestamp
    available_at: UtcTimestamp
    event_types: tuple[NonEmptyString, ...]
    completeness: Literal["complete", "partial", "unknown"]
    gaps: tuple[NonEmptyString, ...]
    source_id: NonEmptyString
    source_hash: Sha256

    @model_validator(mode="after")
    def declared_coverage(self):
        if self.started_at >= self.ended_at or not self.instrument_ids or not self.event_types:
            raise ValueError(
                "explicit nonempty action coverage window, instruments and types required"
            )
        if len(set(self.instrument_ids)) != len(self.instrument_ids) or len(
            set(self.event_types)
        ) != len(self.event_types):
            raise ValueError("duplicate action coverage identities")
        if self.completeness == "complete" and self.gaps:
            raise ValueError("complete action coverage cannot retain gaps")
        return self

    def require_complete(self, *, instrument_ids, started_at, ended_at, as_of, scope, event_types):
        if (
            self.completeness != "complete"
            or self.gaps
            or self.available_at > as_of
            or self.scope != scope
            or scope == "provider-process-date-query"
            or self.started_at > started_at
            or self.ended_at < ended_at
            or not set(instrument_ids) <= set(self.instrument_ids)
            or not set(event_types) <= set(self.event_types)
        ):
            raise ValueError("source does not establish required economic action coverage")


class Holding(VersionedContract):
    instrument: Instrument
    quantity: NonNegative
    cost_basis: NonNegative
    market_value: NonNegative
    price: Positive
    fx_rate: Positive
    marked_at: UtcTimestamp
    # Actual first fill since the last flat position, never a target timestamp.
    # None denotes unknown age in an older externally supplied snapshot.
    opened_at: UtcTimestamp | None = None

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        result = handler(self)
        if self.opened_at is None:
            result.pop("opened_at", None)
        return result


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
    sleeves: dict[str, "SleeveSnapshot"] = Field(default_factory=dict)
    receivables: tuple[DividendReceivable, ...] = ()
    withholding_taxes: NonNegative = Decimal(0)

    @property
    def receivable_balance(self):
        return exact_decimal_sum(r.outstanding for r in self.receivables)

    @model_validator(mode="after")
    def immutable_sleeves(self):
        object.__setattr__(self, "sleeves", _FrozenDict(self.sleeves))
        return self

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        result = handler(self)
        if not self.sleeves:
            result.pop("sleeves", None)
        if not self.receivables:
            result.pop("receivables", None)
        if not self.withholding_taxes:
            result.pop("withholding_taxes", None)
        return result


class SleeveSnapshot(VersionedContract):
    """Reconciled reporting projection; never an independent execution account."""

    portfolio: PortfolioSnapshot
    net_capital_flow: Decimal
    last_rebalanced_at: UtcTimestamp | None
    source_hash: str
    source_event_ids: tuple[str, ...]
    # Exact remainder of finite Decimal portfolio valuation, assigned once.
    valuation_rounding: Decimal = Decimal(0)


PortfolioSnapshot.model_rebuild()


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
    preserve_quantities: dict[str, NonNegative] = Field(default_factory=dict)
    # Each contribution is a fraction of total book NAV, after sleeve blending.
    sleeve_contributions: dict[str, dict[str, Fraction]] = Field(default_factory=dict)
    sleeve_sources: dict[str, str] = Field(default_factory=dict)
    # Preserve actions carry exact actual quantity, avoiding float weight/lot
    # round trips. These may only preserve existing ownership, never create it.
    sleeve_preserve_quantities: dict[str, dict[str, NonNegative]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent(self):
        if len({w.instrument.instrument_id for w in self.weights}) != len(self.weights):
            raise ValueError("duplicate target instrument")
        if sum((w.weight for w in self.weights), Decimal(0)) > 1:
            raise ValueError("target is leveraged")
        if set(self.preserve_quantities) - {w.instrument.instrument_id for w in self.weights}:
            raise ValueError("preserved quantities must be represented in target weights")
        if self.preserve_quantities and self.sleeve_contributions:
            raise ValueError("blended targets require sleeve-specific preservation")
        if self.sleeve_contributions:
            if set(self.sleeve_contributions) != set(self.sleeve_sources):
                raise ValueError("each sleeve contribution needs its immutable source hash")
            for source in self.sleeve_sources.values():
                if len(source) != 64 or any(c not in "0123456789abcdef" for c in source):
                    raise ValueError("sleeve source must be SHA256")
            weights = {w.instrument.instrument_id: w.weight for w in self.weights}
            keys = set(weights) | {k for c in self.sleeve_contributions.values() for k in c}
            if any(
                sum((c.get(k, Decimal(0)) for c in self.sleeve_contributions.values()), Decimal(0))
                != weights.get(k, Decimal(0))
                for k in keys
            ):
                raise ValueError(
                    "sleeve contributions must sum exactly to aggregate target weights"
                )
        elif self.sleeve_sources:
            raise ValueError("sleeve sources require actual contributions")
        if set(self.sleeve_preserve_quantities) - self.sleeve_contributions.keys() or any(
            set(values) - self.sleeve_contributions[name].keys()
            for name, values in self.sleeve_preserve_quantities.items()
        ):
            raise ValueError("preserved quantities require their sleeve contributions")
        object.__setattr__(
            self,
            "sleeve_contributions",
            _FrozenDict({k: _FrozenDict(v) for k, v in self.sleeve_contributions.items()}),
        )
        object.__setattr__(self, "preserve_quantities", _FrozenDict(self.preserve_quantities))
        object.__setattr__(self, "sleeve_sources", _FrozenDict(self.sleeve_sources))
        object.__setattr__(
            self,
            "sleeve_preserve_quantities",
            _FrozenDict({k: _FrozenDict(v) for k, v in self.sleeve_preserve_quantities.items()}),
        )
        return self

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        result = handler(self)
        if not self.preserve_quantities:
            result.pop("preserve_quantities", None)
        if not self.sleeve_contributions:
            result.pop("sleeve_contributions", None)
            result.pop("sleeve_sources", None)
        if not self.sleeve_preserve_quantities:
            result.pop("sleeve_preserve_quantities", None)
        return result


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


class OpeningEvent(VersionedContract):
    """Actual raw opening print; contains no unobserved completed-bar fields."""

    event_id: NonEmptyString
    instrument: Instrument
    opened_at: UtcTimestamp
    price: Positive
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
    def causal_open(self):
        if not self.session_open <= self.opened_at < self.session_close:
            raise ValueError("opening print must be within its declared calendar session")
        if (self.instrument.asset_class == "crypto") != (self.calendar == "24/7"):
            raise ValueError("opening print requires its actual equity/crypto calendar")
        return self

    @classmethod
    def from_bar(cls, bar: ExecutionBar):
        """Project only already-declared opening economics of a historical bar."""
        bar = ExecutionBar.model_validate(bar.model_dump())
        return cls(
            event_id=bar.opening_event_id or bar.event_id,
            instrument=bar.instrument,
            opened_at=bar.opened_at,
            price=bar.open,
            opening_volume=bar.opening_volume,
            fx_rate=bar.fx_rate,
            calendar=bar.calendar,
            calendar_version=bar.calendar_version,
            session=bar.session,
            session_open=bar.session_open,
            session_close=bar.session_close,
            tradable=bar.tradable,
            price_convention=bar.price_convention,
        )


class OpeningSourceIdentity(VersionedContract):
    """Native print identity across capture/feed versions, authenticated externally."""

    provider: Literal["alpaca", "kraken", "binance"]
    instrument: Instrument
    trade_id: Annotated[str, Field(strict=True, min_length=1)]
    wire_timestamp: Annotated[str, Field(strict=True, min_length=1)]

    @model_validator(mode="after")
    def exact_native_identity(self):
        value = self.wire_timestamp
        digits = value[1:] if value.startswith("-") else value
        if not digits.isascii() or not digits.isdecimal() or str(int(value)) != value:
            raise ValueError("wire timestamp requires canonical UTC unix-nanosecond integer text")
        native = self.trade_id
        if self.provider == "alpaca":
            parts = native.split(":")
            if len(parts) != 2 or not parts[0] or not parts[0].isascii() or not parts[0].isalnum():
                raise ValueError("Alpaca trade ID requires its exact exchange:id identity")
            native = parts[1]
        if not native.isascii() or not native.isdecimal() or str(int(native)) != native:
            raise ValueError("native trade ID requires canonical integer text")
        self.projected_opened_at()  # Reject timestamps outside the actual datetime range.
        return self

    def projected_opened_at(self) -> datetime:
        """Exact declared ceil-microsecond-1 projection, never an identity inference."""
        try:
            return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
                microseconds=(int(self.wire_timestamp) + 999) // 1000
            )
        except (ValueError, OverflowError) as exc:
            raise ValueError("native print timestamp is outside the supported clock range") from exc
