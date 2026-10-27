"""Typed venue reports and durable intent ownership across remote submissions.

Network delivery cannot be exactly once. An intent is claimed transactionally
before I/O; ambiguous outcomes remain reserved and require broker reconciliation.
The canonical ledger is the only owner of economic cash/holdings/fill state.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import Field, model_serializer, model_validator

from src.domain.common import (
    Instrument,
    NonEmptyString,
    Sha256,
    UtcTimestamp,
    VersionedContract,
    _utc,
)
from src.domain.portfolio import (
    ExecutionBar,
    Fill,
    FillPolicy,
    InstrumentRisk,
    OpeningEvent,
    OpeningSourceIdentity,
    OrderIntent,
    Positive,
    RiskPolicy,
    exact_decimal_product,
)
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.simulator import PaperExecution

if TYPE_CHECKING:
    from src.execution.opening_sources import OpeningSourceBinding

D = Decimal


class ObservedOpenReceipt(VersionedContract):
    policy: Literal["paper-observed-opening-print-1"] = "paper-observed-opening-print-1"
    account_id: NonEmptyString
    event_id: NonEmptyString
    event_hash: Sha256
    occurred_at: UtcTimestamp
    available_at: UtcTimestamp
    observed_at: UtcTimestamp
    as_of: UtcTimestamp
    evidence_id: NonEmptyString
    evidence_hash: Sha256
    admission_metadata_hashes: tuple[tuple[str, Sha256], ...]

    @model_validator(mode="after")
    def chronological(self):
        if not self.occurred_at <= self.observed_at <= self.available_at <= self.as_of:
            raise ValueError("opening print requires actual causal receipt clocks")
        if self.admission_metadata_hashes != tuple(sorted(set(self.admission_metadata_hashes))):
            raise ValueError("sorted unique admission metadata bindings required")
        return self


def _metadata_hash(metadata):
    import hashlib

    return hashlib.sha256(
        json.dumps(
            {key: value.model_dump(mode="json") for key, value in metadata.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class VenueNotionalReference(VersionedContract):
    """Actual native reference used by a venue's market-order notional filters."""

    price: Positive
    kind: Literal["venue-reference", "weighted-average", "last-trade"]
    average_minutes: int | None = Field(ge=0)
    observed_at: UtcTimestamp
    available_at: UtcTimestamp
    source_hash: Sha256
    # A native reference can override VWAP/last trade. Its absence must also be
    # established by source evidence before using the documented fallback.
    reference_price_absence_hash: Sha256 | None

    @model_validator(mode="after")
    def exact_basis(self):
        if self.observed_at > self.available_at:
            raise ValueError("native notional reference availability precedes observation")
        if self.kind == "venue-reference":
            if self.average_minutes is not None or self.reference_price_absence_hash is not None:
                raise ValueError("native reference cannot claim a fallback averaging basis")
        elif (
            self.reference_price_absence_hash is None
            or (self.kind == "last-trade" and self.average_minutes != 0)
            or (self.kind == "weighted-average" and not self.average_minutes)
        ):
            raise ValueError(
                "native fallback requires exact averaging basis and reference absence proof"
            )
        return self


class VenueRules(VersionedContract):
    venue: NonEmptyString
    instrument: Instrument
    symbol: NonEmptyString
    available_at: UtcTimestamp
    quantity_step: Positive
    price_tick: Positive
    minimum_quantity: Decimal = Field(default=D(0), ge=0)
    minimum_notional: Decimal = Field(default=D(0), ge=0)
    maximum_quantity: Positive | None = None
    maximum_notional: Positive | None = None
    minimum_notional_applies_to_market: bool = True
    maximum_notional_applies_to_market: bool = True
    market_notional_average_minutes: int | None = Field(default=None, ge=0)
    market_notional_reference: VenueNotionalReference | None = None
    market_reference_max_age_seconds: int = Field(default=30, ge=0)
    minimum_price: Positive | None = None
    maximum_price: Positive | None = None
    market_quantity_step: Positive | None = None
    market_minimum_quantity: Positive | None = None
    market_maximum_quantity: Positive | None = None
    order_types: tuple[Literal["market", "limit", "stop", "trailing_stop"], ...] = (
        "market",
        "limit",
    )
    time_in_force: tuple[Literal["GTC", "DAY", "IOC", "FOK"], ...] = ("GTC",)
    market_type: Literal["equity", "spot"]
    active: bool

    @model_validator(mode="after")
    def coherent_limits(self):
        for minimum, maximum in (
            (self.minimum_quantity, self.maximum_quantity),
            (self.minimum_notional, self.maximum_notional),
            (self.minimum_price, self.maximum_price),
            (self.market_minimum_quantity, self.market_maximum_quantity),
        ):
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError("venue minimum exceeds maximum")
        if (
            self.market_notional_reference is not None
            and self.market_notional_average_minutes is None
        ):
            raise ValueError(
                "native market notional reference requires the filter's declared window"
            )
        return self

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        result = handler(self)
        # Existing frozen admission hashes retain their original shape.
        for key, default in (
            ("maximum_notional", None),
            ("minimum_notional_applies_to_market", True),
            ("maximum_notional_applies_to_market", True),
            ("market_notional_average_minutes", None),
            ("market_notional_reference", None),
            ("market_reference_max_age_seconds", 30),
            ("minimum_price", None),
            ("maximum_price", None),
            ("market_quantity_step", None),
            ("market_minimum_quantity", None),
            ("market_maximum_quantity", None),
        ):
            if getattr(self, key) == default:
                result.pop(key, None)
        return result

    def require_order(self, intent: OrderIntent, as_of: datetime):
        as_of = _utc(as_of)
        if not self.active or self.instrument != intent.instrument or self.available_at > as_of:
            raise LedgerError("inactive, mismatched or future venue rules")
        if (as_of - self.available_at).total_seconds() > 86400:
            raise LedgerError("stale venue rules")
        if intent.order_type not in self.order_types or intent.tif not in self.time_in_force:
            raise LedgerError("venue does not support this order type/time in force")
        if (
            intent.quantity % self.quantity_step
            or intent.quantity < self.minimum_quantity
            or (self.maximum_quantity is not None and intent.quantity > self.maximum_quantity)
        ):
            raise LedgerError("venue quantity increment or range violated")
        for price in (intent.limit_price, intent.stop_price):
            if price is not None and (
                price % self.price_tick
                or (self.minimum_price is not None and price < self.minimum_price)
                or (self.maximum_price is not None and price > self.maximum_price)
            ):
                raise LedgerError("venue tick increment or price range violated")
        if intent.trail_price is not None and intent.trail_price % self.price_tick:
            raise LedgerError("venue trailing offset tick increment violated")
        market = intent.order_type == "market"
        if market and (
            (self.market_quantity_step is not None and intent.quantity % self.market_quantity_step)
            or (
                self.market_minimum_quantity is not None
                and intent.quantity < self.market_minimum_quantity
            )
            or (
                self.market_maximum_quantity is not None
                and intent.quantity > self.market_maximum_quantity
            )
        ):
            raise LedgerError("venue market quantity increment or range violated")
        check_min = not market or self.minimum_notional_applies_to_market
        check_max = not market or self.maximum_notional_applies_to_market
        price = (
            intent.limit_price
            if intent.order_type == "limit"
            else intent.stop_price
            if intent.order_type == "stop"
            else intent.reference_price
        )
        if (
            market
            and ((check_min and self.minimum_notional) or (check_max and self.maximum_notional))
            and self.market_notional_average_minutes is not None
        ):
            reference = self.market_notional_reference
            if (
                reference is None
                or reference.available_at > as_of
                or not 0
                <= (as_of - reference.observed_at).total_seconds()
                <= self.market_reference_max_age_seconds
            ):
                raise LedgerError("actual current native market notional reference required")
            if (
                reference.kind != "venue-reference"
                and reference.average_minutes != self.market_notional_average_minutes
            ):
                raise LedgerError("native market notional averaging window differs from filter")
            price = reference.price
        notional = exact_decimal_product(intent.quantity, price)
        if check_min and notional < self.minimum_notional:
            raise LedgerError("venue minimum quote notional violated")
        if check_max and self.maximum_notional is not None and notional > self.maximum_notional:
            raise LedgerError("venue maximum quote notional violated")
        if (self.market_type == "spot") != (intent.instrument.asset_class == "crypto"):
            raise LedgerError("spot/equity venue identity mismatch")


class VenueOrderReport(VersionedContract):
    venue: NonEmptyString
    client_order_id: NonEmptyString
    venue_order_id: NonEmptyString
    observed_at: UtcTimestamp
    status: Literal["new", "partial", "filled", "cancelled", "rejected", "expired", "unknown"]
    cumulative_quantity: Decimal = Field(ge=0)
    fills: tuple[Fill, ...] = ()
    # A response may acknowledge a fill before its executions/fees are available.
    executions_complete: bool = False
    evidence_id: NonEmptyString

    @model_validator(mode="after")
    def consistent(self):
        if len({f.fill_id for f in self.fills}) != len(self.fills):
            raise ValueError("duplicate executions in report")
        if any(f.executed_at > self.observed_at for f in self.fills):
            raise ValueError("future execution in venue report")
        if any(
            fee.available_at > self.observed_at
            for fill in self.fills
            for fee in fill.inventory_fees
        ):
            raise ValueError("future inventory fee source in venue report")
        if sum((f.quantity for f in self.fills), D(0)) > self.cumulative_quantity:
            raise ValueError("execution quantity exceeds broker cumulative quantity")
        return self


class VenueBalances(VersionedContract):
    venue: NonEmptyString
    observed_at: UtcTimestamp
    # Adapters provide actual free+locked balances, converted with explicit FX.
    cash_base: Decimal = Field(ge=0)
    available_cash_base: Decimal = Field(ge=0)
    quantities: dict[str, Decimal]
    open_client_order_ids: tuple[str, ...]
    evidence_id: NonEmptyString

    @model_validator(mode="after")
    def consistent(self):
        if self.available_cash_base > self.cash_base:
            raise ValueError("available cash exceeds total venue cash")
        if any(not quantity.is_finite() or quantity < 0 for quantity in self.quantities.values()):
            raise ValueError("spot/equity venue quantities must be finite and nonnegative")
        if len(set(self.open_client_order_ids)) != len(self.open_client_order_ids):
            raise ValueError("duplicate open client order IDs")
        return self


class VenueAdapter(Protocol):
    venue: str
    mode: Literal["paper", "live"]
    owner: Literal["direct", "lean"]

    def rules(self, instrument: Instrument, as_of: datetime) -> VenueRules: ...
    def submit(
        self, intent: OrderIntent, client_order_id: str, rules: VenueRules, as_of: datetime
    ) -> VenueOrderReport: ...
    def lookup(
        self, intent: OrderIntent, client_order_id: str, venue_order_id: str | None, as_of: datetime
    ) -> VenueOrderReport | None: ...
    def cancel(
        self, intent: OrderIntent, client_order_id: str, venue_order_id: str, as_of: datetime
    ) -> VenueOrderReport: ...
    def balances(self, as_of: datetime) -> VenueBalances: ...


class DefinitiveRejection(ValueError):
    """The venue authoritatively rejected this submission without accepting it."""


class SubmissionNotAttempted(DefinitiveRejection):
    """A verified local preflight denied the order before any submit HTTP call."""


class VenueExecution:
    """One exclusive account owner, aggregate risk, and durable submission outbox.

    Call prepare for typed intents, dispatch after eligibility, and reconcile_all
    on every start and before scheduling new submissions. A None lookup proves
    neither rejection nor absence: ambiguous intents remain blocked/reserved.
    """

    def __init__(
        self,
        ledger: PortfolioLedger,
        account_id: str,
        adapters: dict[str, VenueAdapter],
        routes: dict[str, str],
        *,
        owner: Literal["direct", "lean"],
        risk_policy: RiskPolicy | None = None,
        fill_policy: FillPolicy | None = None,
        allow_live: bool = False,
        max_balance_age_seconds: int = 30,
    ):
        self.ledger, self.account_id, self.adapters, self.routes, self.owner = (
            ledger,
            account_id,
            adapters,
            routes,
            owner,
        )
        self.risk_policy, self.fill_policy = (
            risk_policy or RiskPolicy(),
            fill_policy or FillPolicy(),
        )
        if max_balance_age_seconds < 0:
            raise LedgerError("nonnegative balance evidence age required")
        self.max_balance_age_seconds = max_balance_age_seconds
        spec = ledger.spec(account_id)
        if spec.mode == "live" and not allow_live:
            raise LedgerError("live venue execution requires explicit independent authorization")
        if spec.mode == "backtest" or not adapters or not set(routes.values()) <= set(adapters):
            raise LedgerError("operational account and explicit eligible venue routes required")
        if any(
            key != adapter.venue or adapter.mode != spec.mode or adapter.owner != owner
            for key, adapter in adapters.items()
        ):
            raise LedgerError("venue identity, execution mode or exclusive owner mismatch")
        ledger.db.executescript("""
            CREATE TABLE IF NOT EXISTS execution_books(account TEXT PRIMARY KEY REFERENCES accounts(id), owner TEXT NOT NULL, configuration TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS venue_outbox(account TEXT NOT NULL, intent TEXT NOT NULL, venue TEXT NOT NULL, client_id TEXT NOT NULL, venue_id TEXT, state TEXT NOT NULL, rules TEXT NOT NULL, cumulative TEXT NOT NULL DEFAULT '0', detail TEXT, PRIMARY KEY(account,intent), UNIQUE(venue,client_id), FOREIGN KEY(account,intent) REFERENCES orders(account,id));
            CREATE TABLE IF NOT EXISTS venue_reports(account TEXT NOT NULL, intent TEXT NOT NULL, evidence TEXT NOT NULL, payload TEXT NOT NULL, applied INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(account,intent,evidence));
        """)
        configuration = json.dumps(
            {
                "venues": sorted(adapters),
                "routes": routes,
                "risk": self.risk_policy.model_dump(mode="json"),
                "costs": self.fill_policy.model_dump(mode="json"),
                **(
                    {"max_balance_age_seconds": max_balance_age_seconds}
                    if max_balance_age_seconds != 30
                    else {}
                ),
            },
            sort_keys=True,
        )
        with ledger.atomic():
            if owner != "direct" or set(adapters) != {"paper"}:
                from src.execution.dividends import require_remote_income_account

                require_remote_income_account(ledger, account_id)
            row = ledger.db.execute(
                "SELECT owner,configuration FROM execution_books WHERE account=?", (account_id,)
            ).fetchone()
            if row and (row["owner"] != owner or row["configuration"] != configuration):
                raise LedgerError("account already has an exclusive execution owner/configuration")
            ledger.db.execute(
                "INSERT OR IGNORE INTO execution_books VALUES (?,?,?)",
                (account_id, owner, configuration),
            )

    def state(self, intent_id: str) -> dict:
        row = self.ledger.db.execute(
            "SELECT * FROM venue_outbox WHERE account=? AND intent=?", (self.account_id, intent_id)
        ).fetchone()
        if row is None:
            raise LedgerError("intent has no durable venue route")
        return dict(row)

    def _require_applied_receipts(self):
        if self.ledger.db.execute(
            "SELECT 1 FROM venue_reports WHERE applied=0 AND rowid IN (SELECT MAX(rowid) FROM venue_reports WHERE account=? GROUP BY intent) LIMIT 1",
            (self.account_id,),
        ).fetchone():
            raise LedgerError("recover durable unapplied venue receipts before new risk")

    def admission_metadata(self, intent_id: str) -> dict[str, InstrumentRisk]:
        row = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (self.account_id, "venue-admission-metadata:" + intent_id),
        ).fetchone()
        if row is None:
            raise LedgerError("intent lacks retained admission metadata")
        return {
            key: InstrumentRisk.model_validate(value) for key, value in json.loads(row[0]).items()
        }

    def prepare_target(self, target, metadata, *, sleeve_book=None):
        """Canonical target sizing/netting into this account's durable outbox.

        Superseded remote orders must be reconciled/cancelled by the caller;
        preparation never assumes their reserved cash has been released.
        """
        from src.domain.portfolio import OrderIntent, PortfolioTarget, PriceMark
        from src.execution.routing import target_to_intents

        target = PortfolioTarget.model_validate(target.model_dump())
        event_id = "venue-target:" + target.target_id
        with self.ledger.atomic():
            self._require_applied_receipts()
            previous = self.ledger.db.execute(
                "SELECT payload FROM events WHERE account=? AND id=?", (self.account_id, event_id)
            ).fetchone()
            if previous:
                payload = json.loads(previous[0])
                if payload["target"] != target.model_dump(mode="json"):
                    raise LedgerError("venue target identity changed")
                intents = [OrderIntent.model_validate(o) for o in payload["intents"]]
            else:
                if self.ledger.pending(self.account_id):
                    raise LedgerError("reconcile/cancel outstanding orders before target routing")
                if target.sleeve_contributions:
                    if (
                        sleeve_book is None
                        or sleeve_book.ledger is not self.ledger
                        or sleeve_book.account_id != self.account_id
                    ):
                        raise LedgerError(
                            "blended target requires its canonical account SleeveBook"
                        )
                    intents = sleeve_book.net_target(
                        target,
                        metadata,
                        self.fill_policy,
                        max_price_age_seconds=self.risk_policy.max_price_age_seconds,
                    )
                else:
                    if sleeve_book is not None:
                        raise LedgerError("sleeve book requires explicit target contributions")
                    snapshot = self.ledger.snapshot(
                        self.account_id,
                        target.decision_at,
                        self.risk_policy.max_price_age_seconds,
                        persist=False,
                    )
                    marks = {
                        m.instrument.instrument_id: m
                        for m in (
                            PriceMark.model_validate_json(row[0])
                            for row in self.ledger.db.execute(
                                "SELECT payload FROM marks WHERE account=?", (self.account_id,)
                            )
                        )
                    }
                    intents = target_to_intents(
                        target,
                        snapshot,
                        marks,
                        metadata,
                        self.fill_policy,
                        max_price_age_seconds=self.risk_policy.max_price_age_seconds,
                    )
                self.ledger.record_event(
                    self.account_id,
                    event_id,
                    json.dumps(
                        {
                            "target": target.model_dump(mode="json"),
                            "intents": [o.model_dump(mode="json") for o in intents],
                        },
                        sort_keys=True,
                    ),
                )
        # Remote rules/funding reads happen outside the plan transaction. Each
        # intent's own prepare still atomically rechecks aggregate pending risk.
        return [self.prepare(order, metadata, as_of=target.decision_at) for order in intents]

    @staticmethod
    def client_id(account_id: str, intent_id: str) -> str:
        # Kraken accepts UUIDs, Binance up to 36 characters, Alpaca up to 48.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "quantagentlab:" + account_id + "\0" + intent_id))

    def prepare(self, intent: OrderIntent, metadata: dict[str, InstrumentRisk], *, as_of: datetime):
        intent = OrderIntent.model_validate(intent.model_dump())
        as_of = _utc(as_of)
        if intent.account_id != self.account_id or intent.decision_at != as_of:
            raise LedgerError("intent must use this account and explicit current decision clock")
        venue = self.routes.get(intent.instrument.instrument_id)
        if venue is None:
            raise LedgerError("instrument has no explicit eligible venue route")
        existing = self.ledger.db.execute(
            "SELECT intent FROM venue_outbox WHERE account=? AND intent=?",
            (self.account_id, intent.intent_id),
        ).fetchone()
        if existing:
            self.ledger.submit(intent)
            return self.state(intent.intent_id)
        rules = self.adapters[venue].rules(intent.instrument, as_of)
        if rules.venue != venue:
            raise LedgerError("venue rule source mismatch")
        rules.require_order(intent, as_of)
        balances = self.adapters[venue].balances(as_of)
        if (
            balances.venue != venue
            or not 0
            <= (as_of - balances.observed_at).total_seconds()
            <= self.max_balance_age_seconds
        ):
            raise LedgerError("venue funding evidence clock/identity mismatch")
        with self.ledger.atomic():
            existing = self.ledger.db.execute(
                "SELECT intent FROM venue_outbox WHERE account=? AND intent=?",
                (self.account_id, intent.intent_id),
            ).fetchone()
            if existing:
                self.ledger.submit(intent)  # immutable retry check
                return self.state(intent.intent_id)
            self._require_applied_receipts()
            if self.ledger.db.execute(
                "SELECT 1 FROM venue_outbox WHERE account=? AND state IN ('submitting','ambiguous','cancel_pending','reconciliation_required')",
                (self.account_id,),
            ).fetchone():
                raise LedgerError(
                    "account has unresolved venue outcomes; reconcile before new risk"
                )
            if intent.side == "buy":
                unsubmitted = self.ledger.db.execute(
                    "SELECT intent FROM venue_outbox WHERE account=? AND venue=? AND state='prepared'",
                    (self.account_id, venue),
                ).fetchall()
                reserved = sum(
                    (self.ledger.order(self.account_id, row[0])["reserved"] for row in unsubmitted),
                    D(0),
                )
                if (
                    intent.quantity * intent.reference_price * intent.fx_rate + intent.estimated_fee
                    > balances.available_cash_base - reserved
                ):
                    raise LedgerError("insufficient available funding at the routed venue")
            else:
                reserved_quantity = sum(
                    (
                        self.ledger.order(self.account_id, row[0])["intent"].quantity
                        - self.ledger.order(self.account_id, row[0])["filled"]
                        for row in self.ledger.db.execute(
                            "SELECT intent FROM venue_outbox WHERE account=? AND venue=?",
                            (self.account_id, venue),
                        )
                        if self.ledger.order(self.account_id, row[0])["status"]
                        in ("pending", "partial")
                        and self.ledger.order(self.account_id, row[0])["intent"].side == "sell"
                        and self.ledger.order(self.account_id, row[0])["intent"].instrument
                        == intent.instrument
                    ),
                    D(0),
                )
                if (
                    intent.quantity
                    > balances.quantities.get(intent.instrument.instrument_id, D(0))
                    - reserved_quantity
                ):
                    raise LedgerError("insufficient unreserved holding at the routed venue")
            self.reconcile_balances(as_of=as_of)
            PaperExecution(self.ledger, self.account_id, self.fill_policy, self.risk_policy).submit(
                intent, metadata
            )
            self.ledger.record_event(
                self.account_id,
                "venue-admission-metadata:" + intent.intent_id,
                json.dumps(
                    {key: value.model_dump(mode="json") for key, value in metadata.items()},
                    sort_keys=True,
                ),
            )
            self.ledger.db.execute(
                "INSERT INTO venue_outbox(account,intent,venue,client_id,state,rules) VALUES (?,?,?,?,?,?)",
                (
                    self.account_id,
                    intent.intent_id,
                    venue,
                    self.client_id(self.account_id, intent.intent_id),
                    "prepared",
                    rules.model_dump_json(),
                ),
            )
            return self.state(intent.intent_id)

    def dispatch(self, intent_id: str, *, as_of: datetime):
        as_of = _utc(as_of)
        with self.ledger.atomic():
            self._require_applied_receipts()
            state = self.state(intent_id)
            intent = self.ledger.order(self.account_id, intent_id)["intent"]
            if state["state"] != "prepared":
                return state
            if self.ledger.db.execute(
                "SELECT 1 FROM venue_outbox WHERE account=? AND intent<>? AND state IN ('submitting','ambiguous','cancel_pending','reconciliation_required')",
                (self.account_id, intent_id),
            ).fetchone():
                raise LedgerError("reconcile unresolved outcomes before dispatching another intent")
            if as_of < intent.eligible_at:
                raise LedgerError("intent is not yet eligible for submission")
            if intent.expires_at is not None and as_of > intent.expires_at:
                self.ledger.cancel(self.account_id, intent_id, as_of, "expired before submission")
                self._status(intent_id, "expired")
                return self.state(intent_id)
            VenueRules.model_validate_json(state["rules"]).require_order(intent, as_of)
            self.reconcile_balances(as_of=as_of)
            self._status(intent_id, "submitting")
        # The durable submitting claim precedes I/O. No network call is retried here.
        try:
            report = self.adapters[state["venue"]].submit(
                intent, state["client_id"], VenueRules.model_validate_json(state["rules"]), as_of
            )
        except DefinitiveRejection as exc:
            with self.ledger.atomic():
                self.ledger.cancel(self.account_id, intent_id, as_of, str(exc), rejected=True)
                self._status(intent_id, "rejected", str(exc))
            return self.state(intent_id)
        except Exception as exc:
            with self.ledger.atomic():
                self._status(intent_id, "ambiguous", type(exc).__name__)
            return self.state(intent_id)
        self.ingest(intent_id, report)
        return self.state(intent_id)

    def _status(self, intent_id, state, detail=None):
        self.ledger.db.execute(
            "UPDATE venue_outbox SET state=?,detail=? WHERE account=? AND intent=?",
            (state, detail, self.account_id, intent_id),
        )

    def ingest(self, intent_id: str, report: VenueOrderReport):
        self.ingest_many([(intent_id, report)])

    def _persist_report(self, intent_id: str, report: VenueOrderReport) -> bool:
        # Persist the receipt before attempting economic application, so even an
        # invalid/out-of-order execution is reviewable and recoverable on restart.
        with self.ledger.atomic():
            state = self.state(intent_id)
            if (
                report.venue != state["venue"]
                or report.client_order_id != state["client_id"]
                or (state["venue_id"] and report.venue_order_id != state["venue_id"])
            ):
                raise LedgerError("broker acknowledgement identity mismatch")
            previous = self.ledger.db.execute(
                "SELECT payload,applied FROM venue_reports WHERE account=? AND intent=? AND evidence=?",
                (self.account_id, intent_id, report.evidence_id),
            ).fetchone()
            if previous:
                if VenueOrderReport.model_validate_json(previous["payload"]) != report:
                    raise LedgerError("broker evidence identity reused with different content")
                if previous["applied"]:
                    return False
            self.ledger.db.execute(
                "INSERT OR IGNORE INTO venue_reports(account,intent,evidence,payload) VALUES (?,?,?,?)",
                (self.account_id, intent_id, report.evidence_id, report.model_dump_json()),
            )
        return True

    def ingest_many(self, reports: list[tuple[str, VenueOrderReport]]):
        """Persist all receipts, then book globally ordered executions atomically.

        An incomplete activity history reserves the whole remaining order until
        the adapter supplies complete executions/fees through its cumulative qty.
        Acknowledgement averages never become fabricated individual executions.
        """
        pending = []
        for intent_id, report in reports:
            report = VenueOrderReport.model_validate(report.model_dump())
            if self._persist_report(intent_id, report):
                pending.append((intent_id, report))
        if not pending:
            return
        try:
            with self.ledger.atomic():
                executions = []
                for intent_id, report in pending:
                    self._validate_report(intent_id, report)
                    if report.executions_complete:
                        known = {
                            f.fill_id: f
                            for f in self.ledger.fills(self.account_id)
                            if f.intent_id == intent_id
                        }
                        for fill in report.fills:
                            if fill.fill_id in known and known[fill.fill_id] != fill:
                                raise LedgerError("venue changed an existing execution")
                            known[fill.fill_id] = fill
                        if (
                            sum((f.quantity for f in known.values()), D(0))
                            != report.cumulative_quantity
                        ):
                            raise LedgerError(
                                "complete venue report lacks exact execution coverage"
                            )
                        executions.extend((fill, report.observed_at) for fill in report.fills)
                # One receipt clock for the atomic batch; every fill retains its
                # actual execution timestamp in both the economic journal and payload.
                observed = max(report.observed_at for _, report in pending)
                for fill, _ in sorted(
                    executions, key=lambda pair: (pair[0].executed_at, pair[0].fill_id)
                ):
                    self.ledger.apply_fill(fill, observed_at=observed)
                for intent_id, report in sorted(pending, key=lambda pair: pair[1].observed_at):
                    self._apply_report(intent_id, report, booked_at=observed)
                    self.ledger.db.execute(
                        "UPDATE venue_reports SET applied=1 WHERE account=? AND intent=? AND evidence=?",
                        (self.account_id, intent_id, report.evidence_id),
                    )
        except Exception as exc:
            with self.ledger.atomic():
                for intent_id, _ in pending:
                    self._status(intent_id, "reconciliation_required", type(exc).__name__)
            raise

    def _validate_report(self, intent_id, report):
        state = self.state(intent_id)
        intent = self.ledger.order(self.account_id, intent_id)["intent"]
        if (
            report.cumulative_quantity < D(state["cumulative"])
            or report.cumulative_quantity > intent.quantity
        ):
            raise LedgerError("broker cumulative execution quantity regressed/exceeded intent")
        for fill in report.fills:
            if fill.account_id != self.account_id or fill.intent_id != intent_id:
                raise LedgerError("broker execution account/intent mismatch")

    def _apply_report(self, intent_id, report, *, booked_at):
        self._validate_report(intent_id, report)
        state = self.state(intent_id)
        intent = self.ledger.order(self.account_id, intent_id)["intent"]
        booked = self.ledger.order(self.account_id, intent_id)["filled"]
        complete = report.executions_complete and booked == report.cumulative_quantity
        status = report.status if complete else "reconciliation_required"
        if status == "filled" and booked != intent.quantity:
            raise LedgerError("venue claims filled without full intent quantity")
        if (
            state["state"] in ("filled", "cancelled", "expired", "rejected")
            and status != state["state"]
        ):
            raise LedgerError("terminal venue status cannot regress")
        if status in ("cancelled", "expired", "rejected"):
            self.ledger.cancel(
                self.account_id,
                intent_id,
                booked_at,
                status,
                rejected=status == "rejected",
            )
        self.ledger.db.execute(
            "UPDATE venue_outbox SET venue_id=?,cumulative=?,state=?,detail=NULL WHERE account=? AND intent=?",
            (
                report.venue_order_id,
                str(report.cumulative_quantity),
                status,
                self.account_id,
                intent_id,
            ),
        )

    def reconcile_all(self, *, as_of: datetime) -> list[dict]:
        as_of = _utc(as_of)
        # Crash recovery uses actual durable receipts before requesting new I/O.
        # A later successfully applied receipt supersedes an older failed one;
        # the old immutable content remains retained for review.
        pending = self.ledger.db.execute(
            "SELECT intent,payload,applied FROM venue_reports WHERE rowid IN (SELECT MAX(rowid) FROM venue_reports WHERE account=? GROUP BY intent)",
            (self.account_id,),
        ).fetchall()
        recovered = []
        for row in pending:
            report = VenueOrderReport.model_validate_json(row["payload"])
            if not row["applied"] and report.observed_at <= as_of:
                recovered.append((row["intent"], report))
        self.ingest_many(recovered)
        rows = self.ledger.db.execute(
            "SELECT intent FROM venue_outbox WHERE account=? AND state NOT IN ('filled','cancelled','expired','rejected','prepared') ORDER BY rowid",
            (self.account_id,),
        ).fetchall()
        reports = []
        for row in rows:
            state = self.state(row[0])
            intent = self.ledger.order(self.account_id, row[0])["intent"]
            try:
                report = self.adapters[state["venue"]].lookup(
                    intent, state["client_id"], state["venue_id"], as_of
                )
            except Exception as exc:
                with self.ledger.atomic():
                    self._status(row[0], "ambiguous", type(exc).__name__)
                continue
            if report is not None:
                reports.append((row[0], report))
            else:
                with self.ledger.atomic():
                    self._status(
                        row[0], "ambiguous", "lookup unresolved; never safe to blindly resubmit"
                    )
        # Each report can contain interleaved partial fills. Sort executions,
        # not reports, and defer terminal cancellations until all fills settle.
        self.ingest_many(reports)
        return [self.state(row[0]) for row in rows]

    def cancel(self, intent_id: str, *, as_of: datetime):
        as_of = _utc(as_of)
        with self.ledger.atomic():
            state = self.state(intent_id)
            intent = self.ledger.order(self.account_id, intent_id)["intent"]
            if state["state"] in ("filled", "cancelled", "rejected", "expired"):
                return state
            if state["state"] == "prepared":
                self.ledger.cancel(self.account_id, intent_id, as_of, "cancelled before dispatch")
                self._status(intent_id, "cancelled")
                return self.state(intent_id)
            if state["venue_id"] is None:
                raise LedgerError("reconcile ambiguous submission before cancelling")
            self._status(intent_id, "cancel_pending")
        try:
            report = self.adapters[state["venue"]].cancel(
                intent, state["client_id"], state["venue_id"], as_of
            )
        except Exception as exc:
            with self.ledger.atomic():
                self._status(intent_id, "cancel_pending", type(exc).__name__)
            return self.state(intent_id)
        self.ingest(intent_id, report)
        return self.state(intent_id)

    def reconcile_balances(self, *, as_of: datetime) -> dict:
        as_of = _utc(as_of)
        observations = [adapter.balances(as_of) for adapter in self.adapters.values()]
        if any(
            not 0 <= (as_of - o.observed_at).total_seconds() <= self.max_balance_age_seconds
            for o in observations
        ) or {o.venue for o in observations} != set(self.adapters):
            raise LedgerError("balance snapshot clock/venue coverage differs")
        cash = sum((o.cash_base for o in observations), D(0))
        quantities = {}
        known = {
            r[0]
            for r in self.ledger.db.execute(
                "SELECT client_id FROM venue_outbox WHERE account=?", (self.account_id,)
            )
        }
        for observation in observations:
            if set(observation.open_client_order_ids) - known:
                raise LedgerError("unowned broker orders exist in this exclusive account")
            for key, quantity in observation.quantities.items():
                if quantity < 0:
                    raise LedgerError(
                        "venue short position is incompatible with this long-only book"
                    )
                quantities[key] = quantities.get(key, D(0)) + quantity
        stored = {
            r["instrument"]: D(r["quantity"])
            for r in self.ledger.db.execute(
                "SELECT instrument,quantity FROM holdings WHERE account=?", (self.account_id,)
            )
        }
        if (
            cash != D(self.ledger._account(self.account_id)["cash"])
            or {k: v for k, v in quantities.items() if v} != stored
        ):
            raise LedgerError("venue cash/holdings do not reconcile to durable accounting")
        return {
            "reconciled": True,
            "cash": str(cash),
            "quantities": {k: str(v) for k, v in quantities.items()},
            "evidence": [o.evidence_id for o in observations],
        }


class PaperVenue:
    """Reachable local venue backed by the canonical observed-event simulator.

    submit acknowledges an already reserved intent. advance_open performs actual
    simulated executions against supplied raw opening events; it never fills on
    submission or invents a quote. All economic state stays in PortfolioLedger.
    """

    venue = "paper"
    mode = "paper"
    owner = "direct"

    def __init__(
        self,
        ledger: PortfolioLedger,
        account_id: str,
        rules: dict[str, VenueRules],
        *,
        fill_policy=None,
        risk_policy=None,
    ):
        if ledger.spec(account_id).mode != "paper":
            raise LedgerError("paper venue requires its own paper account")
        self.ledger, self.account_id, self._rules = ledger, account_id, rules
        self.engine = PaperExecution(ledger, account_id, fill_policy, risk_policy)

    def rules(self, instrument, as_of):
        return self._rules[instrument.instrument_id]

    def _report(self, intent, client_order_id, as_of):
        order = self.ledger.order(self.account_id, intent.intent_id)
        fills = tuple(
            f for f in self.ledger.fills(self.account_id) if f.intent_id == intent.intent_id
        )
        status = "new" if order["status"] == "pending" else order["status"]
        content = {
            "status": status,
            "fills": [f.content_hash() for f in fills],
            "as_of": as_of.isoformat(),
        }
        import hashlib

        evidence = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        return VenueOrderReport(
            venue=self.venue,
            client_order_id=client_order_id,
            venue_order_id="paper:" + intent.intent_id,
            observed_at=as_of,
            status=status,
            cumulative_quantity=order["filled"],
            fills=fills,
            executions_complete=True,
            evidence_id=evidence,
        )

    def submit(self, intent, client_order_id, rules, as_of):
        rules.require_order(intent, as_of)
        with self.ledger.atomic():
            self.ledger._clock(self.account_id, as_of)
            self.ledger.record_event(
                self.account_id, "paper-accepted:" + intent.intent_id, intent.content_hash()
            )
            self.ledger.record_event(
                self.account_id,
                "paper-accepted-time:" + intent.intent_id,
                json.dumps(
                    {"at": _utc(as_of).isoformat(), "intent_hash": intent.content_hash()},
                    sort_keys=True,
                ),
            )
        return self._report(intent, client_order_id, as_of)

    def lookup(self, intent, client_order_id, venue_order_id, as_of):
        if intent.intent_id not in self._accepted():
            return None
        return self._report(intent, client_order_id, as_of)

    def cancel(self, intent, client_order_id, venue_order_id, as_of):
        self.ledger.cancel(self.account_id, intent.intent_id, as_of, "paper cancel")
        return self._report(intent, client_order_id, as_of)

    def _accepted(self):
        return {
            row[0][len("paper-accepted:") :]
            for row in self.ledger.db.execute(
                "SELECT id FROM events WHERE account=? AND id LIKE 'paper-accepted:%'",
                (self.account_id,),
            )
        }

    def accepted_intent_ids_before(self, at) -> frozenset[str]:
        """Actual durable acceptance strictly before an observed market instant."""
        at = _utc(at)
        eligible = set()
        for row in self.ledger.db.execute(
            "SELECT id,payload FROM events WHERE account=? AND id LIKE 'paper-accepted-time:%'",
            (self.account_id,),
        ):
            key, receipt = row["id"][len("paper-accepted-time:") :], json.loads(row["payload"])
            order = self.ledger.order(self.account_id, key)["intent"]
            if receipt["intent_hash"] != order.content_hash():
                raise LedgerError("paper acceptance intent hash changed")
            if datetime.fromisoformat(receipt["at"]) < at:
                eligible.add(key)
        return frozenset(eligible)

    def advance_open(self, event: OpeningEvent | ExecutionBar, metadata):
        execute = (
            self.engine.open_bar if isinstance(event, ExecutionBar) else self.engine.open_event
        )
        return execute(
            event, metadata, eligible_intent_ids=self.accepted_intent_ids_before(event.opened_at)
        )

    def observed_open_receipt(self, event_id: str) -> ObservedOpenReceipt | None:
        """Read the committed settlement identity, including original intent IDs.

        After partial/full fills H reconstructs the exact original metadata map
        from these IDs via VenueExecution.admission_metadata, rather than from
        the current pending orders. Absence means no committed settlement.
        """
        row = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (self.account_id, "paper-observed-open:" + event_id),
        ).fetchone()
        if row is None:
            return None
        receipt = ObservedOpenReceipt.model_validate_json(row[0])
        if receipt.account_id != self.account_id or receipt.event_id != event_id:
            raise LedgerError("persisted observed opening receipt identity changed")
        return receipt

    def unbound_opening_receipts(self, instrument: Instrument) -> tuple[ObservedOpenReceipt, ...]:
        from src.execution.opening_sources import unbound_receipts

        return tuple(
            ObservedOpenReceipt.model_validate(r)
            for r in unbound_receipts(self.ledger, self.account_id, instrument)
        )

    def opening_source_binding(self, event_id: str) -> OpeningSourceBinding | None:
        from src.execution.opening_sources import event_binding

        return event_binding(self.ledger, self.account_id, event_id)

    def bind_opening_source(
        self,
        identity: OpeningSourceIdentity,
        *,
        event_id: str,
        evidence_id: str,
        evidence_hash: str,
    ) -> OpeningSourceBinding:
        """Bind externally verified native bytes to an existing exact receipt."""
        from src.execution.opening_sources import bind

        return bind(
            self.ledger,
            self.account_id,
            identity,
            event_id=event_id,
            evidence_id=evidence_id,
            evidence_hash=evidence_hash,
        )

    def settle_open(
        self,
        event: OpeningEvent | ExecutionBar,
        metadata_by_intent,
        *,
        available_at,
        observed_at,
        as_of,
        evidence_id,
        evidence_hash,
        source_identity: OpeningSourceIdentity | None = None,
    ):
        """Settle a documented opening print for previously accepted paper orders.

        Documentary receipt time can follow occurrence. This cannot change a
        historical research availability gate or rewrite published accounting.
        """
        completed_bar = None
        if isinstance(event, ExecutionBar):
            completed_bar = ExecutionBar.model_validate(event.model_dump())
            event = OpeningEvent.from_bar(completed_bar)
        elif isinstance(event, OpeningEvent):
            event = OpeningEvent.model_validate(event.model_dump())
        else:
            raise TypeError("settlement requires an OpeningEvent or ExecutionBar")
        if source_identity is not None:
            source_identity = OpeningSourceIdentity.model_validate(source_identity.model_dump())
        from src.execution.opening_sources import (
            consumed_event,
            require_identity,
            require_new_event_source,
            require_same_opening,
            source_binding,
        )

        if source_identity is not None:
            require_identity(event, source_identity)
        receipt = ObservedOpenReceipt(
            account_id=self.account_id,
            event_id=event.event_id,
            event_hash=event.content_hash(),
            occurred_at=event.opened_at,
            available_at=available_at,
            observed_at=observed_at,
            as_of=as_of,
            evidence_id=evidence_id,
            evidence_hash=evidence_hash,
            admission_metadata_hashes=tuple(
                sorted((key, _metadata_hash(value)) for key, value in metadata_by_intent.items())
            ),
        )
        if completed_bar is not None and completed_bar.closed_at > receipt.as_of:
            raise LedgerError("completed execution bar is not yet closed; use an OpeningEvent")
        with self.ledger.atomic():
            key = "paper-observed-open:" + event.event_id
            existing = self.ledger.db.execute(
                "SELECT payload FROM events WHERE account=? AND id=?", (self.account_id, key)
            ).fetchone()
            if existing:
                previous = ObservedOpenReceipt.model_validate_json(existing[0])
                # Old receipts bound the full completed bar. Retain those exact
                # bytes, but independently prove its opening-only identity before
                # accepting a retry through either public input shape.
                if previous.event_hash != receipt.event_hash:
                    legacy = self.ledger.db.execute(
                        "SELECT payload FROM events WHERE account=? AND id=?",
                        (self.account_id, "open:" + event.event_id),
                    ).fetchone()
                    legacy_bar = ExecutionBar.model_validate_json(legacy[0]) if legacy else None
                    if (
                        legacy_bar is None
                        or legacy_bar.content_hash() != previous.event_hash
                        or OpeningEvent.from_bar(legacy_bar) != event
                    ):
                        raise LedgerError("observed opening receipt changed")
                    receipt = receipt.model_copy(update={"event_hash": previous.event_hash})
                if (
                    previous.model_dump(exclude={"as_of"}) != receipt.model_dump(exclude={"as_of"})
                    or previous.as_of > receipt.as_of
                ):
                    raise LedgerError("observed opening receipt changed")
                consumed = self.ledger.db.execute(
                    "SELECT 1 FROM events WHERE account=? AND id IN (?, ?)",
                    (self.account_id, "opening-event:" + event.event_id, "open:" + event.event_id),
                ).fetchone()
                if consumed is None:
                    raise LedgerError("observed receipt has no committed opening consumption")
                if source_identity is not None:
                    self.bind_opening_source(
                        source_identity,
                        event_id=event.event_id,
                        evidence_id=evidence_id,
                        evidence_hash=evidence_hash,
                    )
                if completed_bar is not None:
                    return self.engine.open_bar(completed_bar, {})
                return self.engine.open_event(event, {})
            first = (
                source_binding(self.ledger, self.account_id, source_identity)
                if source_identity is not None
                else None
            )
            if first is not None:
                require_same_opening(
                    consumed_event(self.ledger, self.account_id, first.consumed_event_id), event
                )
            if (
                first is None
                and self.ledger.db.execute(
                    "SELECT 1 FROM snapshots WHERE account=? AND at>=?",
                    (self.account_id, event.opened_at.isoformat()),
                ).fetchone()
            ):
                raise LedgerError("observed opening print conflicts with published NAV")
            require_new_event_source(self.ledger, self.account_id, event, source_identity)
            accepted = self.accepted_intent_ids_before(event.opened_at)
            expected = {
                state["intent"].intent_id
                for state in self.ledger.pending(self.account_id)
                if state["intent"].intent_id in accepted
                and state["intent"].instrument == event.instrument
                and state["intent"].eligible_at <= event.opened_at
            }
            if set(metadata_by_intent) != expected:
                raise LedgerError(
                    "exact per-order admission metadata coverage required for opening print"
                )
            for intent_id, metadata in metadata_by_intent.items():
                row = self.ledger.db.execute(
                    "SELECT payload FROM events WHERE account=? AND id=?",
                    (self.account_id, "venue-admission-metadata:" + intent_id),
                ).fetchone()
                if row is None or json.loads(row[0]) != {
                    k: v.model_dump(mode="json") for k, v in metadata.items()
                }:
                    raise LedgerError("opening settlement metadata differs from durable admission")
                if any(value.as_of > event.opened_at for value in metadata.values()):
                    raise LedgerError("opening settlement cannot use future risk metadata")
            self.ledger.record_event(self.account_id, key, receipt.model_dump_json())
            if first is not None:
                # A source-authenticated alias only records its own receipt. The
                # original print already consumed capacity, including any unused
                # remainder. No mark or fill is replayed behind a later NAV.
                self.ledger.record_event(
                    self.account_id, "opening-event:" + event.event_id, event.model_dump_json()
                )
                self.bind_opening_source(
                    source_identity,
                    event_id=event.event_id,
                    evidence_id=evidence_id,
                    evidence_hash=evidence_hash,
                )
                if completed_bar is not None:
                    self.engine.open_bar(completed_bar, {})
                return []
            execute = self.engine.open_bar if completed_bar is not None else self.engine.open_event
            fills = execute(
                completed_bar if completed_bar is not None else event,
                {},
                eligible_intent_ids=expected,
                intent_metadata=metadata_by_intent,
                source_identity=source_identity,
            )
            if source_identity is not None:
                self.bind_opening_source(
                    source_identity,
                    event_id=event.event_id,
                    evidence_id=evidence_id,
                    evidence_hash=evidence_hash,
                )
            return fills

    def advance_close(self, bar):
        return self.engine.close_bar(bar)

    def expire_orders(self, *, as_of):
        """Expire due intents after independently authenticated market-close marks."""
        return self.engine.expire_orders(as_of=as_of)

    def balances(self, as_of):
        quantities = {
            row["instrument"]: D(row["quantity"])
            for row in self.ledger.db.execute(
                "SELECT instrument,quantity FROM holdings WHERE account=?", (self.account_id,)
            )
        }
        accepted = self._accepted()
        pending = [
            o for o in self.ledger.pending(self.account_id) if o["intent"].intent_id in accepted
        ]
        cash = D(self.ledger._account(self.account_id)["cash"])
        return VenueBalances(
            venue=self.venue,
            observed_at=as_of,
            cash_base=D(self.ledger._account(self.account_id)["cash"]),
            available_cash_base=cash - sum((o["reserved"] for o in pending), D(0)),
            quantities=quantities,
            open_client_order_ids=tuple(
                VenueExecution.client_id(self.account_id, o["intent"].intent_id) for o in pending
            ),
            evidence_id=f"paper-balance:{as_of.isoformat()}",
        )
