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
from typing import Literal, Protocol

from pydantic import Field, model_validator

from src.domain.common import Instrument, NonEmptyString, UtcTimestamp, VersionedContract, _utc
from src.domain.portfolio import Fill, FillPolicy, InstrumentRisk, OrderIntent, Positive, RiskPolicy
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.simulator import PaperExecution

D = Decimal


class VenueRules(VersionedContract):
    venue: NonEmptyString
    instrument: Instrument
    symbol: NonEmptyString
    available_at: UtcTimestamp
    quantity_step: Positive
    price_tick: Positive
    minimum_quantity: Decimal = Field(default=0, ge=0)
    minimum_notional: Decimal = Field(default=0, ge=0)
    maximum_quantity: Positive | None = None
    order_types: tuple[Literal["market", "limit", "stop"], ...] = ("market", "limit")
    time_in_force: tuple[Literal["GTC", "DAY", "IOC", "FOK"], ...] = ("GTC",)
    market_type: Literal["equity", "spot"]
    active: bool

    def require_order(self, intent: OrderIntent, as_of: datetime):
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
            if price is not None and price % self.price_tick:
                raise LedgerError("venue tick increment violated")
        if intent.quantity * intent.reference_price < self.minimum_notional:
            raise LedgerError("venue minimum quote notional violated")
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
        spec = ledger.spec(account_id)
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
            },
            sort_keys=True,
        )
        with ledger.atomic():
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

    @staticmethod
    def client_id(account_id: str, intent_id: str) -> str:
        # Kraken accepts UUIDs, Binance up to 36 characters, Alpaca up to 48.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "quantagentlab:" + account_id + "\0" + intent_id))

    def prepare(self, intent: OrderIntent, metadata: dict[str, InstrumentRisk], *, as_of: datetime):
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
        if balances.venue != venue or balances.observed_at != as_of:
            raise LedgerError("venue funding evidence clock/identity mismatch")
        with self.ledger.atomic():
            existing = self.ledger.db.execute(
                "SELECT intent FROM venue_outbox WHERE account=? AND intent=?",
                (self.account_id, intent.intent_id),
            ).fetchone()
            if existing:
                self.ledger.submit(intent)  # immutable retry check
                return self.state(intent.intent_id)
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
            PaperExecution(self.ledger, self.account_id, self.fill_policy, self.risk_policy).submit(
                intent, metadata
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
                    return
            self.ledger.db.execute(
                "INSERT OR IGNORE INTO venue_reports(account,intent,evidence,payload) VALUES (?,?,?,?)",
                (self.account_id, intent_id, report.evidence_id, report.model_dump_json()),
            )
        try:
            with self.ledger.atomic():
                self._apply_report(intent_id, report)
                self.ledger.db.execute(
                    "UPDATE venue_reports SET applied=1 WHERE account=? AND intent=? AND evidence=?",
                    (self.account_id, intent_id, report.evidence_id),
                )
        except Exception as exc:
            with self.ledger.atomic():
                self._status(intent_id, "reconciliation_required", type(exc).__name__)
            raise

    def _apply_report(self, intent_id, report):
        state = self.state(intent_id)
        intent = self.ledger.order(self.account_id, intent_id)["intent"]
        if (
            report.cumulative_quantity < D(state["cumulative"])
            or report.cumulative_quantity > intent.quantity
        ):
            raise LedgerError("broker cumulative execution quantity regressed/exceeded intent")
        for fill in sorted(report.fills, key=lambda f: (f.executed_at, f.fill_id)):
            if fill.account_id != self.account_id or fill.intent_id != intent_id:
                raise LedgerError("broker execution account/intent mismatch")
            self.ledger.apply_fill(fill)
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
                report.observed_at,
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
        # Apply executions by actual chronology across venues/reports.
        for intent_id, report in sorted(
            reports,
            key=lambda pair: min(
                (f.executed_at for f in pair[1].fills), default=pair[1].observed_at
            ),
        ):
            self.ingest(intent_id, report)
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
        if any(o.observed_at != as_of for o in observations) or {
            o.venue for o in observations
        } != set(self.adapters):
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
            self.ledger.record_event(
                self.account_id, "paper-accepted:" + intent.intent_id, intent.content_hash()
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

    def advance_open(self, bar, metadata):
        return self.engine.open_bar(bar, metadata, eligible_intent_ids=self._accepted())

    def advance_close(self, bar):
        return self.engine.close_bar(bar)

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
