"""Durable LEAN execution owner over actual runtime tickets and OrderEvents."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal as D
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from src.domain.common import Instrument, NonEmptyString, VersionedContract, _utc
from src.domain.portfolio import (
    Fill,
    InventoryFee,
    PriceMark,
    exact_decimal_product,
    exact_decimal_sum,
)
from src.execution.ccxt_transport import canonical, decimal, digest
from src.execution.dividends import require_remote_income_account
from src.execution.ledger import LedgerError
from src.execution.spot_venue import FeeValuation
from src.execution.venue import SubmissionNotAttempted, VenueBalances, VenueOrderReport, VenueRules


def require_trailing_support(deployment, intent):
    """Only reviewed LEAN brokerage models can execute the canonical trail."""
    if intent.order_type == "trailing_stop" and (
        deployment.brokerage not in ("PaperBrokerage", "AlpacaBrokerage")
        or deployment.brokerage == "AlpacaBrokerage"
        and (intent.instrument.asset_class == "crypto" or intent.quantity % 1)
    ):
        raise LedgerError("reviewed LEAN brokerage/quantity does not support this trailing order")


class LeanInstrument(VersionedContract):
    instrument: Instrument
    ticker: NonEmptyString
    market: NonEmptyString
    base_currency: NonEmptyString | None = None

    @model_validator(mode="after")
    def actual_base(self):
        if (self.instrument.asset_class == "crypto") != (self.base_currency is not None):
            raise ValueError("crypto LEAN mapping requires its actual native base currency")
        return self


class LeanDeployment(VersionedContract):
    deployment_id: NonEmptyString
    account_id: NonEmptyString
    broker_account_id: NonEmptyString
    brokerage: Literal["PaperBrokerage", "AlpacaBrokerage", "BinanceBrokerage", "KrakenBrokerage"]
    mode: Literal["paper", "live"]
    instruments: tuple[LeanInstrument, ...]

    @model_validator(mode="after")
    def coherent(self):
        if not self.instruments or len(
            {item.instrument.instrument_id for item in self.instruments}
        ) != len(self.instruments):
            raise ValueError("unique complete LEAN instrument mappings required")
        if (self.brokerage == "PaperBrokerage") != (self.mode == "paper"):
            raise ValueError("paper LEAN deployment requires the actual PaperBrokerage")
        bases = [item.base_currency for item in self.instruments if item.base_currency is not None]
        if len(set(bases)) != len(bases):
            raise ValueError("one canonical crypto holding per LEAN base currency required")
        return self


class LeanVenue:
    """Prepared intents are shared through SQLite, never through signal weights.

    Only the attached LEAN process may dispatch. A coordinator can construct the
    same adapter without a runtime, read actual retained captures and prepare
    intents; the runtime entry point drains that existing durable outbox.
    """

    venue, owner = "lean", "lean"

    def __init__(
        self,
        ledger,
        account_id,
        *,
        deployment,
        rules,
        runtime=None,
        allow_live=False,
        fee_valuation=None,
        native_reader=None,
        evidence_store=None,
    ):
        self.deployment = LeanDeployment.model_validate(deployment.model_dump())
        self.ledger, self.account_id, self.mode = ledger, account_id, self.deployment.mode
        self.runtime, self._fee_valuation = runtime, fee_valuation
        self.native_reader = native_reader
        self.evidence_store = evidence_store
        self._rules = {
            key: VenueRules.model_validate(value.model_dump()) for key, value in rules.items()
        }
        self.instruments = {
            item.instrument.instrument_id: item.instrument for item in self.deployment.instruments
        }
        spec = ledger.spec(account_id)
        if (
            account_id != self.deployment.account_id
            or spec.mode != self.mode
            or self.mode == "live"
            and not allow_live
        ):
            raise LedgerError(
                "LEAN exact operational account and independent live authorization required"
            )
        if set(self.instruments) != set(self._rules) or any(
            rule.venue != self.venue
            or rule.instrument != self.instruments[key]
            or rule.instrument.quote_currency != spec.base_currency
            for key, rule in self._rules.items()
        ):
            raise LedgerError("LEAN exact same-currency instrument/rule coverage required")
        with ledger.atomic():
            require_remote_income_account(ledger, account_id)
            ledger.record_event(
                account_id, "remote-binding:lean", self.deployment.model_dump_json()
            )
            ledger.db.execute(
                "CREATE TABLE IF NOT EXISTS lean_ticket_bindings(account TEXT NOT NULL,intent TEXT NOT NULL,runtime TEXT NOT NULL,ticket INTEGER NOT NULL,client TEXT NOT NULL,PRIMARY KEY(account,intent),UNIQUE(account,runtime,ticket))"
            )
        if runtime is not None:
            self._runtime_binding()

    def _runtime_binding(self):
        if self.runtime is None:
            raise LedgerError(
                "dispatch only inside the attached LEAN runtime; coordinator prepares intents"
            )
        binding = self.runtime.identity()
        if (
            binding.get("deployment_hash") != self.deployment.content_hash()
            or binding.get("live_mode") is not True
            or not binding.get("runtime_id")
            or not binding.get("engine_version")
        ):
            raise LedgerError("actual LEAN runtime/deployment binding differs")
        with self.ledger.atomic():
            self.ledger.record_event(
                self.account_id, "lean-runtime:" + binding["runtime_id"], canonical(binding)
            )
        return binding

    def _record(self, kind, payload):
        identity = "lean-" + kind + ":" + digest(payload)
        with self.ledger.atomic():
            self.ledger.record_event(self.account_id, identity, canonical(payload))
        return identity

    def _read(self, identity, kind):
        row = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?", (self.account_id, identity)
        ).fetchone()
        if row is None:
            raise LedgerError("LEAN original receipt is missing")
        value = json.loads(row[0])
        if identity != "lean-" + kind + ":" + digest(value):
            raise LedgerError("LEAN original receipt identity changed")
        runtime = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (self.account_id, "lean-runtime:" + value["runtime_id"]),
        ).fetchone()
        if (
            runtime is None
            or json.loads(runtime[0])["deployment_hash"] != self.deployment.content_hash()
        ):
            raise LedgerError("LEAN receipt lacks its immutable runtime/deployment binding")
        return value

    def _latest(self, kind, as_of, client=None, *, required=True):
        for row in self.ledger.db.execute(
            "SELECT id FROM events WHERE account=? AND id LIKE ? ORDER BY rowid DESC",
            (self.account_id, "lean-" + kind + ":%"),
        ):
            value = self._read(row[0], kind)
            if datetime.fromisoformat(value["observed_at"]) <= as_of and (
                client is None or value.get("client_order_id") == client
            ):
                return value, row[0]
        if required:
            raise LedgerError("actual LEAN receipt unavailable before cutoff")
        return None

    def capture(self):
        binding = self._runtime_binding()
        native = self._native_reader().capture() if self.mode == "live" else None
        payload = self.runtime.capture()
        if payload["runtime_id"] != binding["runtime_id"]:
            raise LedgerError("LEAN capture belongs to another runtime")
        if native is not None:
            payload["native_balances"] = native.model_dump(mode="json")
        identity = self._record("balances", payload)
        return self._balances(payload, identity)

    def _native_reader(self):
        from src.execution.alpaca_venue import AlpacaVenue
        from src.execution.binance_venue import BinanceVenue
        from src.execution.kraken_venue import KrakenVenue

        expected = {
            "AlpacaBrokerage": AlpacaVenue,
            "BinanceBrokerage": BinanceVenue,
            "KrakenBrokerage": KrakenVenue,
        }.get(self.deployment.brokerage)
        reader = self.native_reader
        if (
            expected is None
            or type(reader) is not expected
            or not (
                reader.ledger is self.ledger
                or str(reader.ledger.path) != ":memory:"
                and str(self.ledger.path) != ":memory:"
                and Path(reader.ledger.path).resolve() == Path(self.ledger.path).resolve()
            )
            or reader.account_id != self.account_id
            or reader.broker_account_id != self.deployment.broker_account_id
            or reader.mode != "live"
        ):
            raise LedgerError(
                "live LEAN requires its exact actual native account reconciliation reader"
            )
        if self.runtime is not None:
            if reader.venue == "alpaca":
                credential = hashlib.sha256(reader._key.get_secret_value().encode()).hexdigest()
            else:
                stored = self.ledger.db.execute(
                    "SELECT payload FROM events WHERE account=? AND id=?",
                    (self.account_id, "remote-binding:" + reader.venue),
                ).fetchone()
                credential = json.loads(stored[0])["credential_identity"]
            if self._runtime_binding().get("native_credential_identity") != credential:
                raise LedgerError(
                    "LEAN runtime and native reader use different account credentials"
                )
        return reader

    def _verified_native_balance(self, value):
        reader = self._native_reader()
        actual = VenueBalances.model_validate(value)
        row = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (self.account_id, actual.evidence_id),
        ).fetchone()
        if row is None:
            raise LedgerError("LEAN native account receipt missing")
        capture = json.loads(row[0])
        if reader.venue == "alpaca":
            reader._verify_capture(capture, actual.evidence_id)
            verified = reader._balances(capture)
        else:
            if actual.evidence_id != reader.venue + "-capture:" + digest(capture):
                raise LedgerError("LEAN native account receipt changed")
            reader._verify_capture(capture)
            verified = reader._balances(capture, actual.evidence_id)
        if actual != verified:
            raise LedgerError("LEAN native balances differ from original account proof")
        if reader.venue == "alpaca":
            native_ids = {str(order["id"]) for order in capture["orders"]}
        elif reader.venue == "binance":
            native_ids = {str(order["orderId"]) for order in capture["payload"]["orders"]}
        else:
            native_ids = set(reader._open_orders(capture))
        return verified, native_ids

    def _balances(self, payload, identity):
        base = self.ledger.spec(self.account_id).base_currency
        if payload["account_currency"] != base or any(
            decimal(amount) for amount in payload["unsettled_cash"].values()
        ):
            raise LedgerError(
                "LEAN foreign account/unsettled cash requires explicit canonical settlement accounting"
            )
        crypto = {
            item.base_currency: item.instrument.instrument_id
            for item in self.deployment.instruments
            if item.base_currency is not None
        }
        # Crypto is a cash swap in LEAN. Its base-currency CashBook amount is
        # inventory, not additional spendable quote cash or an FX-valued deposit.
        for code, amount in payload["cash_book"].items():
            if code == base:
                continue
            if code not in crypto:
                if decimal(amount):
                    raise LedgerError("LEAN unmapped nonbase cash requires settlement accounting")
            elif decimal(amount) != decimal(payload["quantities"].get(crypto[code], "0")):
                raise LedgerError("LEAN crypto cash-book inventory differs from security holdings")
        if any(
            decimal(payload["quantities"].get(key, "0")) and code not in payload["cash_book"]
            for code, key in crypto.items()
        ):
            raise LedgerError("LEAN crypto holdings lack actual base-currency cash-book proof")
        if set(payload["quantities"]) - set(self.instruments):
            raise LedgerError("LEAN unowned instrument holdings present")
        cash = decimal(payload["cash_book"][base])
        # LEAN's observed margin remaining is its runtime buying-power bound;
        # canonical pending cash reservations remain independently enforced.
        available = max(D(0), min(cash, decimal(payload["margin_remaining"])))
        if self.mode == "live":
            native, native_ids = self._verified_native_balance(payload["native_balances"])
            runtime_ids = [
                identity for order in payload["open_orders"] for identity in order["broker_ids"]
            ]
            if (
                set(runtime_ids) != native_ids
                or len(runtime_ids) != len(set(runtime_ids))
                or any(not order["broker_ids"] for order in payload["open_orders"])
            ):
                raise LedgerError(
                    "LEAN native open orders differ from actual runtime broker identities"
                )
            canonical_clients = []
            for order in payload["open_orders"]:
                if len(order["broker_ids"]) != 1:
                    raise LedgerError("LEAN open order requires a unique original native binding")
                native_key = "lean-native-order:" + digest(
                    [self._native_reader().venue, order["broker_ids"][0]]
                )
                retained = self.ledger.db.execute(
                    "SELECT payload FROM events WHERE account=? AND id=?",
                    (self.account_id, native_key),
                ).fetchone()
                if retained is None:
                    raise LedgerError(
                        "LEAN unowned native open order requires explicit reconciliation"
                    )
                old = json.loads(retained[0])
                intent = self.ledger.order(self.account_id, old["intent_id"])["intent"]
                self._original_native_binding(
                    intent, old["client_order_id"], datetime.fromisoformat(payload["observed_at"])
                )
                canonical_clients.append(old["client_order_id"])
            if (
                not 0
                <= (
                    datetime.fromisoformat(payload["observed_at"]) - native.observed_at
                ).total_seconds()
                <= 30
                or native.cash_base != cash
                or {key: value for key, value in native.quantities.items() if value}
                != {
                    key: decimal(value)
                    for key, value in payload["quantities"].items()
                    if decimal(value)
                }
            ):
                raise LedgerError("LEAN actual native account cash/holdings/cutoff differs")
            available = min(available, native.available_cash_base)
        return VenueBalances(
            venue=self.venue,
            observed_at=payload["observed_at"],
            cash_base=cash,
            available_cash_base=available,
            quantities={key: decimal(value) for key, value in payload["quantities"].items()},
            open_client_order_ids=tuple(
                canonical_clients if self.mode == "live" else payload["open_client_order_ids"]
            ),
            evidence_id=identity,
        )

    def balances(self, as_of):
        payload, identity = self._latest("balances", _utc(as_of))
        return self._balances(payload, identity)

    def runtime_status(self, as_of):
        """Actual persisted runtime health; missing/stale receipts imply no readiness."""
        value, _ = self._latest("runtime-status", _utc(as_of))
        if (
            not 0
            <= (_utc(as_of) - datetime.fromisoformat(value["observed_at"])).total_seconds()
            <= 30
        ):
            raise LedgerError("LEAN runtime status receipt is stale")
        return value

    def rules(self, instrument, as_of):
        if self.instruments.get(instrument.instrument_id) != instrument:
            raise LedgerError("LEAN instrument identity differs")
        self.balances(as_of)
        return self._rules[instrument.instrument_id]

    def _claim(self, intent, client, rules, as_of, *, cancel=False):
        binding = self._runtime_binding()
        now = _utc(self.runtime.now())
        if not 0 <= (now - _utc(as_of)).total_seconds() <= 30:
            raise LedgerError("LEAN operation requires a current measured cutoff")
        row = self.ledger.db.execute(
            "SELECT o.*,b.owner FROM venue_outbox o JOIN execution_books b ON b.account=o.account WHERE o.account=? AND o.intent=?",
            (self.account_id, intent.intent_id),
        ).fetchone()
        if (
            row is None
            or row["owner"] != self.owner
            or row["venue"] != self.venue
            or row["state"] != ("cancel_pending" if cancel else "submitting")
            or row["client_id"] != client
            or self.ledger.order(self.account_id, intent.intent_id)["intent"] != intent
            or not cancel
            and VenueRules.model_validate_json(row["rules"]) != rules
        ):
            raise LedgerError("LEAN operation differs from exact durable execution claim")
        if cancel:
            return binding
        day = None
        try:
            rules.require_order(intent, now)
            require_trailing_support(self.deployment, intent)
            if intent.expires_at is not None and now >= intent.expires_at:
                raise LedgerError("LEAN order expired before actual submission")
            if intent.fx_rate != 1 or intent.tif not in ("GTC", "DAY"):
                raise LedgerError("LEAN requires supported exact cash currency/time-in-force")
            if intent.expires_at is not None and intent.tif != "DAY":
                raise LedgerError("LEAN expiry must use the explicit admitted DAY boundary")
            if self.mode == "live":
                reader = self._native_reader()
                native_rules = reader.rules(intent.instrument, now)
                native_rules.require_order(intent, now)
                if not 0 <= (now - reader.balances(now).observed_at).total_seconds() <= 30:
                    raise LedgerError("LEAN live native funding proof is stale/future")
                if reader.venue == "binance":
                    from src.execution.lean_references import resolve_binance_reference

                    reference = resolve_binance_reference(self, intent, as_of=now)
                    previous = reader._notional_reference
                    try:
                        reader._notional_reference = reference
                        reader._check_order(intent, now)
                    finally:
                        reader._notional_reference = previous
                elif reader.venue == "kraken":
                    reader._check_order(intent, now)
                elif reader.venue == "alpaca":
                    day = reader.validate_day(intent, as_of=now)
            self.runtime.preflight(intent)
            now = self.runtime.now()
            if intent.expires_at is not None and now >= intent.expires_at:
                raise LedgerError("LEAN order expired during actual preflight")
        except (LedgerError, ValueError) as exc:
            raise SubmissionNotAttempted(
                "LEAN preflight denied before submission: " + str(exc)
            ) from None
        with self.ledger.atomic():
            key = "lean-submit-attempt:" + intent.intent_id
            from src.execution.execution_admission import require_execution_admission

            try:
                require_execution_admission(self.ledger, self.account_id)
            except LedgerError as exc:
                raise SubmissionNotAttempted(str(exc)) from None
            if self.ledger.db.execute(
                "SELECT 1 FROM events WHERE account=? AND id=?", (self.account_id, key)
            ).fetchone():
                raise LedgerError(
                    "LEAN submission was attempted; recover original ticket, never resubmit"
                )
            self.ledger.record_event(
                self.account_id,
                key,
                canonical(
                    {
                        "intent_hash": intent.content_hash(),
                        "client_order_id": client,
                        "runtime_id": binding["runtime_id"],
                        "at": now.isoformat(),
                        **({"day_calendar_hash": digest(day)} if day is not None else {}),
                    }
                ),
            )
            from src.execution.execution_admission import _record_submission_claimed

            _record_submission_claimed(self.ledger, self.account_id, intent.intent_id)
        return binding

    def submit(self, intent, client_order_id, rules, as_of):
        self._claim(intent, client_order_id, rules, as_of)
        return self._ticket_report(
            intent, client_order_id, self.runtime.submit(intent, client_order_id)
        )

    def lookup(self, intent, client_order_id, venue_order_id, as_of):
        bound = self.ledger.db.execute(
            "SELECT * FROM lean_ticket_bindings WHERE account=? AND intent=?",
            (self.account_id, intent.intent_id),
        ).fetchone()
        if (
            self.mode == "live"
            and bound
            and (self.runtime is None or bound["runtime"] != self._runtime_binding()["runtime_id"])
        ):
            report = self._recover_native(intent, client_order_id, as_of)
            if venue_order_id is not None and report.venue_order_id != venue_order_id:
                raise LedgerError("LEAN original native ticket identity changed")
            return report
        if self.runtime is None:
            candidates = [
                self._read(row[0], "ticket")
                for row in self.ledger.db.execute(
                    "SELECT id FROM events WHERE account=? AND id LIKE 'lean-ticket:%' ORDER BY rowid DESC",
                    (self.account_id,),
                )
            ]
            payload = next(
                (
                    value
                    for value in candidates
                    if value.get("client_order_id") == client_order_id
                    and datetime.fromisoformat(value["observed_at"]) <= _utc(as_of)
                ),
                None,
            )
            if payload is None:
                return None
        else:
            binding = self._runtime_binding()
            bound = self.ledger.db.execute(
                "SELECT * FROM lean_ticket_bindings WHERE account=? AND intent=?",
                (self.account_id, intent.intent_id),
            ).fetchone()
            if bound and bound["runtime"] != binding["runtime_id"]:
                # LEAN ticket/event IDs are runtime-local. Missing native broker
                # execution history cannot be guessed from a restarted ticket.
                raise LedgerError(
                    "LEAN restart requires original runtime/native execution proof for unresolved tickets"
                )
            payload = self.runtime.lookup(client_order_id, bound["ticket"] if bound else None)
            if payload is None:
                return None
        report = self._ticket_report(intent, client_order_id, payload)
        if venue_order_id is not None and report.venue_order_id != venue_order_id:
            raise LedgerError("LEAN ticket identity changed")
        return report

    def cancel(self, intent, client_order_id, venue_order_id, as_of):
        binding = self._claim(intent, client_order_id, None, as_of, cancel=True)
        bound = self.ledger.db.execute(
            "SELECT * FROM lean_ticket_bindings WHERE account=? AND intent=?",
            (self.account_id, intent.intent_id),
        ).fetchone()
        if (
            bound is None
            or bound["runtime"] != binding["runtime_id"]
            or venue_order_id != bound["runtime"] + ":" + str(bound["ticket"])
        ):
            if (
                self.mode == "live"
                and bound
                and venue_order_id == bound["runtime"] + ":" + str(bound["ticket"])
            ):
                original, _ = self._original_native_binding(intent, client_order_id, as_of)
                actual = self.runtime.lookup_broker_order(original["broker_order_id"])
                if (
                    actual is None
                    or actual["runtime_id"] != binding["runtime_id"]
                    or actual["broker_ids"] != [original["broker_order_id"]]
                ):
                    raise LedgerError(
                        "restarted LEAN cancellation lacks the actual imported native ticket"
                    )
                self._validate_ticket_intent(intent, actual)
                source = self._record("recovered-ticket", actual)
                alias = {
                    "intent_hash": intent.content_hash(),
                    "original_binding": original,
                    "runtime_id": actual["runtime_id"],
                    "ticket_id": actual["ticket_id"],
                    "client_order_id": actual["client_order_id"],
                    "source": source,
                }
                alias_key = "lean-runtime-alias:" + digest(
                    [actual["runtime_id"], actual["ticket_id"]]
                )
                with self.ledger.atomic():
                    prior = self.ledger.db.execute(
                        "SELECT payload FROM events WHERE account=? AND id=?",
                        (self.account_id, alias_key),
                    ).fetchone()
                    if prior is None:
                        self.ledger.record_event(self.account_id, alias_key, canonical(alias))
                    else:
                        retained = json.loads(prior[0])
                        if {key: value for key, value in retained.items() if key != "source"} != {
                            key: value for key, value in alias.items() if key != "source"
                        }:
                            raise LedgerError("LEAN imported runtime ticket identity changed")
                        first = self._read(retained["source"], "recovered-ticket")
                        if any(
                            first[key] != actual[key]
                            for key in ("runtime_id", "ticket_id", "client_order_id", "broker_ids")
                        ):
                            raise LedgerError("LEAN original imported ticket receipt differs")
                        self._validate_ticket_intent(intent, first)
                self.runtime.cancel(actual["ticket_id"], actual["client_order_id"])
                return self._recover_native(intent, client_order_id, as_of)
            raise LedgerError("LEAN cancellation requires exact original runtime ticket")
        self.runtime.cancel(bound["ticket"], client_order_id)
        report = self.lookup(intent, client_order_id, venue_order_id, as_of)
        if report is None:
            raise LedgerError("LEAN cancellation remains unresolved")
        return report

    def _ticket_report(self, intent, client, payload):
        # Archive the complete actual ticket before validating economics.
        evidence = self._record("ticket", payload)
        payload = self._read(evidence, "ticket")
        attempt = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (self.account_id, "lean-submit-attempt:" + intent.intent_id),
        ).fetchone()
        if attempt is None:
            raise LedgerError("LEAN ticket has no durable pre-submission claim")
        attempt = json.loads(attempt[0])
        if (
            attempt["runtime_id"] != payload["runtime_id"]
            or attempt["intent_hash"] != intent.content_hash()
            or attempt["client_order_id"] != client
        ):
            raise LedgerError("LEAN ticket runtime/intent differs from original submission claim")
        if payload["client_order_id"] != client:
            raise LedgerError("LEAN ticket client differs from exact canonical intent")
        self._validate_ticket_intent(intent, payload)
        signed = intent.quantity if intent.side == "buy" else intent.quantity.copy_negate()
        ticket, runtime = payload["ticket_id"], payload["runtime_id"]
        if type(ticket) is not int or ticket < 1:
            raise LedgerError("actual positive LEAN ticket ID required")
        with self.ledger.atomic():
            old = self.ledger.db.execute(
                "SELECT * FROM lean_ticket_bindings WHERE account=? AND intent=?",
                (self.account_id, intent.intent_id),
            ).fetchone()
            if old and (old["runtime"], old["ticket"], old["client"]) != (runtime, ticket, client):
                raise LedgerError("LEAN intent already has another actual runtime ticket")
            self.ledger.db.execute(
                "INSERT OR IGNORE INTO lean_ticket_bindings VALUES (?,?,?,?,?)",
                (self.account_id, intent.intent_id, runtime, ticket, client),
            )
        at = datetime.fromisoformat(payload["observed_at"])
        fills, ids, net_quantities, gross_quantities = [], [], [], []
        for event in payload["events"]:
            event_id = event["event_id"]
            if (
                type(event_id) is not int
                or event_id < 1
                or event_id in ids
                or event["ticket_id"] != ticket
                or event["instrument_id"] != intent.instrument.instrument_id
                or event.get("is_assignment") is not False
            ):
                raise LedgerError("LEAN event identity differs or is duplicated")
            ids.append(event_id)
            native_key = "lean-order-event:" + digest([runtime, ticket, event_id])
            native = {
                "runtime_id": runtime,
                "event": event,
                "client_order_id": client,
                "intent_hash": intent.content_hash(),
            }
            with self.ledger.atomic():
                original = self.ledger.db.execute(
                    "SELECT payload FROM events WHERE account=? AND id=?",
                    (self.account_id, native_key),
                ).fetchone()
                if original:
                    first = json.loads(original[0])
                    if canonical(first["native"]) != canonical(native):
                        raise LedgerError("LEAN event identity changed original execution")
                    source = self._read(first["source"], "ticket")
                    if event not in source["events"] or source["runtime_id"] != runtime:
                        raise LedgerError("LEAN original event missing from ticket source")
                else:
                    first = {"native": native, "source": evidence, "observed_at": at.isoformat()}
                    self.ledger.record_event(self.account_id, native_key, canonical(first))
            quantity, charge = decimal(event["fill_quantity"]), decimal(event["fee_amount"])
            net_quantities.append(quantity.copy_abs())
            if charge < 0 or not quantity and charge:
                raise LedgerError(
                    "LEAN fee-only/rebate event requires explicit posted-fee accounting"
                )
            if not quantity:
                continue
            if (
                quantity * signed <= 0
                or event["status"] not in ("Filled", "PartiallyFilled")
                or event["price_currency"] != self.ledger.spec(self.account_id).base_currency
            ):
                raise LedgerError("LEAN execution side/status/price currency differs")
            occurred = datetime.fromisoformat(event["occurred_at"])
            if occurred > datetime.fromisoformat(first["observed_at"]) or occurred > at:
                raise LedgerError("future LEAN execution event")
            price = decimal(event["fill_price"])
            fee_type = event["fee_type"]
            if fee_type == "QuantConnect.Orders.Fees.ModifiedFillQuantityOrderFee":
                base = next(
                    item.base_currency
                    for item in self.deployment.instruments
                    if item.instrument == intent.instrument
                )
                if intent.side != "buy" or base is None or event["fee_currency"] != base:
                    raise LedgerError(
                        "LEAN modified-quantity fee requires the exact crypto buy asset"
                    )
                quantity = exact_decimal_sum((quantity, charge))
            elif fee_type != "QuantConnect.Orders.Fees.OrderFee":
                raise LedgerError("LEAN fee implementation has unsupported settlement semantics")
            gross_quantities.append(quantity.copy_abs())
            if self.mode == "live":
                # Live runtime fees can be FeeModel estimates, and ticket events
                # are not native execution IDs. Only the native account's actual
                # individual executions may produce canonical live fills.
                continue
            cash_fee, inventory = D(0), ()
            if charge:
                if event["fee_currency"] == event["price_currency"]:
                    cash_fee = charge
                else:
                    expected_asset = next(
                        (
                            item.instrument
                            for item in self.deployment.instruments
                            if item.base_currency == event["fee_currency"]
                        ),
                        None,
                    )
                    if expected_asset == intent.instrument:
                        valuation = FeeValuation(
                            mark=PriceMark(instrument=expected_asset, price=price, as_of=occurred),
                            available_at=first["observed_at"],
                            source_id=native_key,
                            source_hash=digest(first),
                        )
                    elif self._fee_valuation is not None:
                        valuation = FeeValuation.model_validate(
                            self._fee_valuation(event["fee_currency"], occurred).model_dump()
                        )
                    else:
                        raise LedgerError(
                            "LEAN actual third-asset fee needs canonical asset/source valuation"
                        )
                    if (
                        valuation.mark.instrument != expected_asset
                        or self.instruments.get(valuation.mark.instrument.instrument_id)
                        != valuation.mark.instrument
                        or not 0 <= (occurred - valuation.mark.as_of).total_seconds() <= 30
                        or valuation.available_at > datetime.fromisoformat(first["observed_at"])
                    ):
                        raise LedgerError("LEAN fee valuation instrument/age/availability differs")
                    inventory = (
                        InventoryFee(
                            instrument=valuation.mark.instrument,
                            quantity=charge,
                            price=valuation.mark.price,
                            fx_rate=valuation.mark.fx_rate,
                            valued_at=valuation.mark.as_of,
                            available_at=datetime.fromisoformat(first["observed_at"]),
                            source_id=native_key,
                            source_hash=digest(first),
                            valuation_source_id=valuation.source_id,
                            valuation_source_hash=valuation.source_hash,
                        ),
                    )
            fills.append(
                Fill(
                    fill_id=native_key,
                    intent_id=intent.intent_id,
                    account_id=self.account_id,
                    quantity=quantity.copy_abs(),
                    price=price,
                    fee=cash_fee,
                    executed_at=occurred,
                    event_id=native_key,
                    inventory_fees=inventory,
                )
            )
        native_filled = decimal(payload["filled_quantity"])
        if native_filled * signed < 0 or native_filled.copy_abs() > intent.quantity:
            raise LedgerError("LEAN cumulative quantity side/range differs")
        cumulative = exact_decimal_sum(gross_quantities)
        if cumulative > intent.quantity:
            raise LedgerError("LEAN gross execution quantity exceeds original intent")
        status = {
            "New": "new",
            "Submitted": "new",
            "PartiallyFilled": "partial",
            "Filled": "filled",
            "Canceled": "cancelled",
            "Invalid": "rejected",
        }.get(payload["status"], "unknown")
        complete = (
            ids == list(range(1, len(ids) + 1))
            and exact_decimal_sum(net_quantities) == native_filled.copy_abs()
            and status != "unknown"
        )
        if self.mode == "live":
            return self._native_report(intent, client, payload, evidence, cumulative, complete)
        return VenueOrderReport(
            venue=self.venue,
            client_order_id=client,
            venue_order_id=runtime + ":" + str(ticket),
            observed_at=at,
            status=status,
            cumulative_quantity=cumulative,
            fills=tuple(fills),
            executions_complete=complete,
            evidence_id=evidence,
        )

    @staticmethod
    def _validate_ticket_intent(intent, payload):
        if intent.tif == "DAY":
            if (
                payload.get("time_in_force") != "DAY"
                or not payload.get("expires_at")
                or _utc(datetime.fromisoformat(payload["expires_at"])) != intent.expires_at
            ):
                raise LedgerError("LEAN ticket DAY expiry differs from admitted calendar")
        elif payload.get("time_in_force") is not None or payload.get("expires_at") is not None:
            raise LedgerError("LEAN GTC ticket unexpectedly has an expiry")
        signed = intent.quantity if intent.side == "buy" else intent.quantity.copy_negate()
        if (
            payload["instrument_id"] != intent.instrument.instrument_id
            or decimal(payload["quantity"]) != signed
            or payload["order_type"]
            != {
                "market": "Market",
                "limit": "Limit",
                "stop": "StopMarket",
                "trailing_stop": "TrailingStop",
            }[intent.order_type]
        ):
            raise LedgerError("LEAN ticket differs from exact canonical intent")
        if (
            intent.order_type == "limit"
            and decimal(payload["limit_price"]) != intent.limit_price
            or intent.order_type == "stop"
            and decimal(payload["stop_price"]) != intent.stop_price
        ):
            raise LedgerError("LEAN ticket price differs")
        if intent.order_type == "trailing_stop":
            percentage = intent.trail_percent is not None
            expected = (
                exact_decimal_product(intent.trail_percent, D(".01"))
                if percentage
                else intent.trail_price
            )
            if (
                payload.get("trailing_as_percentage") is not percentage
                or decimal(payload.get("trailing_amount")) != expected
                or decimal(payload.get("stop_price")) <= 0
                or payload.get("limit_price") is not None
            ):
                raise LedgerError("LEAN trailing ticket amount/basis/observation differs")
        elif (
            payload.get("trailing_amount") is not None
            or payload.get("trailing_as_percentage") is not None
        ):
            raise LedgerError("LEAN fixed order unexpectedly contains trailing terms")

    def _original_native_binding(self, intent, client, as_of):
        retained = self._latest("native-report", _utc(as_of), client, required=False)
        if retained is not None:
            proof = retained[0]
            source = self._read(proof["ticket_source"], "ticket")
        else:
            # A crash/read failure can occur after retaining the original native
            # ID but before the first native report. The original admitted ticket
            # still binds that exact ID; a new process must never invent a ticket.
            source, ticket_source = self._latest("ticket", _utc(as_of), client)
            if len(source["broker_ids"]) != 1:
                raise LedgerError("LEAN original ticket lacks a unique native order identity")
            proof = {
                "ticket_source": ticket_source,
                "native_binding": "lean-native-order:"
                + digest([self._native_reader().venue, source["broker_ids"][0]]),
            }
        original = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (self.account_id, proof["native_binding"]),
        ).fetchone()
        bound = self.ledger.db.execute(
            "SELECT * FROM lean_ticket_bindings WHERE account=? AND intent=?",
            (self.account_id, intent.intent_id),
        ).fetchone()
        attempt = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (self.account_id, "lean-submit-attempt:" + intent.intent_id),
        ).fetchone()
        if original is None or bound is None or attempt is None:
            raise LedgerError("LEAN original native submission authority is incomplete")
        original, attempt = json.loads(original[0]), json.loads(attempt[0])
        expected = {
            "deployment_hash": self.deployment.content_hash(),
            "venue": self._native_reader().venue,
            "broker_order_id": original["broker_order_id"],
            "runtime_id": bound["runtime"],
            "ticket_id": bound["ticket"],
            "intent_hash": intent.content_hash(),
            "intent_id": intent.intent_id,
            "client_order_id": client,
        }
        if (
            original != expected
            or proof["native_binding"]
            != "lean-native-order:" + digest([original["venue"], original["broker_order_id"]])
            or source["broker_ids"] != [original["broker_order_id"]]
            or source["runtime_id"] != bound["runtime"]
            or source["ticket_id"] != bound["ticket"]
            or source["client_order_id"] != client
            or bound["client"] != client
            or attempt["runtime_id"] != bound["runtime"]
            or attempt["intent_hash"] != intent.content_hash()
            or attempt["client_order_id"] != client
        ):
            raise LedgerError("LEAN original native order binding differs from retained submission")
        self._validate_ticket_intent(intent, source)
        return original, proof["ticket_source"]

    def _recover_native(self, intent, client, as_of):
        original, source = self._original_native_binding(intent, client, as_of)
        reader = self._native_reader()
        native = reader.read_native_order(intent, original["broker_order_id"])
        if native is None or native.venue_order_id != original["broker_order_id"]:
            raise LedgerError("LEAN original native order remains unavailable")
        key = "lean-native-order:" + digest([reader.venue, original["broker_order_id"]])
        prior = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (self.account_id, key + ":client"),
        ).fetchone()
        if prior is None:
            if self._latest("native-report", _utc(as_of), client, required=False) is not None:
                raise LedgerError("LEAN original native client binding is missing")
            with self.ledger.atomic():
                self.ledger.record_event(
                    self.account_id,
                    key + ":client",
                    canonical({"native_client": native.client_order_id}),
                )
        elif json.loads(prior[0]) != {"native_client": native.client_order_id}:
            raise LedgerError("LEAN original native client identity changed during recovery")
        proof = {
            "runtime_id": self.runtime.identity()["runtime_id"]
            if self.runtime
            else original["runtime_id"],
            "observed_at": native.observed_at.isoformat(),
            "client_order_id": client,
            "ticket_source": source,
            "native_binding": key,
            "recovery_policy": "original-native-order-history-1",
            "native_report": native.model_dump(mode="json"),
        }
        return VenueOrderReport(
            venue="lean",
            client_order_id=client,
            venue_order_id=original["runtime_id"] + ":" + str(original["ticket_id"]),
            observed_at=native.observed_at,
            status=native.status,
            cumulative_quantity=native.cumulative_quantity,
            fills=native.fills,
            executions_complete=native.executions_complete,
            evidence_id=self._record("native-report", proof),
        )

    def _native_report(self, intent, client, payload, evidence, cumulative, complete):
        reader = self._native_reader()
        broker_ids = payload["broker_ids"]
        if len(broker_ids) != 1 or not isinstance(broker_ids[0], str) or not broker_ids[0]:
            raise LedgerError("LEAN live history needs exactly one actual native broker order ID")
        order_id = broker_ids[0]
        binding = {
            "deployment_hash": self.deployment.content_hash(),
            "venue": reader.venue,
            "broker_order_id": order_id,
            "runtime_id": payload["runtime_id"],
            "ticket_id": payload["ticket_id"],
            "intent_hash": intent.content_hash(),
            "intent_id": intent.intent_id,
            "client_order_id": client,
        }
        key = "lean-native-order:" + digest([reader.venue, order_id])
        with self.ledger.atomic():
            self.ledger.record_event(self.account_id, key, canonical(binding))
        native = reader.read_native_order(intent, order_id)
        if native is None or native.venue_order_id != order_id:
            raise LedgerError("LEAN actual native order execution proof unavailable")
        with self.ledger.atomic():
            self.ledger.record_event(
                self.account_id,
                key + ":client",
                canonical({"native_client": native.client_order_id}),
            )
        at = max(native.observed_at, datetime.fromisoformat(payload["observed_at"]))
        proof = {
            "runtime_id": payload["runtime_id"],
            "observed_at": at.isoformat(),
            "client_order_id": client,
            "ticket_source": evidence,
            "native_binding": key,
            "native_report": native.model_dump(mode="json"),
        }
        return VenueOrderReport(
            venue="lean",
            client_order_id=client,
            venue_order_id=payload["runtime_id"] + ":" + str(payload["ticket_id"]),
            observed_at=at,
            status=native.status,
            cumulative_quantity=native.cumulative_quantity,
            fills=native.fills,
            executions_complete=complete
            and native.executions_complete
            and cumulative == native.cumulative_quantity,
            evidence_id=self._record("native-report", proof),
        )
