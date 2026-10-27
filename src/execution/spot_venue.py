"""Shared native spot identity, source receipts, fees and durable order claims."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal as D
from functools import wraps

from pydantic import SecretStr, model_validator

from src.domain.common import (
    Instrument,
    NonEmptyString,
    Sha256,
    UtcTimestamp,
    VersionedContract,
    _utc,
)
from src.domain.portfolio import Fill, FillSettlement, InventoryFee, PriceMark
from src.execution.ccxt_transport import CCXTTransport, canonical, decimal, digest
from src.execution.ledger import LedgerError
from src.execution.venue import SubmissionNotAttempted, VenueRules


class NativeSpotMarket(VersionedContract):
    instrument: Instrument
    symbol: NonEmptyString
    native_id: NonEmptyString
    base_asset: NonEmptyString
    quote_asset: NonEmptyString

    @model_validator(mode="after")
    def spot_only(self):
        if self.instrument.asset_class != "crypto" or self.base_asset == self.quote_asset:
            raise ValueError("distinct actual spot asset pair required")
        return self


class FeeValuation(VersionedContract):
    mark: PriceMark
    available_at: UtcTimestamp
    source_id: NonEmptyString
    source_hash: Sha256


def bounded(method):
    @wraps(method)
    def operation(self, *args, **kwargs):
        with self.transport.operation():
            return method(self, *args, **kwargs)

    return operation


class CCXTSpotVenue:
    """Shared machinery used only by the concrete Binance/Kraken native adapters."""

    owner = "direct"
    maximum_operation_delay_seconds = 30

    def __init__(
        self,
        ledger,
        account_id,
        *,
        broker_account_id,
        markets,
        rules,
        api_key,
        api_secret,
        allow_live=False,
        fee_valuation=None,
        client=None,
        clock=None,
        **transport_limits,
    ):
        spec = ledger.spec(account_id)
        if spec.mode not in ("paper", "live") or (spec.mode == "live" and not allow_live):
            raise LedgerError(
                "native spot operation requires independently authorized paper/live mode"
            )
        if self.venue == "kraken" and spec.mode == "paper":
            raise LedgerError("Kraken spot has no verified paper endpoint; use local PaperVenue")
        if broker_account_id is None or not isinstance(broker_account_id, (str, int)):
            raise LedgerError("explicit actual native account identity required")
        credential = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not isinstance(credential, str) or not credential:
            raise LedgerError("native spot credentials unavailable")
        self.ledger, self.account_id, self.mode = ledger, account_id, spec.mode
        self.broker_account_id = str(broker_account_id)
        self.markets = {
            value.instrument.instrument_id: NativeSpotMarket.model_validate(value.model_dump())
            for value in markets
        }
        self._rules = {
            key: VenueRules.model_validate(value.model_dump()) for key, value in rules.items()
        }
        if (
            not self.broker_account_id
            or not self.markets
            or len(self.markets) != len(markets)
            or set(self.markets) != set(self._rules)
            or len({market.base_asset for market in self.markets.values()}) != len(self.markets)
            or len({market.quote_asset for market in self.markets.values()}) != 1
            or any(
                market.instrument != self._rules[key].instrument
                or self._rules[key].venue != self.venue
                or self._rules[key].symbol != market.symbol
                or self._rules[key].market_type != "spot"
                or market.instrument.quote_currency != spec.base_currency
                for key, market in self.markets.items()
            )
        ):
            raise LedgerError(
                "explicit unique native spot markets and same-currency rules required"
            )
        self.cash_asset = next(iter(self.markets.values())).quote_asset
        if self.venue == "binance" and self.cash_asset != spec.base_currency:
            raise LedgerError("Binance quote cash must use its actual currency, not assumed USD FX")
        self._fee_valuation = fee_valuation
        with ledger.atomic():
            from src.execution.dividends import require_remote_income_account

            require_remote_income_account(ledger, account_id)
            for row in ledger.db.execute(
                "SELECT account,payload FROM events WHERE id=?", ("remote-binding:" + self.venue,)
            ):
                other = json.loads(row["payload"])
                if (
                    row["account"] != account_id
                    and other.get("broker_account_id") == self.broker_account_id
                    and other.get("mode") == spec.mode
                ):
                    raise LedgerError("native account already belongs to another canonical book")
            ledger.record_event(
                account_id,
                "remote-binding:" + self.venue,
                canonical(
                    {
                        "broker_account_id": self.broker_account_id,
                        "mode": spec.mode,
                        "base_currency": spec.base_currency,
                        "cash_asset": self.cash_asset,
                        "credential_identity": hashlib.sha256(credential.encode()).hexdigest(),
                        "markets": {
                            key: value.model_dump(mode="json")
                            for key, value in self.markets.items()
                        },
                    }
                ),
            )
            self.transport = CCXTTransport(
                ledger,
                account_id,
                venue=self.venue,
                api_key=api_key,
                api_secret=api_secret,
                sandbox=spec.mode == "paper",
                allow_live=allow_live,
                client=client,
                clock=clock,
                **transport_limits,
            )

    def close(self):
        self.transport.close()

    def _market(self, instrument):
        value = self.markets.get(instrument.instrument_id)
        if value is None or value.instrument != instrument:
            raise LedgerError("native execution instrument differs from immutable mapping")
        return value

    def _source(self, evidence_id):
        evidence, value = self.transport.read(evidence_id)
        if evidence["status"] != 200:
            raise LedgerError("native source is not a successful complete response")
        return evidence, value

    def _save_capture(self, payload, sources):
        capture = {
            "payload": payload,
            "sources": sources,
            "observed_at": self.transport.now().isoformat(),
        }
        identity = self.venue + "-capture:" + digest(capture)
        self._verify_capture(capture)
        result = self._balances(capture, identity)
        with self.ledger.atomic():
            self.ledger.record_event(self.account_id, identity, canonical(capture))
        return result

    def _verify_capture(self, capture):
        if set(capture["sources"]) != set(capture["payload"]):
            raise LedgerError("native capture source roles incomplete")
        for role, identity in capture["sources"].items():
            receipt, value = self._source(identity)
            if canonical(value) != canonical(capture["payload"][role]) or datetime.fromisoformat(
                receipt["observed_at"]
            ) > datetime.fromisoformat(capture["observed_at"]):
                raise LedgerError("native capture differs from original response bytes/clocks")
            self._require_capture_source(role, receipt)

    def _capture(self, as_of):
        as_of = _utc(as_of)
        for row in self.ledger.db.execute(
            "SELECT id,payload FROM events WHERE account=? AND id LIKE ? ORDER BY rowid DESC",
            (self.account_id, self.venue + "-capture:%"),
        ):
            capture = json.loads(row["payload"])
            if datetime.fromisoformat(capture["observed_at"]) > as_of:
                continue
            if row["id"] != self.venue + "-capture:" + digest(capture):
                raise LedgerError("native balance capture identity changed")
            self._verify_capture(capture)
            return capture, row["id"]
        raise LedgerError("capture actual native balances before the decision cutoff")

    def balances(self, as_of):
        capture, identity = self._capture(as_of)
        return self._balances(capture, identity)

    def _claim(self, intent, client_order_id, rules, as_of, *, cancel=False, venue_order_id=None):
        now = self.transport.now()
        if not 0 <= (now - _utc(as_of)).total_seconds() <= self.maximum_operation_delay_seconds:
            raise LedgerError("native submission/cancel requires a recent measured cutoff")
        if intent.account_id != self.account_id or intent.fx_rate != 1:
            raise LedgerError("native exact account and explicit same cash currency required")
        if not self.ledger.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='venue_outbox'"
        ).fetchone():
            raise LedgerError("native operation requires the durable execution owner")
        row = self.ledger.db.execute(
            "SELECT o.*,b.owner FROM venue_outbox o JOIN execution_books b ON b.account=o.account WHERE o.account=? AND o.intent=?",
            (self.account_id, intent.intent_id),
        ).fetchone()
        if (
            row is None
            or row["owner"] != self.owner
            or row["venue"] != self.venue
            or row["state"] != ("cancel_pending" if cancel else "submitting")
            or row["client_id"] != client_order_id
            or self.ledger.order(self.account_id, intent.intent_id)["intent"] != intent
            or (cancel and row["venue_id"] != venue_order_id)
            or (not cancel and VenueRules.model_validate_json(row["rules"]) != rules)
        ):
            raise LedgerError("native operation differs from exact durable owner claim")
        if cancel:
            return
        try:
            rules.require_order(intent, now)
            if intent.expires_at and now > intent.expires_at:
                raise LedgerError("native order expired before submission")
            if not 0 <= (now - self.balances(now).observed_at).total_seconds() <= 30:
                raise LedgerError("refresh native funding before submission")
            self._check_order(intent, now)
        except (LedgerError, ValueError) as exc:
            raise SubmissionNotAttempted(
                "native preflight denied before transmission: " + str(exc)
            ) from None
        with self.ledger.atomic():
            identity = self.venue + "-submit-attempt:" + intent.intent_id
            if self.ledger.db.execute(
                "SELECT 1 FROM events WHERE account=? AND id=?", (self.account_id, identity)
            ).fetchone():
                raise LedgerError(
                    "native submission already attempted; reconcile exact client identity"
                )
            self.ledger.record_event(
                self.account_id,
                identity,
                canonical(
                    {
                        "intent_hash": intent.content_hash(),
                        "client_order_id": client_order_id,
                        "at": now.isoformat(),
                    }
                ),
            )

    def _fill(
        self,
        intent,
        *,
        native_id,
        native_payload,
        quantity,
        price,
        cost,
        fee_asset,
        fee_quantity,
        executed_at,
        observed_at,
        evidence_id,
    ):
        quantity, price, cost, charge = map(decimal, (quantity, price, cost, fee_quantity))
        if min(quantity, price, cost) <= 0 or charge < 0:
            raise LedgerError("actual native trade quantities/cost/commission invalid")
        executed_at, observed_at = _utc(executed_at), _utc(observed_at)
        market = self._market(intent.instrument)
        source, raw = self._source(evidence_id)
        self._require_trade_member(native_payload, raw)
        source_hash = digest(source)
        trade_key = self.venue + "-native-trade:" + digest(str(native_id))
        economics = {
            "native": native_payload,
            "instrument": intent.instrument.model_dump(mode="json"),
            "side": intent.side,
            "quantity": str(quantity),
            "price": str(price),
            "cost": str(cost),
            "fee_asset": fee_asset,
            "fee_quantity": str(charge),
            "executed_at": executed_at.isoformat(),
        }
        retained = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?", (self.account_id, trade_key)
        ).fetchone()
        if retained:
            original = json.loads(retained[0])
            if canonical(original["economics"]) != canonical(economics):
                raise LedgerError("native trade identity changed economics/source fields")
            original_source, original_raw = self._source(original["source_id"])
            self._require_trade_member(native_payload, original_raw)
            if digest(original_source) != original["source_hash"]:
                raise LedgerError("original native trade receipt changed")
            actual = Fill.model_validate(original["fill"])
            if actual.intent_id != intent.intent_id or actual.account_id != self.account_id:
                raise LedgerError("native trade already belongs to another economic intent")
            # Reconstruct against the first genuine source/receipt, not a new
            # lookup clock. This also rechecks the exact fee valuation proof.
            evidence_id, source_hash = original["source_id"], original["source_hash"]
            observed_at = datetime.fromisoformat(original_source["observed_at"])
        fees, cash_fee = (), D(0)
        if charge:
            if fee_asset == self.cash_asset:
                cash_fee = charge
            else:
                fee_market = next(
                    (value for value in self.markets.values() if value.base_asset == fee_asset),
                    None,
                )
                if fee_market is None:
                    raise LedgerError(
                        "native fee asset lacks an exact canonical instrument mapping"
                    )
                if fee_asset == market.base_asset:
                    valuation = FeeValuation(
                        mark=PriceMark(
                            instrument=market.instrument, price=price, as_of=executed_at
                        ),
                        available_at=observed_at,
                        source_id=evidence_id,
                        source_hash=source_hash,
                    )
                elif self._fee_valuation is None:
                    raise LedgerError(
                        "actual third-asset fee requires source-backed causal valuation"
                    )
                else:
                    valuation = FeeValuation.model_validate(
                        self._fee_valuation(fee_market.instrument, executed_at).model_dump()
                    )
                if (
                    valuation.mark.instrument != fee_market.instrument
                    or valuation.available_at > observed_at
                    or not 0 <= (executed_at - valuation.mark.as_of).total_seconds() <= 30
                ):
                    raise LedgerError("native fee valuation identity/availability/age differs")
                fees = (
                    InventoryFee(
                        instrument=fee_market.instrument,
                        quantity=charge,
                        price=valuation.mark.price,
                        fx_rate=valuation.mark.fx_rate,
                        valued_at=valuation.mark.as_of,
                        available_at=max(valuation.available_at, observed_at),
                        source_id=evidence_id,
                        source_hash=source_hash,
                        valuation_source_id=valuation.source_id,
                        valuation_source_hash=valuation.source_hash,
                    ),
                )
        actual = Fill(
            fill_id=self.venue + ":" + str(native_id),
            intent_id=intent.intent_id,
            account_id=self.account_id,
            quantity=quantity,
            price=price,
            fee=cash_fee,
            executed_at=executed_at,
            event_id=evidence_id,
            settlement=FillSettlement(
                gross_quote=cost, source_id=evidence_id, source_hash=source_hash
            ),
            inventory_fees=fees,
        )
        if retained:
            if actual != Fill.model_validate(original["fill"]):
                raise LedgerError("retained native fill differs from original trade economics")
            return actual
        with self.ledger.atomic():
            self.ledger.record_event(
                self.account_id,
                trade_key,
                canonical(
                    {
                        "economics": economics,
                        "source_id": evidence_id,
                        "source_hash": source_hash,
                        "fill": actual.model_dump(mode="json"),
                    }
                ),
            )
        return actual

    @staticmethod
    def _require_trade_member(trade, raw):
        def contains(value):
            if canonical(value) == canonical(trade):
                return True
            children = (
                value.values()
                if isinstance(value, dict)
                else value
                if isinstance(value, list)
                else ()
            )
            return any(contains(child) for child in children)

        if not contains(raw):
            raise LedgerError("native trade absent from original archived response")
