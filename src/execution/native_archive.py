"""Pure, account-bound reads of retained native bytes; no DDL or transport."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import urlsplit

from src.domain.common import _utc
from src.execution.discrepancies import RetainedSourceRef
from src.execution.ledger import LedgerError


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def decode(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise LedgerError("duplicate native JSON key")
            result[key] = item
        return result

    def invalid(value):
        raise LedgerError("nonfinite native JSON value")

    try:
        return json.loads(
            value, parse_float=Decimal, parse_constant=invalid, object_pairs_hook=pairs
        )
    except (ValueError, UnicodeError, TypeError) as exc:
        raise LedgerError("invalid retained native JSON: " + str(exc)) from exc


def exact_time(value, unit):
    """No float, truncation or temporal projection in discrepancy authority."""
    if unit == "iso8601-utc":
        if not isinstance(value, str):
            raise LedgerError("native ISO timestamp must be text")
        import re

        match = re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d+))?(?:Z|\+00:00)", value)
        if match is None or len(match[1] or "") > 6 and any(c != "0" for c in match[1][6:]):
            raise LedgerError("unsupported_timestamp_precision")
        return _utc(datetime.fromisoformat(value))
    if isinstance(value, (bool, float)) or value is None:
        raise LedgerError("native timestamp requires exact text/integer")
    if unit == "unix-milliseconds" and type(value) is not int:
        raise LedgerError("native millisecond timestamp requires an integer")
    if unit not in ("unix-milliseconds", "unix-seconds-decimal"):
        raise LedgerError("unsupported native timestamp unit")
    try:
        amount = Decimal(value)
        # Decimal multiplication with an explicit sufficient precision avoids
        # accepting a rounded integral result for a long source timestamp.
        from decimal import localcontext

        with localcontext() as context:
            context.prec = max(28, len(amount.as_tuple().digits) + 10)
            micros = amount * (1000 if unit == "unix-milliseconds" else 1000000)
        if not micros.is_finite() or micros < 0 or micros != micros.to_integral_value():
            raise LedgerError("unsupported_timestamp_precision")
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=int(micros))
    except (ValueError, ArithmeticError, OverflowError) as exc:
        raise LedgerError("unsupported_timestamp_precision") from exc


class NativeArchive:
    def __init__(self, ledger, account_id, *, as_of):
        self.ledger, self.account_id, self.as_of = ledger, account_id, _utc(as_of)
        self.spec = ledger.spec(account_id)
        self.references = {}

    def table_exists(self, name):
        return (
            self.ledger.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
            is not None
        )

    def reference(self, kind, record_id, payload, *, intent_id=None):
        reference = RetainedSourceRef(
            store_kind=kind,
            account_id=self.account_id,
            record_id=record_id,
            intent_id=intent_id,
            payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        )
        self.references[(kind, intent_id or "", record_id)] = reference
        return reference

    def event(self, record_id, *, content_addressed=False):
        row = self.ledger.db.execute(
            "SELECT rowid,payload FROM events WHERE account=? AND id=?",
            (self.account_id, record_id),
        ).fetchone()
        if row is None:
            raise LedgerError("missing native source: " + record_id)
        value = decode(row["payload"])
        if content_addressed and record_id.rsplit(":", 1)[-1] != digest(value):
            raise LedgerError("retained native source content hash changed")
        return value, self.reference("ledger-event", record_id, row["payload"]), row["rowid"]

    def raw(self, record_id, venue, network):
        if not record_id.startswith(venue + "-http:"):
            raise LedgerError("wrong native raw source namespace")
        value, reference, sequence = self.event(record_id, content_addressed=True)
        at = _utc(datetime.fromisoformat(value["observed_at"]))
        if (
            value.get("complete") is not True
            or value.get("status") != 200
            or value.get("content_encoding") not in ("", "identity")
            or value.get("error")
            or at > self.as_of
            or at < self.spec.opened_at
        ):
            raise LedgerError(
                "native source incomplete/unsuccessful or outside original account cutoff"
            )
        if venue == "alpaca":
            if "response" in value:
                body = value["response"].encode()
            else:
                body = base64.b64decode(value["response_base64"], validate=True)
        else:
            expected = (
                "testnet.binance.vision"
                if network == "binance-testnet"
                else "api.binance.com"
                if venue == "binance"
                else "api.kraken.com"
            )
            endpoint = urlsplit(value["endpoint"])
            if (
                value.get("venue") != venue
                or endpoint.scheme != "https"
                or endpoint.netloc != expected
                or endpoint.query
                or endpoint.fragment
                or endpoint.username
                or endpoint.password
            ):
                raise LedgerError("native source endpoint/network binding differs")
            body = base64.b64decode(value["response_base64"], validate=True)
        if len(body) > 8 * 1024 * 1024:
            raise LedgerError("retained native source exceeds supported byte limit")
        return value, decode(body), reference, sequence

    def source_bindings(self):
        return tuple(
            sorted(("/".join(key), reference) for key, reference in self.references.items())
        )

    def verify_reference(self, reference):
        reference = RetainedSourceRef.model_validate(reference.model_dump())
        if reference.account_id != self.account_id:
            raise LedgerError("native reference belongs to another canonical account")
        if reference.store_kind == "ledger-event":
            value, actual, _ = self.event(reference.record_id)
        elif reference.store_kind == "venue-report":
            row = self.ledger.db.execute(
                "SELECT payload FROM venue_reports WHERE account=? AND intent=? AND evidence=?",
                (self.account_id, reference.intent_id, reference.record_id),
            ).fetchone()
            if row is None:
                raise LedgerError("original native report missing")
            value = decode(row[0])
            actual = self.reference(
                "venue-report", reference.record_id, row[0], intent_id=reference.intent_id
            )
        else:
            raise LedgerError("unsupported_external_valuation_closure")
        if actual != reference:
            raise LedgerError("native source reference differs from actual retained bytes")
        return value
