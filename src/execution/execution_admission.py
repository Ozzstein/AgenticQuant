"""Current compliance gates and the internal pre-transmission outbox state."""

import hashlib
from datetime import datetime
from decimal import Decimal

from src.domain.common import _utc
from src.domain.portfolio import InstrumentRisk, OrderIntent
from src.execution.ledger import LedgerError
from src.execution.native_archive import canonical, decode, digest


def _start_payload(ledger, account, intent_id, at):
    row = ledger.db.execute(
        "SELECT * FROM venue_outbox WHERE account=? AND intent=?", (account, intent_id)
    ).fetchone()
    book = ledger.db.execute("SELECT * FROM execution_books WHERE account=?", (account,)).fetchone()
    intent = ledger.order(account, intent_id)["intent"]
    return {
        "implementation": "venue-submission-start-1",
        "account_id": account,
        "intent_id": intent_id,
        "intent_hash": intent.content_hash(),
        "client_order_id": row["client_id"],
        "venue": row["venue"],
        "owner": book["owner"],
        "execution_book_hash": digest(dict(book)),
        "rules_hash": hashlib.sha256(row["rules"].encode()).hexdigest(),
        "at": _utc(at).isoformat(),
    }


def _record_submission_start(ledger, account, intent_id, at):
    if not ledger.db.in_transaction:
        raise LedgerError("submission start requires the original dispatch transaction")
    ledger.record_event(
        account,
        "venue-submission-start:" + intent_id,
        canonical(_start_payload(ledger, account, intent_id, at)),
    )


def _record_submission_claimed(ledger, account, intent_id):
    if not ledger.db.in_transaction:
        raise LedgerError("submission claim requires the original native transaction")
    start = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?",
        (account, "venue-submission-start:" + intent_id),
    ).fetchone()
    if start is None:
        raise LedgerError("native claim lacks original dispatcher source")
    value = decode(start[0])
    venue = "lean" if value["owner"] == "lean" else value["venue"]
    attempt_id = venue + "-submit-attempt:" + intent_id
    attempt = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?", (account, attempt_id)
    ).fetchone()
    if attempt is None:
        raise LedgerError("native claim lacks its same-transaction original attempt")
    ledger.record_event(
        account,
        "venue-submission-claimed:" + intent_id,
        canonical(
            {
                "implementation": "venue-submission-claimed-1",
                "intent_hash": value["intent_hash"],
                "start_hash": hashlib.sha256(start[0].encode()).hexdigest(),
                "attempt_id": attempt_id,
                "attempt_hash": hashlib.sha256(attempt[0].encode()).hexdigest(),
                "at": decode(attempt[0])["at"],
            }
        ),
    )


def _read_submission_sources(archive, intent, claim):
    """Reexport new source pairs; original complete legacy reports remain readable."""
    ledger, account = archive.ledger, archive.account_id
    start_id, claimed_id = (
        "venue-submission-start:" + intent.intent_id,
        "venue-submission-claimed:" + intent.intent_id,
    )
    rows = ledger.db.execute(
        "SELECT id,payload FROM events WHERE account=? AND id IN (?,?)",
        (account, start_id, claimed_id),
    ).fetchall()
    if not rows:
        return
    if len(rows) != 2:
        raise LedgerError("native submission causal source pair incomplete")
    start, start_ref, _ = archive.event(start_id)
    claimed, _, _ = archive.event(claimed_id)
    at = _utc(datetime.fromisoformat(start["at"]))
    venue = "lean" if start["owner"] == "lean" else start["venue"]
    attempt_id = venue + "-submit-attempt:" + intent.intent_id
    original, attempt_ref, _ = archive.event(attempt_id)
    if (
        start != _start_payload(ledger, account, intent.intent_id, at)
        or not intent.eligible_at <= at <= _utc(datetime.fromisoformat(claim["at"]))
        or (intent.expires_at is not None and at >= intent.expires_at)
    ):
        raise LedgerError("native submission start identity/clock differs")
    if original != claim or claimed != {
        "implementation": "venue-submission-claimed-1",
        "intent_hash": intent.content_hash(),
        "start_hash": start_ref.payload_sha256,
        "attempt_id": attempt_id,
        "attempt_hash": attempt_ref.payload_sha256,
        "at": claim["at"],
    }:
        raise LedgerError("native submission claimed source differs from original attempt")


def _unstarted_submission(archive, row):
    """Prove the exact dispatcher state before a first native claim/transmission.

    No caller can select an excluded case, receipt or reason. Any retained
    attempt/report/native transmission for this identity makes it unresolved.
    The surrounding audit still verifies every original account/source fact.
    """
    from src.execution.venue import VenueExecution, VenueRules

    ledger, account = archive.ledger, archive.account_id
    order = ledger.order(account, row["intent"])
    intent = order["intent"]
    book = ledger.db.execute("SELECT * FROM execution_books WHERE account=?", (account,)).fetchone()
    original = ledger.db.execute(
        "SELECT payload FROM journal WHERE account=? AND id=? AND kind='order'",
        (account, "order:" + intent.intent_id),
    ).fetchone()
    if (
        row["state"] != "submitting"
        or row["venue_id"] is not None
        or row["cumulative"] != "0"
        or order["status"] != "pending"
        or order["filled"] != 0
        or original is None
        or OrderIntent.model_validate_json(original[0]) != intent
        or row["client_id"] != VenueExecution.client_id(account, intent.intent_id)
        or book is None
        or book["owner"] not in ("direct", "lean")
    ):
        return False
    reserved = (
        intent.quantity * intent.reference_price * intent.fx_rate + intent.estimated_fee
        if intent.side == "buy"
        else Decimal(0)
    )
    if order["reserved"] != reserved or ledger.available_cash(account) < 0:
        return False
    if intent.side == "sell" and sum(
        (
            value["remaining"]
            for value in ledger.pending(account)
            if value["intent"].side == "sell" and value["intent"].instrument == intent.instrument
        ),
        Decimal(0),
    ) > ledger.quantity(account, intent.instrument.instrument_id):
        return False
    start, _, _ = archive.event("venue-submission-start:" + intent.intent_id)
    at = _utc(datetime.fromisoformat(start["at"]))
    if (
        start != _start_payload(ledger, account, intent.intent_id, at)
        or at < intent.eligible_at
        or (intent.expires_at is not None and at >= intent.expires_at)
    ):
        return False
    if ledger.db.execute(
        "SELECT 1 FROM events WHERE account=? AND id=?",
        (account, "venue-submission-claimed:" + intent.intent_id),
    ).fetchone():
        return False
    configuration = decode(book["configuration"])
    rules = VenueRules.model_validate_json(row["rules"])
    if (
        configuration["routes"].get(intent.instrument.instrument_id) != row["venue"]
        or rules.venue != row["venue"]
        or rules.instrument != intent.instrument
    ):
        return False
    metadata, _, _ = archive.event("venue-admission-metadata:" + intent.intent_id)
    risk = InstrumentRisk.model_validate(metadata[intent.instrument.instrument_id])
    if risk.instrument != intent.instrument or risk.as_of > intent.decision_at:
        return False
    if (
        ledger.db.execute(
            "SELECT 1 FROM venue_reports WHERE account=? AND intent=?", (account, intent.intent_id)
        ).fetchone()
        or ledger.db.execute(
            "SELECT 1 FROM fills WHERE account=? AND intent=?", (account, intent.intent_id)
        ).fetchone()
    ):
        return False
    client = row["client_id"]
    for record in ledger.db.execute(
        "SELECT id,payload FROM events WHERE account=? ORDER BY rowid", (account,)
    ):
        identity = record["id"]
        if identity in tuple(
            venue + "-submit-attempt:" + intent.intent_id
            for venue in ("alpaca", "binance", "kraken", "lean")
        ):
            return False
        if identity.startswith(("alpaca-http:", "binance-http:", "kraken-http:")):
            value = decode(record["payload"])
            if value.get("method") != "POST":
                continue
            submitted = value.get("request") or value.get("parameters") or {}
            if any(
                submitted.get(name) == client
                for name in ("client_order_id", "newClientOrderId", "cl_ord_id")
            ):
                return False
        elif identity.startswith(("lean-ticket:", "lean-native-order:")):
            if decode(record["payload"]).get("client_order_id") == client:
                return False
    return True


def require_execution_admission(ledger, account_id):
    """Uncached current gate, including inside the final native claim transaction."""
    audit = ledger.execution_compliance(account_id)
    if audit.status != "PASS":
        raise LedgerError(
            "execution compliance FAIL: "
            + ",".join(audit.blocking_codes or ("retained_admission_violation",))
            + "; authority="
            + audit.content_hash()
        )
    return audit
