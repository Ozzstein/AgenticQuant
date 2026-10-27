"""Immutable historical sleeve receipts over the canonical economic journal.

Only the original source prefix is examined. Reading an observation neither
values current holdings nor runs sync, and is valid on a read-only archive.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal as D

from src.domain.portfolio import Fill, OrderIntent, PortfolioSnapshot, SleeveSnapshot
from src.execution.fill_arithmetic import FillArithmetic
from src.execution.ledger import LedgerError
from src.execution.sleeves import digest, exact_sum

IMPLEMENTATION = "historical-sleeve-observation-1"


def _events(book, last=None):
    return list(
        book.ledger.db.execute(
            "SELECT rowid AS source_rowid,* FROM sleeve_events WHERE account=? "
            "AND (? IS NULL OR rowid<=?) ORDER BY rowid",
            (book.account_id, last, last),
        )
    )


def _journals(book, last=None):
    return list(
        book.ledger.db.execute(
            "SELECT * FROM journal WHERE account=? AND (? IS NULL OR sequence<=?) ORDER BY sequence",
            (book.account_id, last, last),
        )
    )


def _prefix(rows, field):
    return {
        "count": len(rows),
        "last": rows[-1][field] if rows else 0,
        "sha256": digest([dict(row) for row in rows]),
    }


def _pending_hash(pending):
    return digest(
        [
            {"intent": row["intent"].model_dump(mode="json"), "reserved": str(row["reserved"])}
            for row in pending
        ]
    )


def _historical_pending(book, journals):
    """Verify historical reservations from original order/fill/cancel receipts."""
    orders = {}
    for row in journals:
        if row["kind"] == "order":
            intent = OrderIntent.model_validate_json(row["payload"])
            if (
                intent.account_id != book.account_id
                or row["id"] != "order:" + intent.intent_id
                or book.ledger.order(book.account_id, intent.intent_id)["intent"] != intent
            ):
                raise LedgerError("historical sleeve order source differs")
            reserved = (
                (intent.quantity * intent.reference_price * intent.fx_rate + intent.estimated_fee)
                if intent.side == "buy"
                else D(0)
            )
            orders[intent.intent_id] = {"intent": intent, "filled": D(0), "reserved": reserved}
        elif row["kind"] == "fill":
            fill = Fill.model_validate_json(row["payload"])
            order = orders.get(fill.intent_id)
            if order is None:
                raise LedgerError("historical sleeve fill lacks its original open order")
            arithmetic = FillArithmetic(fill)
            remaining = exact_sum([order["intent"].quantity, order["filled"].copy_negate()])
            if not 0 < fill.quantity <= remaining:
                raise LedgerError("historical sleeve fill exceeds original quantity")
            order["reserved"] = (
                order["reserved"] * arithmetic.subtract(remaining, fill.quantity) / remaining
            )
            order["filled"] = arithmetic.add(order["filled"], fill.quantity)
            if order["filled"] == order["intent"].quantity:
                del orders[fill.intent_id]
        elif row["id"].startswith("cancel:"):
            if row["kind"] not in ("cancelled", "rejected"):
                raise LedgerError("historical sleeve cancellation kind differs")
            if orders.pop(row["id"][len("cancel:") :], None) is None:
                raise LedgerError("historical sleeve cancellation lacks its original open order")
    return list(orders.values())


def _allocation_sources(book, state, pending):
    ids = {row["intent"].intent_id for row in pending}
    ids.update(
        json.loads(row["payload"])["intent_id"] for row in state["rows"] if row["kind"] == "fill"
    )
    return {identity: digest(book._intent(identity)) for identity in sorted(ids)}


def persist_observation(book, canonical, state, pending, history, output):
    """Called in the publication's existing canonical SQLite transaction."""
    source_hash = next(iter(output.values())).source_hash
    at = canonical.as_of.isoformat()
    old = book.ledger.db.execute(
        "SELECT 1 FROM sleeve_snapshot_receipts WHERE account=? AND at=? AND source_hash=?",
        (book.account_id, at, source_hash),
    ).fetchone()
    if old:
        if read_observation(book, canonical.as_of, source_hash=source_hash) != output:
            raise LedgerError("historical sleeve observation cannot be rewritten")
        return
    events, journals = _events(book), _journals(book)
    base = book.ledger.db.execute(
        "SELECT payload FROM snapshots WHERE account=? AND at=?", (book.account_id, at)
    ).fetchone()
    published_base_hash = (
        canonical.content_hash()
        if base is not None and PortfolioSnapshot.model_validate_json(base[0]) == canonical
        else None
    )
    payload = {
        "implementation": IMPLEMENTATION,
        "account_id": book.account_id,
        "at": at,
        "source_hash": source_hash,
        "canonical_snapshot": canonical.model_dump(mode="json"),
        "canonical_snapshot_hash": canonical.content_hash(),
        "published_base_hash": published_base_hash,
        "event_prefix": _prefix(events, "source_rowid"),
        "journal_prefix": _prefix(journals, "sequence"),
        "allocation_sources": _allocation_sources(book, state, pending),
        "pending_hash": _pending_hash(pending),
        "history": [dict(row) for row in history],
        "sleeves": {key: value.model_dump(mode="json") for key, value in output.items()},
    }
    book.ledger.db.execute(
        "INSERT INTO sleeve_snapshot_receipts VALUES (?,?,?,?,?)",
        (
            book.account_id,
            at,
            source_hash,
            json.dumps(payload, sort_keys=True, allow_nan=False),
            digest(payload),
        ),
    )
    for name, part in output.items():
        values = (
            book.account_id,
            name,
            at,
            source_hash,
            str(part.portfolio.nav),
            str(part.net_capital_flow),
        )
        old_index = book.ledger.db.execute(
            "SELECT * FROM sleeve_observations WHERE account=? AND sleeve=? AND at=? AND source_hash=?",
            values[:4],
        ).fetchone()
        if old_index is not None and tuple(old_index) != values:
            raise LedgerError("historical sleeve observation index differs")
        book.ledger.db.execute(
            "INSERT OR IGNORE INTO sleeve_observations VALUES (?,?,?,?,?,?)", values
        )


def read_observation(book, at, *, source_hash=None):
    with book.ledger._reading():
        exists = book.ledger.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sleeve_snapshot_receipts'"
        ).fetchone()
        if exists is None:
            return None
        rows = book.ledger.db.execute(
            "SELECT * FROM sleeve_snapshot_receipts WHERE account=? AND at=? "
            "AND (? IS NULL OR source_hash=?)",
            (book.account_id, at.isoformat(), source_hash, source_hash),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise LedgerError(
                "multiple historical sleeve observations require explicit source_hash"
            )
        return _verify(book, rows[0], {}, set())


def _verify(book, receipt, cache, visiting):
    key = (receipt["at"], receipt["source_hash"])
    if key in cache:
        return cache[key]
    if key in visiting:
        raise LedgerError("historical sleeve observation history contains a cycle")
    visiting.add(key)
    payload = json.loads(receipt["payload"])
    if (
        digest(payload) != receipt["receipt_hash"]
        or payload["implementation"] != IMPLEMENTATION
        or payload["account_id"] != book.account_id
        or (payload["at"], payload["source_hash"]) != key
    ):
        raise LedgerError("historical sleeve receipt identity/hash differs")
    at = datetime.fromisoformat(receipt["at"])
    canonical = PortfolioSnapshot.model_validate(payload["canonical_snapshot"])
    if (
        canonical.account_id != book.account_id
        or canonical.as_of != at
        or canonical.sleeves
        or canonical.content_hash() != payload["canonical_snapshot_hash"]
    ):
        raise LedgerError("historical sleeve canonical snapshot binding differs")
    saved = book.ledger.db.execute(
        "SELECT payload FROM snapshots WHERE account=? AND at=?", (book.account_id, receipt["at"])
    ).fetchone()
    if payload["published_base_hash"] is not None:
        if (
            saved is None
            or payload["published_base_hash"] != canonical.content_hash()
            or PortfolioSnapshot.model_validate_json(saved[0]) != canonical
        ):
            raise LedgerError("historical sleeve canonical NAV receipt differs")
    events = _events(book, payload["event_prefix"]["last"])
    journals = _journals(book, payload["journal_prefix"]["last"])
    if (
        _prefix(events, "source_rowid") != payload["event_prefix"]
        or _prefix(journals, "sequence") != payload["journal_prefix"]
        or any(datetime.fromisoformat(row["at"]) > at for row in [*events, *journals])
    ):
        raise LedgerError("historical sleeve source/journal prefix changed")
    source = [
        {
            "id": row["id"],
            "at": row["at"],
            "kind": row["kind"],
            "payload": json.loads(row["payload"]),
        }
        for row in events
    ]
    if (
        digest({"events": source, "canonical_snapshot": canonical.model_dump(mode="json")})
        != key[1]
    ):
        raise LedgerError("historical sleeve source hash differs")
    contexts = book._source_context(journals)
    state = book._state(rows=events, contexts=contexts)
    pending = _historical_pending(book, journals)
    if (
        _pending_hash(pending) != payload["pending_hash"]
        or _allocation_sources(book, state, pending) != payload["allocation_sources"]
    ):
        raise LedgerError("historical sleeve pending/allocation sources differ")
    _canonical_parts(canonical, state, pending)
    sequence = book.ledger.db.execute(
        "SELECT rowid FROM sleeve_snapshot_receipts WHERE account=? AND at=? AND source_hash=?",
        (book.account_id, receipt["at"], receipt["source_hash"]),
    ).fetchone()[0]
    original_history = book.ledger.db.execute(
        "SELECT o.* FROM sleeve_observations o LEFT JOIN sleeve_snapshot_receipts p "
        "ON p.account=o.account AND p.at=o.at AND p.source_hash=o.source_hash "
        "WHERE o.account=? AND o.at<=? AND (p.rowid<? OR p.rowid IS NULL) ORDER BY o.rowid",
        (book.account_id, receipt["at"], sequence),
    ).fetchall()
    if [dict(row) for row in original_history] != payload["history"]:
        raise LedgerError("historical sleeve high-water source prefix differs")
    for history in payload["history"]:
        if history["account"] != book.account_id or datetime.fromisoformat(history["at"]) > at:
            raise LedgerError("historical sleeve high-water input has another account/future time")
        prior = book.ledger.db.execute(
            "SELECT * FROM sleeve_snapshot_receipts WHERE account=? AND at=? AND source_hash=?",
            (book.account_id, history["at"], history["source_hash"]),
        ).fetchone()
        if prior is None:
            raise LedgerError("historical sleeve high-water input lacks its original full receipt")
        parts = _verify(book, prior, cache, visiting)
        part = parts.get(history["sleeve"])
        if part is None or (str(part.portfolio.nav), str(part.net_capital_flow)) != (
            history["nav"],
            history["flow"],
        ):
            raise LedgerError("historical sleeve high-water observation differs")
    actual = book._project_snapshot(at, state, canonical, pending, payload["history"])
    retained = {
        name: SleeveSnapshot.model_validate(value) for name, value in payload["sleeves"].items()
    }
    if retained != actual:
        raise LedgerError("historical sleeve full snapshot differs from original projection")
    indexed = book.ledger.db.execute(
        "SELECT * FROM sleeve_observations WHERE account=? AND at=? AND source_hash=? ORDER BY sleeve",
        (book.account_id, receipt["at"], receipt["source_hash"]),
    ).fetchall()
    expected = [
        (
            book.account_id,
            name,
            receipt["at"],
            receipt["source_hash"],
            str(value.portfolio.nav),
            str(value.net_capital_flow),
        )
        for name, value in sorted(actual.items())
    ]
    if [tuple(row) for row in indexed] != expected:
        raise LedgerError("historical sleeve NAV/flow observation index differs")
    visiting.remove(key)
    cache[key] = actual
    return actual


def _canonical_parts(canonical, state, pending):
    for field, actual in (
        ("cash", canonical.cash),
        ("fees", canonical.fees),
        ("realized", canonical.realized_pnl),
        ("taxes", canonical.withholding_taxes),
    ):
        if exact_sum(state[field].values()) != actual:
            raise LedgerError("historical sleeve canonical " + field + " differs")
    reserved = canonical.cash - exact_sum(
        [canonical.cash, *(row["reserved"].copy_negate() for row in pending)]
    )
    if reserved != canonical.reserved_cash:
        raise LedgerError("historical sleeve canonical reservation differs")
    held = {}
    for positions in state["quantities"].values():
        for key, quantity in positions.items():
            held[key] = exact_sum([held.get(key, D(0)), quantity])
    if {key: qty for key, qty in held.items() if qty} != {
        held.instrument.instrument_id: held.quantity for held in canonical.positions
    }:
        raise LedgerError("historical sleeve canonical ownership differs")
    receivables = {
        key: exact_sum(amounts.values()) for key, amounts in state["receivables"].items()
    }
    if {key: value for key, value in receivables.items() if value} != {
        entry.entitlement.action_id: entry.outstanding for entry in canonical.receivables
    }:
        raise LedgerError("historical sleeve canonical entitlement differs")
