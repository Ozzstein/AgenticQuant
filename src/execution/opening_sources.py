"""One native print has one durable liquidity consumption across wire/feed versions.

Callers authenticate native identities against archived provider bytes. This
module binds those identities to immutable actual receipt and execution records;
it never guesses an alias from price or projected time.
"""

import json

from src.domain.common import NonEmptyString, Sha256, VersionedContract
from src.domain.portfolio import ExecutionBar, OpeningEvent, OpeningSourceIdentity
from src.execution.ledger import LedgerError


class OpeningSourceBinding(VersionedContract):
    identity: OpeningSourceIdentity
    event_id: NonEmptyString
    consumed_event_id: NonEmptyString
    evidence_id: NonEmptyString
    evidence_hash: Sha256


def consumed_event(ledger, account, event_id):
    row = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?",
        (account, "opening-event:" + event_id),
    ).fetchone()
    if row:
        event = OpeningEvent.model_validate_json(row[0])
    else:
        row = ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?", (account, "open:" + event_id)
        ).fetchone()
        if row is None:
            raise LedgerError("native print binding requires committed opening consumption")
        event = OpeningEvent.from_bar(ExecutionBar.model_validate_json(row[0]))
    if event.event_id != event_id:
        raise LedgerError("committed opening identity changed")
    return event


def event_binding(ledger, account, event_id):
    row = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?",
        (account, "opening-source-event:" + event_id),
    ).fetchone()
    if row is None:
        return None
    binding = OpeningSourceBinding.model_validate_json(row[0])
    if binding.event_id != event_id:
        raise LedgerError("native opening binding event identity changed")
    first = source_binding(ledger, account, binding.identity)
    if first is None or first.consumed_event_id != binding.consumed_event_id:
        raise LedgerError("native print canonical consumption binding changed")
    _verify_binding_receipt(ledger, account, binding)
    require_same_opening(
        consumed_event(ledger, account, binding.consumed_event_id),
        consumed_event(ledger, account, event_id),
    )
    return binding


def source_binding(ledger, account, identity):
    identity = OpeningSourceIdentity.model_validate(identity.model_dump())
    row = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?",
        (account, "opening-source:" + identity.content_hash()),
    ).fetchone()
    if row is None:
        return None
    binding = OpeningSourceBinding.model_validate_json(row[0])
    if binding.identity != identity or binding.event_id != binding.consumed_event_id:
        raise LedgerError("native print source identity changed")
    event = consumed_event(ledger, account, binding.consumed_event_id)
    require_identity(event, identity)
    _verify_binding_receipt(ledger, account, binding)
    return binding


def _verify_binding_receipt(ledger, account, binding):
    from src.execution.venue import ObservedOpenReceipt

    event = consumed_event(ledger, account, binding.event_id)
    row = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?",
        (account, "paper-observed-open:" + binding.event_id),
    ).fetchone()
    if row is None:
        raise LedgerError("native print binding lost its authoritative receipt")
    receipt = ObservedOpenReceipt.model_validate_json(row[0])
    if (
        receipt.account_id,
        receipt.event_id,
        receipt.evidence_id,
        receipt.evidence_hash,
        receipt.occurred_at,
    ) != (account, binding.event_id, binding.evidence_id, binding.evidence_hash, event.opened_at):
        raise LedgerError("native print binding differs from authoritative receipt")
    if receipt.event_hash != event.content_hash():
        legacy = ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (account, "open:" + binding.event_id),
        ).fetchone()
        legacy_bar = ExecutionBar.model_validate_json(legacy[0]) if legacy else None
        if (
            legacy_bar is None
            or legacy_bar.content_hash() != receipt.event_hash
            or OpeningEvent.from_bar(legacy_bar) != event
        ):
            raise LedgerError("native print binding opening content changed")


def require_identity(event, identity):
    if identity.instrument != event.instrument or identity.projected_opened_at() != event.opened_at:
        raise LedgerError(
            "native identity differs from instrument or declared exact timestamp projection"
        )


def require_same_opening(original, alias):
    if original.model_dump(exclude={"event_id"}) != alias.model_dump(exclude={"event_id"}):
        raise LedgerError("native print alias has different opening economics/calendar")


def unbound_receipts(ledger, account, instrument):
    receipts = []
    for row in ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id LIKE 'paper-observed-open:%' ORDER BY rowid",
        (account,),
    ):
        receipt = json.loads(row[0])
        event = consumed_event(ledger, account, receipt["event_id"])
        if event.instrument.instrument_id != instrument.instrument_id:
            continue
        if event.instrument != instrument:
            raise LedgerError("receipt instrument identity changed")
        if event_binding(ledger, account, event.event_id) is None:
            receipts.append(receipt)
    return receipts


def require_new_event_source(ledger, account, event, identity=None):
    """No new provider print can bypass unmapped receipt or bound-source history."""
    has_prior = False
    for row in ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id LIKE 'paper-observed-open:%'", (account,)
    ):
        receipt = json.loads(row[0])
        if receipt["event_id"] == event.event_id:
            continue
        old = consumed_event(ledger, account, receipt["event_id"])
        if old.instrument.instrument_id != event.instrument.instrument_id:
            continue
        has_prior = True
        if event_binding(ledger, account, old.event_id) is None:
            raise LedgerError(
                "unmapped opening receipt requires verified native-source migration before new print settlement"
            )
    if has_prior and identity is None:
        raise LedgerError(
            "source-bound opening history requires the verified native print identity"
        )
    if identity is not None:
        require_identity(event, identity)


def bind(ledger, account, identity, *, event_id, evidence_id, evidence_hash):
    identity = OpeningSourceIdentity.model_validate(identity.model_dump())
    with ledger.atomic():
        event = consumed_event(ledger, account, event_id)
        require_identity(event, identity)
        row = ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (account, "paper-observed-open:" + event_id),
        ).fetchone()
        if row is None:
            raise LedgerError("native print binding requires its committed observed receipt")
        receipt = json.loads(row[0])
        if (
            receipt["evidence_id"] != evidence_id
            or receipt["evidence_hash"] != evidence_hash
            or receipt["account_id"] != account
        ):
            raise LedgerError("native print proof differs from its committed receipt evidence")
        if receipt["event_hash"] != event.content_hash():
            legacy = ledger.db.execute(
                "SELECT payload FROM events WHERE account=? AND id=?", (account, "open:" + event_id)
            ).fetchone()
            legacy_bar = ExecutionBar.model_validate_json(legacy[0]) if legacy else None
            if (
                legacy_bar is None
                or legacy_bar.content_hash() != receipt["event_hash"]
                or OpeningEvent.from_bar(legacy_bar) != event
            ):
                raise LedgerError("native print receipt content differs from committed opening")
        first = source_binding(ledger, account, identity)
        consumed_id = event_id if first is None else first.consumed_event_id
        if consumed_id != event_id:
            require_same_opening(consumed_event(ledger, account, consumed_id), event)
            if any(fill.event_id == event_id for fill in ledger.fills(account)):
                raise LedgerError(
                    "multiple aliases already filled the same native print; explicit reconciliation required"
                )
        binding = OpeningSourceBinding(
            identity=identity,
            event_id=event_id,
            consumed_event_id=consumed_id,
            evidence_id=evidence_id,
            evidence_hash=evidence_hash,
        )
        _verify_binding_receipt(ledger, account, binding)
        if first is None:
            ledger.record_event(
                account, "opening-source:" + identity.content_hash(), binding.model_dump_json()
            )
        ledger.record_event(account, "opening-source-event:" + event_id, binding.model_dump_json())
        return binding
