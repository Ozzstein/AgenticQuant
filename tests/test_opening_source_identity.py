"""Native source identity/migration fixtures; archived source authentication is external."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from src.domain.portfolio import OpeningEvent, OpeningSourceIdentity
from src.execution.ledger import LedgerError
from tests.test_durable_accounting import A, T, bar, metadata
from tests.test_venue_execution import setup


def identity(event, *, trade_id="X:42"):
    delta = event.opened_at - datetime(1970, 1, 1, tzinfo=timezone.utc)
    ns = ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000
    return OpeningSourceIdentity(
        provider="alpaca", instrument=event.instrument, trade_id=trade_id, wire_timestamp=str(ns)
    )


@pytest.mark.parametrize("invalid", [" 1", "1 ", "+1", "01", "-0", "1.0", "1e3", "١", 1, 1.0])
def test_native_timestamp_never_normalizes_ambiguous_lexical_forms(invalid):
    event = OpeningEvent.from_bar(bar())
    raw = identity(event).model_dump()
    with pytest.raises(ValidationError):
        OpeningSourceIdentity.model_validate({**raw, "wire_timestamp": invalid})


def test_exact_nanoseconds_and_provider_native_id_fields():
    event = OpeningEvent.from_bar(bar())
    source = identity(event)
    ns = int(source.wire_timestamp)
    partial = OpeningSourceIdentity.model_validate(
        {**source.model_dump(), "wire_timestamp": str(ns - 1)}
    )
    assert partial.projected_opened_at() == event.opened_at
    assert partial != source and partial.content_hash() != source.content_hash()
    assert OpeningSourceIdentity.model_validate_json(source.model_dump_json()) == source
    for provider, trade_id in (("kraken", "42"), ("binance", "42"), ("alpaca", "X:42")):
        assert (
            OpeningSourceIdentity.model_validate(
                {**source.model_dump(), "provider": provider, "trade_id": trade_id}
            ).trade_id
            == trade_id
        )
    for provider, trade_id in (
        ("alpaca", "42"),
        ("alpaca", "X:042"),
        ("kraken", "X:42"),
        ("binance", "+42"),
        ("kraken", " 42"),
    ):
        with pytest.raises(ValidationError):
            OpeningSourceIdentity.model_validate(
                {**source.model_dump(), "provider": provider, "trade_id": trade_id}
            )


def initial_print(tmp_path, *, sourced=False):
    ledger, adapter, execution, order = setup(tmp_path)
    execution.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    execution.dispatch(order.intent_id, as_of=order.eligible_at)
    completed = bar(at=order.eligible_at + timedelta(seconds=1), opening_volume="500")
    event = OpeningEvent.from_bar(completed)
    clocks = dict(
        observed_at=completed.closed_at,
        available_at=completed.closed_at,
        as_of=completed.closed_at,
        evidence_id="v1-envelope",
        evidence_hash="1" * 64,
    )
    frozen = {order.intent_id: execution.admission_metadata(order.intent_id)}
    fills = adapter.settle_open(
        completed, frozen, **clocks, **({"source_identity": identity(event)} if sourced else {})
    )
    assert len(fills) == 1 and fills[0].quantity == 5
    return ledger, adapter, execution, order, event, completed, clocks, frozen


def test_v1_v2_alias_fails_closed_until_exact_receipt_migration_then_preserves_capacity(tmp_path):
    ledger, adapter, execution, order, event, completed, clocks, frozen = initial_print(tmp_path)
    # Retain the old full-bar receipt format as an actual pre-migration archive.
    receipt = adapter.observed_open_receipt(event.event_id).model_copy(
        update={"event_hash": completed.content_hash()}
    )
    ledger.db.execute(
        "UPDATE events SET payload=? WHERE id=?",
        (receipt.model_dump_json(), "paper-observed-open:" + event.event_id),
    )
    ledger.db.execute("DELETE FROM events WHERE id=?", ("opening-event:" + event.event_id,))
    alias = event.model_copy(update={"event_id": "operating-v2-new-hash"})
    native = identity(event)
    new_clocks = {**clocks, "evidence_id": "v2-envelope", "evidence_hash": "2" * 64}
    for native_kwargs in ({}, {"source_identity": native}):
        with pytest.raises(LedgerError, match="unmapped opening receipt"):
            adapter.settle_open(alias, frozen, **new_clocks, **native_kwargs)
    with pytest.raises(LedgerError, match="unmapped opening receipt"):
        adapter.engine.open_event(alias, {A.instrument_id: metadata()})
    assert adapter.observed_open_receipt(alias.event_id) is None
    assert adapter.unbound_opening_receipts(A) == (receipt,)
    with pytest.raises(LedgerError, match="receipt evidence"):
        adapter.bind_opening_source(
            native,
            event_id=event.event_id,
            evidence_id=clocks["evidence_id"],
            evidence_hash="f" * 64,
        )
    binding = adapter.bind_opening_source(
        native,
        event_id=event.event_id,
        evidence_id=clocks["evidence_id"],
        evidence_hash=clocks["evidence_hash"],
    )
    assert binding.consumed_event_id == event.event_id
    assert adapter.unbound_opening_receipts(A) == ()
    assert adapter.settle_open(alias, frozen, **new_clocks, source_identity=native) == []
    assert adapter.opening_source_binding(alias.event_id).consumed_event_id == event.event_id
    assert ledger.order(order.account_id, order.intent_id)["remaining"] == 5
    assert len(ledger.fills(order.account_id)) == 1
    ledger.close()
    ledger, adapter, execution, _ = setup(tmp_path)
    assert adapter.settle_open(alias, frozen, **new_clocks, source_identity=native) == []
    later = OpeningEvent.from_bar(bar(at=event.opened_at + timedelta(days=1), opening_volume="500"))
    later_clocks = {
        **new_clocks,
        "observed_at": later.opened_at,
        "available_at": later.opened_at,
        "as_of": later.opened_at,
        "evidence_id": "next-day-native-ID-reuse",
        "evidence_hash": "3" * 64,
    }
    # Same native trade number on a different real day is a different source identity.
    assert (
        len(adapter.settle_open(later, frozen, **later_clocks, source_identity=identity(later)))
        == 1
    )
    assert ledger.quantity(order.account_id, A.instrument_id) == 10
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_source_omission_projection_price_and_identity_substitution_cannot_bypass_binding(tmp_path):
    ledger, adapter, execution, order, event, completed, clocks, frozen = initial_print(
        tmp_path, sourced=True
    )
    alias = event.model_copy(update={"event_id": "other-feed-hash"})
    native = identity(event)
    new_clocks = {**clocks, "evidence_id": "other-feed", "evidence_hash": "4" * 64}
    with pytest.raises(LedgerError, match="verified native print identity"):
        adapter.settle_open(alias, frozen, **new_clocks)
    with pytest.raises(LedgerError, match="verified native print identity"):
        adapter.engine.open_event(alias, {A.instrument_id: metadata()})
    with pytest.raises(LedgerError, match="different opening economics"):
        adapter.settle_open(
            alias.model_copy(update={"price": D(101)}), frozen, **new_clocks, source_identity=native
        )
    with pytest.raises(LedgerError, match="timestamp projection"):
        adapter.settle_open(
            alias,
            frozen,
            **new_clocks,
            source_identity=native.model_copy(
                update={"wire_timestamp": str(int(native.wire_timestamp) + 1000)}
            ),
        )
    with pytest.raises(LedgerError, match="conflicting event retry"):
        adapter.bind_opening_source(
            identity(event, trade_id="X:43"),
            event_id=event.event_id,
            evidence_id=clocks["evidence_id"],
            evidence_hash=clocks["evidence_hash"],
        )
    adapter.advance_close(completed)
    published = ledger.snapshot(order.account_id, completed.closed_at, 0)
    # Verified alias is a documentary no-op, even after later published valuation.
    assert adapter.settle_open(alias, frozen, **new_clocks, source_identity=native) == []
    assert ledger.history(order.account_id)[-1] == published
    assert len(ledger.fills(order.account_id)) == 1
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_authoritative_binding_reader_checks_receipt_bytes_before_alias_retry(tmp_path):
    ledger, adapter, execution, order, event, completed, clocks, frozen = initial_print(
        tmp_path, sourced=True
    )
    receipt = adapter.observed_open_receipt(event.event_id)
    damaged = receipt.model_copy(update={"evidence_hash": "a" * 64})
    ledger.db.execute(
        "UPDATE events SET payload=? WHERE account=? AND id=?",
        (damaged.model_dump_json(), order.account_id, "paper-observed-open:" + event.event_id),
    )
    with pytest.raises(LedgerError, match="authoritative receipt"):
        adapter.opening_source_binding(event.event_id)
    alias = event.model_copy(update={"event_id": "tampered-source-alias"})
    with pytest.raises(LedgerError, match="authoritative receipt"):
        adapter.settle_open(alias, frozen, **clocks, source_identity=identity(event))
    assert adapter.observed_open_receipt(alias.event_id) is None
    assert len(ledger.fills(order.account_id)) == 1
    ledger.close()


def test_distinct_actual_nanoseconds_at_same_projected_clock_keep_separate_liquidity(tmp_path):
    ledger, adapter, execution, order, event, completed, clocks, frozen = initial_print(
        tmp_path, sourced=True
    )
    native = identity(event)
    second_native = native.model_copy(
        update={"wire_timestamp": str(int(native.wire_timestamp) - 1)}
    )
    second = event.model_copy(update={"event_id": "different-real-nanosecond"})
    assert second_native.projected_opened_at() == event.opened_at
    # Only authenticated native IDs determine aliasing; equal price and projected
    # time must not suppress a distinct upstream print's actual capacity.
    assert len(adapter.settle_open(second, frozen, **clocks, source_identity=second_native)) == 1
    assert ledger.quantity(order.account_id, A.instrument_id) == 10
    assert len(ledger.fills(order.account_id)) == 2
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_legacy_receipt_cannot_mask_changed_canonical_opening_projection(tmp_path):
    ledger, adapter, execution, order, event, completed, clocks, frozen = initial_print(tmp_path)
    receipt = adapter.observed_open_receipt(event.event_id).model_copy(
        update={"event_hash": completed.content_hash()}
    )
    ledger.db.execute(
        "UPDATE events SET payload=? WHERE account=? AND id=?",
        (receipt.model_dump_json(), order.account_id, "paper-observed-open:" + event.event_id),
    )
    adapter.bind_opening_source(
        identity(event),
        event_id=event.event_id,
        evidence_id=clocks["evidence_id"],
        evidence_hash=clocks["evidence_hash"],
    )
    ledger.db.execute(
        "UPDATE events SET payload=? WHERE account=? AND id=?",
        (
            event.model_copy(update={"opening_volume": D(1000)}).model_dump_json(),
            order.account_id,
            "opening-event:" + event.event_id,
        ),
    )
    with pytest.raises(LedgerError, match="opening content changed"):
        adapter.opening_source_binding(event.event_id)
    ledger.close()


def test_fully_consumed_order_alias_recovers_empty_metadata_map_and_never_changes_nav(tmp_path):
    ledger, adapter, execution, order = setup(tmp_path)
    execution.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    execution.dispatch(order.intent_id, as_of=order.eligible_at)
    completed = bar(at=order.eligible_at + timedelta(seconds=1), opening_volume="1000")
    event = OpeningEvent.from_bar(completed)
    native = identity(event)
    clocks = dict(
        observed_at=completed.closed_at,
        available_at=completed.closed_at,
        as_of=completed.closed_at,
        evidence_id="complete-v1",
        evidence_hash="1" * 64,
    )
    frozen = {order.intent_id: execution.admission_metadata(order.intent_id)}
    assert len(adapter.settle_open(event, frozen, **clocks, source_identity=native)) == 1
    assert ledger.pending(order.account_id) == []
    adapter.advance_close(completed)
    snapshot = ledger.snapshot(order.account_id, completed.closed_at, 0)
    alias = event.model_copy(update={"event_id": "complete-v2"})
    assert adapter.settle_open(alias, {}, **clocks, source_identity=native) == []
    receipt = adapter.observed_open_receipt(alias.event_id)
    assert receipt.admission_metadata_hashes == ()
    ledger.close()
    ledger, adapter, execution, _ = setup(tmp_path)
    retry_map = {
        intent_id: execution.admission_metadata(intent_id)
        for intent_id, _ in adapter.observed_open_receipt(alias.event_id).admission_metadata_hashes
    }
    assert adapter.settle_open(alias, retry_map, **clocks, source_identity=native) == []
    assert ledger.history(order.account_id)[-1] == snapshot
    assert len(ledger.fills(order.account_id)) == 1
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()
