"""Actual canonical dispatcher/adapter transactions, with offline native protocols."""

import sqlite3
from datetime import timedelta

import pytest

from src.execution.ledger import LedgerError
from src.execution.venue import SubmissionNotAttempted
from tests.test_durable_accounting import T, metadata
from tests.test_execution_compliance import rows
from tests.test_native_authority import alpaca_receipt, readonly_proof


def corrupt_retained_authority(ledger, account):
    # Negative authority fixture, never a substituted successful proof callback.
    with ledger.atomic():
        ledger.record_event(account, "execution-violation:" + "f" * 64, "{}")


def setup_native(path, venue):
    from tests import test_alpaca_venue, test_binance_venue, test_kraken_venue, test_lean_execution

    if venue == "lean":
        runtime, engine, adapter, order, wire, native = test_lean_execution.setup_live(path)
        adapter.capture()
        return runtime, engine, adapter, order, wire, native
    fixtures = {
        "alpaca": test_alpaca_venue,
        "binance": test_binance_venue,
        "kraken": test_kraken_venue,
    }[venue]
    wire, engine, adapter, order = fixtures.setup(path)
    adapter.capture()
    return None, engine, adapter, order, wire, adapter


@pytest.mark.parametrize("venue", ["alpaca", "binance", "kraken", "lean"])
@pytest.mark.parametrize("damage", ["metadata", "reservation"])
def test_native_claim_rechecks_new_failure_after_dispatch_committed_before_transmission(
    tmp_path, monkeypatch, venue, damage
):
    runtime, engine, adapter, order, wire, native = setup_native(tmp_path, venue)
    engine.prepare(
        order, {order.instrument.instrument_id: metadata(instrument=order.instrument)}, as_of=T
    )
    wire.now = order.eligible_at
    if runtime:
        runtime.at = order.eligible_at
    original = adapter.submit
    seen = []

    def race(*args):
        assert not engine.ledger.db.in_transaction
        assert engine.state(order.intent_id)["state"] == "submitting"
        assert engine.ledger.execution_compliance(order.account_id).status == "PASS"
        seen.append(
            engine.ledger.db.execute(
                "SELECT payload FROM events WHERE id=?",
                ("venue-submission-start:" + order.intent_id,),
            ).fetchone()[0]
        )
        if damage == "metadata":
            corrupt_retained_authority(engine.ledger, order.account_id)
        else:
            engine.ledger.db.execute(
                "UPDATE orders SET reserved='0' WHERE account=? AND id=?",
                (order.account_id, order.intent_id),
            )
        return original(*args)

    monkeypatch.setattr(adapter, "submit", race)
    result = engine.dispatch(order.intent_id, as_of=order.eligible_at)
    assert result["state"] == "rejected" and "execution compliance FAIL" in result["detail"]
    assert len(seen) == 1 and wire.order is None
    if runtime:
        assert not runtime.calls
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    assert not engine.ledger.db.execute(
        "SELECT 1 FROM events WHERE id LIKE '%submit-attempt:%' OR id LIKE 'venue-submission-claimed:%'"
    ).fetchone()
    assert not engine.ledger.fills(order.account_id)
    native.close()
    engine.ledger.close()


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_actual_post_expiry_source_blocks_every_new_prepare_before_adapter_work(tmp_path, side):
    from tests.test_alpaca_venue import setup

    alpaca_receipt(tmp_path, expired=True)
    wire, engine, adapter, order = setup(tmp_path)
    at = T + timedelta(seconds=5)
    new = order.model_copy(
        update={
            "intent_id": "new-after-violation",
            "side": side,
            "decision_at": at,
            "eligible_at": at + timedelta(seconds=1),
        }
    )
    before, calls = rows(engine.ledger), len(wire.calls)
    with pytest.raises(LedgerError, match="execution compliance FAIL"):
        engine.prepare(new, {order.instrument.instrument_id: metadata(at=at)}, as_of=at)
    assert rows(engine.ledger) == before and len(wire.calls) == calls
    adapter.close()
    engine.ledger.close()


def test_first_dispatch_cut_is_durable_readonly_and_missing_start_is_not_fresh(
    tmp_path, monkeypatch
):
    runtime, engine, adapter, order, wire, native = setup_native(tmp_path, "alpaca")
    engine.prepare(order, {order.instrument.instrument_id: metadata()}, as_of=T)
    wire.now = order.eligible_at
    saved = tmp_path / "original-dispatch-cut.sqlite"
    original = adapter.submit

    def retain_cut(*args):
        destination = sqlite3.connect(saved)
        engine.ledger.db.backup(destination)
        destination.close()
        return original(*args)

    monkeypatch.setattr(adapter, "submit", retain_cut)
    assert engine.dispatch(order.intent_id, as_of=wire.now)["state"] == "new"
    native.close()
    engine.ledger.close()
    with readonly_proof(saved) as ledger:
        assert ledger.execution_compliance(order.account_id).status == "PASS"
        assert ledger.db.execute("SELECT state FROM venue_outbox").fetchone()[0] == "submitting"
    writable = sqlite3.connect(saved)
    writable.execute("DELETE FROM events WHERE id LIKE 'venue-submission-start:%'")
    writable.commit()
    writable.close()
    with readonly_proof(saved) as ledger:
        audit = ledger.execution_compliance(order.account_id)
        assert audit.status == "FAIL" and "unresolved_native_outbox" in audit.blocking_codes


def test_removed_attempt_index_cannot_recreate_unstarted_native_claim(tmp_path):
    from tests.test_alpaca_venue import admit, setup

    wire, engine, adapter, order = setup(tmp_path)
    admit(wire, engine, adapter, order)
    # Deliberately damaged old archive: retain the independent claimed receipt,
    # remove original attempt/report/transmission and reset mutable flags.
    with engine.ledger.atomic():
        engine.ledger.db.execute(
            "DELETE FROM events WHERE id LIKE 'alpaca-submit-attempt:%' OR id LIKE 'alpaca-order:%'"
        )
        for row in list(
            engine.ledger.db.execute("SELECT id,payload FROM events WHERE id LIKE 'alpaca-http:%'")
        ):
            import json

            if json.loads(row["payload"])["method"] == "POST":
                engine.ledger.db.execute("DELETE FROM events WHERE id=?", (row["id"],))
        engine.ledger.db.execute("DELETE FROM venue_reports")
        engine.ledger.db.execute(
            "UPDATE venue_outbox SET state='submitting',venue_id=NULL,cumulative='0'"
        )
    before = len(wire.calls)
    assert engine.ledger.execution_compliance(order.account_id).status == "FAIL"
    with pytest.raises(SubmissionNotAttempted, match="execution compliance FAIL"):
        adapter.submit(
            order,
            engine.state(order.intent_id)["client_id"],
            adapter._rules[order.instrument.instrument_id],
            wire.now,
        )
    assert len(wire.calls) == before
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    engine.ledger.db.execute("UPDATE venue_outbox SET state='filled'")
    audit = engine.ledger.execution_compliance(order.account_id)
    assert audit.status == "FAIL" and "missing_original_submission_attempt" in audit.blocking_codes
    adapter.close()
    engine.ledger.close()


def test_failure_blocks_new_dispatch_but_preserves_exact_retry_cancellation_and_expiry(tmp_path):
    from tests.test_alpaca_venue import admit, setup

    wire, engine, adapter, order = setup(tmp_path / "cancel")
    admit(wire, engine, adapter, order)
    corrupt_retained_authority(engine.ledger, order.account_id)
    before = len(wire.calls)
    assert (
        engine.prepare(order, {order.instrument.instrument_id: metadata()}, as_of=T)["state"]
        == "new"
    )
    assert engine.dispatch(order.intent_id, as_of=wire.now)["state"] == "new"
    assert len(wire.calls) == before
    assert engine.cancel(order.intent_id, as_of=wire.now)["state"] == "cancelled"
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    adapter.close()
    engine.ledger.close()

    wire, engine, adapter, order = setup(tmp_path / "expired")
    order = order.model_copy(update={"expires_at": T + timedelta(seconds=2)})
    adapter.capture()
    engine.prepare(order, {order.instrument.instrument_id: metadata()}, as_of=T)
    corrupt_retained_authority(engine.ledger, order.account_id)
    with pytest.raises(LedgerError, match="execution compliance FAIL"):
        engine.dispatch(order.intent_id, as_of=order.eligible_at)
    assert engine.state(order.intent_id)["state"] == "prepared" and wire.order is None
    assert engine.dispatch(order.intent_id, as_of=order.expires_at)["state"] == "expired"
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    assert wire.order is None
    adapter.close()
    engine.ledger.close()


@pytest.mark.parametrize("venue", ["alpaca", "binance", "kraken"])
@pytest.mark.parametrize("definite", [True, False])
def test_only_raw_definite_rejection_closes_attempt_despite_mutable_terminal_flags(
    tmp_path, venue, definite
):
    runtime, engine, adapter, order, wire, native = setup_native(tmp_path, venue)
    if venue == "alpaca":
        wire.rejection = (403, 40310000) if definite else (422, "42210000")
    elif venue == "binance":
        wire.reject = -1013 if definite else -2010
    else:
        wire.reject = "EOrder:Insufficient funds" if definite else "EGeneral:Temporary lockout"
    engine.prepare(
        order, {order.instrument.instrument_id: metadata(instrument=order.instrument)}, as_of=T
    )
    wire.now = order.eligible_at
    assert engine.dispatch(order.intent_id, as_of=wire.now)["state"] == (
        "rejected" if definite else "ambiguous"
    )
    engine.ledger.db.execute("UPDATE venue_outbox SET state='rejected'")
    path = engine.ledger.path
    native.close()
    engine.ledger.close()
    with readonly_proof(path) as ledger:
        audit = ledger.execution_compliance(order.account_id)
        assert audit.status == ("PASS" if definite else "FAIL"), audit
        if not definite:
            assert "unresolved_submission_attempt" in audit.blocking_codes


def test_paper_final_acceptance_rechecks_current_failure_inside_local_transaction(
    tmp_path, monkeypatch
):
    from tests.test_venue_execution import setup

    ledger, adapter, engine, order = setup(tmp_path)
    engine.prepare(order, {order.instrument.instrument_id: metadata()}, as_of=T)
    original = adapter.submit

    def race(*args):
        assert not ledger.db.in_transaction
        corrupt_retained_authority(ledger, order.account_id)
        return original(*args)

    monkeypatch.setattr(adapter, "submit", race)
    assert engine.dispatch(order.intent_id, as_of=order.eligible_at)["state"] == "rejected"
    assert not ledger.db.execute("SELECT 1 FROM events WHERE id LIKE 'paper-accepted:%'").fetchone()
    assert not ledger.fills(order.account_id)
    ledger.close()
