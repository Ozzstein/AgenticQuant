"""Actual durable expiry transitions against offline native protocol fixtures."""

import json
from datetime import timedelta

import httpx
import pytest
from pydantic import ValidationError

from src.domain.portfolio import OrderIntent
from src.execution.lean_runtime import LeanOperatingBridge
from src.execution.ledger import LedgerError
from tests.test_alpaca_venue import Wire, admit, setup
from tests.test_durable_accounting import A, T, account, fill, intent, metadata
from tests.test_stop_execution import advance, book, opening


def test_empty_execution_interval_is_invalid_and_fill_at_expiry_is_not_eligible():
    order = intent().model_copy(update={"expires_at": T + timedelta(seconds=2)})
    with pytest.raises(ValidationError, match="nonempty"):
        OrderIntent.model_validate({**order.model_dump(), "expires_at": order.eligible_at})
    ledger = account()
    ledger.submit(order)
    with pytest.raises(LedgerError, match="eligible execution interval"):
        ledger.apply_fill(fill(order, at=order.expires_at))
    assert not ledger.fills(order.account_id)
    assert ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    ledger.close()


def test_elected_stop_cannot_fill_an_open_exactly_at_expiry(tmp_path):
    ledger, engine, order = book(
        tmp_path / "stop.sqlite",
        order_type="stop",
        stop_price="90",
        tif="DAY",
        expires_at=T + timedelta(seconds=4),
    )
    assert advance(engine, opening(3, "80", "4"))[0].quantity == 4
    assert advance(engine, opening(4, "85", "6")) == []
    assert ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    assert ledger.quantity(order.account_id, A.instrument_id) == 6
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


@pytest.mark.parametrize("method", ["dispatch", "expire_orders"])
def test_never_dispatched_order_expires_at_boundary_without_native_submit(tmp_path, method):
    wire, engine, adapter, order = setup(tmp_path)
    try:
        order = order.model_copy(update={"expires_at": T + timedelta(seconds=3)})
        adapter.capture()
        engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
        wire.now = order.expires_at
        if method == "dispatch":
            result = engine.dispatch(order.intent_id, as_of=wire.now)
        else:
            (result,) = engine.expire_orders(as_of=wire.now)
        assert result["state"] == "expired"
        assert all(method == "GET" for method, *_ in wire.calls)
        assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
        assert not engine.ledger.fills(order.account_id)
        assert engine.expire_orders(as_of=wire.now) == ()
    finally:
        adapter.close()
        engine.ledger.close()


class CancellationWire(Wire):
    fail_cancel = False

    def handle(self, request):
        if self.fail_cancel and request.method == "DELETE":
            self.calls.append((request.method, request.url.path, {}, request.content))
            raise httpx.ReadTimeout("fixture lost cancel response", request=request)
        return super().handle(request)


def test_native_late_receipt_partial_cancel_failure_restart_preserves_cash(tmp_path):
    wire, engine, adapter, order = setup(tmp_path, CancellationWire())
    order = order.model_copy(update={"expires_at": T + timedelta(seconds=3)})
    try:
        assert admit(wire, engine, adapter, order)["state"] == "new"
        wire.add_fill("before-expiry", "3", "100", 2)
        wire.order.update(filled_qty="3", status="partially_filled")
        wire.cash, wire.positions = "99700", [{"symbol": "A", "side": "long", "qty": "3"}]
        # The actual execution precedes the deadline; receipt is later.
        wire.now = T + timedelta(seconds=4)
        wire.fail_cancel = True
        (result,) = engine.expire_orders(as_of=wire.now)
        assert result["state"] == "cancel_pending"
        assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 700
        first = engine.ledger.fills(order.account_id)
        assert len(first) == 1 and first[0].executed_at == T + timedelta(seconds=2)
        assert engine.ledger.reconcile(order.account_id)["cash"] == "99700"
        receipt = engine.ledger.db.execute(
            "SELECT payload FROM events WHERE id=?", ("venue-expiry-request:" + order.intent_id,)
        ).fetchone()[0]
        assert json.loads(receipt)["requested_at"] == wire.now.isoformat()
    finally:
        adapter.close()
        engine.ledger.close()
    wire.fail_cancel = False
    wire, engine, adapter, _ = setup(tmp_path, wire)
    try:
        wire.now += timedelta(seconds=1)
        (result,) = engine.expire_orders(as_of=wire.now)
        assert result["state"] == "cancelled"
        assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
        assert engine.ledger.fills(order.account_id) == first
        assert (
            engine.ledger.db.execute(
                "SELECT payload FROM events WHERE id=?",
                ("venue-expiry-request:" + order.intent_id,),
            ).fetchone()[0]
            == receipt
        )
        assert engine.expire_orders(as_of=wire.now) == ()
        assert sum(method == "POST" for method, *_ in wire.calls) == 1
        adapter.capture()
        assert engine.reconcile_balances(as_of=wire.now)["reconciled"]
    finally:
        adapter.close()
        engine.ledger.close()


def test_unknown_submission_cannot_expire_away_reservation(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    try:
        order = order.model_copy(update={"expires_at": T + timedelta(seconds=3)})
        wire.timeout = True
        assert admit(wire, engine, adapter, order)["state"] == "ambiguous"
        wire.now, wire.missing = order.expires_at, True
        (result,) = engine.expire_orders(as_of=wire.now)
        assert result["state"] == "ambiguous"
        assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
        assert sum(method == "POST" for method, *_ in wire.calls) == 1
        assert not any(method == "DELETE" for method, *_ in wire.calls)
        wire.missing = False
        assert engine.expire_orders(as_of=wire.now)[0]["state"] == "cancelled"
        assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    finally:
        adapter.close()
        engine.ledger.close()


@pytest.mark.parametrize("complete", [False, True])
def test_native_expired_status_needs_complete_actual_executions(tmp_path, complete):
    wire, engine, adapter, order = setup(tmp_path)
    try:
        order = order.model_copy(update={"expires_at": T + timedelta(seconds=3)})
        admit(wire, engine, adapter, order)
        wire.order.update(filled_qty="3", status="expired")
        wire.now = order.expires_at
        if complete:
            wire.add_fill("before-expiry", "3", "100", 2)
        (result,) = engine.expire_orders(as_of=wire.now)
        assert result["state"] == ("expired" if complete else "reconciliation_required")
        assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == (
            0 if complete else 1000
        )
        assert len(engine.ledger.fills(order.account_id)) == (1 if complete else 0)
    finally:
        adapter.close()
        engine.ledger.close()


def test_post_deadline_native_execution_retains_original_receipt_and_blocks(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    try:
        order = order.model_copy(update={"expires_at": T + timedelta(seconds=3)})
        admit(wire, engine, adapter, order)
        wire.add_fill("past-deadline", "3", "100", 3)
        wire.order.update(filled_qty="3", status="partially_filled")
        wire.now = T + timedelta(seconds=4)
        with pytest.raises(LedgerError, match="eligible execution interval"):
            engine.expire_orders(as_of=wire.now)
        assert engine.state(order.intent_id)["state"] == "reconciliation_required"
        assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
        assert not engine.ledger.fills(order.account_id)
        receipt = engine.ledger.db.execute(
            "SELECT payload FROM venue_reports WHERE applied=0 ORDER BY rowid DESC"
        ).fetchone()[0]
        assert json.loads(receipt)["fills"][0]["executed_at"] == "2024-01-02T14:00:03Z"
        assert not any(method == "DELETE" for method, *_ in wire.calls)
    finally:
        adapter.close()
        engine.ledger.close()


def test_lean_pump_cancels_due_day_order_before_newer_valuation(tmp_path):
    from tests.test_lean_execution import admit as lean_admit
    from tests.test_lean_execution import setup as lean_setup

    runtime, engine, adapter, order = lean_setup(tmp_path)
    order = order.model_copy(update={"tif": "DAY", "expires_at": T + timedelta(seconds=3)})
    adapter._rules[A.instrument_id] = adapter._rules[A.instrument_id].model_copy(
        update={"time_in_force": ("GTC", "DAY")}
    )
    assert lean_admit(runtime, engine, adapter, order)["state"] == "new"
    runtime.at = order.expires_at
    LeanOperatingBridge(engine).pump()
    assert engine.state(order.intent_id)["state"] == "cancelled"
    assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    assert len([call for call in runtime.calls if call[0] == "cancel"]) == 1
    assert not engine.ledger.fills(order.account_id)
    engine.ledger.close()


def test_actual_native_submission_clock_cannot_cross_expiry_cutoff(tmp_path):
    wire, engine, adapter, order = setup(tmp_path)
    try:
        order = order.model_copy(update={"expires_at": T + timedelta(seconds=3)})
        adapter.capture()
        engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
        wire.now = order.expires_at
        result = engine.dispatch(order.intent_id, as_of=order.eligible_at)
        assert result["state"] == "rejected"
        assert "expired" in result["detail"]
        assert not any(method == "POST" for method, *_ in wire.calls)
        assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    finally:
        adapter.close()
        engine.ledger.close()
