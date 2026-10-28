"""Offline LEAN protocol fixtures; no installed SDK/provider acceptance claim."""

from datetime import timedelta
from decimal import Decimal as D
from types import SimpleNamespace as NS

import pytest

from src.domain.portfolio import PriceMark
from src.execution.lean_runtime import LeanAlgorithmPort
from src.execution.ledger import LedgerError
from tests.test_durable_accounting import A, B, T, intent
from tests.test_lean_execution import admit, setup, setup_live


def trail(adapter, order, field="trail_percent", value="10"):
    adapter._rules[order.instrument.instrument_id] = adapter._rules[
        order.instrument.instrument_id
    ].model_copy(update={"order_types": ("market", "trailing_stop")})
    return order.model_copy(update={"order_type": "trailing_stop", field: D(value)})


@pytest.mark.parametrize("instrument,field", [(A, "trail_percent"), (B, "trail_price")])
def test_paper_trailing_lost_ack_partial_restart_cancel_exact_cost(tmp_path, instrument, field):
    runtime, execution, adapter, order = setup(tmp_path, instrument=instrument)
    order = trail(adapter, order, field)
    runtime.lose_ack = True
    assert admit(runtime, execution, adapter, order)["state"] == "ambiguous"
    assert execution.dispatch(order.intent_id, as_of=runtime.at)["state"] == "ambiguous"
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "new"
    assert not execution.ledger.fills(order.account_id)
    runtime.at += timedelta(seconds=1)
    # The provider's observed stop can change; immutable trail terms cannot.
    runtime.ticket["stop_price"] = "99"
    runtime.add("3", "99", ".3")
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "partial"
    first = execution.ledger.fills(order.account_id)
    execution.ledger.close()
    runtime, execution, adapter, original = setup(tmp_path, instrument=instrument, runtime=runtime)
    assert trail(adapter, original, field) == order
    runtime.at += timedelta(seconds=1)
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "partial"
    assert execution.ledger.fills(order.account_id) == first
    assert execution.cancel(order.intent_id, as_of=runtime.at)["state"] == "cancelled"
    adapter.capture()
    assert execution.reconcile_balances(as_of=runtime.at)["reconciled"]
    execution.ledger.mark(
        order.account_id, [PriceMark(instrument=instrument, price=D(99), as_of=runtime.at)]
    )
    snapshot = execution.ledger.snapshot(order.account_id, runtime.at, 0)
    assert snapshot.cash == D("99702.7") and snapshot.nav == D("99999.7")
    assert snapshot.fees == D(".3") and snapshot.reserved_cash == 0
    assert execution.ledger.reconcile(order.account_id)["reconciled"]
    assert len([call for call in runtime.calls if call[0] == "submit"]) == 1
    execution.ledger.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"trailing_amount": ".2"},
        {"trailing_as_percentage": False},
        {"trailing_as_percentage": 1},
        {"trailing_amount": None},
        {"stop_price": "0"},
        {"limit_price": "100"},
        {"order_type": "StopMarket"},
    ],
)
def test_changed_runtime_trail_retains_reservation_and_books_no_fill(tmp_path, changes):
    runtime, execution, adapter, order = setup(tmp_path)
    order = trail(adapter, order)
    assert admit(runtime, execution, adapter, order)["state"] == "new"
    runtime.at += timedelta(seconds=1)
    runtime.add("3", "100", "1")
    runtime.ticket.update(changes)
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "ambiguous"
    assert not execution.ledger.fills(order.account_id)
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    execution.ledger.close()


@pytest.mark.parametrize(
    "brokerage,instrument,quantity",
    [
        ("BinanceBrokerage", B, "10"),
        ("KrakenBrokerage", B, "10"),
        ("AlpacaBrokerage", B, "10"),
        ("AlpacaBrokerage", A, ".5"),
    ],
)
def test_unsupported_reviewed_brokerage_denied_before_runtime_access(
    brokerage, instrument, quantity
):
    port = object.__new__(LeanAlgorithmPort)
    port.deployment = NS(brokerage=brokerage)
    order = intent(instrument=instrument, qty=quantity).model_copy(
        update={"order_type": "trailing_stop", "trail_percent": D(10)}
    )
    # No algorithm or credentials exist: denial precedes every runtime access.
    for method, args in ((port.preflight, (order,)), (port.submit, (order, "original"))):
        with pytest.raises(LedgerError, match="reviewed LEAN brokerage/quantity"):
            method(*args)


@pytest.mark.parametrize(
    "side,field,value,expected,percentage",
    [
        (
            "buy",
            "trail_percent",
            "12.3456789012345678901234567",
            "0.123456789012345678901234567",
            True,
        ),
        ("sell", "trail_price", "1.25", "1.25", False),
    ],
)
def test_actual_port_overload_and_native_ticket_fields(side, field, value, expected, percentage):
    calls = []
    order = intent(side=side).model_copy(update={"order_type": "trailing_stop", field: D(value)})
    native_order = NS(
        Tag="original-client",
        Symbol="symbol",
        Type="TrailingStop",
        StopPrice=D(99),
        TrailingAmount=D(expected),
        TrailingAsPercentage=percentage,
        BrokerId=["native-id"],
    )
    native_ticket = NS(
        OrderId=51,
        Quantity=D(10 if side == "buy" else -10),
        QuantityFilled=D(0),
        Status="Submitted",
        OrderEvents=[],
    )

    def submit(*args):
        calls.append(args)
        return native_ticket

    port = object.__new__(LeanAlgorithmPort)
    port.deployment, port.symbols = NS(brokerage="PaperBrokerage"), {A.instrument_id: "symbol"}
    port.algorithm = NS(
        TrailingStopOrder=submit, Transactions=NS(GetOrderById=lambda identity: native_order)
    )
    port._native_decimal, port._number, port._properties, port._gtc = D, D, NS, lambda: "gtc"
    port._instrument, port.now, port.runtime_id = (
        lambda symbol: A.instrument_id,
        lambda: T,
        "fixture",
    )
    receipt = port.submit(order, "original-client")
    assert len(calls) == 1
    assert calls[0][:6] == (
        "symbol",
        native_ticket.Quantity,
        D(expected),
        percentage,
        True,
        "original-client",
    )
    assert calls[0][6].TimeInForce == "gtc"
    assert receipt["trailing_amount"] == str(D(expected))
    assert receipt["trailing_as_percentage"] is percentage
    assert receipt["stop_price"] == "99" and receipt["limit_price"] is None
    assert receipt["order_type"] == "TrailingStop" and receipt["events"] == []


@pytest.mark.parametrize("field", ["trail_percent", "trail_price"])
def test_live_alpaca_trail_uses_native_execution_and_original_restart_proof(tmp_path, field):
    runtime, execution, adapter, order, wire, native = setup_live(tmp_path)
    try:
        order = trail(adapter, order, field)
        native._rules[A.instrument_id] = native._rules[A.instrument_id].model_copy(
            update={"order_types": ("market", "trailing_stop")}
        )
        assert admit(runtime, execution, adapter, order)["state"] == "new"
        runtime.at += timedelta(seconds=1)
        runtime.add("3", "100", "7")  # FeeModel output is not a posted native charge.
        wire.add_fill("actual-trail-fill", "3", "100", 2)
        wire.order.update(filled_qty="3", status="partially_filled", hwm="101", stop_price="91")
        wire.cash, wire.positions = "99700", [{"symbol": "A", "side": "long", "qty": "3"}]
        assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "partial"
        fill = execution.ledger.fills(order.account_id)[0]
        assert fill.fill_id == "alpaca:actual-trail-fill" and fill.fee == 0
        runtime.cash = D(wire.cash)
        adapter.capture()
        assert execution.reconcile_balances(as_of=runtime.at)["reconciled"]
        # Imported runtime ticket after restart has another tag/local ID. Native
        # order ownership and the original trail remain mandatory for cancel.
        runtime.runtime_id = "fixture-restarted-trail"
        runtime.at += timedelta(seconds=1)
        runtime.ticket.update(
            runtime_id=runtime.runtime_id,
            ticket_id=88,
            client_order_id="imported",
            events=[],
            trailing_amount="999",
        )
        assert execution.cancel(order.intent_id, as_of=runtime.at)["state"] == "cancel_pending"
        assert not any(call[0] == "cancel" for call in runtime.calls)
        runtime.ticket["trailing_amount"] = ".1" if field == "trail_percent" else "10"
        original_cancel = runtime.cancel

        def cancel(ticket, client):
            original_cancel(ticket, client)
            wire.order["status"] = "canceled"

        runtime.cancel = cancel
        assert execution.cancel(order.intent_id, as_of=runtime.at)["state"] == "cancelled"
        assert execution.ledger.fills(order.account_id) == [fill]
        assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
        assert execution.ledger.reconcile(order.account_id)["reconciled"]
        assert len([call for call in runtime.calls if call[0] == "submit"]) == 1
        assert all(method == "GET" for method, *_ in wire.calls)
    finally:
        native.close()
        execution.ledger.close()
