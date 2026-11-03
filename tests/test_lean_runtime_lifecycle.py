"""Offline regressions; recorded evidence checks do not launch or emulate LEAN."""

import copy
import importlib.util
import json
from datetime import timedelta
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.execution.lean_runtime import LeanAlgorithmPort
from tests.test_lean_execution import admit, setup

FIXTURES = Path(__file__).resolve().parents[1] / "environments" / "lean-smoke"
spec = importlib.util.spec_from_file_location(
    "lifecycle_evidence", FIXTURES / "order_lifecycle_evidence.py"
)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def test_limit_uses_real_async_overload_and_preserves_exact_decimal_and_client():
    """Pinned QCAlgorithm signature has bool before tag; omitted bool must fail."""
    port = object.__new__(LeanAlgorithmPort)
    port.deployment = SimpleNamespace(brokerage="PaperBrokerage")
    port.symbols = {"crypto": object()}
    properties = object()
    port._order_properties = lambda intent: properties
    # This test isolates overload positions; CLR precision is checked in-engine.
    port._native_decimal = lambda amount: amount
    port._ticket = lambda ticket: ticket
    quantity, price = D("0.12345678"), D("1.00")

    def limit(symbol, quantity_arg, price_arg, asynchronous, tag, order_properties):
        assert symbol is port.symbols["crypto"]
        assert quantity_arg == quantity and price_arg == price
        assert asynchronous is True and tag == "original-client"
        assert order_properties is properties
        return "native-ticket"

    port.algorithm = SimpleNamespace(LimitOrder=limit)
    intent = SimpleNamespace(
        order_type="limit",
        side="buy",
        quantity=quantity,
        limit_price=price,
        instrument=SimpleNamespace(instrument_id="crypto"),
    )
    assert port.submit(intent, "original-client") == "native-ticket"


def test_offline_resting_limit_cancel_releases_only_its_reservation(tmp_path):
    """Fake-port regression, explicitly separate from actual-engine evidence."""
    runtime, execution, adapter, order = setup(tmp_path)
    try:
        order = order.model_copy(update={"order_type": "limit", "limit_price": D("1")})
        assert admit(runtime, execution, adapter, order)["state"] == "new"
        original = copy.deepcopy(runtime.ticket)
        reserve = execution.ledger.order(order.account_id, order.intent_id)["reserved"]
        assert reserve > 0 and execution.ledger.fills(order.account_id) == []
        runtime.at += timedelta(seconds=3)
        assert execution.cancel(order.intent_id, as_of=runtime.at)["state"] == "cancelled"
        assert runtime.ticket["client_order_id"] == original["client_order_id"]
        assert runtime.ticket["ticket_id"] == original["ticket_id"]
        assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
        assert execution.ledger.fills(order.account_id) == []
    finally:
        execution.ledger.close()


@pytest.fixture
def recorded():
    # Actual 2026-11-03 output retained verbatim; re-reading it is not a new run.
    return json.loads((FIXTURES / "order_lifecycle_recorded.json").read_text())


def test_recorded_actual_engine_result_contract(recorded):
    evidence.validate_result(recorded)


@pytest.mark.parametrize("corruption", ["tag", "ticket", "reserve", "fill", "cancel", "model"])
def test_evidence_reader_rejects_invalid_acceptance(recorded, corruption):
    final = recorded["snapshots"]["cancelled"]
    if corruption == "tag":
        final["ticket"]["client_order_id"] = "replacement-tag"
    elif corruption == "ticket":
        final["ticket"]["ticket_id"] += 1
    elif corruption == "reserve":
        final["reserved"] = "0.000000000000000000001"
    elif corruption == "fill":
        final["ticket"]["filled_quantity"] = "0.00000001"
    elif corruption == "cancel":
        final["ticket"]["events"] = final["ticket"]["events"][:-1]
    else:
        recorded["fill_model"] = "CustomPartialFillModel"
    with pytest.raises(AssertionError):
        evidence.validate_result(recorded)
