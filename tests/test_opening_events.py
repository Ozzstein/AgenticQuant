"""Opening-only execution fixtures, with no provider acceptance claim."""

import json
from datetime import timedelta
from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from src.domain.portfolio import ExecutionBar, FillPolicy, OpeningEvent, PriceMark
from src.execution.ledger import LedgerError
from src.execution.simulator import PaperExecution
from tests.test_durable_accounting import A, B, T, account, bar, intent, metadata
from tests.test_venue_execution import setup


@pytest.mark.parametrize("instrument", [A, B])
def test_opening_contract_has_only_observed_fields_and_exact_decimal_roundtrip(instrument):
    raw = OpeningEvent.from_bar(bar(instrument=instrument)).model_dump(mode="json")
    raw["price"] = "100.12345678901234567890123456789"
    raw["opening_volume"] = "17.12345678901234567890123456789"
    event = OpeningEvent.model_validate(raw)
    assert (
        OpeningEvent.model_validate_json(event.model_dump_json()).content_hash()
        == event.content_hash()
    )
    assert event.price == D(raw["price"])
    assert not {"open", "high", "low", "close", "volume", "closed_at"} & set(raw)
    for invalid in (
        {**raw, "high": "200"},
        {**raw, "closed_at": raw["session_close"]},
        {**raw, "price_convention": "split_adjusted"},
        {**raw, "opening_volume": "-1"},
        {**raw, "opened_at": raw["session_close"]},
        {**raw, "calendar": "XNYS" if instrument == B else "24/7"},
    ):
        with pytest.raises(ValidationError):
            OpeningEvent.model_validate(invalid)


def test_default_bar_serialization_keeps_legacy_payload_identity():
    completed = bar()
    payload = completed.model_dump(mode="json")
    assert "opening_event_id" not in payload
    assert ExecutionBar.model_validate(payload).content_hash() == completed.content_hash()
    linked = completed.model_copy(update={"opening_event_id": "actual-trade-id"})
    assert linked.model_dump(mode="json")["opening_event_id"] == "actual-trade-id"
    assert OpeningEvent.from_bar(linked).event_id == "actual-trade-id"


def test_completed_and_open_only_use_identical_fill_cost_cash_and_losing_valuation(tmp_path):
    policy = FillPolicy(equity_cost_bps="7.125", slippage_bps="2.25")
    completed = bar(at=T + timedelta(seconds=2), opening_volume="500", close="80")
    observed = OpeningEvent.from_bar(completed)
    ledgers = [account(tmp_path / f"route-{i}.db") for i in range(2)]
    economics = []
    for idx, ledger in enumerate(ledgers):
        engine = PaperExecution(ledger, "backtest:test", policy)
        order = intent(fee=policy.fee(A, D(10), D(100), D(1)))
        engine.submit(order, {A.instrument_id: metadata()})
        fills = (engine.open_bar if idx == 0 else engine.open_event)(
            completed if idx == 0 else observed, {A.instrument_id: metadata()}
        )
        assert len(fills) == 1 and fills[0].quantity == 5
        engine.close_bar(completed)
        snap = ledger.snapshot("backtest:test", completed.closed_at, 0)
        assert snap.nav < 100000 and snap.fees > 0
        assert snap.cash == D(100000) - fills[0].quantity * fills[0].price - fills[0].fee
        economics.append((fills, snap, ledger.reconcile("backtest:test")))
        ledger.close()
    assert economics[0] == economics[1]


@pytest.mark.parametrize("first", ["print", "completed"])
def test_linked_completed_bar_cannot_reuse_capacity_after_restart(tmp_path, first):
    path = tmp_path / "consumed.db"
    completed = bar(at=T + timedelta(seconds=2), opening_volume="500", close="80").model_copy(
        update={"event_id": "daily-bar", "opening_event_id": "actual-print"}
    )
    observed = OpeningEvent.from_bar(completed)
    ledger = account(path)
    engine = PaperExecution(ledger, "backtest:test", FillPolicy(equity_cost_bps=0))
    order = intent()
    engine.submit(order, {A.instrument_id: metadata()})
    fills = (engine.open_event if first == "print" else engine.open_bar)(
        observed if first == "print" else completed, {A.instrument_id: metadata()}
    )
    assert len(fills) == 1 and fills[0].event_id == "actual-print" and fills[0].quantity == 5
    ledger.close()
    ledger = account(path)
    engine = PaperExecution(ledger, "backtest:test", FillPolicy(equity_cost_bps=0))
    assert engine.open_bar(completed, {}) == []
    assert engine.open_event(observed, {}) == []
    engine.close_bar(completed)
    assert engine.open_event(observed, {}) == []
    assert ledger.quantity("backtest:test", A.instrument_id) == 5
    assert ledger.order("backtest:test", order.intent_id)["remaining"] == 5
    assert ledger.snapshot("backtest:test", completed.closed_at, 0).nav == 99900
    assert ledger.reconcile("backtest:test")["reconciled"]
    for field, value in (("open", D(101)), ("opening_volume", D(600)), ("fx_rate", D(2))):
        changed = completed.model_copy(update={field: value, "high": D(101)})
        with pytest.raises(LedgerError, match="conflict|differs|changed"):
            engine.open_bar(changed, {})
        with pytest.raises(LedgerError, match="differs"):
            engine.close_bar(changed)
    assert len(ledger.fills("backtest:test")) == 1
    ledger.close()


def test_legacy_consumed_bar_migrates_without_reusing_liquidity(tmp_path):
    ledger = account(tmp_path / "old.db")
    engine = PaperExecution(ledger, "backtest:test", FillPolicy(equity_cost_bps=0))
    completed = bar(at=T + timedelta(seconds=2), opening_volume="500")
    engine.submit(intent(), {A.instrument_id: metadata()})
    engine.open_bar(completed, {A.instrument_id: metadata()})
    # Exact pre-OpeningEvent database shape: only the complete open:<id> is stored.
    ledger.db.execute("DELETE FROM events WHERE id LIKE 'opening-event:%'")
    assert engine.open_event(OpeningEvent.from_bar(completed), {}) == []
    assert len(ledger.fills("backtest:test")) == 1
    assert (
        ledger.db.execute("SELECT count(*) FROM events WHERE id LIKE 'opening-event:%'").fetchone()[
            0
        ]
        == 1
    )
    assert ledger.reconcile("backtest:test")["reconciled"]
    ledger.close()


def test_paper_print_settles_before_session_close_and_receipt_recovers_exact_map(tmp_path):
    ledger, adapter, execution, order = setup(tmp_path)
    execution.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    execution.dispatch(order.intent_id, as_of=order.eligible_at)
    completed = bar(at=order.eligible_at + timedelta(seconds=1), opening_volume="500", close="80")
    observed = OpeningEvent.from_bar(completed)
    clocks = dict(
        observed_at=observed.opened_at + timedelta(seconds=2),
        available_at=observed.opened_at + timedelta(seconds=3),
        as_of=observed.opened_at + timedelta(seconds=4),
        evidence_id="actual-fixture-trade",
        evidence_hash="c" * 64,
    )
    frozen = {order.intent_id: execution.admission_metadata(order.intent_id)}
    with pytest.raises(LedgerError, match="not yet closed"):
        adapter.settle_open(completed, frozen, **clocks)
    fills = adapter.settle_open(observed, frozen, **clocks)
    assert len(fills) == 1 and fills[0].quantity == 5
    assert fills[0].executed_at == observed.opened_at < clocks["observed_at"] < completed.closed_at
    receipt = adapter.observed_open_receipt(observed.event_id)
    assert receipt.event_hash == observed.content_hash()
    assert receipt.occurred_at == observed.opened_at
    assert not ledger.db.execute("SELECT 1 FROM events WHERE id LIKE 'open:%'").fetchone()
    ledger.close()
    ledger, adapter, execution, _ = setup(tmp_path)
    receipt = adapter.observed_open_receipt(observed.event_id)
    frozen = {
        key: execution.admission_metadata(key) for key, _ in receipt.admission_metadata_hashes
    }
    assert adapter.settle_open(observed, frozen, **clocks) == []
    # The later completed bar carries the same print plus actual close information.
    later = {**clocks, "as_of": completed.closed_at}
    assert adapter.settle_open(completed, frozen, **later) == []
    adapter.advance_close(completed)
    assert ledger.snapshot(order.account_id, completed.closed_at, 0).nav == 99900
    assert ledger.reconcile(order.account_id)["reconciled"]
    assert len(ledger.fills(order.account_id)) == 1
    with pytest.raises(LedgerError, match="receipt changed"):
        adapter.settle_open(
            observed.model_copy(update={"opening_volume": D(1000)}), frozen, **later
        )
    ledger.close()


def test_paper_legacy_full_bar_receipt_retries_as_open_only_without_mutation(tmp_path):
    ledger, adapter, execution, order = setup(tmp_path)
    execution.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    execution.dispatch(order.intent_id, as_of=order.eligible_at)
    completed = bar(at=order.eligible_at + timedelta(seconds=1))
    frozen = {order.intent_id: execution.admission_metadata(order.intent_id)}
    clocks = dict(
        observed_at=completed.closed_at,
        available_at=completed.closed_at,
        as_of=completed.closed_at,
        evidence_id="old-fixture",
        evidence_hash="e" * 64,
    )
    adapter.settle_open(completed, frozen, **clocks)
    receipt = adapter.observed_open_receipt(completed.event_id).model_copy(
        update={"event_hash": completed.content_hash()}
    )
    legacy_bytes = receipt.model_dump_json()
    ledger.db.execute(
        "UPDATE events SET payload=? WHERE id=?",
        (legacy_bytes, "paper-observed-open:" + completed.event_id),
    )
    ledger.db.execute("DELETE FROM events WHERE id LIKE 'opening-event:%'")
    assert adapter.settle_open(OpeningEvent.from_bar(completed), frozen, **clocks) == []
    assert adapter.observed_open_receipt(completed.event_id).model_dump_json() == legacy_bytes
    assert len(ledger.fills(order.account_id)) == 1
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_expiry_bridge_requires_actual_boundary_and_releases_reserved_cash_without_bar(tmp_path):
    ledger, adapter, execution, order = setup(tmp_path)
    close = T + timedelta(hours=7)
    order = order.model_copy(update={"tif": "DAY", "expires_at": close})
    # The venue fixture must honestly advertise DAY support before admission.
    adapter._rules[A.instrument_id] = adapter._rules[A.instrument_id].model_copy(
        update={"time_in_force": ("GTC", "DAY")}
    )
    execution.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    execution.dispatch(order.intent_id, as_of=order.eligible_at)
    assert adapter.expire_orders(as_of=close - timedelta(microseconds=1)) == ()
    assert ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    ledger.mark(order.account_id, [PriceMark(instrument=A, price="80", as_of=close)])
    assert adapter.expire_orders(as_of=close) == (order.intent_id,)
    assert ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    assert adapter.expire_orders(as_of=close) == ()
    assert not ledger.fills(order.account_id)
    assert not ledger.db.execute("SELECT 1 FROM events WHERE id LIKE 'close:%'").fetchone()
    assert ledger.reconcile(order.account_id)["reconciled"]
    assert json.loads(ledger.db.execute("SELECT payload FROM marks").fetchone()[0])["price"] == "80"
    ledger.close()
