"""Actual local ledger transitions; synthetic prices are test fixtures only."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from decimal import localcontext

import pytest
from pydantic import ValidationError

from src.domain.portfolio import (
    FillPolicy,
    OpeningEvent,
    OpeningSourceIdentity,
    OrderIntent,
    PriceMark,
)
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.simulator import PaperExecution
from src.execution.stops import read_stop_state
from tests.test_durable_accounting import A, B, T, account, bar, fill, intent, metadata


def opening(seconds, price, quantity="10", *, instrument=A, tradable=True):
    event = OpeningEvent.from_bar(
        bar(at=T + timedelta(seconds=seconds), instrument=instrument, opening_volume=quantity)
    )
    return event.model_copy(update={"price": D(price), "tradable": tradable})


def book(path, *, side="sell", instrument=A, costs=None, **terms):
    ledger = account(path)
    if side == "sell":
        buy = intent(instrument=instrument)
        ledger.submit(buy)
        ledger.apply_fill(fill(buy))
        ledger.mark(
            buy.account_id,
            [PriceMark(instrument=instrument, price=100, as_of=T + timedelta(seconds=2))],
        )
    costs = costs or FillPolicy(equity_cost_bps=0, crypto_cost_bps=0, opening_participation=1)
    engine = PaperExecution(ledger, "backtest:test", costs)
    raw = intent("protect", side, at=T + timedelta(seconds=2), instrument=instrument).model_dump()
    raw["estimated_fee"] = costs.fee(instrument, D(10), D(100), D(1))
    order = OrderIntent.model_validate({**raw, **terms})
    engine.submit(order, {instrument.instrument_id: metadata(instrument=instrument)})
    return ledger, engine, order


def advance(engine, event):
    return engine.open_event(
        event, {event.instrument.instrument_id: metadata(instrument=event.instrument)}
    )


def test_order_contract_preserves_old_hash_and_exact_trailing_terms():
    ordinary = intent()
    raw = ordinary.model_dump(mode="json")
    assert "trail_price" not in raw and "trail_percent" not in raw
    import hashlib

    assert (
        ordinary.content_hash()
        == hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    )
    trail = OrderIntent.model_validate(
        {**raw, "order_type": "trailing_stop", "trail_percent": "1.2345678901234567890123456789"}
    )
    assert trail.trail_percent == D("1.2345678901234567890123456789")
    assert OrderIntent.model_validate_json(trail.model_dump_json()) == trail
    for changes in (
        {"order_type": "trailing_stop"},
        {"order_type": "trailing_stop", "trail_price": "1", "trail_percent": "1"},
        {"order_type": "trailing_stop", "trail_percent": "100"},
        {"order_type": "trailing_stop", "trail_price": "0"},
        {"order_type": "trailing_stop", "trail_price": "1", "stop_price": "90"},
        {"order_type": "trailing_stop", "trail_price": "1", "tif": "IOC"},
        {"trail_percent": "1"},
    ):
        with pytest.raises(ValidationError):
            OrderIntent.model_validate({**raw, **changes})


@pytest.mark.parametrize("instrument", [A, B])
def test_stop_election_without_capacity_partial_rebound_restart_losing_nav(tmp_path, instrument):
    path = tmp_path / "stops.sqlite"
    ledger, engine, order = book(path, instrument=instrument, order_type="stop", stop_price="90")
    first = opening(3, "80", "0", instrument=instrument)
    assert advance(engine, first) == []
    proof = engine.stop_state(order.intent_id)
    assert proof.elected_event_id == first.event_id and proof.elected_at == first.opened_at
    assert ledger.quantity(order.account_id, instrument.instrument_id) == 10
    part = advance(engine, opening(4, "85", "4", instrument=instrument))
    assert len(part) == 1 and part[0].quantity == 4
    ledger.close()
    ledger = PortfolioLedger(path)
    engine = PaperExecution(ledger, order.account_id, engine.policy)
    assert engine.stop_state(order.intent_id).elected_event_id == first.event_id
    rebound = opening(5, "95", "6", instrument=instrument)
    rest = advance(engine, rebound)
    assert len(rest) == 1 and rest[0].quantity == 6 and rest[0].price == 95
    assert advance(engine, rebound) == []
    assert ledger.quantity(order.account_id, instrument.instrument_id) == 0
    snap = ledger.snapshot(order.account_id, rebound.opened_at, 0)
    assert snap.cash == snap.nav == 99910 and snap.nav < snap.high_water_mark
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


@pytest.mark.parametrize("terms", [{"trail_percent": "10"}, {"trail_price": "12"}])
def test_trailing_highwater_partial_election_survives_rebound_in_another_process(tmp_path, terms):
    path = tmp_path / "trail.sqlite"
    ledger, engine, order = book(path, order_type="trailing_stop", **terms)
    assert advance(engine, opening(3, "100")) == []
    assert advance(engine, opening(4, "120")) == []
    assert engine.stop_state(order.intent_id).anchor == 120
    assert engine.stop_state(order.intent_id).threshold == 108
    part = advance(engine, opening(5, "106", "4"))
    assert part[0].quantity == 4 and part[0].price == 106
    elected = engine.stop_state(order.intent_id)
    ledger.close()
    rebound = opening(6, "130", "6")
    code = """import sys
from src.domain.portfolio import OpeningEvent, FillPolicy, InstrumentRisk
from src.execution.ledger import PortfolioLedger
from src.execution.simulator import PaperExecution
ledger=PortfolioLedger(sys.argv[1])
event=OpeningEvent.model_validate_json(sys.argv[2])
engine=PaperExecution(ledger,'backtest:test',FillPolicy.model_validate_json(sys.argv[3]))
meta=InstrumentRisk.model_validate_json(sys.argv[4])
before=engine.stop_state('protect')
fills=engine.open_event(event,{meta.instrument.instrument_id:meta})
assert len(fills)==1 and fills[0].quantity==6 and fills[0].price==130
after=engine.stop_state('protect')
assert after.anchor==before.anchor==120 and after.threshold==108
assert after.elected_event_id==before.elected_event_id
assert ledger.reconcile('backtest:test')['reconciled']
assert not engine.open_event(event,{meta.instrument.instrument_id:meta})
ledger.close()
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(path),
            rebound.model_dump_json(),
            engine.policy.model_dump_json(),
            metadata().model_dump_json(),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    reopened = PortfolioLedger(path, read_only=True)
    state = read_stop_state(reopened, order.account_id, order.intent_id)
    assert state.elected_at == elected.at
    assert len(reopened.fills(order.account_id)) == 3  # entry plus both actual exits
    reopened.close()


def test_buy_trailing_uses_low_water_and_elected_remaining_stays_marketable(tmp_path):
    ledger, engine, order = book(
        tmp_path / "buy.sqlite", side="buy", order_type="trailing_stop", trail_percent="10"
    )
    assert advance(engine, opening(3, "100")) == []
    assert advance(engine, opening(4, "80")) == []
    assert engine.stop_state(order.intent_id).threshold == 88
    assert advance(engine, opening(5, "90", "4"))[0].quantity == 4
    assert advance(engine, opening(6, "70", "6"))[0].quantity == 6
    assert ledger.quantity(order.account_id, A.instrument_id) == 10
    assert ledger.reconcile(order.account_id)["cash"] == "99220"
    ledger.close()


def test_untradable_or_ineligible_events_and_future_ohlc_do_not_ratchet(tmp_path):
    histories = []
    for future_high in ("125", "999999"):
        ledger, engine, order = book(
            tmp_path / f"causal-{future_high}.sqlite",
            order_type="trailing_stop",
            trail_percent="10",
        )
        assert advance(engine, opening(2, "999")) == []  # before eligibility
        assert engine.stop_state(order.intent_id) is None
        assert advance(engine, opening(3, "500", tradable=False)) == []
        assert engine.stop_state(order.intent_id) is None
        completed = bar(at=T + timedelta(seconds=4)).model_copy(
            update={"high": D(future_high), "low": D(1), "close": D(future_high)}
        )
        assert engine.open_bar(completed, {A.instrument_id: metadata()}) == []
        assert engine.stop_state(order.intent_id).anchor == 100
        # Same opening source is one observation even when a complete bar arrives.
        assert advance(engine, OpeningEvent.from_bar(completed)) == []
        assert advance(engine, opening(5, "95")) == []
        histories.append(engine.stop_state(order.intent_id))
        assert ledger.reconcile(order.account_id)["reconciled"]
        ledger.close()
    assert histories[0] == histories[1]


def test_event_and_stop_and_fill_commit_atomically_on_unexpected_failure(tmp_path, monkeypatch):
    ledger, engine, order = book(tmp_path / "atomic.sqlite", order_type="stop", stop_price="90")
    event = opening(3, "80")
    actual = ledger.apply_fill

    def fail(fill):
        actual(fill)
        raise RuntimeError("fixture process failure after fill before commit")

    monkeypatch.setattr(ledger, "apply_fill", fail)
    with pytest.raises(RuntimeError, match="fixture"):
        advance(engine, event)
    assert engine.stop_state(order.intent_id) is None
    assert ledger.quantity(order.account_id, A.instrument_id) == 10
    assert not ledger.db.execute(
        "SELECT 1 FROM events WHERE id=?", ("opening-event:" + event.event_id,)
    ).fetchone()
    monkeypatch.setattr(ledger, "apply_fill", actual)
    assert advance(engine, event)[0].quantity == 10
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


@pytest.mark.parametrize("damage", ["observation", "opening", "missing", "intent"])
def test_stop_history_and_fill_reconciliation_reject_tampering(tmp_path, damage):
    ledger, engine, order = book(
        tmp_path / "corrupt.sqlite", order_type="trailing_stop", trail_price="10"
    )
    advance(engine, opening(3, "100"))
    advance(engine, opening(4, "89", "4"))
    if damage == "missing":
        ledger.db.execute("DELETE FROM events WHERE id LIKE 'paper-stop:%'")
    elif damage == "intent":
        changed = order.model_copy(update={"trail_price": D(9)})
        ledger.db.execute(
            "UPDATE orders SET payload=? WHERE id=?", (changed.model_dump_json(), order.intent_id)
        )
    else:
        prefix = "paper-stop:%" if damage == "observation" else "opening-event:%"
        row = ledger.db.execute(
            "SELECT id,payload FROM events WHERE id LIKE ? ORDER BY rowid DESC LIMIT 1", (prefix,)
        ).fetchone()
        raw = json.loads(row["payload"])
        raw["threshold" if damage == "observation" else "price"] = "1"
        ledger.db.execute("UPDATE events SET payload=? WHERE id=?", (json.dumps(raw), row["id"]))
    with pytest.raises(LedgerError, match="stop"):
        ledger.reconcile(order.account_id)
    if damage != "intent":
        with pytest.raises(LedgerError, match="stop"):
            advance(engine, opening(5, "120", "6"))
    assert ledger.quantity(order.account_id, A.instrument_id) == 6
    ledger.close()


def test_legacy_partial_stop_requires_actual_crossing_before_restoring_election(tmp_path):
    ledger, engine, order = book(tmp_path / "legacy.sqlite", order_type="stop", stop_price="90")
    first = opening(3, "85", "2")
    advance(engine, first)
    advance(engine, opening(4, "86", "2"))
    # The original simulator retained opens/fills but did not retain trigger state.
    ledger.db.execute("DELETE FROM events WHERE id LIKE 'paper-stop%'")
    assert advance(engine, opening(5, "95", "6"))[0].quantity == 6
    assert engine.stop_state(order.intent_id).elected_event_id == first.event_id
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_expiry_prevents_elected_remaining_from_filling_after_boundary(tmp_path):
    ledger, engine, order = book(
        tmp_path / "expiry.sqlite",
        order_type="stop",
        stop_price="90",
        tif="DAY",
        expires_at=T + timedelta(seconds=4),
    )
    assert advance(engine, opening(3, "80", "4"))[0].quantity == 4
    assert engine.expire_orders(as_of=T + timedelta(seconds=4)) == (order.intent_id,)
    assert advance(engine, opening(5, "85", "6")) == []
    assert ledger.quantity(order.account_id, A.instrument_id) == 6
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_sell_trailing_offset_over_first_price_waits_for_a_positive_trigger(tmp_path):
    ledger, engine, order = book(
        tmp_path / "large-trail.sqlite", order_type="trailing_stop", trail_price="110"
    )
    assert advance(engine, opening(3, "100")) == []
    assert engine.stop_state(order.intent_id).threshold == -10
    assert advance(engine, opening(4, "120")) == []
    assert engine.stop_state(order.intent_id).threshold == 10
    assert advance(engine, opening(5, "9"))[0].price == 9
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_trailing_anchor_and_threshold_preserve_long_decimal_inputs(tmp_path):
    ledger, engine, order = book(
        tmp_path / "decimal.sqlite",
        order_type="trailing_stop",
        trail_percent="1.12345678901234567890123456789",
    )
    event = opening(3, "101.12345678901234567890123456789")
    assert advance(engine, event) == []
    state = engine.stop_state(order.intent_id)
    with localcontext() as context:
        context.prec = 120
        expected = event.price * (1 - order.trail_percent / 100)
    assert state.anchor == event.price and state.threshold == expected
    ledger.close()
    readonly = PortfolioLedger(tmp_path / "decimal.sqlite", read_only=True)
    assert read_stop_state(readonly, order.account_id, order.intent_id) == state
    assert readonly.reconcile(order.account_id)["reconciled"]
    readonly.close()


def test_actual_paper_outbox_observed_receipts_and_restart_use_shared_stop_state(tmp_path):
    from tests.test_venue_execution import setup

    ledger, adapter, execution, original = setup(tmp_path)
    adapter._rules[A.instrument_id] = adapter._rules[A.instrument_id].model_copy(
        update={"order_types": ("market", "trailing_stop")}
    )
    order = OrderIntent.model_validate(
        {**original.model_dump(), "order_type": "trailing_stop", "trail_percent": "10"}
    )
    execution.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    execution.dispatch(order.intent_id, as_of=order.eligible_at)

    def settle(adapter, execution, event):
        return adapter.settle_open(
            event,
            {order.intent_id: execution.admission_metadata(order.intent_id)},
            evidence_id="fixture:" + event.event_id,
            evidence_hash=event.content_hash(),
            observed_at=event.opened_at + timedelta(microseconds=1),
            available_at=event.opened_at + timedelta(microseconds=2),
            as_of=event.opened_at + timedelta(microseconds=3),
            source_identity=OpeningSourceIdentity(
                provider="alpaca",
                instrument=event.instrument,
                trade_id="V:" + str((event.opened_at - T) // timedelta(microseconds=1)),
                wire_timestamp=str(
                    (event.opened_at - datetime(1970, 1, 1, tzinfo=timezone.utc))
                    // timedelta(microseconds=1)
                    * 1000
                ),
            ),
        )

    assert settle(adapter, execution, opening(2, "100")) == []
    assert settle(adapter, execution, opening(3, "80")) == []
    partial = opening(4, "90", "400")
    assert settle(adapter, execution, partial)[0].quantity == 4
    execution.reconcile_all(as_of=partial.opened_at + timedelta(microseconds=3))
    assert execution.state(order.intent_id)["state"] == "partial"
    ledger.close()
    ledger, adapter, execution, _ = setup(tmp_path)
    receipt = adapter.observed_open_receipt(partial.event_id)
    assert receipt.admission_metadata_hashes[0][0] == order.intent_id
    assert settle(adapter, execution, partial) == []
    rebound = opening(5, "70", "600")
    assert settle(adapter, execution, rebound)[0].quantity == 6
    execution.reconcile_all(as_of=rebound.opened_at + timedelta(microseconds=3))
    assert execution.state(order.intent_id)["state"] == "filled"
    assert adapter.engine.stop_state(order.intent_id).elected_event_id == partial.event_id
    assert ledger.quantity(order.account_id, A.instrument_id) == 10
    assert ledger.reconcile(order.account_id)["cash"] == "99220"
    ledger.close()


def test_trailing_partial_costs_slippage_and_cash_reconcile_exactly(tmp_path):
    costs = FillPolicy(
        equity_cost_bps="8.125",
        commission_per_unit=".03",
        slippage_bps="2",
        opening_participation=1,
    )
    ledger, engine, order = book(
        tmp_path / "costs.sqlite",
        side="buy",
        costs=costs,
        order_type="trailing_stop",
        trail_percent="10",
    )
    assert advance(engine, opening(3, "100")) == []
    assert advance(engine, opening(4, "80")) == []
    first = advance(engine, opening(5, "90", "4"))[0]
    assert first.price == D("90.018") and first.fee == D(".4125585")
    assert ledger.order(order.account_id, order.intent_id)["reserved"] == D("600.6675")
    second = advance(engine, opening(6, "70", "6"))[0]
    assert second.price == D("70.014") and second.fee == D(".52131825")
    assert ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    snap = ledger.snapshot(order.account_id, second.executed_at, 0)
    assert snap.cash == D("99218.91012325") and snap.fees == D(".93387675")
    assert snap.nav == D("99918.91012325")
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


@pytest.mark.parametrize("tif,remaining", [("IOC", 6), ("FOK", 10)])
def test_immediate_stop_execution_applies_tif_when_elected(tmp_path, tif, remaining):
    ledger, engine, order = book(
        tmp_path / "immediate.sqlite", order_type="stop", stop_price="90", tif=tif
    )
    assert advance(engine, opening(3, "100")) == []
    assert ledger.order(order.account_id, order.intent_id)["status"] == "pending"
    event = opening(4, "80", "4")
    fills = advance(engine, event)
    assert sum((f.quantity for f in fills), D(0)) == 10 - remaining
    assert engine.stop_state(order.intent_id).elected_event_id == event.event_id
    assert ledger.order(order.account_id, order.intent_id)["status"] == "cancelled"
    assert ledger.quantity(order.account_id, A.instrument_id) == remaining
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()
