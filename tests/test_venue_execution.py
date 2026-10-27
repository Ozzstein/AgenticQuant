"""Durable venue owner/outbox invariants; remote responses are test doubles only."""

from datetime import timedelta
from decimal import Decimal as D

import pytest

from src.domain.portfolio import AccountSpec, Fill, FillPolicy, OpeningEvent
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.venue import (
    PaperVenue,
    VenueBalances,
    VenueExecution,
    VenueOrderReport,
    VenueRules,
)
from tests.test_durable_accounting import A, T, bar, intent, metadata


def setup(tmp_path):
    ledger = PortfolioLedger(tmp_path / "venue.sqlite3")
    ledger.create_account(
        AccountSpec(
            account_id="paper:venue",
            mode="paper",
            base_currency="USD",
            initial_cash=100000,
            opened_at=T,
        )
    )
    rules = VenueRules(
        venue="paper",
        instrument=A,
        symbol=A.symbol,
        available_at=T,
        quantity_step=1,
        price_tick=".01",
        minimum_quantity=1,
        minimum_notional=1,
        market_type="equity",
        active=True,
    )
    costs = FillPolicy(equity_cost_bps=0)
    adapter = PaperVenue(ledger, "paper:venue", {A.instrument_id: rules}, fill_policy=costs)
    engine = VenueExecution(
        ledger,
        "paper:venue",
        {"paper": adapter},
        {A.instrument_id: "paper"},
        owner="direct",
        fill_policy=costs,
    )
    order = intent().model_copy(update={"account_id": "paper:venue"})
    return ledger, adapter, engine, order


def test_real_paper_venue_round_trip_partial_duplicate_and_restart(tmp_path):
    ledger, adapter, engine, order = setup(tmp_path)
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    assert engine.dispatch(order.intent_id, as_of=order.eligible_at)["state"] == "new"
    opening = bar(at=order.eligible_at + timedelta(seconds=1), opening_volume="500")
    fills = adapter.advance_open(opening, {A.instrument_id: metadata()})
    assert sum(f.quantity for f in fills) == 5
    assert engine.reconcile_all(as_of=opening.opened_at)[0]["state"] == "partial"
    assert ledger.quantity(order.account_id, A.instrument_id) == 5
    engine.reconcile_all(as_of=opening.opened_at)
    assert len(ledger.fills(order.account_id)) == 1
    assert engine.reconcile_balances(as_of=opening.opened_at)["reconciled"]
    ledger.close()
    ledger, adapter, engine, same = setup(tmp_path)
    assert engine.prepare(same, {A.instrument_id: metadata()}, as_of=T)["state"] == "partial"
    assert engine.dispatch(order.intent_id, as_of=order.eligible_at)["state"] == "partial"
    engine.cancel(order.intent_id, as_of=order.eligible_at + timedelta(seconds=1))
    assert ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_ambiguous_submission_is_never_blindly_retried(tmp_path, monkeypatch):
    ledger, adapter, engine, order = setup(tmp_path)
    calls = []
    actual_submit = adapter.submit

    def timeout(*args):
        calls.append(1)
        actual_submit(*args)
        raise TimeoutError("response lost after broker acceptance")

    monkeypatch.setattr(adapter, "submit", timeout)
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    assert engine.dispatch(order.intent_id, as_of=order.eligible_at)["state"] == "ambiguous"
    assert engine.dispatch(order.intent_id, as_of=order.eligible_at)["state"] == "ambiguous"
    monkeypatch.setattr(adapter, "lookup", lambda *args: None)
    engine.reconcile_all(as_of=order.eligible_at)
    assert len(calls) == 1 and engine.state(order.intent_id)["state"] == "ambiguous"
    assert ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    ledger.close()
    ledger, adapter, engine, _ = setup(tmp_path)
    # A restart looks up the same client ID; late acknowledgement recovers safely.
    assert engine.reconcile_all(as_of=order.eligible_at)[0]["state"] == "new"
    assert engine.state(order.intent_id)["client_id"] == VenueExecution.client_id(
        order.account_id, order.intent_id
    )
    ledger.close()


def test_report_before_execution_details_keeps_reservations_and_receipt(tmp_path):
    ledger, adapter, engine, order = setup(tmp_path)
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    report = VenueOrderReport(
        venue="paper",
        client_order_id=engine.client_id(order.account_id, order.intent_id),
        venue_order_id="external-1",
        observed_at=order.eligible_at,
        status="filled",
        cumulative_quantity=10,
        executions_complete=False,
        evidence_id="ack",
    )
    engine.ingest(order.intent_id, report)
    assert engine.state(order.intent_id)["state"] == "reconciliation_required"
    assert ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    fill = Fill(
        fill_id="actual-execution",
        intent_id=order.intent_id,
        account_id=order.account_id,
        quantity=10,
        price=100,
        fee=0,
        executed_at=order.eligible_at,
        event_id="broker-trade",
    )
    completed = report.model_copy(
        update={"fills": (fill,), "executions_complete": True, "evidence_id": "execution"}
    )
    engine.ingest(order.intent_id, completed)
    engine.ingest(order.intent_id, report)  # old already-applied receipt is an idempotent no-op
    assert engine.state(order.intent_id)["state"] == "filled"
    assert ledger.reconcile(order.account_id)["reconciled"]
    invalid = completed.model_copy(update={"cumulative_quantity": D(11), "evidence_id": "invalid"})
    with pytest.raises(LedgerError, match="exceeded"):
        engine.ingest(order.intent_id, invalid)
    assert (
        ledger.db.execute("SELECT applied FROM venue_reports WHERE evidence='invalid'").fetchone()[
            0
        ]
        == 0
    )
    assert len(ledger.fills(order.account_id)) == 1
    ledger.close()


def test_exclusive_owner_rules_pending_risk_and_routed_funding(tmp_path, monkeypatch):
    ledger, adapter, engine, order = setup(tmp_path)
    with pytest.raises(LedgerError, match="owner mismatch"):
        VenueExecution(
            ledger, order.account_id, {"paper": adapter}, {A.instrument_id: "paper"}, owner="lean"
        )
    with pytest.raises(LedgerError, match="increment"):
        engine.prepare(
            order.model_copy(update={"quantity": D("1.1")}), {A.instrument_id: metadata()}, as_of=T
        )
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    with pytest.raises(LedgerError, match="risk rejected"):
        engine.prepare(
            order.model_copy(update={"intent_id": "over-cap", "quantity": D(95)}),
            {A.instrument_id: metadata()},
            as_of=T,
        )
    monkeypatch.setattr(
        adapter,
        "balances",
        lambda at: VenueBalances(
            venue="paper",
            observed_at=at,
            cash_base=100,
            available_cash_base=100,
            quantities={},
            open_client_order_ids=(),
            evidence_id="remote",
        ),
    )
    with pytest.raises(LedgerError, match="routed venue"):
        engine.prepare(
            order.model_copy(update={"intent_id": "unfunded"}),
            {A.instrument_id: metadata()},
            as_of=T,
        )
    assert len(ledger.pending(order.account_id)) == 1
    with pytest.raises(LedgerError, match="do not reconcile"):
        engine.reconcile_balances(as_of=T)
    ledger.close()


def test_prepared_unsubmitted_order_cannot_consume_market_liquidity(tmp_path):
    ledger, adapter, engine, order = setup(tmp_path)
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    assert adapter.advance_open(bar(at=order.eligible_at), {A.instrument_id: metadata()}) == []
    assert ledger.quantity(order.account_id, A.instrument_id) == 0
    assert engine.dispatch(order.intent_id, as_of=order.eligible_at)["state"] == "new"
    later = bar(at=order.eligible_at + timedelta(seconds=1))
    assert adapter.advance_open(later, {A.instrument_id: metadata()})
    assert engine.reconcile_all(as_of=later.opened_at)[0]["state"] == "filled"
    ledger.close()


def test_process_exit_after_acceptance_recovers_without_second_submission(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    program = """
import os, sys
from pathlib import Path
from tests.test_venue_execution import setup
from tests.test_durable_accounting import A,T,metadata
ledger,adapter,engine,order=setup(Path(sys.argv[1]))
engine.prepare(order,{A.instrument_id:metadata()},as_of=T)
real_submit=adapter.submit
def accepted_then_die(*args):
    real_submit(*args)
    os._exit(86)
adapter.submit=accepted_then_die
engine.dispatch(order.intent_id,as_of=order.eligible_at)
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(Path.cwd()), "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert completed.returncode == 86
    ledger, adapter, engine, order = setup(tmp_path)
    assert engine.state(order.intent_id)["state"] == "submitting"
    assert engine.reconcile_all(as_of=order.eligible_at)[0]["state"] == "new"
    assert engine.dispatch(order.intent_id, as_of=order.eligible_at)["state"] == "new"
    assert (
        ledger.db.execute(
            "SELECT COUNT(*) FROM events WHERE id LIKE 'paper-accepted:%'"
        ).fetchone()[0]
        == 1
    )
    opening = bar(at=order.eligible_at + timedelta(seconds=1))
    adapter.advance_open(opening, {A.instrument_id: metadata()})
    engine.reconcile_all(as_of=opening.opened_at)
    assert len(ledger.fills(order.account_id)) == 1
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_typed_factory_connects_owned_ledger_risk_and_market_runtime(tmp_path):
    from src.execution.broker import create_typed_paper_execution

    ledger, adapter, engine, order = setup(tmp_path)
    account_spec, rules = ledger.spec(order.account_id), adapter._rules
    ledger.close()
    service = create_typed_paper_execution(
        ledger_path=tmp_path / "venue.sqlite3",
        account=account_spec,
        rules=rules,
        fill_policy=FillPolicy(equity_cost_bps=0),
    )
    service.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    service.dispatch(order.intent_id, as_of=order.eligible_at)
    opening = bar(at=order.eligible_at + timedelta(seconds=1))
    service.adapters["paper"].advance_open(opening, {A.instrument_id: metadata()})
    assert service.reconcile_all(as_of=opening.opened_at)[0]["state"] == "filled"
    service.ledger.close()


def report_for(engine, order, fills, *, at, status="filled", complete=True, evidence="receipt"):
    return VenueOrderReport(
        venue="paper",
        client_order_id=engine.client_id(order.account_id, order.intent_id),
        venue_order_id="venue-" + order.intent_id,
        observed_at=at,
        status=status,
        cumulative_quantity=sum((f.quantity for f in fills), D(0)),
        fills=tuple(fills),
        executions_complete=complete,
        evidence_id=evidence,
    )


def test_venue_rule_defaults_preserve_canonical_hash_on_roundtrip():
    rules = VenueRules(
        venue="paper",
        instrument=A,
        symbol=A.symbol,
        available_at=T,
        quantity_step=1,
        price_tick=".01",
        market_type="equity",
        active=True,
    )
    loaded = VenueRules.model_validate_json(rules.model_dump_json())
    assert isinstance(rules.minimum_quantity, D) and isinstance(rules.minimum_notional, D)
    assert rules.model_dump_json() == loaded.model_dump_json()
    assert rules.content_hash() == loaded.content_hash()


def test_interleaved_actual_fills_after_mark_reconcile_globally_and_restart(tmp_path, monkeypatch):
    from src.domain.portfolio import PriceMark
    from tests.test_durable_accounting import fill

    ledger, adapter, engine, order = setup(tmp_path)
    second = order.model_copy(update={"intent_id": "second", "quantity": D(5)})
    for current in (order, second):
        engine.prepare(current, {A.instrument_id: metadata()}, as_of=T)
    first_at = order.eligible_at
    observed = first_at + timedelta(seconds=4)
    first = fill(order, name="first", qty="2", at=first_at)
    middle = fill(second, name="middle", qty="5", at=first_at + timedelta(seconds=1))
    last = fill(order, name="last", qty="1", at=first_at + timedelta(seconds=2))
    reports = {
        order.intent_id: report_for(engine, order, [first, last], at=observed, status="cancelled"),
        second.intent_id: report_for(engine, second, [middle], at=observed),
    }
    # Price observation may precede knowledge of the actual executions.
    ledger.mark(order.account_id, [PriceMark(instrument=A, price=90, as_of=observed)])
    for current in (order, second):
        engine._status(current.intent_id, "ambiguous")
    monkeypatch.setattr(adapter, "lookup", lambda intent, *args: reports[intent.intent_id])
    engine.reconcile_all(as_of=observed)
    assert [
        r[0]
        for r in ledger.db.execute("SELECT id FROM journal WHERE kind='fill' ORDER BY sequence")
    ] == ["fill:first", "fill:middle", "fill:last"]
    snap = ledger.snapshot(order.account_id, observed, 86400)
    assert (snap.cash, snap.nav, snap.positions[0].quantity) == (D(99200), D(99920), D(8))
    assert snap.positions[0].opened_at == first_at
    assert snap.reserved_cash == 0
    assert engine.state(order.intent_id)["state"] == "cancelled"
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()
    ledger, adapter, engine, order = setup(tmp_path)
    for key, report in reports.items():
        engine.ingest(key, report)
    assert len(ledger.fills(order.account_id)) == 3
    assert ledger.snapshot(order.account_id, observed, 86400) == snap
    ledger.close()


def test_incomplete_activity_history_never_books_a_guessed_partial_prefix(tmp_path):
    from tests.test_durable_accounting import fill

    ledger, adapter, engine, order = setup(tmp_path)
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    execution = fill(order, name="observed", qty="2")
    report = report_for(
        engine, order, [execution], at=execution.executed_at, status="partial", complete=False
    )
    engine.ingest(order.intent_id, report)
    assert not ledger.fills(order.account_id)
    assert ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    engine.ingest(
        order.intent_id,
        report.model_copy(update={"executions_complete": True, "evidence_id": "complete"}),
    )
    assert ledger.quantity(order.account_id, A.instrument_id) == 2
    assert ledger.order(order.account_id, order.intent_id)["reserved"] == 800
    ledger.close()


@pytest.mark.parametrize("conflict", ["published_nav", "later_fill"])
def test_late_receipt_cannot_silently_rewrite_committed_economics(tmp_path, conflict):
    from tests.test_durable_accounting import fill

    ledger, adapter, engine, order = setup(tmp_path)
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    execution = fill(order, name="late", qty="2")
    observed = execution.executed_at + timedelta(seconds=2)
    if conflict == "published_nav":
        ledger.snapshot(order.account_id, observed, 86400)
    else:
        ledger.apply_fill(fill(order, name="already-booked", qty="1", at=observed))
    before = ledger.reconcile(order.account_id)
    report = report_for(engine, order, [execution], at=observed, status="partial")
    if conflict == "later_fill":
        report = report.model_copy(update={"cumulative_quantity": D(3)})
    with pytest.raises(LedgerError, match="NAV reconciliation|economic chronology"):
        engine.ingest(order.intent_id, report)
    assert engine.state(order.intent_id)["state"] == "reconciliation_required"
    assert ledger.reconcile(order.account_id) == before
    assert (
        ledger.db.execute("SELECT applied FROM venue_reports WHERE evidence='receipt'").fetchone()[
            0
        ]
        == 0
    )
    ledger.close()


def test_process_exit_after_durable_receipt_recovers_without_remote_lookup(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    program = """
import os,sys
from pathlib import Path
from tests.test_venue_execution import setup,report_for
from tests.test_durable_accounting import A,T,metadata,fill
ledger,adapter,engine,order=setup(Path(sys.argv[1]))
engine.prepare(order,{A.instrument_id:metadata()},as_of=T)
execution=fill(order)
report=report_for(engine,order,[execution],at=execution.executed_at)
ledger.apply_fill=lambda *args,**kwargs: os._exit(87)
engine.ingest(order.intent_id,report)
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(Path.cwd()), "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert completed.returncode == 87
    ledger, adapter, engine, order = setup(tmp_path)
    with pytest.raises(LedgerError, match="unapplied venue receipts"):
        engine.dispatch(order.intent_id, as_of=order.eligible_at)
    adapter.lookup = lambda *args: pytest.fail("durable terminal receipt needs no remote call")
    engine.reconcile_all(as_of=order.eligible_at)
    assert engine.state(order.intent_id)["state"] == "filled"
    assert len(ledger.fills(order.account_id)) == 1
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_observed_opening_receipt_uses_actual_print_and_frozen_admission_after_capture(tmp_path):
    import json

    from src.execution.venue import ObservedOpenReceipt

    ledger, adapter, engine, order = setup(tmp_path)
    original_metadata = {A.instrument_id: metadata()}
    engine.prepare(order, original_metadata, as_of=T)
    engine.dispatch(order.intent_id, as_of=order.eligible_at)
    opening = bar(at=order.eligible_at + timedelta(seconds=1), opening_volume="500")
    observed = opening.closed_at + timedelta(seconds=1)
    available = observed + timedelta(seconds=1)
    cutoff = available + timedelta(minutes=1)
    kwargs = dict(
        observed_at=observed,
        available_at=available,
        as_of=cutoff,
        evidence_id="actual-test-print",
        evidence_hash="a" * 64,
    )
    frozen = {order.intent_id: engine.admission_metadata(order.intent_id)}
    assert frozen[order.intent_id] == original_metadata
    assert order.intent_id not in adapter.accepted_intent_ids_before(order.eligible_at)
    assert order.intent_id in adapter.accepted_intent_ids_before(opening.opened_at)
    changed = {order.intent_id: {A.instrument_id: metadata(sector="changed")}}
    with pytest.raises(LedgerError, match="differs from durable admission"):
        adapter.settle_open(opening, changed, **kwargs)
    with pytest.raises(ValueError, match="receipt clocks"):
        adapter.settle_open(
            opening, frozen, **{**kwargs, "available_at": observed - timedelta(seconds=1)}
        )
    with pytest.raises(ValueError, match="receipt clocks"):
        adapter.settle_open(
            opening, frozen, **{**kwargs, "as_of": available - timedelta(seconds=1)}
        )
    fills = adapter.settle_open(opening, frozen, **kwargs)
    assert len(fills) == 1 and fills[0].quantity == 5 and fills[0].executed_at == opening.opened_at
    receipt = ObservedOpenReceipt.model_validate_json(
        ledger.db.execute(
            "SELECT payload FROM events WHERE id=?", ("paper-observed-open:" + opening.event_id,)
        ).fetchone()[0]
    )
    assert receipt.occurred_at < receipt.observed_at < receipt.available_at < receipt.as_of
    assert receipt.event_hash == OpeningEvent.from_bar(opening).content_hash()
    assert json.loads(receipt.model_dump_json())["evidence_hash"] == "a" * 64
    adapter.advance_close(opening)
    snap = ledger.snapshot(order.account_id, cutoff, 86400)
    assert snap.positions[0].opened_at == opening.opened_at
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()
    ledger, adapter, engine, _ = setup(tmp_path)
    assert (
        adapter.settle_open(opening, frozen, **{**kwargs, "as_of": cutoff + timedelta(seconds=1)})
        == []
    )
    assert len(ledger.fills(order.account_id)) == 1
    with pytest.raises(LedgerError, match="receipt changed"):
        adapter.settle_open(opening, frozen, **{**kwargs, "evidence_hash": "b" * 64})
    ledger.close()


def test_acceptance_at_open_and_later_cycles_cannot_consume_past_prints(tmp_path):
    ledger, adapter, engine, order = setup(tmp_path)
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    engine.dispatch(order.intent_id, as_of=order.eligible_at)
    opening = bar(at=order.eligible_at)
    frozen = {order.intent_id: engine.admission_metadata(order.intent_id)}
    kwargs = dict(
        observed_at=opening.closed_at,
        available_at=opening.closed_at,
        as_of=opening.closed_at,
        evidence_id="fixture",
        evidence_hash="c" * 64,
    )
    with pytest.raises(LedgerError, match="coverage"):
        adapter.settle_open(opening, frozen, **kwargs)
    assert adapter.settle_open(opening, {}, **kwargs) == []
    assert not ledger.fills(order.account_id)
    ledger.snapshot(order.account_id, opening.closed_at, 86400)
    later = bar(at=opening.opened_at + timedelta(seconds=1))
    with pytest.raises(LedgerError, match="published NAV"):
        adapter.settle_open(
            later,
            frozen,
            **{
                **kwargs,
                "observed_at": later.closed_at,
                "available_at": later.closed_at,
                "as_of": later.closed_at,
            },
        )
    # A caller cannot backdate a new acknowledgement behind committed cycle state.
    with pytest.raises(LedgerError, match="backwards"):
        adapter.submit(
            order,
            engine.client_id(order.account_id, order.intent_id),
            adapter.rules(A, T),
            order.eligible_at,
        )
    ledger.close()


def test_observed_receipt_reader_recovers_original_metadata_after_full_fill_restart(tmp_path):
    ledger, adapter, engine, order = setup(tmp_path)
    engine.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    engine.dispatch(order.intent_id, as_of=order.eligible_at)
    opening = bar(at=order.eligible_at + timedelta(seconds=1))
    clocks = dict(
        observed_at=opening.closed_at,
        available_at=opening.closed_at,
        as_of=opening.closed_at,
        evidence_id="fixture-print",
        evidence_hash="d" * 64,
    )
    assert adapter.observed_open_receipt(opening.event_id) is None
    adapter.settle_open(
        opening, {order.intent_id: engine.admission_metadata(order.intent_id)}, **clocks
    )
    assert not ledger.pending(order.account_id)
    ledger.close()
    ledger, adapter, engine, _ = setup(tmp_path)
    receipt = adapter.observed_open_receipt(opening.event_id)
    original = {key: engine.admission_metadata(key) for key, _ in receipt.admission_metadata_hashes}
    assert set(original) == {order.intent_id}
    assert (
        adapter.settle_open(
            opening, original, **{**clocks, "as_of": opening.closed_at + timedelta(seconds=1)}
        )
        == []
    )
    assert len(ledger.fills(order.account_id)) == 1
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()
