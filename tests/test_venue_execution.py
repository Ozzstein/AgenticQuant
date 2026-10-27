"""Durable venue owner/outbox invariants; remote responses are test doubles only."""

from datetime import timedelta
from decimal import Decimal as D

import pytest

from src.domain.portfolio import AccountSpec, Fill, FillPolicy
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
    opening = bar(at=order.eligible_at, opening_volume="500")
    fills = adapter.advance_open(opening, {A.instrument_id: metadata()})
    assert sum(f.quantity for f in fills) == 5
    assert engine.reconcile_all(as_of=order.eligible_at)[0]["state"] == "partial"
    assert ledger.quantity(order.account_id, A.instrument_id) == 5
    engine.reconcile_all(as_of=order.eligible_at)
    assert len(ledger.fills(order.account_id)) == 1
    assert engine.reconcile_balances(as_of=order.eligible_at)["reconciled"]
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
    adapter.advance_open(bar(at=order.eligible_at), {A.instrument_id: metadata()})
    engine.reconcile_all(as_of=order.eligible_at)
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
    service.adapters["paper"].advance_open(bar(at=order.eligible_at), {A.instrument_id: metadata()})
    assert service.reconcile_all(as_of=order.eligible_at)[0]["state"] == "filled"
    service.ledger.close()
