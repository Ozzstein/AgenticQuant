"""Actual ledger valuation at the later desk clock; provider only is synthetic."""

# ruff: noqa: F811
from datetime import timedelta
from decimal import Decimal as D
from types import SimpleNamespace

import pytest
from src.domain.portfolio import AccountSpec, Fill, OrderIntent, PriceMark
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.sleeves import SleeveBook

from src.agents.contracts import PortfolioReviewContext
from src.agents.graph import analyze_instrument
from src.operations.cycle import PaperCycleService
from tests.test_agent_desk_invariants import (  # noqa: F401
    GenerativeFixture,
    decision_context,
    decision_instrument,
    decision_source,
    desk_config,
    desk_hmm,
    desk_sources,
    jev_service,
)


@pytest.fixture
def clock_ledger(tmp_path, decision_context, decision_instrument):
    at = decision_context.as_of
    ledger = PortfolioLedger(tmp_path / "actual.sqlite")
    account = "paper:actual-ledger"
    ledger.create_account(
        AccountSpec(
            account_id=account,
            mode="paper",
            base_currency="USD",
            initial_cash=D("1000"),
            opened_at=at,
        )
    )
    book = SleeveBook(ledger, account).initialize({"alpha": D("600"), "beta": D("400")})
    order = OrderIntent(
        intent_id="buy",
        account_id=account,
        run_id="earlier",
        instrument=decision_instrument,
        side="buy",
        quantity=D("2"),
        decision_at=at,
        eligible_at=at + timedelta(seconds=1),
        reference_price=D("100"),
        estimated_fee=D("0"),
        strategy_id="alpha",
    )
    book.register_intent(order, {"alpha": D("2")}, source_hash="f" * 64, quantity_quantum=D("1"))
    ledger.submit(order)
    ledger.apply_fill(
        Fill(
            fill_id="f",
            intent_id=order.intent_id,
            account_id=account,
            quantity=D("2"),
            price=D("100"),
            fee=D("0"),
            executed_at=order.eligible_at,
            event_id="print",
        )
    )
    ledger.mark(
        account,
        [PriceMark(instrument=decision_instrument, price=D("101"), as_of=order.eligible_at)],
    )
    service = object.__new__(PaperCycleService)
    service.ledger = ledger
    plan = SimpleNamespace(
        inputs=SimpleNamespace(risk_policy=SimpleNamespace(max_price_age_seconds=10))
    )
    context = decision_context.model_copy(
        update={"as_of": at + timedelta(seconds=3), "account_id": account}
    )
    yield service, plan, context
    ledger.close()


def test_later_snapshot_is_actual_persisted_ledger_and_sleeves(clock_ledger):
    service, plan, context = clock_ledger
    snap = service._desk_snapshot(plan, context)
    assert snap.account_id == context.account_id
    assert snap.as_of == context.as_of
    assert snap.nav == D("1002") and snap.cash == D("800")
    assert snap.positions[0].marked_at < snap.as_of
    assert snap.sleeves["alpha"].portfolio.positions[0].quantity == D("2")
    assert sum(s.portfolio.nav for s in snap.sleeves.values()) == snap.nav
    assert (
        service.ledger.db.execute(
            "SELECT COUNT(*) FROM snapshots WHERE account=? AND at=?",
            (context.account_id, context.as_of.isoformat()),
        ).fetchone()[0]
        == 1
    )
    assert (
        service.ledger.db.execute(
            "SELECT COUNT(*) FROM sleeve_observations WHERE account=? AND at=?",
            (context.account_id, context.as_of.isoformat()),
        ).fetchone()[0]
        == 2
    )
    assert service._desk_snapshot(plan, context) == snap


def test_later_clock_rechecks_mark_freshness_without_creating_prices(clock_ledger):
    service, plan, context = clock_ledger
    late = context.model_copy(update={"as_of": context.as_of + timedelta(seconds=20)})
    with pytest.raises(LedgerError, match="stale/future"):
        service._desk_snapshot(plan, late)
    assert service.ledger.db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 0


def test_later_snapshot_enables_actual_cro_call(
    clock_ledger, tmp_path, desk_config, desk_sources, desk_hmm, decision_instrument
):
    service, plan, context = clock_ledger
    snap = service._desk_snapshot(plan, context)
    calls = []
    decision = analyze_instrument(
        decision_instrument,
        context,
        sources=desk_sources,
        hmm_context=desk_hmm.model_copy(update={"as_of": context.as_of}),
        portfolio_context=PortfolioReviewContext(
            as_of=snap.as_of,
            snapshot_id=snap.content_hash(),
            portfolio_json=snap.model_dump_json(),
            correlation_json='{"asset":{"asset":1}}',
        ),
        config=desk_config,
        decision_service=jev_service(tmp_path),
        llm_factory=lambda role: GenerativeFixture(role, calls),
    )
    assert [r.role for r in decision.reports if r.role == "risk_manager" and r.status == "ok"] == [
        "risk_manager"
    ]
    assert len([role for role, _ in calls if role == "risk_manager"]) == 1
