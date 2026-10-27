"""Actual posting semantics with synthetic vendor fee receipts only."""

from datetime import timedelta
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from src.domain.portfolio import AccountFee, AccountSpec
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.simulator import PaperExecution
from src.execution.sleeves import SleeveBook, exact_sum
from src.execution.venue import VenueExecution
from tests.test_durable_accounting import A, T, account, bar, metadata
from tests.test_sleeve_accounting import setup


def fee(account_id="backtest:test", amount=".019876543210987654321"):
    return AccountFee(
        fee_id="actual-fee",
        account_id=account_id,
        amount=amount,
        currency="USD",
        effective_date=T.date(),
        observed_at=T + timedelta(days=1),
        source_hash="a" * 64,
    )


def test_separately_posted_fee_is_expense_not_fake_trade_or_capital_flow(tmp_path):
    path = tmp_path / "fees.sqlite3"
    ledger = account(path)
    actual = fee()
    assert ledger.apply_account_fee(actual)
    snap = ledger.snapshot(actual.account_id, actual.observed_at, 86400)
    assert snap.cash == snap.nav == D(100000) - actual.amount
    assert snap.fees == actual.amount and snap.realized_pnl == -actual.amount
    assert not ledger.fills(actual.account_id)
    assert ledger.reconcile(actual.account_id)["reconciled"]
    ledger.close()
    ledger = PortfolioLedger(path)
    assert not ledger.apply_account_fee(
        actual.model_copy(update={"observed_at": actual.observed_at + timedelta(days=1)})
    )
    with pytest.raises(LedgerError, match="identity/content changed"):
        ledger.apply_account_fee(actual.model_copy(update={"amount": D(1)}))
    assert ledger.history(actual.account_id) == [snap]
    ledger.close()


def test_account_fee_requires_cash_currency_causality_and_retains_published_nav(tmp_path):
    ledger = account(tmp_path / "invalid.sqlite3", cash="10")
    actual = fee()
    with pytest.raises(LedgerError, match="unreserved cash"):
        ledger.apply_account_fee(actual.model_copy(update={"amount": D(11)}))
    with pytest.raises(LedgerError, match="base-currency"):
        ledger.apply_account_fee(actual.model_copy(update={"currency": "EUR"}))
    snap = ledger.snapshot(actual.account_id, actual.observed_at, 86400)
    with pytest.raises(LedgerError, match="published NAV"):
        ledger.apply_account_fee(actual)
    assert ledger.history(actual.account_id) == [snap]
    assert ledger.reconcile(actual.account_id)["fees"] == "0"
    ledger.close()


def test_posted_fee_attribution_conserves_exact_sleeve_cash_nav_and_restart(tmp_path):
    path = tmp_path / "sleeves.sqlite3"
    ledger, book = setup(path)
    actual = fee(amount=".03")
    allocation = {"alpha": D(".01"), "beta": D(".02")}
    assert book.apply_account_fee(actual, allocation, attribution_source_hash="b" * 64)
    views = book.snapshot(actual.observed_at)
    canonical = ledger.snapshot(actual.account_id, actual.observed_at, 86400)
    assert exact_sum(v.portfolio.nav for v in views.values()) == canonical.nav == D("9999.97")
    assert exact_sum(v.portfolio.fees for v in views.values()) == canonical.fees == actual.amount
    assert [v.net_capital_flow for v in views.values()] == [D(0), D(0)]
    assert not ledger.fills(actual.account_id)
    ledger.close()
    ledger = PortfolioLedger(path)
    book = SleeveBook(ledger, actual.account_id)
    assert not book.apply_account_fee(actual, allocation, attribution_source_hash="b" * 64)
    assert book.snapshot(actual.observed_at) == views
    with pytest.raises(LedgerError, match="immutable account fee attribution"):
        book.apply_account_fee(
            actual, {"alpha": D(".02"), "beta": D(".01")}, attribution_source_hash="b" * 64
        )
    assert ledger.reconcile(actual.account_id)["reconciled"]
    ledger.close()


def test_fee_cannot_implicitly_move_capital_or_leave_partial_projection(tmp_path):
    ledger, book = setup(tmp_path / "funding.sqlite3")
    actual = fee(amount="6000")
    with pytest.raises(LedgerError, match="unreserved sleeve cash"):
        book.apply_account_fee(actual, {"alpha": D(6000)}, attribution_source_hash="b" * 64)
    assert ledger.reconcile(actual.account_id)["cash"] == "10000"
    assert not ledger.db.execute("SELECT 1 FROM sleeve_fees").fetchone()
    ledger.close()


def test_live_namespace_requires_independent_authorization_and_cannot_simulate(tmp_path):
    ledger = PortfolioLedger(tmp_path / "live.sqlite3")
    spec = AccountSpec(
        account_id="live:test", mode="live", base_currency="USD", initial_cash=1000, opened_at=T
    )
    ledger.create_account(spec)
    adapter = SimpleNamespace(venue="fixture", mode="live", owner="direct")
    with pytest.raises(LedgerError, match="independent authorization"):
        VenueExecution(
            ledger,
            spec.account_id,
            {"fixture": adapter},
            {A.instrument_id: "fixture"},
            owner="direct",
        )
    VenueExecution(
        ledger,
        spec.account_id,
        {"fixture": adapter},
        {A.instrument_id: "fixture"},
        owner="direct",
        allow_live=True,
    )
    with pytest.raises(LedgerError, match="cannot receive simulated"):
        PaperExecution(ledger, spec.account_id).open_bar(bar(), {A.instrument_id: metadata()})
    ledger.close()


def test_captured_venue_balances_must_precede_cutoff_and_remain_fresh(tmp_path, monkeypatch):
    from tests.test_venue_execution import setup as venue_setup

    ledger, adapter, service, order = venue_setup(tmp_path)
    capture = adapter.balances(T)
    monkeypatch.setattr(adapter, "balances", lambda at: capture)
    at = T + timedelta(seconds=10)
    candidate = order.model_copy(
        update={"decision_at": at, "eligible_at": at + timedelta(seconds=1)}
    )
    assert (
        service.prepare(candidate, {A.instrument_id: metadata()}, as_of=at)["state"] == "prepared"
    )
    for invalid_at, observed in ((at, at + timedelta(seconds=1)), (T + timedelta(seconds=31), T)):
        capture = capture.model_copy(update={"observed_at": observed})
        candidate = order.model_copy(
            update={
                "intent_id": str(invalid_at),
                "decision_at": invalid_at,
                "eligible_at": invalid_at + timedelta(seconds=1),
            }
        )
        with pytest.raises(LedgerError, match="funding evidence clock"):
            service.prepare(candidate, {A.instrument_id: metadata()}, as_of=invalid_at)
    ledger.close()
