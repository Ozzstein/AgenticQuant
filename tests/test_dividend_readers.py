"""Public dividend readers recover authoritative schedules and receipts after restart."""

from datetime import timedelta
from decimal import Decimal as D

import pytest

from src.execution.ledger import LedgerError, PortfolioLedger
from tests.test_dividend_accounting import buy, distribution, terms
from tests.test_durable_accounting import T, account
from tests.test_paper_dividend_distribution import prepared


def test_admitted_terms_survive_omitted_refresh_and_respect_actual_admission_cutoffs(tmp_path):
    path = tmp_path / "schedule.sqlite"
    ledger = account(path)
    first = terms()
    admitted = T + timedelta(hours=1)
    ledger.register_cash_dividend("backtest:test", first, as_of=admitted)
    assert ledger.cash_dividends("backtest:test", as_of=T) == ()
    assert ledger.cash_dividends("backtest:test", as_of=admitted) == (first,)
    second = terms(
        revision_id="r2",
        supersedes_hash=first.content_hash(),
        cash_per_unit=D(2),
        available_at=T + timedelta(hours=2),
    )
    ledger.register_cash_dividend("backtest:test", second, as_of=second.available_at)
    assert ledger.cash_dividends("backtest:test", as_of=admitted) == (first,)
    assert ledger.cash_dividends("backtest:test", as_of=second.available_at) == (second,)
    ledger.close()
    ledger = account(path)
    assert ledger.cash_dividends("backtest:test", as_of=second.ex_at) == (second,)
    assert ledger.dividend_entitlements("backtest:test", as_of=second.ex_at) == ()
    # H can schedule previously admitted ex events using canonical retained terms.
    with ledger.atomic():
        actual = ledger.accrue_dividend("backtest:test", second.action_id, at=second.ex_at)
        assert ledger.dividend_entitlements("backtest:test", as_of=second.ex_at) == (actual,)
    ledger.close()
    archive = PortfolioLedger(path, read_only=True)
    assert archive.cash_dividends("backtest:test", as_of=second.ex_at) == (second,)
    assert archive.dividend_entitlements("backtest:test", as_of=second.ex_at) == (actual,)
    assert archive.dividend_payments("backtest:test", as_of=second.ex_at) == ()
    archive.close()


def test_original_paper_payment_receipt_and_entitlement_read_back_exactly_after_restart(tmp_path):
    ledger, _, book, original, notice, policy, _ = prepared(tmp_path)
    book.apply_paper_dividend_distribution(notice, policy=policy, as_of=notice.available_at)
    assert (
        ledger.dividend_entitlements(book.account_id, as_of=original.at - timedelta(microseconds=1))
        == ()
    )
    assert ledger.dividend_payments(book.account_id, as_of=notice.observed_at) == ()
    expected = notice.payment(original, paper_policy=policy)
    assert ledger.dividend_payments(book.account_id, as_of=notice.available_at) == (expected,)
    ledger.close()
    archive = PortfolioLedger(tmp_path / "venue.sqlite3", read_only=True)
    assert archive.dividend_payments(
        book.account_id, as_of=notice.available_at + timedelta(days=1)
    ) == (expected,)
    assert archive.dividend_entitlements(book.account_id, as_of=notice.available_at) == (original,)
    archive.close()


def test_public_readers_reject_changed_projection_and_terms_source(tmp_path):
    ledger = account(tmp_path / "tamper.sqlite")
    declared = terms()
    ledger.register_cash_dividend("backtest:test", declared, as_of=T)
    buy(ledger)
    ledger.accrue_dividend("backtest:test", declared.action_id, at=declared.ex_at)
    notice = distribution(declared)
    ledger.apply_dividend_distribution("backtest:test", notice, as_of=notice.available_at)
    ledger.db.execute("DELETE FROM dividend_payments WHERE account=?", ("backtest:test",))
    for reader in (ledger.dividend_entitlements, ledger.dividend_payments):
        with pytest.raises(LedgerError, match="payment projection"):
            reader("backtest:test", as_of=notice.available_at)
    ledger.db.execute(
        "DELETE FROM events WHERE id=?", ("dividend-admission:" + declared.content_hash(),)
    )
    with pytest.raises(LedgerError, match="source chain"):
        ledger.cash_dividends("backtest:test", as_of=notice.available_at)
    ledger.close()
