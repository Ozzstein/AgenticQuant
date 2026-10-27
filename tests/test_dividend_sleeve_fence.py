"""A published sleeve attribution cannot be retroactively changed by income."""

from datetime import datetime, timedelta
from decimal import Decimal as D

import pytest

from src.domain.portfolio import AccountSpec, PriceMark
from src.execution.ledger import LedgerError
from src.execution.sleeves import SleeveBook
from tests.test_dividend_accounting import buy, distribution, terms
from tests.test_durable_accounting import A, T, account


def prepared(path):
    ledger = account(path, cash="10000")
    book = SleeveBook(ledger, "backtest:test").initialize({"owner": D(10000), "empty": D(0)})
    declared = terms()
    ledger.register_cash_dividend(book.account_id, declared, as_of=T)
    buy(ledger, book=book, allocations={"owner": D(10)})
    ledger.mark(
        book.account_id, [PriceMark(instrument=A, price=D(100), as_of=T + timedelta(seconds=1))]
    )
    return ledger, book, declared


@pytest.mark.parametrize("later_seconds", [0, 1])
@pytest.mark.parametrize("event", ["entitlement", "payment"])
def test_sleeve_publication_fences_new_income_after_restart(tmp_path, event, later_seconds):
    path = tmp_path / "published.sqlite"
    ledger, book, declared = prepared(path)
    notice = distribution(declared)
    if event == "payment":
        book.accrue_dividend(declared.action_id, at=declared.entitlement_at)
    event_at = declared.entitlement_at if event == "entitlement" else notice.available_at
    publication_at = event_at + timedelta(seconds=later_seconds)
    before = book.snapshot(publication_at, max_price_age_seconds=864000)
    assert not ledger.history(book.account_id)
    assert datetime.fromisoformat(ledger._account(book.account_id)["last_at"]) < event_at
    rows = [tuple(row) for row in ledger.db.execute("SELECT * FROM sleeve_observations")]
    assert rows
    ledger.close()

    ledger = account(path, cash="10000")
    book = SleeveBook(ledger, "backtest:test")
    with pytest.raises(LedgerError, match="published NAV in sleeve attribution"):
        if event == "entitlement":
            book.accrue_dividend(declared.action_id, at=declared.entitlement_at)
        else:
            book.apply_dividend_distribution(notice, as_of=notice.available_at)
    assert [tuple(row) for row in ledger.db.execute("SELECT * FROM sleeve_observations")] == rows
    assert book.snapshot(publication_at, 864000, persist=False) == before
    assert not ledger.db.execute("SELECT 1 FROM dividend_payments").fetchone()
    assert not ledger.db.execute(
        "SELECT 1 FROM events WHERE id LIKE 'dividend-distribution:%'"
    ).fetchone()
    if event == "entitlement":
        assert not ledger.dividend_entitlements(book.account_id, as_of=publication_at)
    else:
        assert ledger.dividend_receivables(book.account_id)[0].outstanding == 10
    assert ledger.reconcile(book.account_id)["reconciled"]
    ledger.close()


def test_unpublished_preview_and_other_account_do_not_fence_income(tmp_path):
    ledger, book, declared = prepared(tmp_path / "isolated.sqlite")
    other = "backtest:other"
    ledger.create_account(
        AccountSpec(
            account_id=other, mode="backtest", base_currency="USD", initial_cash=D(100), opened_at=T
        )
    )
    later = declared.entitlement_at + timedelta(seconds=1)
    SleeveBook(ledger, other).initialize({"other": D(100)}).snapshot(later)
    book.snapshot(later, persist=False)
    earned = book.accrue_dividend(declared.action_id, at=declared.entitlement_at)
    assert earned.quantity == 10
    notice = distribution(declared)
    book.snapshot(notice.available_at, 864000, persist=False)
    assert book.apply_dividend_distribution(notice, as_of=notice.available_at)
    book.snapshot(notice.available_at, 864000)
    # These exact events are already incorporated in the published attribution.
    assert book.accrue_dividend(declared.action_id, at=declared.entitlement_at) == earned
    assert not book.apply_dividend_distribution(notice, as_of=notice.available_at)
    assert ledger.reconcile(book.account_id)["reconciled"]
    ledger.close()
