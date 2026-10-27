"""Local paper income uses real-sized ledger ownership over synthetic source fixtures."""

from datetime import timedelta
from decimal import Decimal as D

import pytest

from src.domain.portfolio import AccountSpec, PaperDividendPolicy, PriceMark, exact_decimal_sum
from src.execution.alpaca_venue import AlpacaVenue
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.sleeves import SleeveBook
from tests.test_dividend_accounting import distribution, terms
from tests.test_durable_accounting import A, T, bar, metadata
from tests.test_venue_execution import setup


def prepared(tmp_path, *, quantity=10):
    ledger, adapter, execution, order = setup(tmp_path)
    book = SleeveBook(ledger, order.account_id).initialize({"owner": D(100000), "empty": D(0)})
    order = order.model_copy(update={"quantity": D(quantity)})
    book.register_intent(order, {"owner": D(quantity)}, source_hash="a" * 64, quantity_quantum=D(1))
    execution.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    execution.dispatch(order.intent_id, as_of=order.eligible_at)
    opening = bar(at=order.eligible_at + timedelta(seconds=1))
    adapter.settle_open(
        opening,
        {order.intent_id: execution.admission_metadata(order.intent_id)},
        observed_at=opening.closed_at,
        available_at=opening.closed_at,
        as_of=opening.closed_at,
        evidence_id="actual-test-print",
        evidence_hash="f" * 64,
    )
    adapter.advance_close(opening)
    book.sync()
    # Archived terms existed before ex. They may be processed in a later catch-up
    # cycle while the committed ledger is still before ex, without a prior run.
    declared = terms()
    ledger.register_cash_dividend(order.account_id, declared, as_of=opening.closed_at)
    entitlement = book.accrue_dividend(declared.action_id, at=declared.ex_at)
    ledger.mark(order.account_id, [PriceMark(instrument=A, price=99, as_of=declared.ex_at)])
    snapshot = book.snapshot(declared.ex_at)
    notice = distribution(declared)
    policy = PaperDividendPolicy(
        policy_id="local-source-policy",
        account_id=order.account_id,
        distribution_hash=notice.content_hash(),
        convention="source-confirmed-paper-distribution-1",
        fee_tax_convention="source-confirmed-per-unit-deductions-1",
        available_at=notice.available_at,
        source_id="deductions-applicability",
        source_hash="9" * 64,
    )
    return ledger, adapter, book, entitlement, notice, policy, snapshot


def test_local_paper_actual_entitlement_net_cash_sleeves_and_restart(tmp_path):
    ledger, adapter, book, original, notice, policy, before = prepared(tmp_path)
    assert original.quantity == 10
    assert before["owner"].portfolio.receivable_balance == 10
    assert book.apply_paper_dividend_distribution(notice, policy=policy, as_of=notice.available_at)
    payment = notice.payment(original, paper_policy=policy)
    assert (
        payment.origin == "paper-distribution"
        and payment.sizing_policy_hash == policy.content_hash()
    )
    assert payment.gross_amount == 10 and payment.net_amount == 8
    views = book.snapshot(notice.available_at)
    assert views["owner"].portfolio.cash == 99008 and views["empty"].portfolio.nav == 0
    assert views["owner"].portfolio.nav == 99998
    assert views["owner"].portfolio.fees == D(".5")
    assert views["owner"].portfolio.withholding_taxes == D("1.5")
    assert views["owner"].portfolio.receivable_balance == 0
    for field in ("cash", "nav", "fees", "withholding_taxes", "receivable_balance"):
        actual = ledger.snapshot(book.account_id, notice.available_at, 345600, persist=False)
        assert exact_decimal_sum(getattr(v.portfolio, field) for v in views.values()) == getattr(
            actual, field
        )
    assert ledger.reconcile(book.account_id)["reconciled"]
    ledger.close()
    ledger, adapter, _, order = setup(tmp_path)
    book = SleeveBook(ledger, order.account_id)
    assert not book.apply_paper_dividend_distribution(
        notice, policy=policy, as_of=notice.available_at
    )
    assert ledger.reconcile(order.account_id)["reconciled"]
    # Source and actual application policy are part of reconciliation, not labels.
    ledger.db.execute(
        "DELETE FROM events WHERE account=? AND id=?",
        (order.account_id, "paper-dividend-policy:" + policy.content_hash()),
    )
    with pytest.raises(LedgerError, match="retained sizing policy"):
        ledger.reconcile(order.account_id)
    ledger.close()


def test_same_actual_notice_sizes_each_local_entitlement_without_reusing_other_account_amount(
    tmp_path,
):
    payments = []
    for name, quantity in (("candidate", 10), ("control", 5)):
        ledger, _, book, original, notice, policy, _ = prepared(tmp_path / name, quantity=quantity)
        book.apply_paper_dividend_distribution(notice, policy=policy, as_of=notice.available_at)
        payments.append(notice.payment(original, paper_policy=policy))
        assert ledger.reconcile(book.account_id)["reconciled"]
        ledger.close()
    assert payments[0].source_hash == payments[1].source_hash
    assert payments[0].net_amount == 8 and payments[1].net_amount == 4
    assert payments[0].entitlement_hash != payments[1].entitlement_hash


def test_paper_policy_wrong_account_notice_timing_and_published_nav_fail_atomically(tmp_path):
    ledger, _, book, original, notice, policy, _ = prepared(tmp_path)
    for changed in (
        policy.model_copy(update={"account_id": "paper:someone-else"}),
        policy.model_copy(update={"distribution_hash": "0" * 64}),
        policy.model_copy(update={"available_at": notice.available_at + timedelta(seconds=1)}),
    ):
        with pytest.raises(ValueError, match="paper dividend policy"):
            book.apply_paper_dividend_distribution(
                notice, policy=changed, as_of=notice.available_at
            )
    with pytest.raises(LedgerError, match="source clocks"):
        book.apply_paper_dividend_distribution(notice, policy=policy, as_of=notice.observed_at)
    assert ledger.db.execute("SELECT count(*) FROM dividend_payments").fetchone()[0] == 0
    assert (
        ledger.db.execute(
            "SELECT count(*) FROM events WHERE id LIKE 'paper-dividend-policy:%'"
        ).fetchone()[0]
        == 0
    )
    ledger.snapshot(book.account_id, notice.available_at, 345600)
    with pytest.raises(LedgerError, match="published NAV"):
        book.apply_paper_dividend_distribution(notice, policy=policy, as_of=notice.available_at)
    assert ledger.dividend_receivables(book.account_id)[0].outstanding == 10
    ledger.close()


def test_local_paper_cannot_become_remote_or_relabel_sized_amount_as_broker_posting(tmp_path):
    ledger, adapter, book, original, notice, policy, _ = prepared(tmp_path)
    book.apply_paper_dividend_distribution(notice, policy=policy, as_of=notice.available_at)
    with pytest.raises(LedgerError, match="remote execution owner"):
        AlpacaVenue(
            ledger,
            book.account_id,
            broker_account_id="fixture",
            api_key="fixture",
            api_secret="fixture",
            rules={A.instrument_id: adapter.rules(A, T).model_copy(update={"venue": "alpaca"})},
        )
    posted = notice.payment(original).model_copy(
        update={"origin": "posted-account", "payment_id": "broker-id"}
    )
    with pytest.raises(LedgerError, match="cannot mix broker-posted"):
        ledger.apply_dividend_payment(posted, as_of=notice.available_at)
    assert ledger.reconcile(book.account_id)["reconciled"]
    ledger.close()


def test_remote_paper_binding_cannot_receive_local_sized_distribution(tmp_path):
    ledger, _, book, _, notice, policy, _ = prepared(tmp_path)
    ledger.record_event(book.account_id, "remote-binding:alpaca", '{"broker_account_id":"fixture"}')
    with pytest.raises(LedgerError, match="exclusively local"):
        book.apply_paper_dividend_distribution(notice, policy=policy, as_of=notice.available_at)
    assert ledger.dividend_receivables(book.account_id)[0].outstanding == 10
    ledger.close()


def test_terms_first_available_after_ex_and_processing_after_account_advanced_stay_blocked(
    tmp_path,
):
    ledger, _, _, order = setup(tmp_path)
    late = terms(available_at=T + timedelta(days=2))
    with pytest.raises(LedgerError, match="causal"):
        ledger.register_cash_dividend(order.account_id, late, as_of=late.available_at)
    ledger.snapshot(order.account_id, T + timedelta(days=2), 0)
    with pytest.raises(LedgerError, match="backdate"):
        ledger.register_cash_dividend(order.account_id, terms(), as_of=T)
    ledger.close()


@pytest.mark.parametrize("mode", ["live", "backtest"])
def test_local_paper_api_never_synthesizes_live_or_silently_changes_replay_policy(tmp_path, mode):
    ledger = PortfolioLedger(tmp_path / "other-mode.sqlite")
    account = mode + ":other"
    ledger.create_account(
        AccountSpec(
            account_id=account, mode=mode, base_currency="USD", initial_cash=10000, opened_at=T
        )
    )
    notice = distribution(terms())
    policy = PaperDividendPolicy(
        policy_id="wrong-mode",
        account_id=account,
        distribution_hash=notice.content_hash(),
        convention="source-confirmed-paper-distribution-1",
        fee_tax_convention="source-confirmed-per-unit-deductions-1",
        available_at=notice.available_at,
        source_id="policy",
        source_hash="a" * 64,
    )
    with pytest.raises(LedgerError, match="exclusively local"):
        ledger.apply_paper_dividend_distribution(
            account, notice, policy=policy, as_of=notice.available_at
        )
    assert ledger.reconcile(account)["cash"] == "10000"
    ledger.close()
