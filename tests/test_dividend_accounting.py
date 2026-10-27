"""Actual ledger transitions over synthetic dividend notices, never provider proof."""

from datetime import timedelta
from decimal import Decimal as D

import pytest

from src.domain.portfolio import CashDividend, DividendDistribution, PriceMark, exact_decimal_sum
from src.execution.ledger import LedgerError
from src.execution.sleeves import SleeveBook
from tests.test_durable_accounting import A, T, account, fill, intent


def terms(**changes):
    ex = T + timedelta(days=1)
    return CashDividend(
        **{
            "action_id": "div-1",
            "revision_id": "r1",
            "instrument": A,
            "ex_at": ex,
            "entitlement_at": ex,
            "record_date": ex.date(),
            "payable_date": (ex + timedelta(days=3)).date(),
            "published_at": None,
            "available_at": T,
            "cash_per_unit": D(1),
            "currency": "USD",
            "convention": "pre-ex-beneficial-ownership-1",
            "entitlement_source_id": "beneficial-owner-rule",
            "entitlement_source_hash": "a" * 64,
            "source_id": "actual-fixture-terms",
            "source_hash": "b" * 64,
            **changes,
        }
    )


def distribution(t, **changes):
    paid = T + timedelta(days=4)
    return DividendDistribution(
        **{
            "distribution_id": "actual-paid-notice",
            "action_id": t.action_id,
            "terms_hash": t.content_hash(),
            "effective_date": paid.date(),
            "observed_at": paid,
            "available_at": paid + timedelta(seconds=1),
            "currency": "USD",
            "gross_per_unit": D(1),
            "net_per_unit": D(".8"),
            "fee_per_unit": D(".05"),
            "withholding_per_unit": D(".15"),
            "fee_source_hash": "c" * 64,
            "tax_source_hash": "d" * 64,
            "source_id": "fixture-distribution",
            "source_hash": "e" * 64,
            **changes,
        }
    )


def buy(ledger, *, book=None, allocations=None, quantity="10", at=T, name="buy"):
    order = intent(name, qty=quantity, at=at)
    if book is not None:
        book.register_intent(order, allocations, source_hash="1" * 64, quantity_quantum=D(1))
    ledger.submit(order)
    ledger.apply_fill(fill(order, name))
    if book is not None:
        book.sync()
    return order


def test_entitlement_preserves_nav_but_not_spendable_cash_then_actual_payment(tmp_path):
    ledger = account(tmp_path / "div.db", cash="10000")
    t = terms()
    ledger.register_cash_dividend("backtest:test", t, as_of=T)
    buy(ledger)
    original = ledger.accrue_dividend("backtest:test", t.action_id, at=t.ex_at)
    ledger.mark("backtest:test", [PriceMark(instrument=A, price="99", as_of=t.ex_at)])
    snap = ledger.snapshot("backtest:test", t.ex_at, 0)
    assert snap.cash == ledger.available_cash("backtest:test") == 9000
    assert snap.receivable_balance == 10 and snap.nav == 10000 and snap.realized_pnl == 10
    assert original.quantity == 10
    ledger.close()
    ledger = account(tmp_path / "div.db", cash="10000")
    assert ledger.accrue_dividend("backtest:test", t.action_id, at=t.ex_at) == original
    notice = distribution(t)
    assert ledger.apply_dividend_distribution("backtest:test", notice, as_of=notice.available_at)
    assert not ledger.apply_dividend_distribution(
        "backtest:test", notice, as_of=notice.available_at
    )
    ledger.mark("backtest:test", [PriceMark(instrument=A, price="99", as_of=notice.available_at)])
    paid = ledger.snapshot("backtest:test", notice.available_at, 0)
    assert paid.cash == 9008 and paid.receivable_balance == 0
    assert paid.nav == 9998 and paid.fees == D(".5") and paid.withholding_taxes == D("1.5")
    assert paid.realized_pnl == 8
    assert ledger.reconcile("backtest:test")["receivables"] == "0"
    ledger.close()


def test_dividend_does_not_follow_post_ex_sale_or_new_buyer_and_sleeves_conserve(tmp_path):
    ledger = account(tmp_path / "sleeves.db", cash="10000")
    book = SleeveBook(ledger, "backtest:test").initialize(
        {"original": D(5000), "new": D(5000), "empty": D(0)}
    )
    t = terms()
    ledger.register_cash_dividend("backtest:test", t, as_of=T)
    buy(ledger, book=book, allocations={"original": D(10)})
    original = book.accrue_dividend(t.action_id, at=t.ex_at)
    ledger.mark("backtest:test", [PriceMark(instrument=A, price="100", as_of=t.ex_at)])
    opening = book.snapshot(t.ex_at, persist=False)
    sale = intent("sale", side="sell", qty="10", at=t.ex_at + timedelta(seconds=1))
    book.register_intent(sale, {"original": D(10)}, source_hash="2" * 64, quantity_quantum=D(1))
    ledger.submit(sale)
    ledger.apply_fill(fill(sale, "sale"))
    book.sync()
    buy(
        ledger,
        book=book,
        allocations={"new": D(10)},
        at=t.ex_at + timedelta(seconds=3),
        name="new-owner",
    )
    before = book.snapshot(t.ex_at + timedelta(seconds=4), persist=False)
    assert before["original"].portfolio.receivable_balance == 10
    assert before["new"].portfolio.receivable_balance == 0
    assert before["original"].portfolio.positions == ()
    assert (
        before["original"].portfolio.receivables[0].entitlement.parent_entitlement_hash
        == original.content_hash()
    )
    notice = distribution(t)
    assert book.apply_dividend_distribution(notice, as_of=notice.available_at)
    paid = book.snapshot(notice.available_at, persist=False)
    assert paid["original"].portfolio.cash == 5008 and paid["new"].portfolio.cash == 4000
    assert paid["original"].portfolio.withholding_taxes == D("1.5")
    assert paid["empty"].portfolio.nav == 0
    canonical = ledger.snapshot("backtest:test", notice.available_at, 345600, persist=False)
    for attr in ("nav", "cash", "fees", "realized_pnl", "withholding_taxes", "receivable_balance"):
        assert exact_decimal_sum(getattr(s.portfolio, attr) for s in paid.values()) == getattr(
            canonical, attr
        )
    assert opening["original"].portfolio.nav == 5010
    assert ledger.reconcile("backtest:test")["reconciled"]
    ledger.close()


def test_partial_tiny_payments_keep_original_receivable_owners_without_negative_residual(tmp_path):
    ledger = account(tmp_path / "tiny.db")
    book = SleeveBook(ledger, "backtest:test").initialize(
        {"a": D(50000), "z": D(50000), "empty": D(0)}
    )
    t = terms(cash_per_unit=D("0.000000000000000000123456789123456789"))
    ledger.register_cash_dividend("backtest:test", t, as_of=T)
    buy(ledger, book=book, allocations={"a": D(9), "z": D(1)})
    book.accrue_dividend(t.action_id, at=t.ex_at)
    half = t.cash_per_unit / 2
    for index in (1, 2):
        notice = distribution(
            t,
            distribution_id=f"partial-{index}",
            gross_per_unit=half,
            net_per_unit=half,
            fee_per_unit=D(0),
            withholding_per_unit=D(0),
            fee_source_hash=None,
            tax_source_hash=None,
        )
        book.apply_dividend_distribution(notice, as_of=notice.available_at)
        assert all(
            v >= 0 for amounts in book.sync()["receivables"].values() for v in amounts.values()
        )
    assert ledger.dividend_receivables("backtest:test") == []
    assert ledger.reconcile("backtest:test")["reconciled"]
    ledger.close()


def test_source_revision_late_entitlement_payment_tamper_fail_closed(tmp_path):
    ledger = account(tmp_path / "proof.db")
    t = terms()
    ledger.register_cash_dividend("backtest:test", t, as_of=T)
    buy(ledger)
    with pytest.raises(LedgerError, match="causal"):
        ledger.register_cash_dividend(
            "backtest:test",
            terms(action_id="late", available_at=t.ex_at + timedelta(seconds=1)),
            as_of=t.ex_at + timedelta(seconds=1),
        )
    ledger.accrue_dividend("backtest:test", t.action_id, at=t.ex_at)
    with pytest.raises(LedgerError, match="post-entitlement"):
        ledger.register_cash_dividend(
            "backtest:test",
            terms(revision_id="r2", supersedes_hash=t.content_hash()),
            as_of=t.ex_at,
        )
    notice = distribution(t)
    payment = notice.payment(ledger.accrue_dividend("backtest:test", t.action_id, at=t.ex_at))
    with pytest.raises(LedgerError, match="exact actual per-unit"):
        ledger.apply_dividend_payment(payment, as_of=notice.available_at)
    ledger.apply_dividend_distribution("backtest:test", notice, as_of=notice.available_at)
    ledger.db.execute(
        "UPDATE dividend_payments SET payload=replace(payload, 'actual-fixture', 'tampered')"
    )
    ledger.db.execute("UPDATE dividend_entitlements SET outstanding='1'")
    with pytest.raises(LedgerError, match="projection"):
        ledger.reconcile("backtest:test")
    ledger.close()


def test_same_boundary_offsetting_trades_cannot_retroactively_accrue(tmp_path):
    ledger = account(tmp_path / "boundary.db")
    t = terms()
    ledger.register_cash_dividend("backtest:test", t, as_of=T)
    buy(ledger)
    purchase = intent("same-in", at=t.ex_at - timedelta(seconds=1))
    sale = intent("same-out", side="sell", at=t.ex_at - timedelta(seconds=1))
    ledger.submit(purchase)
    ledger.submit(sale)
    ledger.apply_fill(fill(purchase, "same-in", at=t.ex_at))
    ledger.apply_fill(fill(sale, "same-out", at=t.ex_at))
    assert ledger.quantity("backtest:test", A.instrument_id) == 10
    with pytest.raises(LedgerError, match="same-boundary"):
        ledger.accrue_dividend("backtest:test", t.action_id, at=t.ex_at)
    assert not ledger.dividend_receivables("backtest:test")
    ledger.close()


def replay_dividends(*, pay=True):
    from tests.test_core_backtest import START, replay_inputs
    from tests.test_core_backtest import A as replay_instrument

    inputs = replay_inputs()
    ex = START + timedelta(days=2)
    t = terms(
        instrument=replay_instrument,
        ex_at=ex,
        entitlement_at=ex,
        record_date=ex.date(),
        payable_date=ex.date(),
    )
    notice = distribution(
        t,
        effective_date=ex.date(),
        observed_at=ex + timedelta(hours=1),
        available_at=ex + timedelta(hours=2),
    )
    inputs.update(cash_dividends=[t], dividend_distributions=[notice] if pay else [])
    return inputs


def test_replay_receivables_payments_validation_artifacts_and_exact_own_cost_entitlement(tmp_path):
    from src.core.backtester import (
        PortfolioReplay,
        load_replay_artifact,
        persist_replay_artifact,
        recorded_replay_inputs,
        require_matching_inputs,
    )
    from src.core.benchmark import BenchmarkRunner, require_matched_replay
    from src.core.validation import ValidationSuite
    from src.execution.ledger import PortfolioLedger

    inputs = replay_dividends()
    path = tmp_path / "base.sqlite"
    result = PortfolioReplay(path).run(**inputs)
    assert result.metrics.total_return < 0 and result.metrics.max_drawdown < 0
    assert result.snapshots[-1].nav == 8880
    assert result.snapshots[-1].fees == 5 and result.snapshots[-1].withholding_taxes == 15
    assert [e["kind"] for e in result.economic_events] == [
        "dividend_entitlement",
        "dividend_payment",
    ]
    report = ValidationSuite().validate(result, ledger_path=path, _accounting_only=True)
    assert report.overall == "RELIABLE", report.summary
    reference = persist_replay_artifact(result, path, tmp_path / "result.json")
    assert load_replay_artifact(reference) == result
    assert PortfolioReplay(path).run(**inputs) == result
    ledger = PortfolioLedger(path, read_only=True)
    try:
        recovered = recorded_replay_inputs(ledger, result)
    finally:
        ledger.close()
    require_matching_inputs(result, recovered)
    costs = BenchmarkRunner().cost_sensitivity(inputs=inputs, output_dir=tmp_path / "costs")
    for cost in costs:
        require_matched_replay(result, cost, cost_variant=True)
        entitlement, payment = [e["payload"] for e in cost.economic_events]
        assert D(payment["gross_amount"]) == D(entitlement["quantity"])
        assert D(payment["net_amount"]) == D(entitlement["quantity"]) * D(".8")
    unpaid_inputs = replay_dividends(pay=False)
    unpaid = PortfolioReplay(tmp_path / "unpaid.sqlite").run(**unpaid_inputs)
    assert unpaid.snapshots[-1].receivable_balance == 100
    assert unpaid.snapshots[-1].cash == 0
    assert unpaid.snapshots[-1].nav == 8900
    assert (
        ValidationSuite()
        .validate(unpaid, ledger_path=tmp_path / "unpaid.sqlite", _accounting_only=True)
        .overall
        == "RELIABLE"
    )


def test_replay_changed_source_tampered_income_or_nav_cannot_validate(tmp_path):
    from src.core.backtester import PortfolioReplay, require_matching_inputs
    from src.core.validation import ValidationSuite
    from src.utils.exceptions import BacktestError

    inputs = replay_dividends()
    path = tmp_path / "proof.sqlite"
    result = PortfolioReplay(path).run(**inputs)
    changed = dict(inputs)
    changed["dividend_distributions"] = [
        inputs["dividend_distributions"][0].model_copy(update={"source_hash": "9" * 64})
    ]
    with pytest.raises(BacktestError, match="manifest"):
        require_matching_inputs(result, changed)
    corrupt = result.model_copy(deep=True)
    corrupt.economic_events[1]["payload"]["net_amount"] = "1000"
    report = ValidationSuite().validate(corrupt, ledger_path=path, _accounting_only=True)
    assert report.overall == "NOT RELIABLE"
    assert all(
        c.verdict.value == "FAIL"
        for c in report.checks
        if c.name in ("nav_arithmetic", "durable_accounting")
    )


def test_process_date_and_unknown_protocol_scopes_are_not_economic_coverage():
    from src.domain.portfolio import ActionCoverage

    coverage = ActionCoverage(
        coverage_id="process-query",
        instrument_ids=(A.instrument_id,),
        venue="alpaca",
        scope="provider-process-date-query",
        started_at=T,
        ended_at=T + timedelta(days=10),
        available_at=T + timedelta(days=10),
        event_types=("all-provider-types",),
        completeness="complete",
        gaps=(),
        source_id="raw-process-query",
        source_hash="f" * 64,
    )
    with pytest.raises(ValueError, match="economic action coverage"):
        coverage.require_complete(
            instrument_ids=(A.instrument_id,),
            started_at=T,
            ended_at=T + timedelta(days=1),
            as_of=T + timedelta(days=10),
            scope="effective-event-stream",
            event_types=("cash_dividend",),
        )
    unknown = coverage.model_copy(
        update={
            "scope": "protocol-state-events",
            "completeness": "unknown",
            "gaps": ("protocol actions not reported by market endpoint",),
        }
    )
    with pytest.raises(ValueError, match="economic action coverage"):
        unknown.require_complete(
            instrument_ids=(A.instrument_id,),
            started_at=T,
            ended_at=T + timedelta(days=1),
            as_of=T + timedelta(days=10),
            scope="protocol-state-events",
            event_types=("all-provider-types",),
        )


def test_posted_account_payment_partial_restart_and_nonspendable_nav_reserve(tmp_path):
    from src.domain.portfolio import AccountSpec, RiskPolicy
    from src.execution.ledger import PortfolioLedger
    from src.execution.risk_controls import evaluate_order
    from tests.test_durable_accounting import metadata

    path = tmp_path / "actual-paper.db"
    ledger = PortfolioLedger(path)
    spec = AccountSpec(
        account_id="paper:dividends",
        mode="paper",
        base_currency="USD",
        initial_cash=1000,
        opened_at=T,
    )
    ledger.create_account(spec)
    t = terms(cash_per_unit=D(100))
    ledger.register_cash_dividend(spec.account_id, t, as_of=T)
    purchase = intent(qty="9").model_copy(update={"account_id": spec.account_id})
    ledger.submit(purchase)
    ledger.apply_fill(fill(purchase))
    entitlement = ledger.accrue_dividend(spec.account_id, t.action_id, at=t.ex_at)
    ledger.mark(spec.account_id, [PriceMark(instrument=A, price="100", as_of=t.ex_at)])
    snapshot = ledger.snapshot(spec.account_id, t.ex_at, 0)
    assert snapshot.nav == 1900 and snapshot.cash == 100
    extra = intent("unfunded", qty="2", at=t.ex_at).model_copy(
        update={"account_id": spec.account_id}
    )
    with pytest.raises(LedgerError, match="cash"):
        ledger.submit(extra)
    risk = evaluate_order(
        extra.model_copy(update={"quantity": D(1)}),
        snapshot,
        [],
        {A.instrument_id: metadata(t.ex_at)},
        RiskPolicy(equity_weight=1, sector_weight=1, min_cash=".1", max_daily_turnover=1),
    )
    assert not risk.passed
    notice = distribution(
        t,
        gross_per_unit=D(50),
        net_per_unit=D(40),
        fee_per_unit=D("2.5"),
        withholding_per_unit=D("7.5"),
    )
    payment = notice.payment(entitlement).model_copy(
        update={"origin": "posted-account", "payment_id": "actual-credit-1"}
    )
    with pytest.raises(LedgerError, match="exclusive"):
        ledger.apply_dividend_distribution(spec.account_id, notice, as_of=notice.available_at)
    assert ledger.apply_dividend_payment(payment, as_of=payment.available_at)
    assert ledger.dividend_receivables(spec.account_id)[0].outstanding == 450
    assert ledger.reconcile(spec.account_id)["cash"] == "460"
    ledger.close()
    ledger = PortfolioLedger(path)
    assert not ledger.apply_dividend_payment(
        payment.model_copy(
            update={
                "observed_at": payment.observed_at + timedelta(seconds=10),
                "available_at": payment.available_at + timedelta(seconds=10),
            }
        ),
        as_of=payment.available_at + timedelta(seconds=10),
    )
    assert ledger.apply_dividend_payment(
        payment.model_copy(update={"payment_id": "actual-credit-2"}), as_of=payment.available_at
    )
    assert not ledger.dividend_receivables(spec.account_id)
    assert ledger.reconcile(spec.account_id)["cash"] == "820"
    assert ledger.reconcile(spec.account_id)["withholding_taxes"] == "135.0"
    with pytest.raises(LedgerError, match="exceeds"):
        ledger.apply_dividend_payment(
            payment.model_copy(update={"payment_id": "over-credit"}), as_of=payment.available_at
        )
    ledger.close()


def test_callback_and_sleeve_replay_bind_receivable_snapshot_and_future_payment_changes(tmp_path):
    from src.core.backtester import PortfolioReplay, recorded_decisions
    from src.core.validation import ValidationSuite
    from src.domain.portfolio import PortfolioTarget, TargetWeight
    from src.execution.ledger import PortfolioLedger
    from tests.test_core_backtest import START
    from tests.test_core_backtest import A as replay_instrument

    def provider(at, snapshot):
        if at == START + timedelta(hours=6):
            return PortfolioTarget(
                target_id="buy-before-ex",
                decision_at=at,
                universe_version="test-universe",
                strategy_id="hold-income",
                artifact_version="fixture",
                data_snapshot_id="fixture",
                weights=(TargetWeight(instrument=replay_instrument, weight=D(".5")),),
            )
        return None

    inputs = replay_dividends()
    inputs.pop("signals")
    inputs.update(
        decision_provider=provider,
        decision_recipe_hash="6" * 64,
        decision_times=[START + timedelta(hours=6), START + timedelta(days=2, minutes=10)],
    )
    path = tmp_path / "callback.sqlite"
    result = PortfolioReplay(path).run(**inputs)
    ledger = PortfolioLedger(path, read_only=True)
    records = recorded_decisions(ledger, result)
    ledger.close()
    assert records[1].snapshot.receivable_balance == 50
    assert records[1].snapshot.cash == 5000
    report = ValidationSuite().validate(result, ledger_path=path, _accounting_only=True)
    assert report.overall == "RELIABLE", report.summary
    altered = dict(inputs)
    altered["dividend_distributions"] = [
        inputs["dividend_distributions"][0].model_copy(
            update={
                "net_per_unit": D(".7"),
                "withholding_per_unit": D(".25"),
                "source_hash": "7" * 64,
            }
        )
    ]
    future = PortfolioReplay(tmp_path / "future.sqlite").run(**altered)
    assert future.snapshots[:-1] == result.snapshots[:-1]
    assert future.trades == result.trades
    assert future.snapshots[-1].nav == result.snapshots[-1].nav - 5
    assert future.snapshots[-1].withholding_taxes == D("12.5")


def test_dividend_sleeve_replay_artifact_recomputes_income_and_exact_period_returns(tmp_path):
    from src.core.backtester import PortfolioReplay, load_replay_artifact, persist_replay_artifact
    from src.domain.portfolio import PortfolioTarget, TargetWeight
    from src.execution.ledger import PortfolioLedger
    from tests.test_core_backtest import START
    from tests.test_core_backtest import A as replay_instrument

    inputs = replay_dividends()
    at = START + timedelta(hours=6)
    inputs["signals"] = []
    inputs["targets"] = [
        PortfolioTarget(
            target_id="two-owners",
            decision_at=at,
            universe_version="test-universe",
            strategy_id="blend",
            artifact_version="fixture",
            data_snapshot_id="fixture",
            weights=(TargetWeight(instrument=replay_instrument, weight=D(".6")),),
            sleeve_contributions={"a": {"A": D(".2")}, "b": {"A": D(".4")}, "empty": {}},
            sleeve_sources={"a": "a" * 64, "b": "b" * 64, "empty": "c" * 64},
        )
    ]
    inputs["sleeve_capital"] = {"a": D(5000), "b": D(5000), "empty": D(0)}
    path = tmp_path / "sleeve-replay.sqlite"
    result = PortfolioReplay(path).run(**inputs)
    reference = persist_replay_artifact(result, path, tmp_path / "sleeves.json")
    assert load_replay_artifact(reference) == result
    ledger = PortfolioLedger(path)
    book = SleeveBook(ledger, inputs["account"].account_id)
    parts = book.snapshot(inputs["context"].as_of)
    assert parts["a"].portfolio.withholding_taxes == 3
    assert parts["b"].portfolio.withholding_taxes == 6
    assert parts["empty"].portfolio.nav == 0
    assert exact_decimal_sum(s.portfolio.nav for s in parts.values()) == result.snapshots[-1].nav
    assert all(s.portfolio.receivable_balance == 0 for s in parts.values())
    ledger.close()


def test_long_precision_cash_survives_nonmonetary_cancellation_and_split(tmp_path):
    from src.domain.portfolio import CorporateAction

    ledger = account(tmp_path / "exact-cash.db")
    book = SleeveBook(ledger, "backtest:test").initialize({"owner": D(100000), "empty": D(0)})
    rate = D("0.000000000000000000123456789123456789")
    t = terms(cash_per_unit=rate)
    ledger.register_cash_dividend("backtest:test", t, as_of=T)
    buy(ledger, book=book, allocations={"owner": D(10)})
    book.accrue_dividend(t.action_id, at=t.ex_at)
    notice = distribution(
        t,
        gross_per_unit=rate,
        net_per_unit=rate,
        fee_per_unit=D(0),
        withholding_per_unit=D(0),
        fee_source_hash=None,
        tax_source_hash=None,
    )
    book.apply_dividend_distribution(notice, as_of=notice.available_at)
    expected = D(ledger._account("backtest:test")["cash"])
    pending = intent("cancel-after-income", qty="1", at=notice.available_at)
    book.register_intent(pending, {"owner": D(1)}, source_hash="a" * 64, quantity_quantum=D(1))
    ledger.submit(pending)
    ledger.cancel("backtest:test", pending.intent_id, notice.available_at + timedelta(seconds=1))
    ledger.corporate_action(
        "backtest:test",
        CorporateAction(
            action_id="split-after-income",
            instrument=A,
            effective_at=notice.available_at + timedelta(seconds=2),
            kind="split",
            value=2,
            source_id="fixture",
        ),
    )
    assert D(ledger.reconcile("backtest:test")["cash"]) == expected
    assert book.sync()["cash"]["owner"] == expected
    assert book.sync()["cash"]["empty"] == 0
    ledger.close()


def test_cli_parses_actual_dividend_bundle_and_rejects_late_terms_without_performance(tmp_path):
    import json

    from typer.testing import CliRunner

    from scripts.run_backtest import app
    from tests.test_core_backtest import write_bundle

    inputs = replay_dividends()
    bundle = tmp_path / "bundle.json"
    write_bundle(bundle, inputs)
    output = tmp_path / "run.json"
    run = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(bundle),
            "--ledger",
            str(tmp_path / "cli.sqlite"),
            "--output",
            str(output),
        ],
    )
    assert run.exit_code == 0, run.output
    assert json.loads(output.read_text())["snapshots"][-1]["withholding_taxes"] == "15.00"
    inputs["cash_dividends"] = [
        inputs["cash_dividends"][0].model_copy(
            update={"available_at": inputs["cash_dividends"][0].ex_at + timedelta(seconds=1)}
        )
    ]
    write_bundle(bundle, inputs)
    failed = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(bundle),
            "--ledger",
            str(tmp_path / "late.sqlite"),
            "--output",
            str(tmp_path / "failed.json"),
        ],
    )
    assert failed.exit_code == 1
    assert not (tmp_path / "failed.json").exists()
    payload = json.loads((tmp_path / "failed.json.failure.json").read_text())
    assert payload["status"] == "failed" and "metrics" not in payload


def test_coverage_gaps_are_durable_and_cannot_be_replaced_by_a_complete_claim(tmp_path):
    from src.domain.portfolio import ActionCoverage

    ledger = account(tmp_path / "coverage.sqlite")
    cutoff = T + timedelta(days=1)
    coverage = ActionCoverage(
        coverage_id="source-scope",
        instrument_ids=(A.instrument_id,),
        venue="alpaca",
        scope="provider-process-date-query",
        started_at=T,
        ended_at=cutoff,
        available_at=cutoff,
        event_types=("cash_dividend", "split", "merger", "spinoff"),
        completeness="partial",
        gaps=(
            "process-date pagination does not establish ex-date coverage",
            "merger/spinoff economics unsupported",
        ),
        source_id="actual-fixture",
        source_hash="a" * 64,
    )
    assert ledger.record_action_coverage("backtest:test", coverage, as_of=cutoff)
    assert not ledger.record_action_coverage("backtest:test", coverage, as_of=cutoff)
    with pytest.raises(LedgerError, match="conflicting"):
        ledger.record_action_coverage(
            "backtest:test",
            coverage.model_copy(update={"completeness": "complete", "gaps": ()}),
            as_of=cutoff,
        )
    assert ledger.reconcile("backtest:test")["cash"] == "100000"
    ledger.close()


def test_entitlement_and_posted_payment_never_rewrite_already_published_nav(tmp_path):
    ledger = account(tmp_path / "nav-fence.sqlite")
    t = terms()
    ledger.register_cash_dividend("backtest:test", t, as_of=T)
    buy(ledger)
    ledger.mark("backtest:test", [PriceMark(instrument=A, price="99", as_of=t.ex_at)])
    published = ledger.snapshot("backtest:test", t.ex_at, 0)
    with pytest.raises(LedgerError, match="published NAV"):
        ledger.accrue_dividend("backtest:test", t.action_id, at=t.ex_at)
    assert ledger.history("backtest:test")[-1] == published
    assert not ledger.dividend_receivables("backtest:test")
    ledger.close()
    ledger = account(tmp_path / "payment-fence.sqlite")
    ledger.register_cash_dividend("backtest:test", t, as_of=T)
    buy(ledger)
    ledger.accrue_dividend("backtest:test", t.action_id, at=t.ex_at)
    notice = distribution(t)
    ledger.mark("backtest:test", [PriceMark(instrument=A, price="90", as_of=notice.available_at)])
    published = ledger.snapshot("backtest:test", notice.available_at, 0)
    with pytest.raises(LedgerError, match="published NAV"):
        ledger.apply_dividend_distribution("backtest:test", notice, as_of=notice.available_at)
    assert ledger.history("backtest:test")[-1] == published
    assert ledger.dividend_receivables("backtest:test")[0].outstanding == 10
    assert not ledger.db.execute("SELECT 1 FROM dividend_payments").fetchone()
    ledger.close()
