"""Hand-calculated synthetic fixtures; no provider acceptance claimed."""

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

from src.domain.common import Instrument
from src.domain.portfolio import (
    AccountSpec,
    CorporateAction,
    ExecutionBar,
    Fill,
    FillPolicy,
    InstrumentRisk,
    OrderIntent,
    PriceMark,
    RiskPolicy,
)
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.paper_trader import PaperTrader
from src.execution.risk_controls import evaluate_order
from src.execution.simulator import PaperExecution
from src.utils.schemas import Order, OrderSide

T = datetime(2024, 1, 2, 14, tzinfo=timezone.utc)
A = Instrument(instrument_id="us:A", symbol="A", asset_class="equity", quote_currency="USD")
B = Instrument(
    instrument_id="spot:BTC", symbol="BTC/USD", asset_class="crypto", quote_currency="USD"
)


def account(path=":memory:", cash="100000"):
    ledger = PortfolioLedger(path)
    ledger.create_account(
        AccountSpec(
            account_id="backtest:test",
            mode="backtest",
            base_currency="USD",
            initial_cash=cash,
            opened_at=T,
        )
    )
    return ledger


def intent(name="buy", side="buy", qty="10", price="100", at=T, instrument=A, fee="0"):
    return OrderIntent(
        intent_id=name,
        account_id="backtest:test",
        run_id="run",
        instrument=instrument,
        side=side,
        quantity=qty,
        decision_at=at,
        eligible_at=at + timedelta(seconds=1),
        reference_price=price,
        estimated_fee=fee,
        strategy_id="test",
    )


def fill(order, name="f1", qty=None, price="100", fee="0", at=None):
    return Fill(
        fill_id=name,
        intent_id=order.intent_id,
        account_id=order.account_id,
        quantity=qty or order.quantity,
        price=price,
        fee=fee,
        executed_at=at or order.eligible_at,
        event_id="quote-" + name,
    )


def metadata(at=T, instrument=A, sector="tech"):
    return InstrumentRisk(
        instrument=instrument, sector=sector, adv_base="100000000", as_of=at, lot_size="1"
    )


def bar(at=T + timedelta(days=1), opening_volume="10000", instrument=A, close="100"):
    return ExecutionBar(
        event_id=at.isoformat() + instrument.instrument_id,
        instrument=instrument,
        opened_at=at,
        closed_at=at + timedelta(hours=6),
        open="100",
        high=max(D(100), D(close)),
        low=min(D(100), D(close)),
        close=close,
        volume="100000",
        opening_volume=opening_volume,
        calendar="24/7" if instrument.asset_class == "crypto" else "XNYS",
        calendar_version="fixture-1",
        session=at.date(),
        session_open=at,
        session_close=at + timedelta(hours=6),
    )


def test_fee_cash_basis_and_realized_reconcile():
    ledger = account(cash="10000")
    buy = intent(qty="10", fee="1")
    ledger.submit(buy)
    ledger.apply_fill(fill(buy, fee="1"))
    t2 = T + timedelta(days=1)
    sell = intent("sell", "sell", "4", "110", t2, fee="2")
    ledger.submit(sell)
    ledger.apply_fill(fill(sell, "f2", price="110", fee="2"))
    ledger.mark("backtest:test", [PriceMark(instrument=A, price="90", as_of=sell.eligible_at)])
    snap = ledger.snapshot("backtest:test", sell.eligible_at, 0)
    assert snap.cash == D("9437")
    assert snap.positions[0].cost_basis == D("600.6")
    assert snap.realized_pnl == D("37.6")
    assert snap.nav == D("9977")
    assert snap.fees == 3
    assert ledger.reconcile("backtest:test")["reconciled"]


def test_partial_duplicate_and_reservations():
    ledger = account(cash="1000")
    buy = intent(qty="9", fee="9")
    ledger.submit(buy)
    assert ledger.available_cash(buy.account_id) == 91
    first = fill(buy, qty="3", fee="3")
    assert ledger.apply_fill(first)
    assert not ledger.apply_fill(first)
    assert ledger.order(buy.account_id, buy.intent_id)["reserved"] == 606
    assert ledger.order(buy.account_id, buy.intent_id)["status"] == "partial"
    with pytest.raises(LedgerError, match="different content"):
        ledger.apply_fill(first.model_copy(update={"fee": D(4)}))
    ledger.apply_fill(fill(buy, "f2", qty="6", fee="6", at=T + timedelta(seconds=2)))
    assert ledger.quantity(buy.account_id, A.instrument_id) == 9
    assert ledger.available_cash(buy.account_id) == 91
    assert ledger.reconcile(buy.account_id)["reconciled"]


def test_failed_fill_rolls_back_all_state():
    ledger = account(cash="1000")
    buy = intent(qty="9")
    ledger.submit(buy)
    with pytest.raises(LedgerError, match="cash"):
        ledger.apply_fill(fill(buy, price="200"))
    assert ledger.order(buy.account_id, buy.intent_id)["filled"] == 0
    assert ledger.quantity(buy.account_id, A.instrument_id) == 0
    assert ledger.reconcile(buy.account_id)["cash"] == "1000"


def test_restart_new_process_and_duplicate_fill(tmp_path):
    path = tmp_path / "ledger.db"
    ledger = account(path)
    buy = intent()
    ledger.submit(buy)
    f = fill(buy)
    ledger.apply_fill(f)
    ledger.close()
    code = """import sys
from src.execution.ledger import PortfolioLedger
from src.domain.portfolio import Fill
l=PortfolioLedger(sys.argv[1])
assert not l.apply_fill(Fill.model_validate_json(sys.argv[2]))
assert l.quantity('backtest:test','us:A')==10
assert l.reconcile('backtest:test')['cash']=='99000'
"""
    subprocess.run(
        [sys.executable, "-c", code, str(path), f.model_dump_json()],
        check=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )


def test_sell_to_zero_and_nav_cash_reserve():
    ledger = account()
    buy = intent(qty="100")
    ledger.submit(buy)
    ledger.apply_fill(fill(buy))
    t2 = T + timedelta(days=1)
    ledger.mark(buy.account_id, [PriceMark(instrument=A, price="100", as_of=t2)])
    snapshot = ledger.snapshot(buy.account_id, t2, 0)
    sell = intent("sell", "sell", "100", at=t2)
    result = evaluate_order(sell, snapshot, [], {A.instrument_id: metadata(t2)}, RiskPolicy())
    assert result.passed, result.details
    # Reproduced defect: 900 spend with only 1000 cash and 100000 NAV.
    low_cash = snapshot.model_copy(update={"cash": D(1000)})
    result = evaluate_order(
        intent("other", qty="9", at=t2), low_cash, [], {A.instrument_id: metadata(t2)}, RiskPolicy()
    )
    assert "cash_buffer" in result.failed_checks


def test_pending_exposure_sector_crypto_and_buy_halt():
    ledger = account()
    at = T
    snap = ledger.snapshot("backtest:test", at, 0)
    pending = {"intent": intent(qty="60"), "remaining": D(60), "reserved": D(6000)}
    result = evaluate_order(
        intent("more", qty="50"), snap, [pending], {A.instrument_id: metadata()}, RiskPolicy()
    )
    assert "position_size" in result.failed_checks
    result = evaluate_order(
        intent(instrument=B, qty="45"),
        snap,
        [],
        {B.instrument_id: metadata(instrument=B)},
        RiskPolicy(crypto_total="0.04"),
    )
    assert "crypto_total" in result.failed_checks
    result = evaluate_order(
        intent(),
        snap.model_copy(update={"high_water_mark": D(120000)}),
        [],
        {A.instrument_id: metadata()},
        RiskPolicy(),
    )
    assert "drawdown_cb" in result.failed_checks


def test_partial_open_liquidity_survives_restart(tmp_path):
    path = tmp_path / "ledger.db"
    ledger = account(path)
    engine = PaperExecution(ledger, "backtest:test")
    engine.submit(intent(qty="10", fee="1"), {A.instrument_id: metadata()})
    event = bar(opening_volume="300")
    assert engine.open_bar(event, {A.instrument_id: metadata()})[0].quantity == 3
    ledger.close()
    ledger = PortfolioLedger(path)
    engine = PaperExecution(ledger, "backtest:test")
    assert engine.open_bar(event, {A.instrument_id: metadata()}) == []
    assert ledger.quantity("backtest:test", A.instrument_id) == 3
    event2 = bar(at=T + timedelta(days=2), opening_volume="700")
    assert engine.open_bar(event2, {A.instrument_id: metadata()})[0].quantity == 7
    assert D(ledger.reconcile("backtest:test")["cash"]) == D("98999")


def test_split_dividend_delisting_conserve_accounting():
    ledger = account(cash="10000")
    buy = intent(qty="10")
    ledger.submit(buy)
    ledger.apply_fill(fill(buy))
    ledger.mark(buy.account_id, [PriceMark(instrument=A, price="100", as_of=buy.eligible_at)])
    t2 = T + timedelta(days=1)
    split = CorporateAction(
        action_id="split",
        instrument=A,
        kind="split",
        value="2",
        effective_at=t2,
        source_id="fixture",
    )
    ledger.corporate_action(buy.account_id, split)
    ledger.corporate_action(buy.account_id, split)
    snap = ledger.snapshot(buy.account_id, t2, 0)
    assert snap.positions[0].quantity == 20
    assert snap.positions[0].cost_basis == 1000
    assert snap.nav == 10000
    ledger.corporate_action(
        buy.account_id,
        CorporateAction(
            action_id="div",
            instrument=A,
            kind="dividend",
            value="1",
            effective_at=t2 + timedelta(seconds=1),
            source_id="fixture",
        ),
    )
    ledger.corporate_action(
        buy.account_id,
        CorporateAction(
            action_id="delist",
            instrument=A,
            kind="delist",
            value="40",
            effective_at=t2 + timedelta(seconds=2),
            source_id="fixture",
        ),
    )
    assert ledger.reconcile(buy.account_id)["cash"] == "9820"
    assert ledger.reconcile(buy.account_id)["realized_pnl"] == "-180"


def test_crypto_weekend_and_stale_marks():
    ledger = account()
    engine = PaperExecution(ledger, "backtest:test")
    engine.submit(intent(instrument=B, fee="2"), {B.instrument_id: metadata(instrument=B)})
    saturday = datetime(2024, 1, 6, 0, tzinfo=timezone.utc)
    event = bar(at=saturday, instrument=B)
    assert engine.open_bar(event, {B.instrument_id: metadata(instrument=B)})
    with pytest.raises(LedgerError, match="stale"):
        ledger.snapshot("backtest:test", saturday + timedelta(days=3), 86400)


def test_no_short_double_sell_or_cross_account_fill():
    ledger = account()
    buy = intent()
    ledger.submit(buy)
    ledger.apply_fill(fill(buy))
    sell = intent("s1", "sell", "10", at=T + timedelta(seconds=2))
    ledger.submit(sell)
    with pytest.raises(LedgerError, match="holdings"):
        ledger.submit(intent("s2", "sell", "1", at=T + timedelta(seconds=2)))
    with pytest.raises(LedgerError, match="unknown order"):
        ledger.apply_fill(fill(sell).model_copy(update={"account_id": "backtest:other"}))


def test_legacy_paper_restart_uses_canonical_ledger(tmp_path):
    path = tmp_path / "paper.db"
    trader = PaperTrader(
        10000, slippage_bps=0, commission_per_share=0, min_commission=0, ledger_path=path
    )
    order = Order(ticker="A", side=OrderSide.BUY, quantity=10)
    trader.execute_order(order, {"A": 100})
    trader.ledger.close()
    restarted = PaperTrader(
        10000, slippage_bps=0, commission_per_share=0, min_commission=0, ledger_path=path
    )
    assert restarted.portfolio.cash == 9000
    assert restarted.execute_order(order, {"A": 100})
    assert restarted.portfolio.positions["A"].quantity == 10
    assert restarted.ledger.reconcile("paper:default")["reconciled"]


def test_fx_and_currency_basis_reconcile():
    foreign = Instrument(
        instrument_id="eu:X", symbol="X", asset_class="equity", quote_currency="EUR"
    )
    ledger = account(cash="1000")
    order = intent(instrument=foreign, qty="2", price="100", fee="1").model_copy(
        update={"fx_rate": D("1.2")}
    )
    ledger.submit(order)
    ledger.apply_fill(fill(order, fee="1").model_copy(update={"fx_rate": D("1.2")}))
    ledger.mark(
        order.account_id,
        [PriceMark(instrument=foreign, price="100", fx_rate="1.3", as_of=order.eligible_at)],
    )
    snap = ledger.snapshot(order.account_id, order.eligible_at, 0)
    assert snap.cash == 759
    assert snap.positions[0].cost_basis == 241
    assert snap.nav == 1019
    assert ledger.reconcile(order.account_id)["reconciled"]


def test_pending_sector_exposure_and_strict_adv_boundary():
    ledger = account()
    snap = ledger.snapshot("backtest:test", T, 0)
    other = Instrument(instrument_id="us:C", symbol="C", asset_class="equity", quote_currency="USD")
    pending = [{"intent": intent("one", qty="60"), "remaining": D(60), "reserved": D(6000)}]
    buy = intent("two", qty="60", instrument=other)
    meta = {A.instrument_id: metadata(), other.instrument_id: metadata(instrument=other)}
    result = evaluate_order(buy, snap, pending, meta, RiskPolicy(sector_weight=".10"))
    assert "sector" in result.failed_checks
    exact = metadata().model_copy(update={"adv_base": D(1000000)})
    result = evaluate_order(intent(qty="100"), snap, [], {A.instrument_id: exact}, RiskPolicy())
    assert "market_impact" in result.failed_checks


def test_limit_stop_fok_and_cancellation_use_open_events():
    ledger = account()
    engine = PaperExecution(ledger, "backtest:test", FillPolicy(equity_cost_bps=0))
    market = intent(qty="10")
    limit = market.model_copy(update={"order_type": "limit", "limit_price": D(90)})
    engine.submit(limit, {A.instrument_id: metadata()})
    event = bar()
    # Intrabar lows cannot invent a fill when the observed open is not marketable.
    event = event.model_copy(update={"low": D(80)})
    assert engine.open_bar(event, {A.instrument_id: metadata()}) == []
    assert ledger.quantity("backtest:test", A.instrument_id) == 0
    ledger.cancel("backtest:test", limit.intent_id, event.opened_at, "replace")
    order = intent("fok", qty="10", at=event.opened_at).model_copy(update={"tif": "FOK"})
    engine.submit(order, {A.instrument_id: metadata()})
    assert (
        engine.open_bar(
            bar(at=T + timedelta(days=2), opening_volume="300"), {A.instrument_id: metadata()}
        )
        == []
    )
    assert ledger.order("backtest:test", "fok")["status"] == "cancelled"
    assert ledger.available_cash("backtest:test") == 100000


def test_underestimated_fees_and_conflicting_intents_fail_before_execution():
    ledger = account()
    engine = PaperExecution(ledger, "backtest:test")
    with pytest.raises(LedgerError, match="fee"):
        engine.submit(intent(), {A.instrument_id: metadata()})
    engine.submit(intent(fee="1"), {A.instrument_id: metadata()})
    with pytest.raises(LedgerError, match="different content"):
        engine.submit(intent(qty="11", fee="1.1"), {A.instrument_id: metadata()})


def test_process_crash_inside_fill_transaction_recovers_without_lost_liquidity(tmp_path):
    path = tmp_path / "crash.db"
    ledger = account(path)
    engine = PaperExecution(ledger, "backtest:test")
    engine.submit(intent(fee="1"), {A.instrument_id: metadata()})
    ledger.close()
    event = bar()
    code = """import os, sys
from src.execution.ledger import PortfolioLedger
from src.execution.simulator import PaperExecution
from src.domain.portfolio import ExecutionBar, InstrumentRisk
l=PortfolioLedger(sys.argv[1])
original=l.apply_fill
def crash(fill):
    original(fill)
    os._exit(23)
l.apply_fill=crash
meta=InstrumentRisk.model_validate_json(sys.argv[3])
PaperExecution(l,'backtest:test').open_bar(ExecutionBar.model_validate_json(sys.argv[2]),{meta.instrument.instrument_id:meta})
"""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(path),
            event.model_dump_json(),
            metadata().model_dump_json(),
        ],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    assert process.returncode == 23
    ledger = PortfolioLedger(path)
    assert ledger.quantity("backtest:test", A.instrument_id) == 0
    assert ledger.order("backtest:test", "buy")["status"] == "pending"
    engine = PaperExecution(ledger, "backtest:test")
    assert len(engine.open_bar(event, {A.instrument_id: metadata()})) == 1
    assert ledger.quantity("backtest:test", A.instrument_id) == 10
    assert ledger.reconcile("backtest:test")["reconciled"]


def test_full_liquidation_consumes_exact_basis_before_reentry(tmp_path):
    # A finite Decimal multiply/divide round trip need not equal its input.
    # Closing 3.001 units previously left a hidden -1E-26 basis residue.
    path = tmp_path / "precision.sqlite3"
    ledger = account(path)
    basis = D("37.84564546343524345566432112")
    quantity = D("3.001")
    buy_fee = basis - quantity * D(12)
    expected_realized = D(0)
    for cycle in range(2):
        for side, price, fee in (("buy", "12", buy_fee), ("sell", "13", D(".001"))):
            step = cycle * 2 + (side == "sell")
            order = intent(
                name=f"precision-{step}",
                side=side,
                qty=str(quantity),
                price=price,
                fee=str(fee),
                at=T + timedelta(seconds=2 * step),
            )
            ledger.submit(order)
            ledger.apply_fill(fill(order, name=f"precision-{step}", price=price, fee=str(fee)))
            if side == "sell":
                expected_realized += (quantity * D(price) - fee) - basis
                assert ledger.quantity(order.account_id, A.instrument_id) == 0
            assert ledger.reconcile(order.account_id)["realized_pnl"] == str(expected_realized)
            ledger.close()
            ledger = PortfolioLedger(path)
    assert D(ledger._account(order.account_id)["realized"]) == expected_realized
    ledger.close()


def test_actual_position_open_time_survives_additions_and_restart_resets_after_flat(tmp_path):
    path = tmp_path / "position-age.sqlite3"
    ledger = account(path)
    first_open = None
    for index, (side, qty) in enumerate(
        (("buy", "5"), ("buy", "5"), ("sell", "4"), ("sell", "6"), ("buy", "2"))
    ):
        order = intent(name=f"age-{index}", side=side, qty=qty, at=T + timedelta(days=index))
        ledger.submit(order)
        execution = fill(order, name=f"age-fill-{index}")
        ledger.apply_fill(execution)
        if index == 0:
            first_open = execution.executed_at
        expected = None if index == 3 else execution.executed_at if index == 4 else first_open
        ledger.close()
        ledger = PortfolioLedger(path)
        assert ledger.position_opened_at(order.account_id, A.instrument_id) == expected
        assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()
