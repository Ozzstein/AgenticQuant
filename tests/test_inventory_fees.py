"""Canonical noncash fee and native gross settlement invariants over test fixtures."""

import json
from datetime import timedelta
from decimal import Decimal as D

import pytest

from src.domain.common import Instrument
from src.domain.portfolio import (
    AccountSpec,
    Fill,
    FillPolicy,
    FillSettlement,
    InventoryFee,
    PriceMark,
    exact_decimal_sum,
)
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.sleeves import SleeveBook
from src.execution.venue import PaperVenue, VenueExecution, VenueOrderReport, VenueRules
from tests.test_durable_accounting import B, T, account, fill, intent, metadata

BNB = Instrument(
    instrument_id="spot:BNB", symbol="BNB/USD", asset_class="crypto", quote_currency="USD"
)


def fee(instrument=B, quantity=".03", at=T + timedelta(seconds=1), **changes):
    return InventoryFee(
        **{
            "instrument": instrument,
            "quantity": quantity,
            "price": D(100),
            "valued_at": at,
            "available_at": at,
            "source_id": "actual-trade-commission",
            "source_hash": "a" * 64,
            "valuation_source_id": "actual-causal-fee-price",
            "valuation_source_hash": "b" * 64,
            **changes,
        }
    )


def execute(
    ledger,
    name="buy",
    quantity="3",
    instrument=B,
    at=T,
    *,
    inventory_fees=(),
    settlement=None,
    book=None,
    allocations=None,
    side="buy",
    price="100",
):
    order = intent(name, qty=quantity, instrument=instrument, at=at, side=side, price=price)
    if book:
        book.register_intent(
            order, allocations, source_hash="c" * 64, quantity_quantum=D(".00000001")
        )
    ledger.submit(order)
    actual = fill(order, name, price=price).model_copy(
        update={"inventory_fees": inventory_fees, "settlement": settlement}
    )
    ledger.apply_fill(actual)
    if book:
        book.sync()
    return order, actual


def test_base_asset_fee_keeps_gross_fill_and_actual_net_holding_loss_restart(tmp_path):
    path = tmp_path / "base-fee.sqlite"
    ledger = account(path, cash="10000")
    order, actual = execute(ledger, inventory_fees=(fee(),))
    assert ledger.fills(order.account_id) == [actual] and actual.quantity == 3
    assert ledger.quantity(order.account_id, B.instrument_id) == D("2.97")
    ledger.mark(order.account_id, [PriceMark(instrument=B, price=100, as_of=actual.executed_at)])
    snap = ledger.snapshot(order.account_id, actual.executed_at, 0)
    assert snap.cash == 9700 and snap.nav == 9997 and snap.fees == 3 and snap.realized_pnl == -3
    assert snap.positions[0].cost_basis == 297
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()
    ledger = account(path, cash="10000")
    assert not ledger.apply_fill(actual)
    assert ledger.quantity(order.account_id, B.instrument_id) == D("2.97")
    execute(ledger, "exit", quantity="2.97", side="sell", price="90", at=T + timedelta(seconds=2))
    assert ledger.position_opened_at(order.account_id, B.instrument_id) is None
    final = ledger.snapshot(order.account_id, T + timedelta(seconds=3), 10)
    assert final.positions == () and final.cash == D("9967.30") and final.nav < snap.nav
    assert ledger.reconcile(order.account_id)["realized_pnl"] == "-32.70"
    ledger.close()


def test_actual_native_gross_settlement_does_not_invent_price_or_cash_rounding(tmp_path):
    ledger = account(tmp_path / "actual-cost.sqlite", cash="10000")
    native = FillSettlement(
        gross_quote="299.99999999", source_id="native-trade-cost", source_hash="d" * 64
    )
    order, actual = execute(ledger, settlement=native)
    assert (
        actual.quantity == 3 and actual.price == 100 and actual.base_notional == D("299.99999999")
    )
    assert ledger.available_cash(order.account_id) == D("9700.00000001")
    ledger.mark(order.account_id, [PriceMark(instrument=B, price=100, as_of=actual.executed_at)])
    assert (
        ledger.snapshot(order.account_id, actual.executed_at, 0).positions[0].cost_basis
        == native.gross_quote
    )
    assert ledger.reconcile(order.account_id)["reconciled"]
    assert (
        Fill.model_validate_json(actual.model_dump_json()).content_hash() == actual.content_hash()
    )
    assert "inventory_fees" not in actual.model_dump()
    assert "settlement" not in fill(order).model_dump()
    ledger.close()


def test_third_asset_fee_reduces_only_actual_fee_inventory_and_never_cash(tmp_path):
    ledger = account(tmp_path / "third.sqlite", cash="10000")
    execute(ledger, "fee-token", quantity="1", instrument=BNB)
    at = T + timedelta(seconds=2)
    order, actual = execute(
        ledger, at=at, inventory_fees=(fee(BNB, ".02", at=at + timedelta(seconds=1), price=D(150)),)
    )
    assert ledger.quantity(order.account_id, B.instrument_id) == 3
    assert ledger.quantity(order.account_id, BNB.instrument_id) == D(".98")
    assert ledger.available_cash(order.account_id) == 9600
    ledger.mark(
        order.account_id,
        [
            PriceMark(instrument=B, price=100, as_of=actual.executed_at),
            PriceMark(instrument=BNB, price=150, as_of=actual.executed_at),
        ],
    )
    snapshot = ledger.snapshot(order.account_id, actual.executed_at, 0)
    assert snapshot.fees == 3 and snapshot.realized_pnl == -2
    assert snapshot.nav == 10047  # BNB appreciated by 50, then an actual value-3 fee.
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_missing_fee_inventory_future_source_and_reserved_units_roll_back_whole_fill(tmp_path):
    ledger = account(tmp_path / "rejected.sqlite")
    order = intent(instrument=B)
    ledger.submit(order)
    for charge, error in (
        (fee(BNB), "actually owned"),
        (fee(quantity="11"), "exceeds owned"),
        (fee(available_at=T + timedelta(seconds=2)), "source is unavailable"),
    ):
        with pytest.raises(LedgerError, match=error):
            ledger.apply_fill(fill(order).model_copy(update={"inventory_fees": (charge,)}))
        assert (
            ledger.fills(order.account_id) == []
            and ledger.quantity(order.account_id, B.instrument_id) == 0
        )
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.close()


def test_partial_fill_inventory_fee_allocates_actual_owner_quantities_and_conserves_sleeves(
    tmp_path,
):
    path = tmp_path / "owners.sqlite"
    ledger = account(path, cash="10000")
    book = SleeveBook(ledger, "backtest:test").initialize(
        {"one": D(8000), "two": D(2000), "empty": D(0)}
    )
    order = intent(qty="6", instrument=B)
    book.register_intent(
        order, {"one": D(4), "two": D(2)}, source_hash="e" * 64, quantity_quantum=D(1)
    )
    ledger.submit(order)
    actual = fill(order, qty=D(3)).model_copy(update={"inventory_fees": (fee(),)})
    ledger.apply_fill(actual)
    book.sync()
    ledger.mark(order.account_id, [PriceMark(instrument=B, price=100, as_of=actual.executed_at)])
    views = book.snapshot(actual.executed_at)
    assert views["one"].portfolio.positions[0].quantity == D("1.98")
    assert views["two"].portfolio.positions[0].quantity == D(".99")
    assert views["one"].portfolio.fees == 2 and views["two"].portfolio.fees == 1
    assert views["empty"].portfolio.nav == 0
    for field in ("cash", "nav", "fees", "realized_pnl"):
        snapshot = ledger.snapshot(order.account_id, actual.executed_at, 0, persist=False)
        assert exact_decimal_sum(
            getattr(view.portfolio, field) for view in views.values()
        ) == getattr(snapshot, field)
    ledger.close()
    ledger = account(path, cash="10000")
    book = SleeveBook(ledger, order.account_id)
    second_at = T + timedelta(seconds=2)
    second = fill(order, "second", qty=D(3), at=second_at).model_copy(
        update={"inventory_fees": (fee(at=second_at),)}
    )
    ledger.apply_fill(second)
    book.sync()
    assert ledger.quantity(order.account_id, B.instrument_id) == D("5.94")
    assert ledger.reconcile(order.account_id)["fees"] == "6.00"
    assert not ledger.apply_fill(second)
    book.sync()
    ledger.close()


def test_missing_or_changed_fee_journal_is_not_successful_reconciliation(tmp_path):
    ledger = account(tmp_path / "corrupt.sqlite")
    order, _ = execute(ledger, inventory_fees=(fee(),))
    row = ledger.db.execute("SELECT id,payload FROM journal WHERE kind='inventory_fee'").fetchone()
    payload = json.loads(row["payload"])
    payload["removed_basis"] = "2"
    ledger.db.execute("UPDATE journal SET payload=? WHERE id=?", (json.dumps(payload), row["id"]))
    with pytest.raises(LedgerError, match="source/basis"):
        ledger.reconcile(order.account_id)
    ledger.db.execute("DELETE FROM journal WHERE id=?", (row["id"],))
    with pytest.raises(LedgerError, match="fee coverage"):
        ledger.reconcile(order.account_id)
    ledger.close()


def test_actual_long_native_cost_preserves_every_cash_digit_and_sleeve_basis(tmp_path):
    ledger = account(tmp_path / "long.sqlite", cash="10000")
    book = SleeveBook(ledger, "backtest:test").initialize({"owner": D(10000), "empty": D(0)})
    cost = D("299.999999999999999999999999999999999999")
    order, actual = execute(
        ledger,
        settlement=FillSettlement(gross_quote=cost, source_id="cost", source_hash="f" * 64),
        inventory_fees=(fee(),),
        book=book,
        allocations={"owner": D(3)},
    )
    ledger.mark(order.account_id, [PriceMark(instrument=B, price=100, as_of=actual.executed_at)])
    views = book.snapshot(actual.executed_at)
    expected_cash = exact_decimal_sum([D(10000), cost.copy_negate()])
    assert actual.base_notional == cost
    assert ledger.available_cash(order.account_id) == expected_cash
    assert views["owner"].portfolio.cash == expected_cash
    assert views["empty"].portfolio.nav == 0
    assert ledger.reconcile(order.account_id)["cash"] == str(expected_cash)
    ledger.close()


def test_fee_never_spends_quantity_reserved_for_another_sell(tmp_path):
    ledger = account(tmp_path / "reserved.sqlite")
    execute(ledger, "fee-asset", instrument=BNB, quantity="1")
    reserved = intent("reserved", instrument=BNB, side="sell", qty="1", at=T + timedelta(seconds=2))
    ledger.submit(reserved)
    order = intent("other-buy", instrument=B, qty="3", at=T + timedelta(seconds=3))
    ledger.submit(order)
    before = ledger.reconcile(order.account_id)
    with pytest.raises(LedgerError, match="outstanding sell quantities"):
        ledger.apply_fill(
            fill(order).model_copy(update={"inventory_fees": (fee(BNB, at=order.eligible_at),)})
        )
    assert ledger.reconcile(order.account_id) == before
    assert ledger.order(order.account_id, order.intent_id)["filled"] == 0
    assert ledger.quantity(order.account_id, BNB.instrument_id) == 1
    ledger.close()


def test_full_fee_asset_debit_closes_actual_age_and_never_charges_other_sleeve(tmp_path):
    ledger = account(tmp_path / "ownership.sqlite", cash="10000")
    book = SleeveBook(ledger, "backtest:test").initialize({"owner": D(5000), "other": D(5000)})
    execute(
        ledger, "fee-asset", instrument=BNB, quantity="1", book=book, allocations={"other": D(1)}
    )
    at = T + timedelta(seconds=2)
    with pytest.raises(LedgerError, match="explicit ownership transfer"):
        with ledger.atomic():
            execute(
                ledger,
                at=at,
                book=book,
                allocations={"owner": D(3)},
                inventory_fees=(fee(BNB, "1", at=at + timedelta(seconds=1)),),
            )
    assert len(ledger.fills(book.account_id)) == 1
    assert ledger.quantity(book.account_id, BNB.instrument_id) == 1
    # The actual fill's owner owns the fee asset in this second attempt.
    execute(
        ledger,
        at=at,
        book=book,
        allocations={"other": D(3)},
        inventory_fees=(fee(BNB, "1", at=at + timedelta(seconds=1)),),
    )
    assert ledger.quantity(book.account_id, BNB.instrument_id) == 0
    assert ledger.position_opened_at(book.account_id, BNB.instrument_id) is None
    assert book.sync()["fees"] == {"owner": D(0), "other": D(100)}
    assert ledger.reconcile(book.account_id)["reconciled"]
    ledger.close()


def test_actual_operational_report_binds_fee_receipt_and_restart(tmp_path):
    path = tmp_path / "operational.sqlite"
    ledger = PortfolioLedger(path)
    ledger.create_account(
        AccountSpec(
            account_id="paper:fee",
            mode="paper",
            base_currency="USD",
            initial_cash=D(10000),
            opened_at=T,
        )
    )
    rule = VenueRules(
        venue="paper",
        instrument=B,
        symbol=B.symbol,
        available_at=T,
        quantity_step=D(".01"),
        price_tick=D(".01"),
        market_type="spot",
        active=True,
    )
    policy = FillPolicy(crypto_cost_bps=0)
    adapter = PaperVenue(ledger, "paper:fee", {B.instrument_id: rule}, fill_policy=policy)
    engine = VenueExecution(
        ledger,
        "paper:fee",
        {"paper": adapter},
        {B.instrument_id: "paper"},
        owner="direct",
        fill_policy=policy,
    )
    order = intent(qty="3", instrument=B).model_copy(update={"account_id": "paper:fee"})
    book = SleeveBook(ledger, order.account_id).initialize({"owner": D(10000)})
    book.register_intent(order, {"owner": D(3)}, source_hash="c" * 64, quantity_quantum=D(".01"))
    engine.prepare(order, {B.instrument_id: metadata(instrument=B)}, as_of=T)
    observed = order.eligible_at + timedelta(seconds=1)
    actual = fill(order).model_copy(update={"inventory_fees": (fee(available_at=observed),)})
    report = VenueOrderReport(
        venue="paper",
        client_order_id=engine.client_id(order.account_id, order.intent_id),
        venue_order_id="native-id",
        observed_at=observed,
        status="filled",
        cumulative_quantity=D(3),
        fills=(actual,),
        executions_complete=True,
        evidence_id="actual-native-fixture",
    )
    engine.ingest(order.intent_id, report)
    book.sync()
    assert engine.state(order.intent_id)["state"] == "filled"
    assert ledger.reconcile(order.account_id)["fees"] == "3.00"
    ledger.close()
    ledger = PortfolioLedger(path)
    assert not ledger.apply_fill(actual, observed_at=observed)
    assert ledger.quantity(order.account_id, B.instrument_id) == D("2.97")
    assert ledger.reconcile(order.account_id)["reconciled"]
    ledger.db.execute("DELETE FROM events WHERE id=?", ("fill-receipt:" + actual.fill_id,))
    with pytest.raises(LedgerError, match="source is unavailable"):
        ledger.reconcile(order.account_id)
    ledger.close()


def test_future_fee_valuation_and_future_report_availability_are_rejected():
    order = intent(qty="3", instrument=B)
    with pytest.raises(ValueError, match="future market valuation"):
        Fill.model_validate(
            {
                **fill(order).model_dump(),
                "inventory_fees": (fee(at=order.eligible_at + timedelta(seconds=1)),),
            }
        )
    actual = fill(order).model_copy(
        update={"inventory_fees": (fee(available_at=order.eligible_at + timedelta(seconds=1)),)}
    )
    with pytest.raises(ValueError, match="future inventory fee"):
        VenueOrderReport(
            venue="fixture",
            client_order_id="client",
            venue_order_id="native",
            observed_at=order.eligible_at,
            status="filled",
            cumulative_quantity=D(3),
            fills=(actual,),
            executions_complete=True,
            evidence_id="receipt",
        )
