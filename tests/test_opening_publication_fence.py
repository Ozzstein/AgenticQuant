"""Paper observation boundary fixtures; native source authentication is external."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

from src.domain.portfolio import AccountSpec, FillPolicy, OpeningEvent, PriceMark
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.sleeves import SleeveBook
from src.execution.venue import PaperVenue, VenueExecution, VenueRules
from tests.test_durable_accounting import B, bar, intent, metadata
from tests.test_opening_source_identity import identity

START = datetime(2024, 1, 2, 23, 59, 50, tzinfo=timezone.utc)
MIDNIGHT = START + timedelta(seconds=10)
ACCOUNT = "paper:opening-publication"


def rows(ledger, *, exclude=()):
    names = [
        row[0]
        for row in ledger.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        if row[0] not in exclude
    ]
    return {
        name: tuple(
            tuple(row) for row in ledger.db.execute('SELECT * FROM "' + name + '" ORDER BY rowid')
        )
        for name in names
    }


def components(path):
    ledger = PortfolioLedger(path / "opening.sqlite3")
    ledger.create_account(
        AccountSpec(
            account_id=ACCOUNT,
            mode="paper",
            base_currency="USD",
            initial_cash=D(10000),
            opened_at=START,
        )
    )
    costs = FillPolicy(crypto_cost_bps=0)
    rules = VenueRules(
        venue="paper",
        instrument=B,
        symbol=B.symbol,
        available_at=START,
        quantity_step=D(1),
        price_tick=D(".01"),
        market_type="spot",
        active=True,
    )
    adapter = PaperVenue(ledger, ACCOUNT, {B.instrument_id: rules}, fill_policy=costs)
    execution = VenueExecution(
        ledger,
        ACCOUNT,
        {"paper": adapter},
        {B.instrument_id: "paper"},
        owner="direct",
        fill_policy=costs,
    )
    return ledger, adapter, execution, SleeveBook(ledger, ACCOUNT)


def observed(at, price, volume, trade_id):
    completed = bar(at=at, instrument=B, opening_volume=volume).model_copy(
        update={"open": D(price), "low": D(price)}
    )
    event = OpeningEvent.from_bar(completed)
    native = identity(event, trade_id="X:" + trade_id).model_copy(
        update={"provider": "kraken", "trade_id": trade_id}
    )
    clocks = dict(
        observed_at=at + timedelta(seconds=1),
        available_at=at + timedelta(seconds=2),
        as_of=at + timedelta(seconds=3),
        evidence_id="fixture-receipt-" + trade_id,
        evidence_hash=trade_id * 64,
        source_identity=native,
    )
    return event, completed, clocks


def prepared(path):
    ledger, adapter, execution, book = components(path)
    book.initialize({"owner": D(10000), "empty": D(0)})
    ledger.mark(ACCOUNT, [PriceMark(instrument=B, price=D(100), as_of=START)])
    order = intent(at=START, instrument=B, qty="3").model_copy(update={"account_id": ACCOUNT})
    book.register_intent(order, {"owner": D(3)}, source_hash="a" * 64, quantity_quantum=D(1))
    execution.prepare(order, {B.instrument_id: metadata(at=START, instrument=B)}, as_of=START)
    execution.dispatch(order.intent_id, as_of=order.eligible_at)
    frozen = {order.intent_id: execution.admission_metadata(order.intent_id)}
    first, _, clocks = observed(START + timedelta(seconds=2), "100", "100", "1")
    fills = adapter.settle_open(first, frozen, **clocks)
    assert len(fills) == 1 and fills[0].quantity == 1
    book.sync()
    assert ledger.quantity(ACCOUNT, B.instrument_id) == 1
    return ledger, adapter, execution, book, order, frozen, first, clocks


@pytest.mark.parametrize("publication", ["base", "sleeve", "legacy-only", "full-only"])
@pytest.mark.parametrize("restart", [False, True])
def test_new_same_clock_open_rejects_before_any_mark_order_or_receipt_change(
    tmp_path, publication, restart
):
    ledger, adapter, _, book, order, frozen, _, _ = prepared(tmp_path)
    ledger.mark(ACCOUNT, [PriceMark(instrument=B, price=D(90), as_of=MIDNIGHT)])
    if publication == "base":
        assert ledger.snapshot(ACCOUNT, MIDNIGHT, 0).nav == 9990
    else:
        published = book.snapshot(MIDNIGHT)
        assert published["owner"].portfolio.nav == 9990
        assert not ledger.history(ACCOUNT)
        if publication == "legacy-only":
            ledger.db.execute("DROP TABLE sleeve_snapshot_receipts")
        elif publication == "full-only":
            ledger.db.execute("DELETE FROM sleeve_observations")
    if restart:
        ledger.close()
        ledger, adapter, _, book = components(tmp_path)
    before = rows(ledger)
    event, _, clocks = observed(MIDNIGHT, "91", "200", "2")
    try:
        with pytest.raises(
            LedgerError, match="observed opening print conflicts with published NAV"
        ):
            adapter.settle_open(event, frozen, **clocks)
        assert rows(ledger) == before
        assert ledger.quantity(ACCOUNT, B.instrument_id) == 1
        assert ledger.order(ACCOUNT, order.intent_id)["remaining"] == 2
        assert ledger.order(ACCOUNT, order.intent_id)["reserved"] == 200
        assert adapter.observed_open_receipt(event.event_id) is None
        assert adapter.opening_source_binding(event.event_id) is None
        assert ledger.reconcile(ACCOUNT)["reconciled"]
    finally:
        ledger.close()


@pytest.mark.parametrize("publication", ["unpublished", "preview", "other-account", "earlier"])
def test_causal_same_clock_close_open_group_and_nonpublication_do_not_block(tmp_path, publication):
    ledger, adapter, _, book, _, frozen, _, _ = prepared(tmp_path)
    try:
        if publication == "earlier":
            book.snapshot(START + timedelta(seconds=3))
        ledger.mark(ACCOUNT, [PriceMark(instrument=B, price=D(90), as_of=MIDNIGHT)])
        if publication == "preview":
            assert book.snapshot(MIDNIGHT, persist=False)["owner"].portfolio.nav == 9990
        elif publication == "other-account":
            ledger.create_account(
                AccountSpec(
                    account_id="paper:other",
                    mode="paper",
                    base_currency="USD",
                    initial_cash=D(50),
                    opened_at=START,
                )
            )
            SleeveBook(ledger, "paper:other").initialize({"other": D(50)}).snapshot(MIDNIGHT)
        event, _, clocks = observed(MIDNIGHT, "91", "200", "2")
        fills = adapter.settle_open(event, frozen, **clocks)
        assert len(fills) == 1 and fills[0].quantity == 2
        assert ledger.quantity(ACCOUNT, B.instrument_id) == 3
        result = book.snapshot(MIDNIGHT)
        assert result["owner"].portfolio.nav == 9991
        assert result["empty"].portfolio.nav == 0
        assert ledger.reconcile(ACCOUNT)["reconciled"]
    finally:
        ledger.close()


def test_exact_retry_and_native_alias_preserve_published_sleeves_after_restart(tmp_path):
    ledger, adapter, _, book, _, frozen, first, clocks = prepared(tmp_path)
    ledger.mark(ACCOUNT, [PriceMark(instrument=B, price=D(90), as_of=MIDNIGHT)])
    published = book.snapshot(MIDNIGHT)
    source_hash = published["owner"].source_hash
    receipt = adapter.observed_open_receipt(first.event_id)
    ledger.close()
    ledger, adapter, _, book = components(tmp_path)
    try:
        later = {**clocks, "as_of": MIDNIGHT + timedelta(seconds=3)}
        before = rows(ledger)
        assert adapter.settle_open(first, frozen, **later) == []
        assert rows(ledger) == before
        assert adapter.observed_open_receipt(first.event_id) == receipt
        with pytest.raises(LedgerError, match="receipt changed"):
            adapter.settle_open(first.model_copy(update={"price": D(101)}), frozen, **later)
        assert rows(ledger) == before
        alias = first.model_copy(update={"event_id": "same-native-trade-different-wire-version"})
        alias_clocks = {**later, "evidence_id": "alias-fixture", "evidence_hash": "c" * 64}
        economic_before = rows(ledger, exclude=("events",))
        assert adapter.settle_open(alias, frozen, **alias_clocks) == []
        assert rows(ledger, exclude=("events",)) == economic_before
        assert adapter.opening_source_binding(alias.event_id).consumed_event_id == first.event_id
        assert book.observation(MIDNIGHT, source_hash=source_hash) == published
        after_alias = rows(ledger)
        assert adapter.settle_open(alias, frozen, **alias_clocks) == []
        assert rows(ledger) == after_alias
        assert ledger.reconcile(ACCOUNT)["reconciled"]
    finally:
        ledger.close()


def test_completed_bar_rejection_rolls_back_enclosing_transaction(tmp_path):
    ledger, adapter, _, book, _, frozen, _, _ = prepared(tmp_path)
    ledger.mark(ACCOUNT, [PriceMark(instrument=B, price=D(90), as_of=MIDNIGHT)])
    book.snapshot(MIDNIGHT)
    before = rows(ledger)
    _, completed, clocks = observed(MIDNIGHT, "91", "200", "2")
    clocks["as_of"] = completed.closed_at
    try:
        with pytest.raises(
            LedgerError, match="observed opening print conflicts with published NAV"
        ):
            with ledger.atomic():
                ledger.record_event(ACCOUNT, "outer-attempt", "fixture")
                adapter.settle_open(completed, frozen, **clocks)
        assert rows(ledger) == before
    finally:
        ledger.close()
