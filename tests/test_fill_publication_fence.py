"""A new fill cannot mutate already published sleeve economic history."""

from datetime import timedelta
from decimal import Decimal as D

import pytest

from src.domain.portfolio import AccountSpec, PriceMark
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.sleeves import SleeveBook
from tests.test_alpaca_venue import admit, setup
from tests.test_durable_accounting import A, T
from tests.test_durable_accounting import fill as synthetic_fill

TABLES = (
    "accounts",
    "holdings",
    "orders",
    "journal",
    "marks",
    "snapshots",
    "sleeve_events",
    "sleeve_observations",
    "sleeve_snapshot_receipts",
)


def economics(ledger):
    return {
        name: [tuple(row) for row in ledger.db.execute("SELECT * FROM " + name)] for name in TABLES
    }


def prepared(path):
    wire, engine, adapter, order = setup(path)
    book = SleeveBook(engine.ledger, order.account_id).initialize(
        {"owner": D(100000), "empty": D(0)}
    )
    engine.ledger.mark(order.account_id, [PriceMark(instrument=A, price=D(100), as_of=T)])
    book.register_intent(
        order, {"owner": order.quantity}, source_hash="a" * 64, quantity_quantum=D(1)
    )
    assert admit(wire, engine, adapter, order)["state"] == "new"
    return wire, engine, adapter, order, book


@pytest.mark.parametrize("later", [0, 1])
@pytest.mark.parametrize("restart", [False, True])
def test_late_native_fill_cannot_rewrite_published_sleeve_history(tmp_path, later, restart):
    wire, engine, adapter, order, book = prepared(tmp_path)
    publication_at = T + timedelta(seconds=2 + later)
    original = book.snapshot(publication_at)
    assert not engine.ledger.history(order.account_id)
    before = economics(engine.ledger)
    if restart:
        adapter.close()
        engine.ledger.close()
        wire, engine, adapter, same = setup(tmp_path, wire)
        assert order == same
        book = SleeveBook(engine.ledger, order.account_id)
    try:
        wire.now = T + timedelta(seconds=4)
        wire.add_fill("actual-after-publication", "3", "100", 2)
        wire.order.update(status="partially_filled", filled_qty="3")
        with pytest.raises(LedgerError, match="published.*sleeve"):
            engine.reconcile_all(as_of=wire.now)
        assert economics(engine.ledger) == before
        assert not engine.ledger.fills(order.account_id)
        assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
        assert engine.state(order.intent_id)["state"] == "reconciliation_required"
        assert engine.ledger.db.execute("SELECT 1 FROM venue_reports WHERE applied=0").fetchone()
        assert not engine.ledger.db.execute(
            "SELECT 1 FROM events WHERE id LIKE 'fill-receipt:%'"
        ).fetchone()
        assert (
            book.observation(publication_at, source_hash=original["owner"].source_hash) == original
        )
        assert engine.ledger.reconcile(order.account_id)["reconciled"]
    finally:
        adapter.close()
        engine.ledger.close()


@pytest.mark.parametrize("publication", ["preview", "other-account", "earlier"])
def test_preview_other_account_and_earlier_publication_do_not_block_new_fill(tmp_path, publication):
    wire, engine, adapter, order, book = prepared(tmp_path)
    try:
        at = T + timedelta(seconds=3)
        if publication == "preview":
            book.snapshot(at, persist=False)
        elif publication == "other-account":
            engine.ledger.create_account(
                AccountSpec(
                    account_id="paper:other",
                    mode="paper",
                    base_currency="USD",
                    initial_cash=D(100),
                    opened_at=T,
                )
            )
            SleeveBook(engine.ledger, "paper:other").initialize({"other": D(100)}).snapshot(at)
        else:
            book.snapshot(T + timedelta(seconds=1))
        wire.now = at
        wire.add_fill("valid-fill", "3", "100", 2)
        wire.order.update(status="partially_filled", filled_qty="3")
        assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "partial"
        assert engine.ledger.quantity(order.account_id, A.instrument_id) == 3
        original = book.snapshot(at)
        fills = engine.ledger.fills(order.account_id)
        before = economics(engine.ledger)
        # Exact existing executions were included in publication; their retries
        # keep original receipt identity and must not become a new late fill.
        assert engine.reconcile_all(as_of=wire.now)[0]["state"] == "partial"
        assert engine.ledger.fills(order.account_id) == fills
        assert economics(engine.ledger) == before
        assert book.observation(at, source_hash=original["owner"].source_hash) == original
        assert engine.ledger.reconcile(order.account_id)["reconciled"]
    finally:
        adapter.close()
        engine.ledger.close()


def test_original_fill_retry_after_restart_keeps_first_receipt_and_detects_changes(tmp_path):
    wire, engine, adapter, order, book = prepared(tmp_path)
    wire.now = T + timedelta(seconds=3)
    wire.add_fill("original-fill", "3", "100", 2)
    wire.order.update(status="partially_filled", filled_qty="3")
    engine.reconcile_all(as_of=wire.now)
    original_fill = engine.ledger.fills(order.account_id)[0]
    book.snapshot(wire.now)
    receipt = engine.ledger.db.execute(
        "SELECT payload FROM events WHERE id=?", ("fill-receipt:" + original_fill.fill_id,)
    ).fetchone()[0]
    adapter.close()
    engine.ledger.close()
    ledger = PortfolioLedger(tmp_path / "alpaca.sqlite3")
    try:
        before = economics(ledger)
        assert not ledger.apply_fill(original_fill, observed_at=T + timedelta(seconds=4))
        assert (
            ledger.db.execute(
                "SELECT payload FROM events WHERE id=?", ("fill-receipt:" + original_fill.fill_id,)
            ).fetchone()[0]
            == receipt
        )
        with pytest.raises(LedgerError, match="reused with different content"):
            ledger.apply_fill(
                original_fill.model_copy(update={"price": D(101)}),
                observed_at=T + timedelta(seconds=4),
            )
        assert economics(ledger) == before
    finally:
        ledger.close()


@pytest.mark.parametrize("retained", ["legacy-index", "full-receipt"])
def test_either_original_publication_authority_fences_a_fill(tmp_path, retained):
    _, engine, adapter, order, book = prepared(tmp_path)
    try:
        book.snapshot(T + timedelta(seconds=3))
        if retained == "legacy-index":
            engine.ledger.db.execute("DROP TABLE sleeve_snapshot_receipts")
        else:
            engine.ledger.db.execute("DELETE FROM sleeve_observations")
        with pytest.raises(LedgerError, match="published.*sleeve"):
            engine.ledger.apply_fill(
                synthetic_fill(order, qty="3", at=T + timedelta(seconds=2)),
                observed_at=T + timedelta(seconds=4),
            )
        assert not engine.ledger.fills(order.account_id)
        assert engine.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    finally:
        adapter.close()
        engine.ledger.close()


@pytest.mark.parametrize("published", ["base", "sleeve"])
@pytest.mark.parametrize("observed", [False, True])
def test_fill_fence_rolls_back_callers_transaction_and_cannot_be_avoided_by_receipt_mode(
    tmp_path, published, observed
):
    _, engine, adapter, order, book = prepared(tmp_path)
    try:
        at = T + timedelta(seconds=3)
        if published == "base":
            engine.ledger.snapshot(order.account_id, at, 3600)
        else:
            book.snapshot(at)
        before = economics(engine.ledger)
        with pytest.raises(LedgerError, match="published NAV"):
            with engine.ledger.atomic():
                engine.ledger.record_event(order.account_id, "transient-fill-attempt", "fixture")
                engine.ledger.mark(
                    order.account_id, [PriceMark(instrument=A, price=D(99), as_of=at)]
                )
                engine.ledger.apply_fill(
                    synthetic_fill(order, qty="3", at=T + timedelta(seconds=2)),
                    observed_at=T + timedelta(seconds=4) if observed else None,
                )
        assert economics(engine.ledger) == before
        assert not engine.ledger.db.execute(
            "SELECT 1 FROM events WHERE id='transient-fill-attempt'"
        ).fetchone()
        assert not engine.ledger.db.execute(
            "SELECT 1 FROM events WHERE id LIKE 'fill-receipt:%'"
        ).fetchone()
    finally:
        adapter.close()
        engine.ledger.close()
