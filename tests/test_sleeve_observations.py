"""Historical receipt integrity over actual SQLite economics, synthetic inputs."""

import hashlib
import json
import sqlite3
from datetime import timedelta
from decimal import Decimal as D

import pytest

from src.domain.portfolio import PriceMark
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.sleeves import SleeveBook, digest
from tests.test_durable_accounting import A, T, fill, intent
from tests.test_sleeve_accounting import allocated_buy, setup


def published(path):
    ledger, book = setup(path)
    order = allocated_buy(ledger, book)
    first = fill(order, qty="5", fee=".50")
    ledger.apply_fill(first)
    ledger.snapshot(order.account_id, first.executed_at, 3600)
    original = book.snapshot(first.executed_at)
    second = fill(
        order, name="next", qty="5", fee=".50", at=first.executed_at + timedelta(seconds=1)
    )
    ledger.apply_fill(second)
    later = T + timedelta(days=1)
    ledger.mark(order.account_id, [PriceMark(instrument=A, price=D(90), as_of=later)])
    book.transfer_cash("actual-later-transfer", "alpha", "beta", D(100), later, "e" * 64)
    ledger.snapshot(order.account_id, later, 0)
    current = book.snapshot(later)
    return ledger, book, first.executed_at, original, later, current


def test_exact_readonly_receipt_survives_later_fills_transfers_and_current_state(
    monkeypatch, tmp_path
):
    path = tmp_path / "observations.sqlite"
    ledger, _, at, original, later, current = published(path)
    account_id = ledger.spec("backtest:test").account_id
    assert original["alpha"].portfolio.positions[0].quantity == 2
    assert current["alpha"].portfolio.positions[0].quantity == 3
    ledger.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    readonly = PortfolioLedger(path, read_only=True)
    mutation_ops = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_ALTER_TABLE,
    }
    readonly.db.set_authorizer(
        lambda action, *_: sqlite3.SQLITE_DENY if action in mutation_ops else sqlite3.SQLITE_OK
    )

    def no_current(*args, **kwargs):
        raise AssertionError("historical read must not reconstruct current state")

    monkeypatch.setattr(readonly, "snapshot", no_current)
    monkeypatch.setattr(readonly, "pending", no_current)
    monkeypatch.setattr(readonly, "reconcile", no_current)
    monkeypatch.setattr(SleeveBook, "sync", no_current)
    reader = SleeveBook(readonly, account_id)
    assert reader.observation(at) == original
    assert reader.observation(later) == current
    assert reader.observation(at + timedelta(microseconds=1)) is None
    assert reader.observation(at, source_hash="0" * 64) is None
    assert readonly.db.total_changes == 0
    readonly.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_same_cutoff_multiple_ownership_receipts_require_exact_source(tmp_path):
    ledger, book = setup(tmp_path / "same-clock.sqlite")
    first = book.snapshot(T)
    book.transfer_cash("same-clock", "alpha", "beta", D(100), T, "1" * 64)
    second = book.snapshot(T)
    assert first["alpha"].source_hash != second["alpha"].source_hash
    with pytest.raises(LedgerError, match="explicit source_hash"):
        book.observation(T)
    for parts in (first, second):
        assert book.observation(T, source_hash=parts["alpha"].source_hash) == parts
    assert book.snapshot(T) == second
    assert ledger.db.execute("SELECT COUNT(*) FROM sleeve_snapshot_receipts").fetchone()[0] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "receipt",
        "rehash_output",
        "events",
        "journal",
        "fill",
        "allocation",
        "canonical",
        "index",
        "history",
        "prefix",
        "omit_history",
    ],
)
def test_original_receipt_and_economic_source_corruption_rejected(tmp_path, mutation):
    ledger, book, at, original, later, _ = published(tmp_path / "corrupt.sqlite")
    account_id = book.account_id
    with ledger.atomic():
        if mutation in ("receipt", "rehash_output", "omit_history"):
            changed_at = later if mutation == "omit_history" else at
            row = ledger.db.execute(
                "SELECT * FROM sleeve_snapshot_receipts WHERE at=?", (changed_at.isoformat(),)
            ).fetchone()
            payload = json.loads(row["payload"])
            if mutation == "omit_history":
                payload["history"] = []
                for sleeve in payload["sleeves"].values():
                    sleeve["portfolio"]["high_water_mark"] = sleeve["portfolio"]["nav"]
            else:
                payload["sleeves"]["alpha"]["portfolio"]["positions"][0]["quantity"] = "99"
            ledger.db.execute(
                "UPDATE sleeve_snapshot_receipts SET payload=?,receipt_hash=? WHERE at=?",
                (
                    json.dumps(payload),
                    digest(payload) if mutation != "receipt" else row["receipt_hash"],
                    changed_at.isoformat(),
                ),
            )
        elif mutation == "events":
            ledger.db.execute("UPDATE sleeve_events SET at=? WHERE kind='fill'", (T.isoformat(),))
        elif mutation == "journal":
            ledger.db.execute("UPDATE journal SET cash_delta='-900' WHERE kind='fill'")
        elif mutation == "fill":
            ledger.db.execute("UPDATE fills SET realized='1'")
        elif mutation == "allocation":
            row = ledger.db.execute("SELECT payload FROM sleeve_intents").fetchone()
            payload = json.loads(row[0])
            payload["allocations"] = {"alpha": "7", "beta": "3"}
            ledger.db.execute("UPDATE sleeve_intents SET payload=?", (json.dumps(payload),))
        elif mutation == "canonical":
            row = ledger.db.execute(
                "SELECT payload FROM snapshots WHERE at=?", (at.isoformat(),)
            ).fetchone()
            payload = json.loads(row[0])
            payload["reserved_cash"] = "100"
            ledger.db.execute(
                "UPDATE snapshots SET payload=? WHERE at=?", (json.dumps(payload), at.isoformat())
            )
        elif mutation in ("index", "history"):
            ledger.db.execute(
                "UPDATE sleeve_observations SET nav='1' WHERE at=? AND sleeve='alpha'",
                (at.isoformat(),),
            )
        else:
            ledger.db.execute("DELETE FROM journal WHERE kind='order'")
    with pytest.raises(LedgerError):
        book.observation(
            later if mutation in ("history", "omit_history") else at,
            source_hash=None
            if mutation in ("history", "omit_history")
            else original["alpha"].source_hash,
        )
    # A bad receipt cannot change the original book by merely being read.
    assert (
        ledger.db.execute("SELECT cash FROM accounts WHERE id=?", (account_id,)).fetchone()[0]
        == "8999.00"
    )


def test_missing_full_legacy_receipt_is_not_reconstructed(tmp_path):
    path = tmp_path / "legacy.sqlite"
    ledger, book, at, _, later, _ = published(path)
    with ledger.atomic():
        ledger.db.execute("DELETE FROM sleeve_snapshot_receipts WHERE at=?", (at.isoformat(),))
    assert book.observation(at) is None
    with pytest.raises(LedgerError, match="lacks its original full receipt"):
        book.observation(later)
    ledger.db.execute("DROP TABLE sleeve_snapshot_receipts")
    ledger.close()
    readonly = PortfolioLedger(path, read_only=True)
    assert SleeveBook(readonly, "backtest:test").observation(at) is None
    assert (
        readonly.db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='sleeve_snapshot_receipts'"
        ).fetchone()
        is None
    )
    readonly.close()


def test_unpublished_future_sources_and_other_account_do_not_become_receipts(tmp_path):
    ledger, book = setup(tmp_path / "unpublished.sqlite")
    book.snapshot(T, persist=False)
    assert book.observation(T) is None
    original = book.snapshot(T)
    # A later mutable position/cache is deliberately irrelevant to original proof.
    ledger.db.execute("UPDATE accounts SET cash='777777'")
    ledger.db.execute("DELETE FROM marks")
    assert book.observation(T) == original
    ledger.db.execute("UPDATE accounts SET cash='10000'")
    book.transfer_cash("future", "alpha", "beta", D(10), T + timedelta(days=1), "f" * 64)
    with pytest.raises(LedgerError, match="future ownership"):
        book.snapshot(T)
    assert book.observation(T) == original


def test_reservation_proof_keeps_exact_finite_decimal_operation_order(tmp_path):
    ledger, book = setup(tmp_path / "reservation.sqlite")
    order = intent(qty="7", fee=".123456789123456789123456789")
    book.register_intent(order, {"alpha": D(7)}, source_hash="1" * 64, quantity_quantum=D(1))
    ledger.submit(order)
    for i, qty in enumerate(("1", "2", "1")):
        executed = fill(
            order, name=str(i), qty=qty, fee=".01", at=order.eligible_at + timedelta(seconds=i)
        )
        ledger.apply_fill(executed)
        parts = book.snapshot(executed.executed_at)
        assert book.observation(executed.executed_at) == parts
        assert (
            parts["alpha"].portfolio.reserved_cash
            == ledger.order(order.account_id, order.intent_id)["reserved"]
        )


def test_original_nonspendable_entitlement_survives_later_payment_fees_and_tax(tmp_path):
    from tests.test_dividend_accounting import buy, distribution, terms

    path = tmp_path / "income-observation.sqlite"
    ledger, book = setup(path)
    declared = terms()
    ledger.register_cash_dividend(book.account_id, declared, as_of=T)
    buy(ledger, book=book, allocations={"alpha": D(10)})
    book.accrue_dividend(declared.action_id, at=declared.ex_at)
    ledger.mark(book.account_id, [PriceMark(instrument=A, price=D(99), as_of=declared.ex_at)])
    ledger.snapshot(book.account_id, declared.ex_at, 0)
    entitled = book.snapshot(declared.ex_at)
    notice = distribution(declared)
    book.apply_dividend_distribution(notice, as_of=notice.available_at)
    ledger.mark(book.account_id, [PriceMark(instrument=A, price=D(99), as_of=notice.available_at)])
    ledger.snapshot(book.account_id, notice.available_at, 0)
    paid = book.snapshot(notice.available_at)
    ledger.close()
    readonly = PortfolioLedger(path, read_only=True)
    reader = SleeveBook(readonly, "backtest:test")
    assert reader.observation(declared.ex_at) == entitled
    assert reader.observation(notice.available_at) == paid
    assert entitled["alpha"].portfolio.receivable_balance == 10
    assert entitled["alpha"].portfolio.cash == 4000
    assert paid["alpha"].portfolio.cash == 4008
    assert paid["alpha"].portfolio.fees == D(".5")
    assert paid["alpha"].portfolio.withholding_taxes == D("1.5")
    readonly.close()
