"""Actual historical canonical/sleeve authorities, using synthetic ledger economics."""

from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

from src.domain.common import Instrument
from src.domain.portfolio import AccountSpec, Fill, OrderIntent, PriceMark
from src.execution.ledger import PortfolioLedger
from src.execution.sleeves import SleeveBook
from src.research.registry import RegistryError
from src.research.strategy_experiments import _verify_review_snapshot

T = datetime(2024, 1, 2, 14, tzinfo=timezone.utc)
A = Instrument(
    instrument_id="US:A", symbol="A", asset_class="equity", quote_currency="USD", venue="XNYS"
)


def historical_review(path, *, persist=True):
    ledger = PortfolioLedger(path)
    ledger.create_account(
        AccountSpec(
            account_id="paper:advice",
            mode="paper",
            base_currency="USD",
            initial_cash="10000",
            opened_at=T,
        )
    )
    book = SleeveBook(ledger, "paper:advice").initialize({"alpha": D(5000), "beta": D(5000)})
    order = OrderIntent(
        intent_id="partial",
        account_id="paper:advice",
        run_id="original",
        instrument=A,
        side="buy",
        quantity="10",
        decision_at=T,
        eligible_at=T + timedelta(seconds=1),
        reference_price="100",
        estimated_fee="1",
        strategy_id="blended",
    )
    book.register_intent(
        order, {"alpha": D(3), "beta": D(7)}, source_hash="a" * 64, quantity_quantum=D(1)
    )
    ledger.submit(order)
    ledger.apply_fill(
        Fill(
            fill_id="actual-partial",
            intent_id=order.intent_id,
            account_id=order.account_id,
            quantity="5",
            price="100",
            fee=".5",
            executed_at=order.eligible_at,
            event_id="synthetic-print",
        )
    )
    at = T + timedelta(seconds=2)
    ledger.mark(order.account_id, [PriceMark(instrument=A, price="110", as_of=at)])
    base = ledger.snapshot(order.account_id, at, 0)
    parts = book.snapshot(at, 0, persist=persist)
    review = base.model_copy(update={"sleeves": parts})
    ledger.close()
    return review


def test_original_readonly_review_survives_later_actual_ownership_and_nav(tmp_path):
    path = tmp_path / "original.sqlite"
    review = historical_review(path)
    ledger = PortfolioLedger(path)
    book = SleeveBook(ledger, review.account_id)
    later = review.as_of + timedelta(days=1)
    ledger.cancel(review.account_id, "partial", later)
    book.transfer_cash("new-owner-flow", "alpha", "beta", D(1000), later, "b" * 64)
    ledger.mark(review.account_id, [PriceMark(instrument=A, price="120", as_of=later)])
    ledger.snapshot(review.account_id, later, 0)
    current = book.snapshot(later, 0)
    assert current != review.sleeves
    ledger.close()
    with closing(PortfolioLedger(path, read_only=True)) as archive:
        _verify_review_snapshot(archive, review)
        assert (
            SleeveBook(archive, review.account_id).observation(
                review.as_of, source_hash=review.sleeves["alpha"].source_hash
            )
            == review.sleeves
        )


def test_same_clock_ownership_receipts_resolve_only_original_shared_source_hash(tmp_path):
    path = tmp_path / "same-clock.sqlite"
    ledger = PortfolioLedger(path)
    ledger.create_account(
        AccountSpec(
            account_id="paper:advice",
            mode="paper",
            base_currency="USD",
            initial_cash="10000",
            opened_at=T,
        )
    )
    book = SleeveBook(ledger, "paper:advice").initialize({"alpha": D(5000), "beta": D(5000)})
    at = T + timedelta(seconds=1)
    base = ledger.snapshot("paper:advice", at, 0)
    first = book.snapshot(at, 0)
    book.transfer_cash("same-clock-flow", "alpha", "beta", D(1000), at, "a" * 64)
    second = book.snapshot(at, 0)
    assert first["alpha"].source_hash != second["alpha"].source_hash
    ledger.close()
    with closing(PortfolioLedger(path, read_only=True)) as archive:
        for parts in (first, second):
            _verify_review_snapshot(archive, base.model_copy(update={"sleeves": parts}))
        false_alias = {
            name: part.model_copy(update={"source_hash": second[name].source_hash})
            for name, part in first.items()
        }
        with pytest.raises(RegistryError, match="sleeves"):
            _verify_review_snapshot(archive, base.model_copy(update={"sleeves": false_alias}))


@pytest.mark.parametrize(
    "mutation", ["partial", "cash", "holding", "events", "mixed_source", "false_source"]
)
def test_original_sleeve_claims_must_equal_the_complete_exact_receipt(tmp_path, mutation):
    path = tmp_path / "original.sqlite"
    review = historical_review(path)
    parts = dict(review.sleeves)
    alpha = parts["alpha"]
    if mutation == "partial":
        parts.pop("beta")
    elif mutation == "cash":
        parts["alpha"] = alpha.model_copy(
            update={
                "portfolio": alpha.portfolio.model_copy(
                    update={"cash": alpha.portfolio.cash + D(1)}
                )
            }
        )
    elif mutation == "holding":
        holding = alpha.portfolio.positions[0].model_copy(update={"quantity": D(999)})
        parts["alpha"] = alpha.model_copy(
            update={"portfolio": alpha.portfolio.model_copy(update={"positions": (holding,)})}
        )
    elif mutation == "events":
        parts["alpha"] = alpha.model_copy(
            update={"source_event_ids": alpha.source_event_ids + ("invented",)}
        )
    elif mutation == "mixed_source":
        parts["alpha"] = alpha.model_copy(update={"source_hash": "f" * 64})
    else:
        parts = {name: s.model_copy(update={"source_hash": "f" * 64}) for name, s in parts.items()}
    with closing(PortfolioLedger(path, read_only=True)) as archive:
        with pytest.raises(RegistryError, match="sleeve"):
            _verify_review_snapshot(archive, review.model_copy(update={"sleeves": parts}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("cash", D(999)),
        ("reserved_cash", D(0)),
        ("fees", D(0)),
        ("high_water_mark", D(99999)),
        ("as_of", T),
        ("account_id", "paper:other"),
    ],
)
def test_base_review_still_requires_exact_account_economics_and_clock(tmp_path, field, value):
    path = tmp_path / "original.sqlite"
    review = historical_review(path)
    with closing(PortfolioLedger(path, read_only=True)) as archive:
        with pytest.raises(RegistryError, match="portfolio"):
            _verify_review_snapshot(archive, review.model_copy(update={field: value}))


def test_unpersisted_sleeves_cannot_be_recreated_from_present_state(tmp_path):
    path = tmp_path / "unarchived.sqlite"
    review = historical_review(path, persist=False)
    with closing(PortfolioLedger(path, read_only=True)) as archive:
        with pytest.raises(RegistryError, match="sleeves"):
            _verify_review_snapshot(archive, review)
        # An original observation with no sleeve claims remains base-verifiable.
        _verify_review_snapshot(archive, review.model_copy(update={"sleeves": {}}))


def test_historical_source_tampering_is_not_covered_by_unchanged_base_nav(tmp_path):
    path = tmp_path / "corrupt.sqlite"
    review = historical_review(path)
    ledger = PortfolioLedger(path)
    ledger.db.execute(
        "UPDATE sleeve_events SET payload='{}' WHERE account=? AND kind='fill'",
        (review.account_id,),
    )
    ledger.close()
    with closing(PortfolioLedger(path, read_only=True)) as archive:
        with pytest.raises(RegistryError, match="authority"):
            _verify_review_snapshot(archive, review)


def test_legacy_nav_only_archive_cannot_supply_full_sleeve_authority(tmp_path):
    path = tmp_path / "legacy.sqlite"
    review = historical_review(path)
    ledger = PortfolioLedger(path)
    ledger.db.execute("DROP TABLE sleeve_snapshot_receipts")
    ledger.close()
    with closing(PortfolioLedger(path, read_only=True)) as archive:
        with pytest.raises(RegistryError, match="sleeves"):
            _verify_review_snapshot(archive, review)
        assert (
            archive.db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='sleeve_snapshot_receipts'"
            ).fetchone()
            is None
        )
