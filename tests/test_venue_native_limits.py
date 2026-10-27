"""Exact native limit enforcement over synthetic documented source receipts."""

from datetime import timedelta
from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from src.execution.ledger import LedgerError
from src.execution.venue import VenueNotionalReference, VenueRules
from tests.test_durable_accounting import A, T, intent


def rule(**changes):
    return VenueRules(
        **{
            "venue": "paper",
            "instrument": A,
            "symbol": A.symbol,
            "available_at": T,
            "quantity_step": "1",
            "price_tick": ".01",
            "market_type": "equity",
            "active": True,
            "order_types": ("market", "limit", "stop"),
            **changes,
        }
    )


def reference(**changes):
    return VenueNotionalReference(
        **{
            "price": "100",
            "kind": "weighted-average",
            "average_minutes": 5,
            "observed_at": T,
            "available_at": T,
            "source_hash": "a" * 64,
            "reference_price_absence_hash": "b" * 64,
            **changes,
        }
    )


def test_minimum_and_maximum_notional_use_actual_order_price_and_exact_inclusive_bounds():
    limits = rule(minimum_notional="999.99999999999999999999999999", maximum_notional="1000")
    limits.require_order(intent(), T)
    with pytest.raises(LedgerError, match="maximum quote notional"):
        limits.require_order(intent(qty="11"), T)
    with pytest.raises(LedgerError, match="minimum quote notional"):
        limits.require_order(intent(qty="9"), T)
    # A good indicative reference cannot disguise the actual limit/stop price.
    for order_type, price_key in (("limit", "limit_price"), ("stop", "stop_price")):
        order = intent().model_copy(update={"order_type": order_type, price_key: D(101)})
        with pytest.raises(LedgerError, match="maximum quote notional"):
            limits.require_order(order, T)


def test_native_market_price_is_explicit_causal_and_exactly_matches_filter_basis():
    limits = rule(maximum_notional="1000", market_notional_average_minutes=5)
    with pytest.raises(LedgerError, match="actual current native"):
        limits.require_order(intent(price="90"), T)
    limits = limits.model_copy(update={"market_notional_reference": reference(price="101")})
    with pytest.raises(LedgerError, match="maximum quote notional"):
        limits.require_order(intent(price="90"), T)
    limits = limits.model_copy(update={"market_notional_reference": reference()})
    limits.require_order(intent(price="101"), T)
    with pytest.raises(LedgerError, match="actual current native"):
        limits.require_order(intent(), T + timedelta(seconds=31))
    for price in (
        reference(available_at=T + timedelta(seconds=1)),
        reference(observed_at=T + timedelta(seconds=1), available_at=T + timedelta(seconds=1)),
    ):
        with pytest.raises(LedgerError, match="actual current native"):
            limits.model_copy(update={"market_notional_reference": price}).require_order(
                intent(), T
            )
    with pytest.raises(LedgerError, match="averaging window"):
        limits.model_copy(
            update={"market_notional_reference": reference(average_minutes=2)}
        ).require_order(intent(), T)
    override = reference(
        kind="venue-reference", average_minutes=None, reference_price_absence_hash=None
    )
    limits.model_copy(update={"market_notional_reference": override}).require_order(intent(), T)
    for changes in (
        {"reference_price_absence_hash": None},
        {"kind": "last-trade", "average_minutes": 5},
        {"kind": "venue-reference"},
    ):
        with pytest.raises(ValidationError):
            reference(**changes)


def test_market_applicability_is_independent_for_minimum_and_maximum():
    order = intent()
    for limits in (
        rule(minimum_notional=2000, minimum_notional_applies_to_market=False),
        rule(maximum_notional=500, maximum_notional_applies_to_market=False),
    ):
        limits.require_order(order, T)
        with pytest.raises(LedgerError, match="quote notional"):
            limits.require_order(
                order.model_copy(update={"order_type": "limit", "limit_price": D(100)}), T
            )
    limits = rule(
        minimum_notional=500, maximum_notional=1000, maximum_notional_applies_to_market=False
    )
    with pytest.raises(LedgerError, match="minimum"):
        limits.require_order(intent(qty="4"), T)
    limits.require_order(intent(qty="11"), T)


def test_market_specific_lots_and_explicit_price_range_are_additional_constraints():
    limits = rule(
        quantity_step="1",
        maximum_quantity=20,
        market_quantity_step="2",
        market_minimum_quantity=4,
        market_maximum_quantity=10,
        minimum_price=99,
        maximum_price=101,
    )
    limits.require_order(intent(), T)
    for quantity in (2, 5, 12):
        with pytest.raises(LedgerError, match="market quantity"):
            limits.require_order(intent(qty=str(quantity)), T)
    limits.require_order(
        intent(qty="5").model_copy(update={"order_type": "limit", "limit_price": D(100)}), T
    )
    for price in (D(98), D(102), D("100.001")):
        with pytest.raises(LedgerError, match="price range"):
            limits.require_order(
                intent().model_copy(update={"order_type": "limit", "limit_price": price}), T
            )
    with pytest.raises(ValidationError, match="minimum exceeds"):
        rule(minimum_quantity=10, maximum_quantity=5)


def test_admission_rule_hashes_round_trip_and_old_shape_is_preserved():
    legacy = rule()
    assert not set(legacy.model_dump()) & {
        "maximum_notional",
        "minimum_notional_applies_to_market",
        "maximum_notional_applies_to_market",
        "market_notional_average_minutes",
        "market_notional_reference",
        "market_reference_max_age_seconds",
        "minimum_price",
        "maximum_price",
        "market_quantity_step",
        "market_minimum_quantity",
        "market_maximum_quantity",
    }
    for limits in (
        legacy,
        rule(
            maximum_notional="1000",
            market_notional_average_minutes=5,
            market_notional_reference=reference(),
        ),
    ):
        assert (
            VenueRules.model_validate_json(limits.model_dump_json()).content_hash()
            == limits.content_hash()
        )
