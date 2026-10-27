"""Actual raw source parsing and native notional/quantity enforcement offline."""

from datetime import timedelta
from decimal import Decimal

import httpx
import pandas as pd
import pytest
from src.domain.portfolio import OrderIntent
from src.execution.ledger import LedgerError
from src.execution.venue import VenueRules

from src.data.adapters.base import ProviderError, ReadHTTP
from src.data.adapters.operating import OperatingCapture
from src.data.binance_reference import (
    parse_binance_notional_payload,
    parse_binance_notional_reference,
)
from tests.test_operating_capture import context, primary, read
from tests.test_operating_capture import crypto_capture as crypto_capture

pytestmark = pytest.mark.parametrize("crypto_capture", ["binance"], indirect=True)


def refresh(f):
    result = f.adapter.reference(
        f.instrument, f.req, bar_ids=f.bar_ids, fx_evidence_id=f.fx.record.evidence_id
    )
    proof = primary(result, "operating_reference")
    return proof, read(f.store, proof, f.clock)


def native_sources(f, value):
    selected = {}
    for source in value["source_manifest"]["sources"]:
        envelope = f.store.get(source["evidence_id"])
        if envelope.kind == "operating_raw":
            meta = f.store.normalized(envelope.record.evidence_id, as_of=f.clock())
            selected[meta["request"]["path"].split("/")[-1]] = envelope.record.evidence_id
    return selected


def test_non_null_native_wins_and_pure_reader_binds_exact_source(crypto_capture):
    f = crypto_capture
    value = read(f.store, f.reference, f.clock)
    ids = native_sources(f, value)
    actual = parse_binance_notional_reference(
        f.store,
        f.instrument,
        context(f.clock()),
        reference_evidence_id=ids["referencePrice"],
        average_minutes=5,
    )
    assert actual.kind == "venue-reference" and actual.price == Decimal("110.123456789123456789")
    assert actual.source_hash == f.store.get(ids["referencePrice"]).content_hash()
    assert actual.reference_price_absence_hash is None
    assert not any(r.url.path.endswith("avgPrice") for r in f.requests)
    with pytest.raises(ProviderError):
        parse_binance_notional_reference(
            f.store,
            f.instrument,
            f.req,
            reference_evidence_id=ids["referencePrice"],
            average_minutes=5,
        )
    late = context(f.clock() + timedelta(seconds=31))
    with pytest.raises(ProviderError, match="stale"):
        parse_binance_notional_reference(
            f.store,
            f.instrument,
            late,
            reference_evidence_id=ids["referencePrice"],
            average_minutes=5,
        )


@pytest.mark.parametrize("native_error", [False, True])
def test_exact_null_or_2043_absence_allows_only_exact_native_average(crypto_capture, native_error):
    f = crypto_capture
    f.state["reference_price"] = None
    if native_error:
        f.state["reference_error"] = -2043
    _, value = refresh(f)
    ref = value["venue_rules"][0]["market_notional_reference"]
    ids = native_sources(f, value)
    assert ref["kind"] == "weighted-average" and ref["average_minutes"] == 5
    assert ref["price"] == "109.123456789123456789"
    assert ref["source_hash"] == f.store.get(ids["avgPrice"]).content_hash()
    assert ref["reference_price_absence_hash"] == f.store.get(ids["referencePrice"]).content_hash()
    f.state["average_mins"] = 1
    with pytest.raises(ProviderError, match="window_differs"):
        refresh(f)


def test_zero_window_uses_real_last_individual_trade_and_wrong_error_is_not_absence(crypto_capture):
    f = crypto_capture
    f.state["reference_price"] = None
    f.state["filter_changes"] = {"MIN_NOTIONAL": {"avgPriceMins": 0}}
    _, value = refresh(f)
    ref = value["venue_rules"][0]["market_notional_reference"]
    assert ref["kind"] == "last-trade" and ref["average_minutes"] == 0
    assert ref["price"] == "108.123456789123456789"
    assert f.requests[-1].url.params["limit"] == "1"
    f.state["reference_error"] = -1121
    with pytest.raises(ProviderError, match="absence_not_proved"):
        refresh(f)
    assert f.requests[-1].url.path.endswith("referencePrice")


def test_independent_quantity_constraints_and_exact_risk_lcm(crypto_capture):
    f = crypto_capture
    f.state["filter_changes"] = {
        "LOT_SIZE": {"stepSize": ".02"},
        "MARKET_LOT_SIZE": {"stepSize": ".03"},
    }
    _, value = refresh(f)
    rules = VenueRules.model_validate(value["venue_rules"][0])
    assert rules.quantity_step == Decimal(".02") and rules.market_quantity_step == Decimal(".03")
    assert value["instrument_metadata"][0]["lot_size"] == "0.06"
    assert Decimal(value["instrument_metadata"][0]["min_notional"]) == Decimal(5) * Decimal(".9997")
    order = OrderIntent(
        intent_id="lot",
        account_id="paper:test",
        run_id="native",
        instrument=f.instrument,
        side="buy",
        quantity=Decimal(".08"),
        reference_price=110,
        strategy_id="test",
        decision_at=f.clock(),
        eligible_at=f.clock() + timedelta(microseconds=1),
    )
    with pytest.raises(LedgerError, match="market quantity"):
        rules.require_order(order, f.clock())
    rules.require_order(order.model_copy(update={"quantity": Decimal(".06")}), f.clock())


def test_missing_and_differing_native_windows_fail_after_raw_capture(crypto_capture):
    f = crypto_capture
    f.state["omit_window"] = True
    with pytest.raises(ProviderError, match="window_missing"):
        refresh(f)
    f.state["omit_window"] = False
    f.state["maximum_notional"] = True
    f.state["filter_changes"] = {"NOTIONAL": {"avgPriceMins": 1}}
    with pytest.raises(ProviderError, match="independent_native_notional_windows"):
        refresh(f)
    assert f.requests[-1].url.path.endswith("exchangeInfo")


def test_microsecond_header_is_retained_and_missing_unit_proof_rejected(crypto_capture):
    f = crypto_capture
    observed = f.clock() - timedelta(microseconds=1)
    http = ReadHTTP(
        "binance",
        "https://api.binance.com",
        min_interval=0,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "symbol": "BTCUSDT",
                        "referencePrice": "111.000000000000000001",
                        "timestamp": pd.Timestamp(observed).value // 1000,
                    },
                )
            )
        ),
    )
    capture = OperatingCapture(f.store, clock=f.clock)
    source, _ = capture.wire(
        http,
        "/api/v3/referencePrice",
        context(f.clock()),
        upstream="binance-spot-v3",
        params={"symbol": "BTCUSDT"},
        instrument=f.instrument,
        timestamp_unit="MICROSECOND",
        headers={"X-MBX-TIME-UNIT": "MICROSECOND"},
    )
    actual = parse_binance_notional_reference(
        f.store,
        f.instrument,
        context(f.clock()),
        reference_evidence_id=source.record.evidence_id,
        average_minutes=5,
    )
    assert actual.observed_at == observed
    assert actual.price == Decimal("111.000000000000000001")
    unstamped, _ = capture.wire(
        http,
        "/api/v3/referencePrice",
        context(f.clock()),
        upstream="binance-spot-v3",
        params={"symbol": "BTCUSDT"},
        instrument=f.instrument,
        timestamp_unit="MICROSECOND",
    )
    with pytest.raises(ProviderError, match="timestamp_unit_unproved"):
        parse_binance_notional_reference(
            f.store,
            f.instrument,
            context(f.clock()),
            reference_evidence_id=unstamped.record.evidence_id,
            average_minutes=5,
        )


def test_native_error_archiving_does_not_relax_authentication_or_default_errors(crypto_capture):
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(400, json={"code": -2043}))
    )
    http = ReadHTTP("binance", "https://api.binance.com", min_interval=0, client=client)
    with pytest.raises(ProviderError, match="http_failure"):
        http.get("/api/v3/referencePrice")
    assert http.get("/api/v3/referencePrice", allowed_error_statuses=(400,)) == b'{"code":-2043}'
    with pytest.raises(ValueError):
        http.get("/api/v3/referencePrice", allowed_error_statuses=(401, 403))


@pytest.mark.parametrize("policy", ["absent", "zero", "not_market"])
def test_explicit_native_filter_list_does_not_invent_an_applicable_minimum(crypto_capture, policy):
    f = crypto_capture
    if policy == "absent":
        f.state["omit_filters"] = ("MIN_NOTIONAL",)
    else:
        f.state["filter_changes"] = {
            "MIN_NOTIONAL": {"minNotional": "0"} if policy == "zero" else {"applyToMarket": False}
        }
    before = len(f.requests)
    _, value = refresh(f)
    rules = VenueRules.model_validate(value["venue_rules"][0])
    assert rules.minimum_notional == 0 and rules.maximum_notional is None
    assert rules.market_notional_reference is None
    assert Decimal(value["instrument_metadata"][0]["min_notional"]) == 0
    assert [r.url.path for r in f.requests[before:]] == ["/api/v3/exchangeInfo"]


def test_pure_byte_parser_preserves_values_but_cannot_supply_source_authority(crypto_capture):
    parsed = parse_binance_notional_payload(
        b'[{"id":9007199254740993123,"time":1711065960123456,"price":"108.123456789123456789"}]',
        operation="trades",
        symbol="BTCUSDT",
        timestamp_unit="MICROSECOND",
    )
    assert parsed.trade_id == "9007199254740993123"
    assert parsed.price == Decimal("108.123456789123456789")
    assert pd.Timestamp(parsed.observed_at).value == 1711065960123456000
    assert not hasattr(parsed, "source_hash") and not hasattr(parsed, "available_at")
    absent = parse_binance_notional_payload(
        b'{"code":-2043}',
        operation="referencePrice",
        symbol="BTCUSDT",
        timestamp_unit="MILLISECOND",
    )
    assert absent.kind == "reference-absent" and absent.observed_at is None
    for body in (
        b'{"symbol":"BTCUSDT","referencePrice":null,"timestamp":1,"timestamp":2}',
        b'{"symbol":"ETHUSDT","referencePrice":null,"timestamp":1}',
        b'{"symbol":"BTCUSDT","referencePrice":null,"timestamp":true}',
        b'{"code":-2043,"referencePrice":"111"}',
        b'{"code":-1121}',
    ):
        with pytest.raises(ProviderError):
            parse_binance_notional_payload(
                body, operation="referencePrice", symbol="BTCUSDT", timestamp_unit="MILLISECOND"
            )


def test_store_reader_rejects_valid_native_bytes_with_wrong_source_authority(crypto_capture):
    f = crypto_capture
    source_id = native_sources(f, read(f.store, f.reference, f.clock))["referencePrice"]
    raw = f.store.raw(source_id)
    variants = (
        {"origin": "https://untrusted.example"},
        {"path": "/api/v3/avgPrice"},
        {"params": {"symbol": "ETHUSDT"}},
        {"upstream": "binance-unverified"},
    )
    capture = OperatingCapture(f.store, clock=f.clock)
    for variant in variants:
        with httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=raw))
        ) as client:
            http = ReadHTTP(
                "binance",
                variant.get("origin", "https://api.binance.com"),
                client=client,
                min_interval=0,
            )
            envelope, _ = capture.wire(
                http,
                variant.get("path", "/api/v3/referencePrice"),
                context(f.clock()),
                upstream=variant.get("upstream", "binance-spot-v3"),
                params=variant.get("params", {"symbol": "BTCUSDT"}),
                instrument=f.instrument,
                timestamp_unit="MILLISECOND",
            )
        with pytest.raises(ProviderError, match="source_identity_differs"):
            parse_binance_notional_reference(
                f.store,
                f.instrument,
                context(f.clock()),
                reference_evidence_id=envelope.record.evidence_id,
                average_minutes=5,
            )
