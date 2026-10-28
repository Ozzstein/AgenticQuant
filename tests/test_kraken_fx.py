"""Synthetic native fixtures through actual measured FX capture and certification."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from src.data.adapters.base import ProviderError, ReadHTTP
from src.data.adapters.operating import (
    KrakenOperatingAdapter,
    OperatingCapture,
    _fx,
    verify_operating_capture,
)
from src.data.evidence.store import EvidenceStore
from src.data.kraken_fx import parse_kraken_fx_reference, read_legacy_fx_archive
from src.domain.common import Instrument, RunContext

T = datetime(2026, 10, 28, tzinfo=timezone.utc)
FX = Instrument(
    instrument_id="spot:kraken:usdt-usd",
    symbol="USDT/USD",
    asset_class="crypto",
    quote_currency="USD",
    venue="kraken",
)


def context(at):
    return RunContext(
        run_id="fx-test", mode="paper", as_of=at, config_hash="0" * 64, code_version="test"
    )


def capture_fx(store, clock, ctx, *, rate="0.9997", mutation=None):
    stamp = str(Decimal(str(clock().timestamp())) - Decimal("0.876543211"))
    pair = dict(
        wsname="USDT/USD",
        aclass_base="currency",
        aclass_quote="currency",
        base="USDT",
        quote="ZUSD",
        lot="unit",
        lot_multiplier=1,
        status="online",
    )
    assets = {
        "USDT": dict(altname="USDT", aclass="currency", status="enabled"),
        "ZUSD": dict(altname="USD", aclass="currency", status="enabled"),
    }
    rows = [[rate, "12.345", "NATIVE_TIMESTAMP", "b", "m", "", 7]]
    if mutation == "pair":
        pair["quote"] = "ZEUR"
    if mutation == "asset":
        assets["ZUSD"]["altname"] = "USDT"
    if mutation == "ambiguous":
        rows.append(["1.0001", "1", "NATIVE_TIMESTAMP", "s", "l", "", 8])
    if mutation == "duplicate":
        rows.append(rows[0])
    if mutation == "no_id":
        rows[0].pop()
    if mutation == "future":
        stamp = str(Decimal(stamp) + 100)
    if mutation == "stale":
        stamp = str(Decimal(stamp) - 301)
    if mutation == "sub_ns":
        stamp += "1"
    calls = []

    def handler(request):
        calls.append(request)
        clock.at += timedelta(seconds=2)
        body = {
            "AssetPairs": {"USDTZUSD": pair},
            "Assets": assets,
            "Trades": {"USDTZUSD": rows, "last": "unused-opaque-cursor"},
        }[request.url.path.rsplit("/", 1)[-1]]
        raw = json.dumps(dict(error=[], result=body)).replace('"NATIVE_TIMESTAMP"', stamp).encode()
        return httpx.Response(200, content=raw)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        adapter = KrakenOperatingAdapter(
            store,
            clock=clock,
            http=ReadHTTP("kraken", "https://api.kraken.com", client=client, min_interval=0),
        )
        result = adapter.fx_reference(FX, ctx, market_id="USDTZUSD")
    finally:
        client.close()
    return next(e for e in result.evidence if e.kind == "fx"), calls


class Clock:
    def __init__(self, at=T):
        self.at = at

    def __call__(self):
        return self.at


def setup(tmp_path):
    store, clock = EvidenceStore(tmp_path / "fx.sqlite"), Clock()
    proof, calls = capture_fx(store, clock, context(T))
    return SimpleNamespace(store=store, clock=clock, proof=proof, calls=calls)


def test_exact_three_authorities_units_ns_and_original_receipts(tmp_path):
    f = setup(tmp_path)
    value = parse_kraken_fx_reference(
        f.store,
        FX,
        context(f.clock()),
        evidence_id=f.proof.record.evidence_id,
        valuation_at=f.clock(),
    )
    assert value["rate"] == "0.9997" and value["units"] == "USD/USDT"
    assert value["trades"][0]["timestamp_ns"].endswith("123456789")
    assert (
        value["reference_type"] == "last-trade-indicative"
        and value["executable_conversion"] is False
    )
    assert [r.url.path for r in f.calls] == [
        "/0/public/AssetPairs",
        "/0/public/Assets",
        "/0/public/Trades",
    ]
    assert all("assetVersion" not in r.url.params for r in f.calls)
    assert f.proof.record.ingested_at == f.proof.record.available_at == T + timedelta(seconds=6)
    assert verify_operating_capture(f.store, f.proof.record.evidence_id, as_of=f.clock()) == value
    assert (
        verify_operating_capture(
            f.store, f.proof.record.evidence_id, as_of=f.clock() + timedelta(days=1)
        )
        == value
    )


@pytest.mark.parametrize(
    "mutation", ["pair", "asset", "ambiguous", "duplicate", "no_id", "future", "sub_ns", "stale"]
)
def test_native_invalid_or_ambiguous_sources_keep_raw_without_fx(tmp_path, mutation):
    store, clock = EvidenceStore(tmp_path / "fx.sqlite"), Clock()
    with pytest.raises(ProviderError):
        capture_fx(store, clock, context(T), mutation=mutation)
    assert store.query(as_of=clock(), kind="operating_raw").records
    assert not store.query(as_of=clock(), kind="fx").records


def test_freshness_cutoffs_and_exact_expected_instrument(tmp_path):
    f = setup(tmp_path)
    key = f.proof.record.evidence_id
    for at, max_age in [
        (T + timedelta(seconds=5), 300),
        (T + timedelta(seconds=31), 30),
        (T + timedelta(seconds=301), 300),
    ]:
        with pytest.raises(ProviderError):
            parse_kraken_fx_reference(
                f.store, FX, context(at), evidence_id=key, valuation_at=at, max_age_seconds=max_age
            )
    with pytest.raises(ProviderError):
        parse_kraken_fx_reference(
            f.store,
            FX.model_copy(update={"instrument_id": "another"}),
            context(f.clock()),
            evidence_id=key,
            valuation_at=f.clock(),
        )
    for age in (0, 301, True):
        with pytest.raises(ProviderError):
            parse_kraken_fx_reference(
                f.store,
                FX,
                context(f.clock()),
                evidence_id=key,
                valuation_at=f.clock(),
                max_age_seconds=age,
            )


def test_certified_fx_operating_and_late_open_rejection(tmp_path):
    f = setup(tmp_path)
    spot = Instrument(
        instrument_id="spot:btc-usdt",
        symbol="BTC/USDT",
        quote_currency="USDT",
        venue="binance",
        asset_class="crypto",
    )
    capture = OperatingCapture(f.store, clock=f.clock)
    rate, _ = _fx(capture, spot, context(f.clock()), f.proof.record.evidence_id, at=f.clock())
    assert rate == Decimal(".9997")
    parent = capture.derived(
        context(f.clock()),
        (f.proof,),
        lambda _: dict(
            opening_events=[
                dict(
                    instrument=spot.model_dump(mode="json"),
                    opened_at=T.isoformat(),
                    fx_rate=".9997",
                )
            ]
        ),
        kind="operating_prints",
        logical_key="late-fx",
    )
    with pytest.raises(ProviderError):
        verify_operating_capture(f.store, parent.record.evidence_id, as_of=f.clock())


def test_forged_normalization_and_legacy_are_not_certified(tmp_path):
    f = setup(tmp_path)
    value = f.store.normalized(f.proof.record.evidence_id, as_of=f.clock())
    value["rate"] = "1"
    bad = f.store.put(
        f.proof.record.model_copy(update={"evidence_id": "forged-fx"}),
        f.store.raw(f.proof.record.evidence_id),
        kind="fx",
        logical_key="forged",
        normalized=value,
    )
    with pytest.raises(ProviderError):
        parse_kraken_fx_reference(
            f.store,
            FX,
            context(f.clock()),
            evidence_id=bad.record.evidence_id,
            valuation_at=f.clock(),
        )
    assert (
        read_legacy_fx_archive(f.store, bad.record.evidence_id, as_of=f.clock())["scope"]
        == "legacy-unverified-fx"
    )
    with pytest.raises(ProviderError):
        verify_operating_capture(f.store, bad.record.evidence_id, as_of=f.clock())


@pytest.mark.parametrize(
    "field,value",
    [
        ("rate", "1"),
        ("units", "USDT/USD"),
        ("executable_conversion", True),
        ("native_quote", "USDT"),
        ("observed_at", T.isoformat()),
        ("reference_type", "executable"),
        ("market_id", "OTHER"),
    ],
)
def test_entire_normalization_is_recomputed_not_hash_shaped_claims(tmp_path, field, value):
    f = setup(tmp_path)
    payload = f.store.normalized(f.proof.record.evidence_id, as_of=f.clock())
    payload[field] = value
    bad = f.store.put(
        f.proof.record.model_copy(update={"evidence_id": "forged-" + field}),
        f.store.raw(f.proof.record.evidence_id),
        kind="fx",
        logical_key="forged",
        normalized=payload,
    )
    with pytest.raises(ProviderError):
        parse_kraken_fx_reference(
            f.store,
            FX,
            context(f.clock()),
            evidence_id=bad.record.evidence_id,
            valuation_at=f.clock(),
        )


def test_parent_raw_changes_and_source_order_cannot_certify(tmp_path, monkeypatch):
    f = setup(tmp_path)
    payload = f.store.normalized(f.proof.record.evidence_id, as_of=f.clock())
    trade_id = payload["source_manifest"]["sources"][-1]["evidence_id"]
    original = f.store.raw
    monkeypatch.setattr(
        f.store,
        "raw",
        lambda key: (
            original(key).replace(b'"0.9997"', b'"1.9997"') if key == trade_id else original(key)
        ),
    )
    with pytest.raises(ProviderError):
        parse_kraken_fx_reference(
            f.store,
            FX,
            context(f.clock()),
            evidence_id=f.proof.record.evidence_id,
            valuation_at=f.clock(),
        )


def test_reference_cannot_refresh_stale_event_or_use_ambient_newer_source(tmp_path):
    f = setup(tmp_path)
    old = f.proof.record.evidence_id
    f.clock.at += timedelta(seconds=301)
    newer, _ = capture_fx(f.store, f.clock, context(f.clock()), rate="0.9999")
    with pytest.raises(ProviderError):
        parse_kraken_fx_reference(
            f.store, FX, context(f.clock()), evidence_id=old, valuation_at=f.clock()
        )
    assert (
        parse_kraken_fx_reference(
            f.store,
            FX,
            context(f.clock()),
            evidence_id=newer.record.evidence_id,
            valuation_at=f.clock(),
        )["rate"]
        == "0.9999"
    )


def test_transitive_event_rate_and_future_parent_receipt_are_bound(tmp_path):
    f = setup(tmp_path)
    spot = Instrument(
        instrument_id="spot:btc-usdt",
        symbol="BTC/USDT",
        quote_currency="USDT",
        venue="binance",
        asset_class="crypto",
    )
    capture = OperatingCapture(f.store, clock=f.clock)
    for rate in (".9997", "1"):
        parent = capture.derived(
            context(f.clock()),
            (f.proof,),
            lambda _: dict(
                opening_events=[
                    dict(
                        instrument=spot.model_dump(mode="json"),
                        opened_at=f.clock().isoformat(),
                        fx_rate=rate,
                    )
                ]
            ),
            kind="operating_prints",
            logical_key="fx-" + rate,
        )
        if rate == "1":
            with pytest.raises(ProviderError, match="fx_source_value"):
                verify_operating_capture(f.store, parent.record.evidence_id, as_of=f.clock())
        else:
            assert (
                verify_operating_capture(f.store, parent.record.evidence_id, as_of=f.clock())[
                    "opening_events"
                ][0]["fx_rate"]
                == rate
            )
