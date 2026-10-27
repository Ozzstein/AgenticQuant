"""Synthetic official-shaped bodies through real capture, proof and feature paths."""

import json
from datetime import timedelta
from decimal import Decimal, localcontext
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from src.data.adapters.base import ProviderError, ReadHTTP
from src.data.adapters.coingecko_context import ORIGINS, verify_dominance_record
from src.data.adapters.context import CoinGeckoAdapter
from src.data.evidence.service import EvidenceService
from src.data.evidence.store import EvidenceStore
from src.data.features.contextual import materialize_evidence_fields
from src.data.features.crypto import crypto_context_bindings
from src.data.features.dataset import feature_dataset
from src.domain.common import Instrument
from src.domain.market import FeatureSpec
from tests.test_crypto_context_sources import T, ctx, funding, native_case, observations


@pytest.fixture
def coingecko(tmp_path):
    clients = []

    def factory(bodies, *, tier="demo", key=None, store=None, now=T):
        f = SimpleNamespace(calls=[], now=now)

        def get(request):
            f.now += timedelta(seconds=1)
            f.calls.append(request)
            body = bodies[len(f.calls) - 1]
            if isinstance(body, int):
                return httpx.Response(body)
            return httpx.Response(
                200, content=body if isinstance(body, bytes) else json.dumps(body).encode()
            )

        client = httpx.Client(transport=httpx.MockTransport(get))
        clients.append(client)
        f.store = store or EvidenceStore(tmp_path / (tier + str(len(clients)) + ".sqlite"))
        f.adapter = CoinGeckoAdapter(
            f.store,
            api_key=key,
            tier=tier,
            clock=lambda: f.now,
            http=ReadHTTP("coingecko", ORIGINS[tier], client=client, attempts=1, min_interval=0),
        )
        return f

    yield factory
    for client in clients:
        client.close()


def global_body(at=T - timedelta(hours=1)):
    return {"data": {"updated_at": int(at.timestamp()), "market_cap_percentage": {"btc": 60}}}


def chart(rows):
    return {"market_cap_chart": {"market_cap": rows}}


def ms(at):
    return pd.Timestamp(at).value // 1_000_000


def forged(store, original, payload, name):
    return store.put(
        original.record.model_copy(update={"evidence_id": name}),
        store.raw(original.record.evidence_id),
        kind=original.kind,
        logical_key=original.logical_key,
        normalized=payload,
    )


def test_current_native_precision_dates_receipt_and_key_redaction(coingecko):
    native = (
        '{"data":{"updated_at":%d,"market_cap_percentage":'
        '{"btc":60.12345678901234567890123456789}}}' % int(T.timestamp())
    ).encode()
    f = coingecko([native], key="fixture-not-real-key")
    result = f.adapter.current_dominance(ctx())
    raw, value = result.evidence
    payload = verify_dominance_record(f.store, value.record.evidence_id, as_of=f.now)
    assert payload["value"] == "0.6012345678901234567890123456789"
    assert value.record.observed_at == T
    assert value.record.available_at == value.record.ingested_at == T + timedelta(seconds=1)
    assert f.store.raw(raw.record.evidence_id) == native
    assert f.calls[0].headers["x-cg-demo-api-key"] == "fixture-not-real-key"
    for item in result.evidence:
        assert "fixture-not-real-key" not in item.model_dump_json()
        assert "fixture-not-real-key" not in str(
            f.store.normalized(item.record.evidence_id, as_of=f.now)
        )
    with pytest.raises(ValueError, match="available"):
        f.store.get(value.record.evidence_id, as_of=T)


@pytest.mark.parametrize(
    "body,code",
    [
        ({}, "malformed"),
        (global_body(T + timedelta(hours=1)), "after_actual_receipt"),
        ({"data": {"updated_at": "1", "market_cap_percentage": {"btc": 60}}}, "updated_at"),
        ({"data": {"updated_at": 1, "market_cap_percentage": {"btc": -1}}}, "decimal"),
        ({"data": {"updated_at": 1, "market_cap_percentage": {"btc": 101}}}, "outside_range"),
        (b'{"data":{},"data":{}}', "duplicate_native"),
    ],
)
def test_bad_current_body_is_archived_and_never_becomes_neutral(coingecko, body, code):
    f = coingecko([body])
    with pytest.raises(ProviderError, match=code):
        f.adapter.current_dominance(ctx())
    raw = f.store.query(as_of=f.now, kind="crypto_raw")
    assert len(raw.records) == 1
    assert not f.store.query(as_of=f.now, kind="btc_dominance").records


@pytest.mark.parametrize("status", [401, 403, 429])
def test_access_denial_is_not_empty_success(coingecko, status):
    f = coingecko([status])
    with pytest.raises(ProviderError):
        f.adapter.current_dominance(ctx())
    assert not f.store.query(as_of=f.now, kind="btc_dominance").records


def test_historical_exact_native_join_and_independent_decimal_proof(coingecko):
    at = T - timedelta(days=1)
    f = coingecko(
        [chart([[ms(at), 3]]), {"market_caps": [[ms(at), 1]]}],
        tier="pro",
        key="fixture-not-real-key",
    )
    result = f.adapter.dominance_history(ctx(), days="30")
    assert result.status == "available" and len(f.calls) == 2
    assert f.calls[0].url.path == "/api/v3/global/market_cap_chart"
    assert dict(f.calls[0].url.params) == {"days": "30", "vs_currency": "usd"}
    assert f.calls[1].url.path == "/api/v3/coins/bitcoin/market_chart/range"
    assert dict(f.calls[1].url.params)["precision"] == "full"
    value = observations(result, "btc_dominance")[0]
    payload = verify_dominance_record(f.store, value.record.evidence_id, as_of=f.now)
    with localcontext() as arithmetic:
        arithmetic.prec = 50
        assert Decimal(payload["value"]) == Decimal(1) / Decimal(3)
    assert payload["btc_market_cap"] == "1" and payload["total_market_cap"] == "3"
    assert payload["quote_currency"] == "USD"
    assert payload["bitcoin_pointer"] == ["market_caps", 0]
    assert payload["global_pointer"] == ["market_cap_chart", "market_cap", 0]
    assert len(payload["source_manifest"]) == 2
    assert value.record.observed_at == at < value.record.available_at
    assert value.record.available_at == T + timedelta(seconds=2)
    coverage = observations(result, "crypto_coverage")[0]
    assert (
        f.store.normalized(coverage.record.evidence_id, as_of=f.now)["completeness"]
        == "not-certified"
    )


def test_unmatched_one_millisecond_is_a_gap_without_interpolation(coingecko):
    a, b = ms(T - timedelta(days=2)), ms(T - timedelta(days=1))
    f = coingecko(
        [chart([[a, 10], [b, 20]]), {"market_caps": [[a + 1, 6], [b, 12]]}],
        tier="pro",
        key="fixture-only",
    )
    result = f.adapter.dominance_history(ctx())
    assert result.status == "partial"
    assert len(observations(result, "btc_dominance")) == 1
    coverage = f.store.normalized(
        observations(result, "crypto_coverage")[0].record.evidence_id, as_of=f.now
    )
    assert len(coverage["unmatched_global"]) == len(coverage["unmatched_bitcoin"]) == 1
    assert coverage["paired_observations"] == 1


@pytest.mark.parametrize("rows", [[], [[ms(T - timedelta(days=1)), 10]]])
def test_empty_or_unmatched_history_cannot_claim_complete_zero(coingecko, rows):
    f = coingecko([chart(rows), {"market_caps": []}], tier="pro", key="fixture-only")
    result = f.adapter.dominance_history(ctx())
    assert result.status == "missing" and not observations(result, "btc_dominance")
    assert len(f.calls) == (2 if rows else 1)
    assert (
        f.store.normalized(
            observations(result, "crypto_coverage")[0].record.evidence_id, as_of=f.now
        )["completeness"]
        == "not-certified"
    )


@pytest.mark.parametrize(
    "tier,key,days", [("demo", None, "30"), ("pro", None, "30"), ("pro", "fixture", "2")]
)
def test_paid_history_entitlement_and_documented_window_required_before_http(
    coingecko, tier, key, days
):
    f = coingecko([], tier=tier, key=key)
    with pytest.raises(ProviderError):
        f.adapter.dominance_history(ctx(), days=days)
    assert not f.calls


@pytest.mark.parametrize(
    "field,value",
    [("value", "0.99"), ("bitcoin_pointer", ["market_caps", -1]), ("quote_currency", "USDT")],
)
def test_hash_shaped_scalar_or_pointer_cannot_override_native_history(coingecko, field, value):
    at = ms(T - timedelta(days=1))
    f = coingecko([chart([[at, 10]]), {"market_caps": [[at, 6]]}], tier="pro", key="fixture")
    result = f.adapter.dominance_history(ctx())
    original = observations(result, "btc_dominance")[0]
    payload = f.store.normalized(original.record.evidence_id, as_of=f.now)
    payload[field] = value
    bad = forged(f.store, original, payload, "forged-" + field)
    with pytest.raises(ProviderError):
        verify_dominance_record(f.store, bad.record.evidence_id, as_of=f.now)


def test_all_four_crypto_context_fields_flow_from_actual_producers(tmp_path, coingecko):
    at = T - timedelta(hours=1)
    f = native_case(tmp_path, rows=[funding(at)])
    funding_result = f.adapter.derivatives(
        f.instrument, ctx(), capability="funding", since=T - timedelta(days=1), end=T
    )
    f.rows = [
        dict(
            symbol="BTCUSDT",
            timestamp=ms(at),
            sumOpenInterest="123.456789",
            sumOpenInterestValue="999999",
        )
    ]
    oi_result = f.adapter.derivatives(
        f.instrument, ctx(), capability="open_interest", since=T - timedelta(days=1), end=T
    )
    cg = coingecko([global_body(at)], store=f.store, now=f.now)
    dominance = cg.adapter.current_dominance(ctx())
    spot = Instrument(
        instrument_id="spot:btc:usdt",
        symbol="BTC/USDT",
        quote_currency="USDT",
        asset_class="crypto",
        venue="binance",
    )
    f.exchange = f.adapter.exchange
    markets = {f.instrument.symbol: f.market, spot.symbol: {"spot": True}}
    f.exchange.load_markets = lambda: markets
    f.exchange.parse_timeframe = lambda _: 86400
    f.exchange.fetch_ohlcv = lambda *args, **kwargs: [
        [ms(T - timedelta(days=1)), "100", "110", "90", "105.5", "5"]
    ]
    bars = f.adapter.bars(spot, ctx(), ingested_at=cg.now, start=T - timedelta(days=1), end=T)
    bindings = crypto_context_bindings(
        spot, perpetual=f.instrument, benchmark=spot, open_interest_units="base:BTC"
    )
    selected = tuple(
        v
        for result in (funding_result, oi_result, dominance, bars)
        for v in result.evidence
        if v.kind in ("funding", "open_interest", "btc_dominance", "bar")
    )
    index = pd.MultiIndex.from_tuples([(T, spot.instrument_id)], names=["datetime", "instrument"])
    frozen = frozenset(item.record.evidence_id for item in selected)
    strict = materialize_evidence_fields(
        f.store, index, ctx(cg.now), bindings, allowed_evidence_ids=frozen
    )
    assert strict.values.isna().all().all()
    current = materialize_evidence_fields(
        f.store,
        index,
        ctx(cg.now),
        bindings,
        allowed_evidence_ids=frozen,
        availability_policy="captured-vintage-at-fit-1",
    )
    assert current.values.iloc[0].to_dict() == dict(
        benchmark_close=105.5,
        btc_dominance=0.6,
        funding_rate=float(funding(at)["fundingRate"]),
        open_interest=123.456789,
    )
    assert current.values.index.equals(index)
    assert current.observed_at.iloc[0].to_dict() == dict(
        benchmark_close=T, btc_dominance=at, funding_rate=at, open_interest=at
    )
    assert current.source_ids.iloc[0] == tuple(sorted(frozen))
    assert current.available_at.iloc[0] == cg.now
    snapshot = EvidenceService(f.store).market_snapshot(
        (spot,), ctx(cg.now), start=T - timedelta(days=1), market_type="spot"
    )
    specs = tuple(
        FeatureSpec(
            name=b.name, expression="$" + b.name, required_fields=(b.name,), market_type="spot"
        )
        for b in bindings
    )
    dataset = feature_dataset(snapshot, specs, evidence_fields=current)
    pd.testing.assert_frame_equal(dataset.features.sort_index(axis=1), current.values)
    assert dataset.feature_available_at.iloc[0] == cg.now
    assert dataset.snapshot_hash != snapshot.content_hash()
    assert dataset.eligible_training(cg.now).features.empty  # No fabricated future label.
    wrong_units = tuple(
        b.model_copy(
            update={"filters": tuple((k, "contracts" if k == "units" else v) for k, v in b.filters)}
        )
        if b.name == "open_interest"
        else b
        for b in bindings
    )
    missing = materialize_evidence_fields(
        f.store,
        index,
        ctx(cg.now),
        wrong_units,
        allowed_evidence_ids=frozen,
        availability_policy="captured-vintage-at-fit-1",
    )
    assert pd.isna(missing.values.iloc[0]["open_interest"]) and pd.isna(
        missing.available_at.iloc[0]
    )


def test_exact_decimal_conflicts_do_not_disappear_in_float_features(tmp_path):
    at = T - timedelta(hours=1)
    f = native_case(
        tmp_path,
        rows=[funding(at, "0.000100000000000000000001"), funding(at, "0.000100000000000000000002")],
    )
    assert float(f.rows[0]["fundingRate"]) == float(f.rows[1]["fundingRate"])
    result = f.adapter.derivatives(
        f.instrument, ctx(), capability="funding", since=T - timedelta(days=1), end=T
    )
    spot = Instrument(
        instrument_id="spot:btc",
        symbol="BTC/USDT",
        asset_class="crypto",
        quote_currency="USDT",
        venue="binance",
    )
    binding = crypto_context_bindings(
        spot, perpetual=f.instrument, benchmark=spot, open_interest_units="base:BTC"
    )[0]
    index = pd.MultiIndex.from_tuples([(T, spot.instrument_id)], names=["datetime", "instrument"])
    fields = materialize_evidence_fields(
        f.store,
        index,
        ctx(f.now),
        (binding,),
        allowed_evidence_ids=frozenset(
            v.record.evidence_id for v in observations(result, "funding")
        ),
        availability_policy="captured-vintage-at-fit-1",
    )
    assert pd.isna(fields.values.iloc[0, 0]) and pd.isna(fields.available_at.iloc[0])
    assert fields.coverage[0][-1] == "conflicting"


def test_conflicting_native_cap_pairs_remain_separate_evidence(coingecko, tmp_path):
    at = T - timedelta(days=1)
    f = coingecko(
        [chart([[ms(at), 10], [ms(at), 20]]), {"market_caps": [[ms(at), 6]]}],
        tier="pro",
        key="fixture",
    )
    result = f.adapter.dominance_history(ctx())
    values = observations(result, "btc_dominance")
    assert result.status == "partial" and len(values) == 2
    assert {
        verify_dominance_record(f.store, item.record.evidence_id, as_of=f.now)["value"]
        for item in values
    } == {"0.3", "0.6"}
    coverage = f.store.normalized(
        observations(result, "crypto_coverage")[0].record.evidence_id, as_of=f.now
    )
    assert coverage["conflicting_observations"] == [at.isoformat()]
    perp = native_case(tmp_path).instrument
    spot = Instrument(
        instrument_id="spot:btc",
        symbol="BTC/USDT",
        asset_class="crypto",
        quote_currency="USDT",
        venue="binance",
    )
    binding = crypto_context_bindings(
        spot, perpetual=perp, benchmark=spot, open_interest_units="base:BTC"
    )[2]
    index = pd.MultiIndex.from_tuples([(at, spot.instrument_id)], names=["datetime", "instrument"])
    fields = materialize_evidence_fields(
        f.store, index, ctx(f.now), (binding,), availability_policy="captured-vintage-at-fit-1"
    )
    assert pd.isna(fields.values.iloc[0, 0]) and fields.coverage[0][-1] == "conflicting"


def test_contextual_materialization_reparses_native_dominance_proof(coingecko, tmp_path):
    f = coingecko([global_body()])
    original = observations(f.adapter.current_dominance(ctx()), "btc_dominance")[0]
    payload = f.store.normalized(original.record.evidence_id, as_of=f.now)
    payload["value"] = "0.99"
    bad = forged(f.store, original, payload, "forged-current-dominance")
    perp = native_case(tmp_path).instrument
    spot = Instrument(
        instrument_id="spot:btc",
        symbol="BTC/USDT",
        asset_class="crypto",
        quote_currency="USDT",
        venue="binance",
    )
    binding = crypto_context_bindings(
        spot, perpetual=perp, benchmark=spot, open_interest_units="base:BTC"
    )[2]
    index = pd.MultiIndex.from_tuples([(T, spot.instrument_id)], names=["datetime", "instrument"])
    with pytest.raises(ProviderError, match="normalization_differs"):
        materialize_evidence_fields(
            f.store,
            index,
            ctx(f.now),
            (binding,),
            allowed_evidence_ids=frozenset({bad.record.evidence_id}),
            availability_policy="captured-vintage-at-fit-1",
        )


@pytest.mark.parametrize("body", [chart([[ms(T - timedelta(days=1)), 0]]), chart([[True, 10]])])
def test_invalid_denominator_and_non_native_timestamp_fail_with_raw_retained(coingecko, body):
    f = coingecko(
        [body, {"market_caps": [[ms(T - timedelta(days=1)), 6]]}], tier="pro", key="fixture"
    )
    with pytest.raises(ProviderError):
        f.adapter.dominance_history(ctx())
    assert f.store.query(as_of=f.now, kind="crypto_raw").records
    assert not f.store.query(as_of=f.now, kind="btc_dominance").records


def test_no_alternate_origin_or_unbounded_request_budget(coingecko):
    f = coingecko([global_body()])
    f.adapter.http.base_url = "https://unapproved.example/api/v3"
    with pytest.raises(ProviderError, match="origin"):
        f.adapter.current_dominance(ctx())
    assert not f.calls
    at = ms(T - timedelta(days=1))
    f = coingecko([chart([[at, 10]]), {"market_caps": [[at, 6]]}], tier="pro", key="fixture")
    f.adapter.http.max_requests = 1
    with pytest.raises(ProviderError, match="budget"):
        f.adapter.dominance_history(ctx())
    assert len(f.calls) == len(f.store.query(as_of=f.now, kind="crypto_raw").records) == 1


def test_bindings_reject_spot_perpetual_and_implicit_usd_stablecoin_parity(tmp_path):
    f = native_case(tmp_path)
    spot = Instrument(
        instrument_id="spot:btc",
        symbol="BTC/USDT",
        asset_class="crypto",
        quote_currency="USDT",
        venue="binance",
    )
    usd = spot.model_copy(
        update={"instrument_id": "spot:usd", "symbol": "BTC/USD", "quote_currency": "USD"}
    )
    with pytest.raises(ValueError, match="FX"):
        crypto_context_bindings(
            spot, perpetual=f.instrument, benchmark=usd, open_interest_units="base:BTC"
        )
    with pytest.raises(ValueError, match="perpetual"):
        crypto_context_bindings(
            spot, perpetual=spot, benchmark=spot, open_interest_units="base:BTC"
        )
    with pytest.raises(ValueError, match="unit"):
        crypto_context_bindings(
            spot, perpetual=f.instrument, benchmark=spot, open_interest_units="quote:USDT"
        )
