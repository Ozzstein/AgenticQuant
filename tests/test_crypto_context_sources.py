"""Synthetic native responses through the actual immutable crypto producers."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from src.data.adapters.base import ProviderError
from src.data.adapters.crypto_context import verify_derivative_record
from src.data.adapters.market import CCXTAdapter
from src.data.evidence.store import EvidenceStore
from src.data.features.contextual import NumericEvidenceBinding, materialize_evidence_fields
from src.domain.common import Instrument, RunContext

T = datetime(2026, 10, 27, tzinfo=timezone.utc)


def ctx(at=T):
    return RunContext(
        run_id="crypto-native", mode="paper", as_of=at, config_hash="0" * 64, code_version="test"
    )


def native_case(tmp_path, *, inverse=False, venue="binance", rows=None):
    quote, settle = ("USD", "BTC") if inverse else ("USDT", "USDT")
    if venue == "krakenfutures":
        quote, settle = "USD", "USD"
    instrument = Instrument(
        instrument_id="perp:" + venue + ":" + settle,
        symbol=f"BTC/{quote}:{settle}",
        asset_class="crypto",
        quote_currency=quote,
        venue=venue,
    )
    native_id = "PF_XBTUSD" if venue == "krakenfutures" else "BTCUSD_PERP" if inverse else "BTCUSDT"
    market = dict(
        symbol=instrument.symbol,
        id=native_id,
        base="BTC",
        quote=quote,
        settle=settle,
        swap=True,
        spot=False,
        contract=True,
        future=False,
        linear=not inverse,
        inverse=inverse,
        contractSize=100 if inverse else 1,
        info=dict(
            symbol=native_id,
            contractType="PERPETUAL",
            baseAsset="BTC",
            quoteAsset=quote,
            marginAsset=settle,
            pair="BTCUSD",
            contractSize=100 if inverse else 1,
        ),
    )
    case = SimpleNamespace(
        instrument=instrument,
        market=market,
        calls=[],
        now=T,
        rows=rows or [],
        failure=None,
        repeat=False,
    )
    exchange = SimpleNamespace(
        id=venue, load_markets=lambda: {instrument.symbol: market}, last_http_response=None
    )

    def fetch(url, method):
        assert method == "GET"
        case.now += timedelta(seconds=1)
        request = urlsplit(url)
        args = {k: v[0] for k, v in parse_qs(request.query).items()}
        case.calls.append((request, args))
        if case.failure is not None:
            body = case.failure
        elif venue == "krakenfutures":
            body = {"result": "success", "rates": case.rows}
        else:
            body = (
                case.rows
                if case.repeat
                else [
                    r
                    for r in case.rows
                    if int(args["startTime"])
                    <= r.get("fundingTime", r.get("timestamp"))
                    <= int(args["endTime"])
                ][: int(args["limit"])]
            )
        exchange.last_http_response = json.dumps(body, separators=(",", ":"))
        return body

    exchange.fetch = fetch
    case.store = EvidenceStore(tmp_path / "crypto.sqlite")
    case.adapter = CCXTAdapter(case.store, exchange, clock=lambda: case.now)
    return case


def funding(at, value="-0.000100000000000000000001", symbol="BTCUSDT"):
    return dict(symbol=symbol, fundingTime=pd.Timestamp(at).value // 1_000_000, fundingRate=value)


def observations(result, kind):
    return tuple(e for e in result.evidence if e.kind == kind)


def test_funding_pages_preserve_wire_decimals_real_clocks_and_pit_features(tmp_path):
    since = T - timedelta(days=2)
    rows = [funding(since + timedelta(hours=8 * i)) for i in range(6)]
    f = native_case(tmp_path, rows=rows)
    result = f.adapter.derivatives(
        f.instrument,
        ctx(),
        capability="funding",
        since=since,
        end=T,
        limit=2,
        max_pages=4,
        ingested_at=since,
    )
    values = observations(result, "funding")
    assert len(values) == 6 and len(f.calls) == 4
    assert f.calls[1][1]["startTime"] == str(rows[1]["fundingTime"] + 1)
    for item in values:
        data = verify_derivative_record(f.store, item.record.evidence_id, as_of=f.now)
        assert data["value"] == "-0.000100000000000000000001"
        assert item.record.units == "fraction-per-funding-event"
        assert item.record.available_at == item.record.ingested_at > T
        assert item.record.observed_at < T and data["funding_interval_hours"] is None
    index = pd.MultiIndex.from_tuples(
        [(since + timedelta(hours=8), "spot:btc")], names=["datetime", "instrument"]
    )
    binding = NumericEvidenceBinding(
        name="funding_rate",
        target_instrument_id="spot:btc",
        source_instrument_id=f.instrument.instrument_id,
        kind="funding",
        value_path=("value",),
        observation_date_path=("observed_at",),
        source_market="perpetual",
        max_age_seconds=86400,
    )
    strict = materialize_evidence_fields(f.store, index, ctx(f.now), (binding,))
    assert strict.values.isna().all().all()
    current = materialize_evidence_fields(
        f.store,
        index,
        ctx(f.now),
        (binding,),
        availability_policy="captured-vintage-at-fit-1",
        allowed_evidence_ids=frozenset(v.record.evidence_id for v in values),
    )
    assert current.values.iloc[0, 0] == float(rows[1]["fundingRate"])
    assert current.observed_at.iloc[0, 0] == since + timedelta(hours=8)
    assert current.available_at.iloc[0] > T
    coverage = observations(result, "crypto_coverage")[0]
    assert (
        f.store.normalized(coverage.record.evidence_id, as_of=f.now)["completeness"]
        == "not-certified"
    )


@pytest.mark.parametrize("inverse", [False, True])
def test_native_oi_contract_amount_is_not_quote_notional_or_spot(tmp_path, inverse):
    row = dict(
        timestamp=pd.Timestamp(T - timedelta(hours=1)).value // 1_000_000,
        sumOpenInterest="123.000000000000000001",
        sumOpenInterestValue="999999999.99",
    )
    row.update(pair="BTCUSD", contractType="PERPETUAL") if inverse else row.update(symbol="BTCUSDT")
    f = native_case(tmp_path, inverse=inverse, rows=[row])
    result = f.adapter.derivatives(
        f.instrument,
        ctx(),
        capability="open_interest",
        since=T - timedelta(days=1),
        end=T,
        timeframe="1h",
    )
    item = observations(result, "open_interest")[0]
    value = verify_derivative_record(f.store, item.record.evidence_id, as_of=f.now)
    assert value["value"] == "123.000000000000000001"
    assert value["units"] == ("contracts" if inverse else "base:BTC")
    assert value["contract_size"] == ("100" if inverse else "1")
    assert value["settle_currency"] == ("BTC" if inverse else "USDT")
    assert f.calls[0][0].hostname == ("dapi.binance.com" if inverse else "fapi.binance.com")
    assert f.calls[0][1]["period"] == "1h"
    assert item.record.instrument_ids == (f.instrument.instrument_id,)


def test_kraken_futures_funding_distinct_and_no_oi_or_spot_history_invention(tmp_path):
    f = native_case(
        tmp_path,
        venue="krakenfutures",
        rows=[
            dict(
                timestamp=(T - timedelta(hours=4)).isoformat(),
                fundingRate="12",
                relativeFundingRate="-0.00001234567890123456789",
            )
        ],
    )
    result = f.adapter.derivatives(
        f.instrument, ctx(), capability="funding", since=T - timedelta(days=1), end=T
    )
    data = verify_derivative_record(
        f.store, observations(result, "funding")[0].record.evidence_id, as_of=f.now
    )
    assert data["value"] == "-0.00001234567890123456789" and len(f.calls) == 1
    assert f.calls[0][1] == {"symbol": "PF_XBTUSD"}
    with pytest.raises(ProviderError, match="not_supported"):
        f.adapter.derivatives(
            f.instrument, ctx(), capability="open_interest", since=T - timedelta(days=1), end=T
        )
    with pytest.raises(ProviderError, match="bounded_supported"):
        f.adapter.derivatives(
            f.instrument.model_copy(update={"venue": "kraken"}),
            ctx(),
            capability="funding",
            since=T - timedelta(days=1),
            end=T,
        )


@pytest.mark.parametrize(
    "case",
    [
        "wrong_settle",
        "spot",
        "missing_size",
        "wrong_native_contract",
        "old_oi",
        "missing_end",
        "budget",
        "repeat",
        "native_error",
        "special",
    ],
)
def test_native_history_fails_explicitly_and_preserves_actual_received_material(tmp_path, case):
    f = native_case(
        tmp_path, rows=[funding(T - timedelta(hours=16)), funding(T - timedelta(hours=8))]
    )
    args = dict(capability="funding", since=T - timedelta(days=1), end=T, limit=1, max_pages=3)
    if case == "wrong_settle":
        f.market["settle"] = "USD"
    if case == "spot":
        f.market["spot"] = True
    if case == "missing_size":
        f.market.pop("contractSize")
    if case == "wrong_native_contract":
        f.market["info"]["contractType"] = "CURRENT_QUARTER"
    if case == "old_oi":
        args.update(capability="open_interest", since=T - timedelta(days=90))
    if case == "missing_end":
        args.pop("end")
    if case == "budget":
        args["max_pages"] = 1
    if case == "repeat":
        f.repeat = True
    if case == "native_error":
        f.failure = {"code": -1121, "msg": "fixture invalid symbol"}
    if case == "special":
        f.rows[0]["rateType"] = "Special"
    with pytest.raises(ProviderError):
        f.adapter.derivatives(f.instrument, ctx(), **args)
    assert len(f.calls) <= args["max_pages"]
    if case in ("native_error", "special", "budget", "repeat"):
        assert f.store.query(as_of=f.now, kind="crypto_raw").records
    if case in ("wrong_settle", "spot", "missing_size", "wrong_native_contract"):
        assert f.store.query(as_of=f.now, kind="crypto_contract").records


def test_independent_native_reparse_rejects_forged_scalar_and_keeps_conflicts(tmp_path):
    f = native_case(
        tmp_path,
        rows=[funding(T - timedelta(hours=8), "0.001"), funding(T - timedelta(hours=8), "0.002")],
    )
    result = f.adapter.derivatives(
        f.instrument, ctx(), capability="funding", since=T - timedelta(days=1), end=T
    )
    values = observations(result, "funding")
    assert len(values) == 2
    assert f.store.query(as_of=f.now, kind="funding").status == "conflicting"
    original = values[0]
    payload = f.store.normalized(original.record.evidence_id, as_of=f.now)
    payload["value"] = "99"
    from src.data.adapters.base import store_observation

    forged = store_observation(
        f.store,
        raw=f.store.raw(original.record.evidence_id),
        normalized=payload,
        provider=original.record.provider,
        version=original.record.provider_version,
        source_uri=original.record.source_uri,
        kind="funding",
        logical_key="forged",
        context=ctx(),
        ingested_at=f.now,
        instrument=f.instrument,
        observed_at=original.record.observed_at,
        units=original.record.units,
    )
    with pytest.raises(ProviderError, match="normalization_differs"):
        verify_derivative_record(f.store, forged.record.evidence_id, as_of=f.now)
