"""Dated dominance normalization for the existing CoinGecko adapter."""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

import pandas as pd

from src.data.adapters.base import FetchResult, ProviderError
from src.data.adapters.crypto_context import (
    VERSION,
    CryptoCapture,
    decimal,
    fail,
    instant_ms,
    native_json,
)
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest

DAYS = ("1", "7", "14", "30", "90", "180", "365", "max")
ORIGINS = {
    "demo": "https://api.coingecko.com/api/v3",
    "pro": "https://pro-api.coingecko.com/api/v3",
}


def _capture(adapter, context):
    if adapter.http.base_url != ORIGINS[adapter.tier]:
        fail("fixed_coingecko_origin_required")
    return CryptoCapture(adapter.store, context, "coingecko", clock=adapter.clock)


def _get(adapter):
    return lambda base, path, params: adapter.http.get(
        path, params=params, headers=adapter._cg_headers
    )


def _current(body):
    data = body["data"]
    stamp = data["updated_at"]
    if type(stamp) is not int or stamp < 0:
        fail("native_global_updated_at_required")
    observed = pd.Timestamp(stamp, unit="s", tz="UTC").to_pydatetime()
    percent = decimal(data["market_cap_percentage"]["btc"])
    if percent > 100:
        fail("btc_dominance_percentage_outside_range")
    sign, digits, exponent = percent.as_tuple()
    value = Decimal((sign, digits, exponent - 2))
    return dict(
        value=str(value),
        observed_at=observed.isoformat(),
        units="fraction-of-global-market-cap",
        market_type="global",
        numerator_scope="bitcoin",
        denominator_scope="coingecko-tracked-global-market",
        convention="reported-btc-market-cap-percentage-divided-by-100-1",
        source_pointer=["data", "market_cap_percentage", "btc"],
    ), observed


def current_dominance(adapter, context):
    capture = _capture(adapter, context)
    source, body = capture.wire(_get(adapter), adapter.http.base_url, "/global", {})
    try:
        value, observed = _current(body)
        if observed > source.record.ingested_at:
            fail("global_update_after_actual_receipt")
        record = capture.put(
            adapter.store.raw(source.record.evidence_id),
            value,
            kind="btc_dominance",
            key="bitcoin:dominance:" + observed.isoformat(),
            uri=adapter.http.base_url + "/global",
            observed=observed,
            units=value["units"],
            sources=(source,),
        )
        return FetchResult(
            "available",
            (source, record),
            "current observed global percentage; no historical availability claim",
        )
    except ProviderError:
        raise
    except (KeyError, TypeError, ValueError, ArithmeticError):
        fail("malformed_current_global_dominance_archived")


def _chart(rows):
    if not isinstance(rows, list):
        fail("native_market_cap_array_required")
    result = {}
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 2:
            fail("native_market_cap_pair_required")
        observed, value = instant_ms(row[0]), decimal(row[1])
        result.setdefault(observed, []).append((value, index))
    return result


def _ratio(btc, total, observed, bitcoin_pointer, global_pointer):
    if total <= 0 or not 0 <= btc <= total:
        fail("global_market_cap_denominator_or_scope_invalid")
    with localcontext() as arithmetic:
        arithmetic.prec, arithmetic.rounding = 50, ROUND_HALF_EVEN
        ratio = btc / total
    return dict(
        value=str(ratio),
        observed_at=observed.isoformat(),
        units="fraction-of-global-market-cap",
        market_type="global",
        numerator_scope="bitcoin",
        denominator_scope="coingecko-tracked-global-market",
        quote_currency="USD",
        btc_market_cap=str(btc),
        total_market_cap=str(total),
        convention="equal-native-timestamp-cap-ratio-decimal50-half-even-1",
        bitcoin_pointer=bitcoin_pointer,
        global_pointer=global_pointer,
    )


def dominance_history(adapter, context, *, days):
    if adapter.tier != "pro" or not adapter._cg_headers.get("x-cg-pro-api-key"):
        fail("historical_global_market_cap_requires_pro_entitlement")
    if not isinstance(days, str) or days not in DAYS:
        fail("documented_global_history_days_required")
    capture = _capture(adapter, context)
    source, body = capture.wire(
        _get(adapter),
        adapter.http.base_url,
        "/global/market_cap_chart",
        {"days": days, "vs_currency": "usd"},
    )
    try:
        total = _chart(body["market_cap_chart"]["market_cap"])
        selected = sorted(at for at in total if at <= context.as_of)
        evidence = [source]
        btc = {}
        bitcoin = None
        if selected:
            start = int(pd.Timestamp(selected[0]).value // 1_000_000_000) - 1
            end = int((pd.Timestamp(selected[-1]).value + 999999999) // 1_000_000_000) + 1
            bitcoin, raw = capture.wire(
                _get(adapter),
                adapter.http.base_url,
                "/coins/bitcoin/market_chart/range",
                {
                    "vs_currency": "usd",
                    "from": start,
                    "to": end,
                    "interval": "daily",
                    "precision": "full",
                },
            )
            evidence.append(bitcoin)
            btc = _chart(raw["market_caps"])
        paired, conflicting = [], []
        for observed in selected:
            if observed not in btc:
                continue
            values = []
            for cap, i in btc[observed]:
                for global_cap, j in total[observed]:
                    value = _ratio(
                        cap,
                        global_cap,
                        observed,
                        ["market_caps", i],
                        ["market_cap_chart", "market_cap", j],
                    )
                    item = capture.put(
                        canonical_bytes(
                            {"bitcoin": bitcoin.record.raw_hash, "global": source.record.raw_hash}
                        ),
                        value,
                        kind="btc_dominance",
                        key="bitcoin:dominance:" + observed.isoformat(),
                        uri=adapter.http.base_url + "/global/market_cap_chart",
                        observed=observed,
                        units=value["units"],
                        sources=(source, bitcoin),
                    )
                    evidence.append(item)
                    values.append(value["value"])
            paired.append(observed)
            if len(set(values)) > 1:
                conflicting.append(observed.isoformat())
        missing = [at.isoformat() for at in selected if at not in btc]
        sources = (source, bitcoin) if bitcoin else (source,)
        coverage = capture.put(
            canonical_bytes({"days": days, "currency": "usd"}),
            dict(
                days=days,
                quote_currency="USD",
                scope="exact-native-global-timestamp-grid",
                completeness="not-certified",
                paired_observations=len(paired),
                unmatched_global=missing,
                unmatched_bitcoin=[at.isoformat() for at in sorted(btc) if at not in total],
                conflicting_observations=conflicting,
                gaps=[
                    "capture-time revised history; requires actual archived availability for historical PIT use"
                ],
            ),
            kind="crypto_coverage",
            key="bitcoin:dominance:" + days,
            uri=adapter.http.base_url + "/global/market_cap_chart",
            sources=sources,
        )
        evidence.append(coverage)
        return FetchResult(
            "missing" if not paired else "partial" if missing or conflicting else "available",
            tuple(evidence),
            "exact native timestamp join only; Pro access and full-history entitlement unverified",
        )
    except ProviderError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, ArithmeticError):
        fail("malformed_historical_dominance_archived")


def verify_dominance_record(store: EvidenceStore, evidence_id: str, *, as_of: datetime) -> dict:
    """Reparse official native source bodies; never trust a supplied ratio."""
    try:
        envelope = store.get(evidence_id, as_of=as_of)
        payload = store.normalized(evidence_id, as_of=as_of)
        if (
            envelope.kind != "btc_dominance"
            or envelope.record.provider != "coingecko"
            or envelope.record.provider_version != VERSION
            or envelope.record.instrument_ids
        ):
            fail("unsupported_dominance_record")
        sources = tuple(
            store.get(item["evidence_id"], as_of=as_of) for item in payload["source_manifest"]
        )
        for ref, item in zip(payload["source_manifest"], sources):
            item.record.require_available(envelope.record.ingested_at)
            if (
                ref
                != {"evidence_id": item.record.evidence_id, "envelope_hash": item.content_hash()}
                or item.kind != "crypto_raw"
                or item.record.provider != "coingecko"
                or item.record.provider_version != VERSION
                or item.record.instrument_ids
                or item.record.available_at < item.record.ingested_at
            ):
                fail("dominance_native_source_identity_differs")
            meta = store.normalized(item.record.evidence_id, as_of=as_of)
            request = meta["request"]
            if (
                request["origin"] not in ORIGINS.values()
                or item.record.source_uri != request["origin"] + request["path"]
                or meta["wire_hash"] != digest(store.raw(item.record.evidence_id))
            ):
                fail("dominance_native_origin_or_raw_differs")
        if len(sources) == 1:
            source = sources[0]
            request = store.normalized(source.record.evidence_id, as_of=as_of)["request"]
            if (
                request["path"] != "/global"
                or request["params"] != {}
                or store.raw(evidence_id) != store.raw(source.record.evidence_id)
            ):
                fail("current_dominance_source_scope_differs")
            expected, observed = _current(native_json(store.raw(source.record.evidence_id)))
            if observed > source.record.ingested_at:
                fail("global_update_after_actual_receipt")
        elif len(sources) == 2:
            source, bitcoin = sources
            requests = [
                store.normalized(item.record.evidence_id, as_of=as_of)["request"]
                for item in sources
            ]
            if (
                requests[0]["origin"] != ORIGINS["pro"]
                or requests[1]["origin"] != ORIGINS["pro"]
                or requests[0]["path"] != "/global/market_cap_chart"
                or requests[1]["path"] != "/coins/bitcoin/market_chart/range"
                or requests[0]["params"]
                != {"days": requests[0]["params"].get("days"), "vs_currency": "usd"}
                or requests[0]["params"]["days"] not in DAYS
                or requests[1]["params"].get("vs_currency") != "usd"
                or requests[1]["params"].get("interval") != "daily"
                or requests[1]["params"].get("precision") != "full"
                or set(requests[1]["params"])
                != {"vs_currency", "from", "to", "interval", "precision"}
                or type(requests[1]["params"]["from"]) is not int
                or type(requests[1]["params"]["to"]) is not int
                or requests[1]["params"]["from"] >= requests[1]["params"]["to"]
            ):
                fail("historical_dominance_source_scope_differs")
            global_pointer, bitcoin_pointer = payload["global_pointer"], payload["bitcoin_pointer"]
            if (
                not isinstance(global_pointer, list)
                or not isinstance(bitcoin_pointer, list)
                or global_pointer[:-1] != ["market_cap_chart", "market_cap"]
                or bitcoin_pointer[:-1] != ["market_caps"]
                or type(global_pointer[-1]) is not int
                or type(bitcoin_pointer[-1]) is not int
                or min(global_pointer[-1], bitcoin_pointer[-1]) < 0
            ):
                fail("native_cap_pointer_differs")
            global_row = native_json(store.raw(source.record.evidence_id))["market_cap_chart"][
                "market_cap"
            ][global_pointer[-1]]
            btc_row = native_json(store.raw(bitcoin.record.evidence_id))["market_caps"][
                bitcoin_pointer[-1]
            ]
            observed = instant_ms(global_row[0])
            if (
                observed != instant_ms(btc_row[0])
                or observed > envelope.record.requested_as_of
                or not requests[1]["params"]["from"] * 1_000_000_000
                <= pd.Timestamp(observed).value
                <= requests[1]["params"]["to"] * 1_000_000_000
            ):
                fail("native_cap_timestamp_mismatch")
            expected = _ratio(
                decimal(btc_row[1]),
                decimal(global_row[1]),
                observed,
                bitcoin_pointer,
                global_pointer,
            )
            if store.raw(evidence_id) != canonical_bytes(
                {"bitcoin": bitcoin.record.raw_hash, "global": source.record.raw_hash}
            ):
                fail("native_cap_source_manifest_differs")
        else:
            fail("dominance_exact_source_roles_required")
        expected["source_manifest"] = payload["source_manifest"]
        if (
            expected != payload
            or envelope.record.observed_at != observed
            or envelope.record.units != expected["units"]
            or envelope.record.available_at < envelope.record.ingested_at
            or envelope.record.source_uri != sources[0].record.source_uri
        ):
            fail("dominance_normalization_differs")
        return expected
    except ProviderError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, ArithmeticError):
        fail("invalid_native_dominance_proof")
