"""Shared native crypto context capture; existing adapters own public methods."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import urlencode

import pandas as pd

from src.data.adapters.base import FetchResult, ProviderError, store_observation
from src.data.evidence.store import canonical_bytes, digest
from src.domain.common import Instrument, _utc

VERSION = "native-crypto-context-2"
OI_PERIODS = ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")


def fail(code):
    raise ProviderError("crypto-context", code)


def decimal(value, *, signed=False):
    if isinstance(value, bool):
        fail("invalid_native_decimal")
    result = Decimal(str(value))
    if not result.is_finite() or (not signed and result < 0):
        fail("invalid_native_decimal")
    return result


def instant_ms(value):
    if type(value) is not int or value < 0:
        fail("native_integer_milliseconds_required")
    return pd.Timestamp(value, unit="ms", tz="UTC").to_pydatetime()


def native_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail("duplicate_native_json_field")
            result[key] = value
        return result

    try:
        return json.loads(raw, parse_float=str, object_pairs_hook=unique)
    except (ValueError, UnicodeError):
        fail("malformed_native_json_archived")


class CryptoCapture:
    def __init__(self, store, context, provider, *, version=VERSION, clock=None):
        self.store, self.context, self.provider, self.version = store, context, provider, version
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.previous = None

    def now(self):
        at = _utc(self.clock())
        if at < self.context.as_of or (self.previous is not None and at < self.previous):
            fail("measured_capture_clock_precedes_request")
        self.previous = at
        return at

    def put(
        self, raw, value, *, kind, key, uri, instrument=None, observed=None, units=None, sources=()
    ):
        at = self.now()
        for source in sources:
            source.record.require_available(at)
        if sources:
            value = {
                **value,
                "source_manifest": [
                    {
                        "evidence_id": source.record.evidence_id,
                        "envelope_hash": source.content_hash(),
                    }
                    for source in sources
                ],
            }
        return store_observation(
            self.store,
            raw=raw,
            normalized=value,
            provider=self.provider,
            version=self.version,
            source_uri=uri,
            kind=kind,
            logical_key=key,
            context=self.context,
            ingested_at=at,
            instrument=instrument,
            observed_at=observed,
            units=units,
        )

    def wire(self, get, base, path, params, *, instrument=None):
        raw = get(base, path, params)
        if not isinstance(raw, bytes):
            fail("complete_native_response_bytes_required")
        source = self.put(
            raw,
            {"request": {"origin": base, "path": path, "params": params}, "wire_hash": digest(raw)},
            kind="crypto_raw",
            key=base + path + ":" + digest(canonical_bytes(params)),
            uri=base + path,
            instrument=instrument,
        )

        return source, native_json(raw)


def contract_market(market, instrument, exchange_id):
    if (
        instrument.asset_class != "crypto"
        or instrument.venue != exchange_id
        or market.get("symbol") != instrument.symbol
        or market.get("swap") is not True
        or market.get("spot") is not False
        or market.get("contract") is not True
        or market.get("future") is not False
    ):
        fail("verified_perpetual_market_required")
    base, rest = instrument.symbol.split("/")
    quote, settle = rest.split(":")
    if (
        market.get("base") != base
        or market.get("quote") != quote
        or market.get("settle") != settle
        or quote != instrument.quote_currency
        or type(market.get("linear")) is not bool
        or type(market.get("inverse")) is not bool
        or market["linear"] == market["inverse"]
        or not re.fullmatch(r"[A-Za-z0-9_.]+", market.get("id", ""))
    ):
        fail("perpetual_settlement_identity_differs")
    size = decimal(market["contractSize"])
    if size <= 0:
        fail("explicit_native_contract_size_required")
    info = market["info"]
    if exchange_id == "binance" and (
        info.get("symbol") != market["id"]
        or info.get("contractType") != "PERPETUAL"
        or info.get("baseAsset") != base
        or info.get("quoteAsset") != quote
        or info.get("marginAsset") != settle
        or (market["inverse"] and decimal(info["contractSize"]) != size)
    ):
        fail("native_contract_metadata_differs")
    if exchange_id == "krakenfutures" and info.get("symbol", "").lower() != market["id"].lower():
        fail("native_contract_metadata_differs")
    return dict(
        native_id=market["id"],
        base_currency=base,
        quote_currency=quote,
        settle_currency=settle,
        contract_size=str(size),
        linear=market["linear"],
        inverse=market["inverse"],
    )


def native_derivative_row(row, market, instrument, exchange_id, capability, timeframe):
    details = contract_market(market, instrument, exchange_id)
    if exchange_id == "binance":
        if capability == "funding" or details["linear"]:
            if row.get("symbol") != details["native_id"]:
                fail("native_derivative_symbol_differs")
        elif row.get("pair") != market["info"]["pair"] or row.get("contractType") != "PERPETUAL":
            fail("native_inverse_contract_identity_differs")
        observed = instant_ms(row["fundingTime" if capability == "funding" else "timestamp"])
        if capability == "funding":
            if row.get("rateType", "Regular") != "Regular":
                fail("nonregular_funding_requires_explicit_feature_policy")
            value, units = decimal(row["fundingRate"], signed=True), "fraction-per-funding-event"
        else:
            value = decimal(row["sumOpenInterest"])
            units = "base:" + details["base_currency"] if details["linear"] else "contracts"
    elif exchange_id == "krakenfutures" and capability == "funding":
        observed = _utc(pd.Timestamp(row["timestamp"]).ceil("us").to_pydatetime())
        value, units = (
            decimal(row["relativeFundingRate"], signed=True),
            "fraction-per-funding-event",
        )
    else:
        fail("unsupported_native_derivative_capability")
    return {
        "value": str(value),
        "observed_at": observed.isoformat(),
        "units": units,
        "market_type": "perpetual",
        "instrument": instrument.model_dump(mode="json"),
        **details,
        "sampling_period": timeframe if capability == "open_interest" else "native-funding-events",
        "funding_interval_hours": None,
    }, observed


def derivative_history(
    adapter, instrument, context, *, capability, since, end, timeframe, limit, max_pages
):
    if capability not in ("funding", "open_interest") or ":" not in instrument.symbol:
        fail("explicit_perpetual_symbol_and_supported_capability_required")
    if end is None:
        fail("explicit_derivative_end_required")
    start, end = _utc(since), _utc(end)
    exchange = adapter.exchange
    if (
        exchange.id not in ("binance", "krakenfutures")
        or instrument.venue != exchange.id
        or start >= end
        or end > context.as_of
        or type(max_pages) is not int
        or not 1 <= max_pages <= 1000
    ):
        fail("bounded_supported_derivative_window_required")
    if capability == "open_interest" and exchange.id == "krakenfutures":
        fail("kraken_futures_open_interest_history_not_supported")
    if capability == "open_interest" and timeframe not in OI_PERIODS:
        fail("explicit_native_open_interest_period_required")
    maximum = 1000 if capability == "funding" else 500
    limit = maximum if limit is None else limit
    if type(limit) is not int or not 1 <= limit <= maximum:
        fail("native_derivative_page_limit_invalid")
    capture = CryptoCapture(
        adapter.store,
        context,
        "ccxt:" + exchange.id,
        version=adapter.version + ":" + VERSION,
        clock=adapter.clock,
    )
    try:
        try:
            markets = exchange.load_markets()
        except Exception:
            fail("perpetual_contract_metadata_unavailable")
        market = markets[instrument.symbol]
        raw_market = canonical_bytes(market)
        metadata = capture.put(
            raw_market,
            {"sdk_market": market, "serialization": "complete-ccxt-market-sdk-output"},
            kind="crypto_contract",
            key=instrument.instrument_id,
            uri="ccxt://" + exchange.id + "/market/" + instrument.symbol,
            instrument=instrument,
        )
        details = contract_market(market, instrument, exchange.id)
        if capability == "open_interest" and start < (
            pd.Timestamp(capture.now()) - pd.DateOffset(months=1)
        ):
            fail("binance_open_interest_one_month_scope_requires_archive")
        if exchange.id == "binance":
            base = "https://fapi.binance.com" if details["linear"] else "https://dapi.binance.com"
            path = (
                ("/fapi/v1/" if details["linear"] else "/dapi/v1/") + "fundingRate"
                if capability == "funding"
                else "/futures/data/openInterestHist"
            )
        else:
            base, path = (
                "https://futures.kraken.com",
                "/derivatives/api/v3/historical-funding-rates",
            )

        request_failed = [False]

        def get(origin, endpoint, params):
            exchange.last_http_response = None
            request_failed[0] = False
            try:
                exchange.fetch(origin + endpoint + "?" + urlencode(params), "GET")
            except Exception:
                request_failed[0] = True
            raw = exchange.last_http_response
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            if raw is None:
                fail("native_derivative_response_unavailable")
            return raw

        cursor = pd.Timestamp(start).value // 1_000_000
        until = (pd.Timestamp(end).value + 999999) // 1_000_000 - 1
        evidence, seen, observations = [metadata], set(), []
        for _ in range(max_pages):
            if exchange.id == "binance":
                params = dict(startTime=cursor, endTime=until, limit=limit)
                if capability == "funding" or details["linear"]:
                    params["symbol"] = details["native_id"]
                else:
                    params.update(pair=market["info"]["pair"], contractType="PERPETUAL")
                if capability == "open_interest":
                    params["period"] = timeframe
            else:
                params = {"symbol": details["native_id"].upper()}
            wire, body = capture.wire(get, base, path, params, instrument=instrument)
            evidence.append(wire)
            if request_failed[0]:
                fail("native_derivative_transport_error_archived")
            rows = (
                body.get("rates")
                if exchange.id == "krakenfutures" and isinstance(body, dict)
                else body
            )
            if not isinstance(rows, list) or (
                exchange.id == "krakenfutures" and body.get("result") != "success"
            ):
                fail("native_derivative_error_archived")
            if exchange.id == "binance" and len(rows) > limit:
                fail("native_derivative_page_exceeds_limit")
            previous, newest = None, None
            for position, row in enumerate(rows):
                value, observed = native_derivative_row(
                    row, market, instrument, exchange.id, capability, timeframe
                )
                stamp = pd.Timestamp(observed).value // 1_000_000
                if exchange.id == "binance" and not cursor <= stamp <= until:
                    fail("native_derivative_response_outside_requested_page")
                if previous is not None and stamp < previous:
                    fail("native_derivative_page_not_ascending")
                previous, newest = stamp, stamp
                if not start <= observed < end:
                    continue
                identity = (stamp, canonical_bytes(value))
                if identity in seen:
                    continue
                seen.add(identity)
                value.update(
                    source_pointer=["rates", position]
                    if exchange.id == "krakenfutures"
                    else [position],
                    request_window={"since": start.isoformat(), "end": end.isoformat()},
                )
                item = capture.put(
                    adapter.store.raw(wire.record.evidence_id),
                    value,
                    kind=capability,
                    key=instrument.instrument_id + ":" + capability + ":" + observed.isoformat(),
                    uri=base + path,
                    instrument=instrument,
                    observed=observed,
                    units=value["units"],
                    sources=(metadata, wire),
                )
                evidence.append(item)
                observations.append(observed)
            if (
                exchange.id == "krakenfutures"
                or not rows
                or len(rows) < limit
                or (newest is not None and newest >= until)
            ):
                break
            if newest is None or newest < cursor:
                fail("native_derivative_pagination_did_not_advance")
            cursor = newest + 1
        else:
            fail("native_derivative_pagination_budget_exhausted")
        coverage = capture.put(
            canonical_bytes({"since": start.isoformat(), "end": end.isoformat()}),
            {
                "capability": capability,
                "since": start.isoformat(),
                "end": end.isoformat(),
                "observations": len(observations),
                "first_observed_at": min(observations).isoformat() if observations else None,
                "last_observed_at": max(observations).isoformat() if observations else None,
                "completeness": "not-certified",
                "gaps": [
                    "funding cadence is not assumed"
                    if capability == "funding"
                    else "no-trade periods may be omitted; historical API limited to latest month"
                ],
            },
            kind="crypto_coverage",
            key=instrument.instrument_id + ":" + capability,
            uri=base + path,
            instrument=instrument,
            sources=tuple(
                source for source in evidence if source.kind in ("crypto_raw", "crypto_contract")
            ),
        )
        evidence.append(coverage)
        return FetchResult(
            "available" if observations else "missing",
            tuple(evidence),
            "capture-time availability; no certified historical completeness",
        )
    except ProviderError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, ArithmeticError):
        fail("malformed_native_derivative_or_contract")


def verify_derivative_record(store, evidence_id, *, as_of):
    """Reconstruct a scalar from full original native bytes and contract output."""
    try:
        envelope = store.get(evidence_id, as_of=as_of)
        payload = store.normalized(evidence_id, as_of=as_of)
        if envelope.kind not in (
            "funding",
            "open_interest",
        ) or not envelope.record.provider_version.endswith(":" + VERSION):
            fail("unsupported_native_derivative_record")
        instrument = Instrument.model_validate(payload["instrument"])
        exchange_id = envelope.record.provider.removeprefix("ccxt:")
        sources = tuple(
            store.get(source["evidence_id"], as_of=as_of) for source in payload["source_manifest"]
        )
        if len(sources) != 2 or tuple(source.kind for source in sources) != (
            "crypto_contract",
            "crypto_raw",
        ):
            fail("native_derivative_source_roles_differ")
        for expected, source in zip(payload["source_manifest"], sources):
            source.record.require_available(envelope.record.ingested_at)
            if expected != {
                "evidence_id": source.record.evidence_id,
                "envelope_hash": source.content_hash(),
            }:
                fail("native_derivative_source_hash_differs")
            if (
                source.record.provider != envelope.record.provider
                or source.record.provider_version != envelope.record.provider_version
                or source.record.instrument_ids != (instrument.instrument_id,)
            ):
                fail("native_derivative_source_identity_differs")
            if source.record.available_at < source.record.ingested_at:
                fail("native_source_availability_precedes_capture")
        metadata, wire = sources
        market = store.normalized(metadata.record.evidence_id, as_of=as_of)["sdk_market"]
        if store.raw(metadata.record.evidence_id) != canonical_bytes(market):
            fail("native_contract_sdk_output_differs")
        details = contract_market(market, instrument, exchange_id)
        request = store.normalized(wire.record.evidence_id, as_of=as_of)["request"]
        origin = "https://fapi.binance.com" if details["linear"] else "https://dapi.binance.com"
        path = (
            ("/fapi/v1/" if details["linear"] else "/dapi/v1/") + "fundingRate"
            if envelope.kind == "funding"
            else "/futures/data/openInterestHist"
        )
        if exchange_id == "krakenfutures":
            origin, path = (
                "https://futures.kraken.com",
                "/derivatives/api/v3/historical-funding-rates",
            )
        params = request["params"]
        if (
            request["origin"] != origin
            or request["path"] != path
            or wire.record.source_uri != origin + path
            or envelope.record.source_uri != origin + path
        ):
            fail("native_derivative_endpoint_differs")
        if exchange_id == "binance":
            identity = (
                {"symbol": details["native_id"]}
                if envelope.kind == "funding" or details["linear"]
                else {"pair": market["info"]["pair"], "contractType": "PERPETUAL"}
            )
            if any(params.get(key) != value for key, value in identity.items()):
                fail("native_derivative_request_identity_differs")
        elif params != {"symbol": details["native_id"].upper()}:
            fail("native_derivative_request_identity_differs")
        raw = store.raw(wire.record.evidence_id)
        if (
            raw != store.raw(evidence_id)
            or digest(raw) != store.normalized(wire.record.evidence_id, as_of=as_of)["wire_hash"]
        ):
            fail("native_derivative_raw_differs")
        body, pointer = native_json(raw), payload["source_pointer"]
        expected_prefix = ["rates"] if exchange_id == "krakenfutures" else []
        if pointer[:-1] != expected_prefix or type(pointer[-1]) is not int or pointer[-1] < 0:
            fail("native_derivative_pointer_differs")
        row = body["rates"][pointer[-1]] if expected_prefix else body[pointer[-1]]
        expected, observed = native_derivative_row(
            row, market, instrument, exchange_id, envelope.kind, payload["sampling_period"]
        )
        if exchange_id == "binance":
            expected_keys = (
                set(identity)
                | {"startTime", "endTime", "limit"}
                | ({"period"} if envelope.kind == "open_interest" else set())
            )
            stamp = pd.Timestamp(observed).value // 1_000_000
            if (
                set(params) != expected_keys
                or type(params["startTime"]) is not int
                or type(params["endTime"]) is not int
                or type(params["limit"]) is not int
                or not 1 <= params["limit"] <= (1000 if envelope.kind == "funding" else 500)
                or not params["startTime"] <= stamp <= params["endTime"]
            ):
                fail("native_derivative_page_scope_differs")
        window = payload["request_window"]
        start, end = (
            _utc(datetime.fromisoformat(window["since"])),
            _utc(datetime.fromisoformat(window["end"])),
        )
        if not start <= observed < end or end > envelope.record.requested_as_of:
            fail("native_derivative_observation_window_differs")
        if envelope.kind == "open_interest" and (
            payload["sampling_period"] not in OI_PERIODS
            or params.get("period") != payload["sampling_period"]
        ):
            fail("native_open_interest_period_differs")
        expected.update(
            source_pointer=pointer,
            request_window=window,
            source_manifest=payload["source_manifest"],
        )
        if (
            expected != payload
            or envelope.record.observed_at != observed
            or envelope.record.units != expected["units"]
            or envelope.record.instrument_ids != (instrument.instrument_id,)
            or envelope.record.available_at < envelope.record.ingested_at
        ):
            fail("native_derivative_normalization_differs")
        return expected
    except ProviderError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, ArithmeticError):
        fail("invalid_native_derivative_proof")
