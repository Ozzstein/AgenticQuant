"""Exact native Binance notional-price proof, reusable by read-only consumers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from src.data.adapters.operating import (
    BinanceOperatingAdapter,
    _crypto_identity,
    _decimal,
    _epoch,
    _fail,
    verify_operating_capture,
)
from src.data.evidence.store import EvidenceStore
from src.domain.common import Instrument, RunContext

if TYPE_CHECKING:
    from src.execution.venue import VenueNotionalReference


BINANCE_REFERENCE_ORIGINS = {
    "mainnet": ("https://api.binance.com", "https://data-api.binance.vision"),
    "testnet": ("https://testnet.binance.vision",),
}


def _source(store, key, instrument, context, operation, params, network):
    envelope = store.get(key, as_of=context.as_of)
    meta = verify_operating_capture(store, key, as_of=context.as_of)
    request = meta["request"]
    if (
        envelope.kind != "operating_raw"
        or meta["upstream"] != "binance-spot-v3"
        or envelope.record.instrument_ids != (instrument.instrument_id,)
        or request["path"] != "/api/v3/" + operation
        or request["params"] != params
        or envelope.record.source_uri
        not in tuple(base + "/api/v3/" + operation for base in BINANCE_REFERENCE_ORIGINS[network])
        or (
            "source_origin" in meta
            and meta["source_origin"] + "/api/v3/" + operation != envelope.record.source_uri
        )
    ):
        _fail("binance_notional_source_identity_differs")
    unit = request.get("timestamp_unit")
    header = request.get("time_unit_header")
    if (
        unit not in ("MILLISECOND", "MICROSECOND")
        or (unit == "MICROSECOND" and header not in ("MICROSECOND", "microsecond"))
        or (unit == "MILLISECOND" and header is not None)
    ):
        _fail("binance_notional_timestamp_unit_unproved")
    return envelope, store.raw(key), unit


def _fresh(source, observed, context, max_age):
    if (
        not observed <= source.record.available_at <= context.as_of
        or not 0 <= (context.as_of - observed).total_seconds() <= max_age
    ):
        _fail("binance_notional_reference_stale_or_future")


def _time(value, unit):
    if type(value) is not int or value < 0:
        _fail("binance_integer_wire_timestamp_required")
    return _epoch(value, unit)


@dataclass(frozen=True)
class NativeBinanceNotionalObservation:
    """Parsed wire values only; no source authenticity or absence authority."""

    kind: str
    price: Decimal | None
    observed_at: datetime | None
    average_minutes: int | None = None
    trade_id: str | None = None


def parse_binance_notional_payload(
    raw: bytes, *, operation: str, symbol: str, timestamp_unit: str
) -> NativeBinanceNotionalObservation:
    """Pure native-byte validation for transport reuse, never certification.

    The caller supplies the actual endpoint, requested symbol and timestamp
    unit. Only the store reader below authenticates those facts and the clocks.
    A -2043 response has no event timestamp; its actual receipt remains required.
    """

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("binance_duplicate_wire_field")
            result[key] = value
        return result

    try:
        if (
            not isinstance(raw, bytes)
            or not isinstance(symbol, str)
            or not re.fullmatch(r"[A-Z0-9]+", symbol)
            or timestamp_unit not in ("MILLISECOND", "MICROSECOND")
        ):
            _fail("binance_native_payload_arguments_invalid")
        unit = 1_000_000 if timestamp_unit == "MILLISECOND" else 1_000
        body = json.loads(raw, parse_float=str, object_pairs_hook=unique_object)
        if operation == "referencePrice":
            if isinstance(body, dict) and "code" in body:
                if (
                    type(body["code"]) is int
                    and body["code"] == -2043
                    and "referencePrice" not in body
                    and body.get("symbol", symbol) == symbol
                ):
                    return NativeBinanceNotionalObservation("reference-absent", None, None)
                _fail("binance_reference_absence_not_proved")
            if (
                not isinstance(body, dict)
                or body.get("symbol") != symbol
                or "referencePrice" not in body
            ):
                _fail("binance_reference_absence_not_proved")
            observed = _time(body["timestamp"], unit)
            if body["referencePrice"] is None:
                return NativeBinanceNotionalObservation("reference-absent", None, observed)
            return NativeBinanceNotionalObservation(
                "venue-reference", _decimal(body["referencePrice"]), observed
            )
        if operation == "avgPrice":
            if not isinstance(body, dict) or type(body.get("mins")) is not int or body["mins"] <= 0:
                _fail("binance_native_average_window_invalid")
            return NativeBinanceNotionalObservation(
                "weighted-average",
                _decimal(body["price"]),
                _time(body["closeTime"], unit),
                body["mins"],
            )
        if operation == "trades":
            if (
                not isinstance(body, list)
                or len(body) != 1
                or type(body[0]["id"]) is not int
                or body[0]["id"] < 0
            ):
                _fail("binance_last_individual_trade_required")
            return NativeBinanceNotionalObservation(
                "last-trade",
                _decimal(body[0]["price"]),
                _time(body[0]["time"], unit),
                average_minutes=0,
                trade_id=str(body[0]["id"]),
            )
        _fail("unsupported_binance_notional_operation")
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        ArithmeticError,
        AttributeError,
        UnicodeError,
    ):
        _fail("binance_native_payload_malformed")


def parse_binance_notional_reference(
    store: EvidenceStore,
    instrument: Instrument,
    context: RunContext,
    *,
    reference_evidence_id: str,
    average_minutes: int,
    average_evidence_id: str | None = None,
    last_trade_evidence_id: str | None = None,
    max_age_seconds: int = 30,
    network: Literal["mainnet", "testnet"] = "mainnet",
) -> VenueNotionalReference:
    """Authenticate stored raw bodies; no fetch, caller price or generic fallback.

    Native non-null reference wins. Only explicit null or native -2043 proves
    absence; then actual avgPrice exact mins/closeTime or a last individual trade
    for a zero-minute filter is required. Record hashes identify full envelopes.
    """
    from src.execution.venue import VenueNotionalReference

    try:
        _crypto_identity(instrument, "binance")
        if (
            type(average_minutes) is not int
            or average_minutes < 0
            or type(max_age_seconds) is not int
            or not 0 <= max_age_seconds <= 30
            or network not in BINANCE_REFERENCE_ORIGINS
        ):
            _fail("binance_notional_window_or_age_invalid")
        symbol = BinanceOperatingAdapter._symbol(instrument)
        source, raw, unit = _source(
            store,
            reference_evidence_id,
            instrument,
            context,
            "referencePrice",
            {"symbol": symbol},
            network,
        )
        native = parse_binance_notional_payload(
            raw, operation="referencePrice", symbol=symbol, timestamp_unit=unit
        )
        observed = native.observed_at or source.record.ingested_at
        _fresh(source, observed, context, max_age_seconds)
        if native.kind == "venue-reference":
            return VenueNotionalReference(
                price=native.price,
                kind="venue-reference",
                average_minutes=None,
                observed_at=observed,
                available_at=source.record.available_at,
                source_hash=source.content_hash(),
                reference_price_absence_hash=None,
            )
        if average_minutes:
            if average_evidence_id is None or last_trade_evidence_id is not None:
                _fail("binance_exact_average_source_required")
            fallback, raw, unit = _source(
                store,
                average_evidence_id,
                instrument,
                context,
                "avgPrice",
                {"symbol": symbol},
                network,
            )
            native = parse_binance_notional_payload(
                raw, operation="avgPrice", symbol=symbol, timestamp_unit=unit
            )
            if native.average_minutes != average_minutes:
                _fail("binance_native_average_window_differs")
        else:
            if last_trade_evidence_id is None or average_evidence_id is not None:
                _fail("binance_actual_last_trade_source_required")
            fallback, raw, unit = _source(
                store,
                last_trade_evidence_id,
                instrument,
                context,
                "trades",
                {"symbol": symbol, "limit": 1},
                network,
            )
            native = parse_binance_notional_payload(
                raw, operation="trades", symbol=symbol, timestamp_unit=unit
            )
        _fresh(fallback, native.observed_at, context, max_age_seconds)
        return VenueNotionalReference(
            price=native.price,
            kind=native.kind,
            average_minutes=average_minutes,
            observed_at=native.observed_at,
            available_at=max(source.record.available_at, fallback.record.available_at),
            source_hash=fallback.content_hash(),
            reference_price_absence_hash=source.content_hash(),
        )
    except (KeyError, TypeError, ValueError, IndexError, ArithmeticError, AttributeError):
        _fail("binance_notional_source_malformed_or_unavailable")
