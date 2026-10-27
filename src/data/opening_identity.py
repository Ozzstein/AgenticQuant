"""Reconstruct upstream print identity from immutable operating wire receipts.

Event serialization, selected Alpaca feed and Python's microsecond clock are not
capacity identities. The native trade ID and exact UTC nanosecond instant are.
Legacy receipts without the original individual trade cannot establish an alias.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd

from src.data.adapters.operating import _fail, verify_operating_capture
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import Instrument

if TYPE_CHECKING:
    from src.domain.portfolio import OpeningSourceIdentity


def _native_id(value) -> str:
    text = str(value)
    if isinstance(value, bool) or not re.fullmatch(r"0|[1-9][0-9]*", text):
        _fail("native_integer_trade_id_required")
    return text


def _iso_ns(value) -> int:
    # pandas silently truncates fractions beyond nanoseconds. Reject them.
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})",
        value,
    ):
        _fail("exact_timezone_nanosecond_timestamp_required")
    instant = pd.Timestamp(value)
    return instant.tz_convert("UTC").value


def _epoch_ns(value, multiplier) -> int:
    number = Decimal(str(value)) * multiplier
    if not number.is_finite() or number != number.to_integral_value():
        _fail("exact_nanosecond_epoch_required")
    return int(number)


def _raw_prints(store, value, instrument, convention, as_of):
    """Yield native identities/economics only from the closed individual APIs."""
    providers = {
        "alpaca-iex-regular-sale-individual-print-1": "alpaca",
        "alpaca-sip-regular-sale-individual-print-1": "alpaca",
        "kraken-individual-spot-print-1": "kraken",
        "binance-individual-spot-print-1": "binance",
    }
    if convention not in providers:
        _fail("unsupported_opening_source_convention")
    provider = providers[convention]
    for source in value["source_manifest"]["sources"]:
        key = source["evidence_id"]
        envelope = store.get(key, as_of=as_of)
        if envelope.kind != "operating_raw":
            continue  # Explicit FX/calendar ancestors cannot supply a trade.
        wire = store.normalized(key, as_of=as_of)
        request = wire["request"]
        path, params = request["path"], request["params"]
        expected = {
            "alpaca": ("alpaca-market-data", "/v2/stocks/trades"),
            "kraken": ("kraken-spot-rest-v0", "/0/public/Trades"),
            "binance": ("binance-spot-v3", "/api/v3/trades"),
        }[provider]
        if (wire.get("upstream"), path) != expected:
            _fail("opening_upstream_or_endpoint_differs")
        if envelope.record.instrument_ids != (instrument.instrument_id,):
            _fail("opening_wire_instrument_differs")
        raw = json.loads(store.raw(key), parse_float=str)
        if provider == "alpaca":
            feed = convention.split("-")[1]
            if (
                instrument.asset_class not in ("equity", "etf")
                or params.get("symbols") != instrument.symbol
                or params.get("feed") != feed
                or set(raw.get("trades", {})) - {instrument.symbol}
            ):
                _fail("opening_wire_market_differs")
            for row in raw.get("trades", {}).get(instrument.symbol, []):
                exchange = row["x"]
                if not isinstance(exchange, str) or not exchange or ":" in exchange:
                    _fail("native_exchange_trade_id_required")
                if row.get("c") == ["@"]:
                    yield (
                        provider,
                        exchange + ":" + _native_id(row["i"]),
                        _iso_ns(row["t"]),
                        Decimal(str(row["p"])),
                        Decimal(str(row["s"])),
                    )
        elif provider == "kraken":
            pair = params.get("pair")
            result = raw.get("result", {})
            if (
                instrument.asset_class != "crypto"
                or instrument.venue != "kraken"
                or raw.get("error")
                or not pair
                or set(result) != {pair, "last"}
            ):
                _fail("opening_wire_market_differs")
            for row in result[pair]:
                if len(row) < 7:
                    _fail("native_individual_trade_id_required")
                yield (
                    provider,
                    _native_id(row[6]),
                    _epoch_ns(row[2], 1_000_000_000),
                    Decimal(str(row[0])),
                    Decimal(str(row[1])),
                )
        else:
            if (
                instrument.asset_class != "crypto"
                or instrument.venue != "binance"
                or params.get("symbol") != instrument.symbol.replace("/", "")
                or not isinstance(raw, list)
            ):
                _fail("opening_wire_market_differs")
            for row in raw:
                if row.get("isBestMatch") is True:
                    yield (
                        provider,
                        _native_id(row["id"]),
                        _epoch_ns(_native_id(row["time"]), 1_000_000),
                        Decimal(str(row["price"])),
                        Decimal(str(row["qty"])),
                    )


def verified_opening_source_identity(
    store: EvidenceStore, evidence_id: str, event_id: str, *, as_of: datetime
) -> OpeningSourceIdentity:
    """Authenticate one v1/v2 event and return accounting's upstream identity.

    Both committed wire versions declare ``ceil-microsecond-1``. Identity uses
    the original nanoseconds, independently of that lossy event projection. No
    caller-supplied alias, inferred timestamp or mutable provider query is used.
    """
    from src.domain.portfolio import ExecutionBar, OpeningEvent, OpeningSourceIdentity

    try:
        envelope = store.get(evidence_id, as_of=as_of)
        value = verify_operating_capture(store, evidence_id, as_of=as_of)
        if envelope.kind != "operating_prints":
            _fail("individual_opening_capture_required")
        legacy = envelope.record.provider_version == "operating-wire-1"
        key = "execution_bars" if legacy else "opening_events"
        if ("opening_events" if legacy else "execution_bars") in value:
            _fail("opening_capture_version_differs")
        rows = [row for row in value[key] if row["event_id"] == event_id]
        prints = [row for row in value["prints"] if row["event_id"] == event_id]
        liquidity = [row for row in value["opening_liquidity"] if row["event_id"] == event_id]
        if len(rows) != 1 or len(prints) != 1 or len(liquidity) != 1:
            _fail("exact_unique_opening_source_required")
        event = (ExecutionBar if legacy else OpeningEvent).model_validate(rows[0])
        instrument = Instrument.model_validate(rows[0]["instrument"])
        if envelope.record.instrument_ids != (instrument.instrument_id,):
            _fail("opening_capture_instrument_differs")
        if value.get("timestamp_projection") != "ceil-microsecond-1":
            _fail("unproved_opening_timestamp_projection")
        printed = prints[0]
        ns = _iso_ns(printed["event_time_wire"])
        projected = pd.Timestamp(ns, unit="ns", tz="UTC").ceil("us").to_pydatetime()
        if event.opened_at != projected:
            _fail("opening_projection_differs_from_wire")
        expected = [instrument.model_dump(mode="json"), value["convention"], printed["trade_id"]]
        if not legacy:
            expected.append(ns)
        if digest(canonical_bytes(expected)) != event_id:
            _fail("opening_event_serialization_differs")
        actual_liquidity = {
            name: rows[0][name]
            for name in ("event_id", "instrument", "opened_at", "opening_volume", "fx_rate")
        }
        if liquidity[0] != actual_liquidity:
            _fail("opening_liquidity_differs")
        matches = [
            row
            for row in _raw_prints(store, value, instrument, value["convention"], as_of)
            if row[1:3] == (printed["trade_id"], ns)
        ]
        price = event.open if legacy else event.price
        if not matches or any(row[3:] != (price, event.opening_volume) for row in matches):
            _fail("opening_trade_missing_or_conflicting_with_wire")
        provider = matches[0][0]
        return OpeningSourceIdentity(
            provider=provider,
            instrument=instrument,
            trade_id=printed["trade_id"],
            wire_timestamp=str(ns),
        )
    except (KeyError, IndexError, TypeError, ValueError, ArithmeticError, AttributeError):
        _fail("malformed_or_unproved_opening_source_identity")
