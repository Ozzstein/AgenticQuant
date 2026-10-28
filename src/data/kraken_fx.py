"""Closed indicative USDT/USD source; never a cash conversion or executable quote."""

from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import Decimal, localcontext

import pandas as pd

from src.data.adapters.operating import VERSION, _decimal, _fail, verify_operating_capture
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import Instrument, RunContext, _utc

POLICY = "kraken-usdt-usd-last-trade-1"
ORIGIN = "https://api.kraken.com"


def _identity(instrument, market_id):
    if (
        not isinstance(instrument, Instrument)
        or instrument.asset_class != "crypto"
        or instrument.symbol != "USDT/USD"
        or instrument.quote_currency != "USD"
        or instrument.venue != "kraken"
        or not isinstance(market_id, str)
        or not re.fullmatch(r"[A-Z0-9]+", market_id)
    ):
        _fail("explicit_kraken_usdt_usd_identity_required")


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("fx_duplicate_native_json_key")
            result[key] = value
        return result

    value = json.loads(raw, parse_float=Decimal, object_pairs_hook=unique)
    if value.get("error") != [] or not isinstance(value.get("result"), dict):
        _fail("fx_native_error_or_shape")
    return value["result"]


def _pair(body, market_id):
    if set(body) != {market_id}:
        _fail("fx_native_pair_key_differs")
    row = body[market_id]
    if (
        row["wsname"] != "USDT/USD"
        or row["aclass_base"] != "currency"
        or row["aclass_quote"] != "currency"
        or row["lot"] != "unit"
        or type(row["lot_multiplier"]) is not int
        or row["lot_multiplier"] != 1
        or row["status"] != "online"
        or row["base"] == row["quote"]
        or any(
            not isinstance(row[k], str) or not re.fullmatch(r"[A-Z0-9]+", row[k])
            for k in ("base", "quote")
        )
    ):
        _fail("fx_native_pair_identity_differs")
    return row


def _normalize(instrument, market_id, pair_body, asset_body, trade_body, received_at):
    _identity(instrument, market_id)
    pair = _pair(pair_body, market_id)
    if set(asset_body) != {pair["base"], pair["quote"]}:
        _fail("fx_native_asset_keys_differ")
    for native, name in ((pair["base"], "USDT"), (pair["quote"], "USD")):
        if (
            asset_body[native]["altname"] != name
            or asset_body[native]["aclass"] != "currency"
            or asset_body[native]["status"] != "enabled"
        ):
            _fail("fx_native_asset_identity_differs")
    if set(trade_body) != {market_id, "last"}:
        _fail("fx_native_trade_pair_differs")
    rows = trade_body[market_id]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        _fail("fx_recent_individual_trades_missing")
    parsed, ids = [], set()
    for index, row in enumerate(rows):
        if (
            not isinstance(row, list)
            or len(row) != 7
            or type(row[6]) is not int
            or row[6] < 0
            or row[6] in ids
            or row[3] not in ("b", "s")
            or row[4] not in ("m", "l")
            or row[5] != ""
            or isinstance(row[2], bool)
            or not isinstance(row[2], (int, Decimal))
            or not isinstance(row[0], str)
            or not isinstance(row[1], str)
        ):
            _fail("fx_native_individual_trade_invalid")
        ids.add(row[6])
        stamp = str(row[2])
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]{1,9})?", stamp):
            _fail("fx_exact_unix_nanoseconds_required")
        with localcontext() as arithmetic:
            arithmetic.prec = max(50, len(stamp) + 10)
            ns = int(Decimal(stamp) * 1_000_000_000)
        observed = pd.Timestamp(ns, unit="ns", tz="UTC").ceil("us").to_pydatetime()
        if observed > received_at:
            _fail("fx_trade_after_measured_receipt")
        price = _decimal(row[0])
        _decimal(row[1])
        parsed.append(
            (
                ns,
                observed,
                price,
                dict(
                    trade_id=str(row[6]),
                    timestamp_wire=stamp,
                    timestamp_ns=str(ns),
                    price=row[0],
                    size=row[1],
                    pointer=["result", market_id, index],
                ),
            )
        )
    latest = max(row[0] for row in parsed)
    selected = [row for row in parsed if row[0] == latest]
    if len({row[2] for row in selected}) != 1:
        _fail("fx_ambiguous_latest_trade_price")
    return dict(
        instrument=instrument.model_dump(mode="json"),
        market_id=market_id,
        native_base=pair["base"],
        native_quote=pair["quote"],
        base_currency="USDT",
        quote_currency="USD",
        units="USD/USDT",
        rate=selected[0][3]["price"],
        observed_at=selected[0][1].isoformat(),
        trades=[row[3] for row in selected],
        convention=POLICY,
        reference_type="last-trade-indicative",
        executable_conversion=False,
    )


def capture_kraken_fx(adapter, instrument, context, *, market_id):
    _identity(instrument, market_id)
    if adapter.http.base_url != ORIGIN:
        _fail("fx_fixed_kraken_origin_required")
    pair, body = adapter._get(
        "AssetPairs", context, instrument, dict(pair=market_id, aclass_base="currency")
    )
    info = _pair(body, market_id)
    assets, _ = adapter._get(
        "Assets",
        context,
        instrument,
        dict(asset=info["base"] + "," + info["quote"], aclass="currency"),
    )
    trades, _ = adapter._get("Trades", context, instrument, dict(pair=market_id, count=1000))
    value = _normalize(
        instrument,
        market_id,
        _json(adapter.store.raw(pair.record.evidence_id)),
        _json(adapter.store.raw(assets.record.evidence_id)),
        _json(adapter.store.raw(trades.record.evidence_id)),
        trades.record.ingested_at,
    )
    observed = _utc(pd.Timestamp(value["observed_at"]).to_pydatetime())

    def admitted(captured):
        if not 0 <= (captured - observed).total_seconds() <= 300:
            _fail("fx_missing_stale_or_future")
        return value

    proof = adapter.derived(
        context,
        (pair, assets, trades),
        admitted,
        kind="fx",
        logical_key=instrument.instrument_id,
        instrument=instrument,
        policy=POLICY,
        observed_at=observed,
        units="USD/USDT",
    )
    return adapter.result(
        (pair, assets, trades, proof),
        detail="indicative USDT/USD source; no executable conversion or history claim",
    )


def parse_kraken_fx_reference(
    store: EvidenceStore,
    fx_instrument: Instrument,
    context: RunContext,
    *,
    evidence_id: str,
    valuation_at: datetime,
    max_age_seconds: int = 300,
) -> dict:
    """Certify explicit immutable receipts, never discover or fetch a newer price."""
    return _verify_reference(
        store,
        fx_instrument,
        context.as_of,
        evidence_id=evidence_id,
        valuation_at=valuation_at,
        max_age_seconds=max_age_seconds,
    )


def _verify_reference(
    store, fx_instrument, as_of, *, evidence_id, valuation_at, max_age_seconds=300
):
    try:
        at = _utc(valuation_at)
        if type(max_age_seconds) is not int or not 1 <= max_age_seconds <= 300 or at > _utc(as_of):
            _fail("fx_invalid_valuation_cutoff_or_age")
        envelope = store.get(evidence_id, as_of=at)
        value = store.normalized(evidence_id, as_of=at)
        manifest = value["source_manifest"]
        if (
            envelope.kind != "fx"
            or envelope.record.provider != "operating"
            or envelope.record.provider_version != VERSION
            or envelope.record.source_uri != "operating://fx"
            or envelope.record.units != "USD/USDT"
            or envelope.record.instrument_ids != (fx_instrument.instrument_id,)
            or manifest["implementation"] != VERSION
            or manifest["policy"] != POLICY
            or len(manifest["sources"]) != 3
            or store.raw(evidence_id) != canonical_bytes(manifest)
            or envelope.record.available_at < envelope.record.ingested_at
        ):
            _fail("fx_certified_native_source_required")
        sources, bodies = [], []
        market_id = value["market_id"]
        for ref, operation in zip(
            manifest["sources"], ("AssetPairs", "Assets", "Trades"), strict=True
        ):
            source = store.get(ref["evidence_id"], as_of=at)
            meta = verify_operating_capture(store, ref["evidence_id"], as_of=at)
            expected = (
                dict(pair=market_id, aclass_base="currency")
                if operation == "AssetPairs"
                else (
                    dict(
                        asset=value["native_base"] + "," + value["native_quote"], aclass="currency"
                    )
                    if operation == "Assets"
                    else dict(pair=market_id, count=1000)
                )
            )
            if (
                source.kind != "operating_raw"
                or source.record.provider_version != VERSION
                or source.record.requested_as_of != envelope.record.requested_as_of
                or source.record.ingested_at < envelope.record.requested_as_of
                or source.record.instrument_ids != (fx_instrument.instrument_id,)
                or source.record.source_uri != ORIGIN + "/0/public/" + operation
                or meta["upstream"] != "kraken-spot-rest-v0"
                or meta["request"] != dict(path="/0/public/" + operation, params=expected)
                or ref
                != dict(
                    evidence_id=source.record.evidence_id,
                    envelope_hash=source.content_hash(),
                    raw_hash=source.record.raw_hash,
                )
            ):
                _fail("fx_native_authority_or_request_differs")
            source.record.require_available(envelope.record.ingested_at)
            sources.append(source)
            bodies.append(_json(store.raw(source.record.evidence_id)))
        if len({s.record.evidence_id for s in sources}) != 3:
            _fail("fx_distinct_native_authorities_required")
        if [s.record.ingested_at for s in sources] != sorted(s.record.ingested_at for s in sources):
            _fail("fx_native_receipt_order_differs")
        expected = _normalize(fx_instrument, market_id, *bodies, sources[-1].record.ingested_at)
        observed = _utc(pd.Timestamp(expected["observed_at"]).to_pydatetime())
        if (
            not observed <= envelope.record.available_at <= at
            or not 0 <= (at - observed).total_seconds() <= max_age_seconds
            or envelope.record.observed_at != observed
        ):
            _fail("fx_missing_stale_or_future")
        expected["source_manifest"] = manifest
        if expected != value:
            _fail("fx_native_normalization_differs")
        return expected
    except (KeyError, IndexError, TypeError, ValueError, ArithmeticError):
        _fail("fx_invalid_native_proof")


def read_legacy_fx_archive(store, evidence_id, *, as_of):
    """Explicitly unverified inspection; never usable by a certified consumer."""
    envelope = store.get(evidence_id, as_of=as_of)
    if envelope.kind != "fx" or digest(store.raw(evidence_id)) != envelope.record.raw_hash:
        _fail("legacy_fx_archive_differs")
    return dict(
        scope="legacy-unverified-fx",
        envelope=envelope.model_dump(mode="json"),
        normalized=store.normalized(evidence_id, as_of=as_of),
    )
