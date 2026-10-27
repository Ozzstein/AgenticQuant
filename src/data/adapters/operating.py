"""Measured read-only operating captures and source-bound paper economics.

Provider wire responses are archived before parsing. Current references are
capture-time observations, never historical membership/fundamentals. Individual
prints supply execution liquidity; candles supply research prices and ADV only.
See docs/implementation/data-quant.md for source/API and projection conventions.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from functools import wraps
from typing import Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd

from src.data.adapters.base import FetchResult, ProviderError, ReadHTTP, store_observation
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import Instrument, RunContext, _utc
from src.domain.market import MarketBar, MarketSnapshot

VERSION = "operating-wire-2"
SUPPORTED_VERSIONS = frozenset({"operating-wire-1", VERSION})
KINDS = (
    "opening_events",
    "instrument_metadata",
    "universe_history",
    "corporate_actions",
    "venue_rules",
    "calendar",
    "opening_liquidity",
    "availability_audit",
)
LEGACY_KINDS = ("execution_bars", *KINDS[1:])
UTC = timezone.utc


def _fail(code):
    raise ProviderError("operating", code)


def _operation(method):
    @wraps(method)
    def guarded(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except ProviderError:
            raise
        except (KeyError, IndexError, TypeError, ValueError, ArithmeticError, AttributeError):
            _fail("malformed_" + method.__name__ + "_or_missing_coverage")

    return guarded


def _decimal(value, *, positive=True):
    try:
        number = Decimal(str(value))
        if not number.is_finite() or (number <= 0 if positive else number < 0):
            raise ValueError()
        return number
    except (ValueError, TypeError, ArithmeticError):
        _fail("invalid_decimal")


def _instant(value):
    """Conservative microsecond projection; exact wire timestamp stays archived."""
    try:
        instant = pd.Timestamp(value)
        if instant.tzinfo is None:
            raise ValueError()
        return instant.tz_convert("UTC").ceil("us").to_pydatetime()
    except (TypeError, ValueError, OverflowError):
        _fail("invalid_timestamp")


def _epoch(value, unit):
    # No float round-trip for exchange seconds/milliseconds/nanoseconds.
    return _instant(pd.Timestamp(int(Decimal(str(value)) * unit), unit="ns", tz="UTC"))


def _window(start, end, context):
    start, end = _utc(start), _utc(end)
    if start >= end or end > context.as_of:
        _fail("invalid_requested_window")
    return start, end


def _usd(instrument):
    if instrument.quote_currency != "USD":
        _fail("explicit_usd_fx_evidence_required")


class OperatingCapture:
    """Owned measured archive clock. Do not wrap this store in CaptureStore.

    ``clock`` is injectable for deterministic tests; production uses measured UTC.
    There is no caller-controlled ingestion timestamp in any public operation.
    All returned records use provider='operating'; actual upstream identities and
    versioned URLs are immutable in their wire/source manifests.
    """

    def __init__(self, store: EvidenceStore, *, clock: Callable[[], datetime] | None = None):
        self.store = store
        self.clock = clock or (lambda: datetime.now(UTC))
        self._last_clock = None

    def now(self):
        now = _utc(self.clock())
        if self._last_clock is not None and now < self._last_clock:
            _fail("archive_clock_moved_backwards")
        self._last_clock = now
        return now

    def wire(
        self,
        http: ReadHTTP,
        path,
        context,
        *,
        upstream,
        params=None,
        headers=None,
        instrument=None,
        allowed_error_statuses=(),
        timestamp_unit=None,
    ):
        options = (
            {"allowed_error_statuses": tuple(allowed_error_statuses)}
            if allowed_error_statuses
            else {}
        )
        raw = http.get(path, params=params, headers=headers, **options)
        observed = self.now()  # After the blocking request, before any normalization.
        safe = {
            k: v
            for k, v in (params or {}).items()
            if k.lower() not in {"apikey", "api_key", "token"}
        }
        envelope = store_observation(
            self.store,
            raw=raw,
            normalized={
                "upstream": upstream,
                **({"source_origin": http.base_url} if timestamp_unit is not None else {}),
                "request": {
                    "path": path,
                    "params": safe,
                    **(
                        {
                            "timestamp_unit": timestamp_unit,
                            "time_unit_header": next(
                                (
                                    v
                                    for k, v in (headers or {}).items()
                                    if k.lower() == "x-mbx-time-unit"
                                ),
                                None,
                            ),
                        }
                        if timestamp_unit is not None
                        else {}
                    ),
                },
                "wire_hash": digest(raw),
            },
            provider="operating",
            version=VERSION,
            source_uri=http.base_url + path,
            kind="operating_raw",
            logical_key=digest(canonical_bytes([upstream, path, safe])),
            context=context,
            ingested_at=observed,
            instrument=instrument,
        )
        try:
            # Preserve provider decimal lexemes instead of parsing through floats.
            value = json.loads(raw, parse_float=str)
        except (ValueError, UnicodeError):
            _fail("malformed_json_wire_archived")
        return envelope, value

    def derived(
        self, context, sources, builder, *, kind, logical_key, instrument=None, policy=None
    ):
        captured = self.now()
        if captured < context.as_of:
            _fail("archive_clock_precedes_request")
        sources = tuple(sources)
        if not sources:
            _fail("actual_immutable_source_lineage_required")
        for envelope in sources:
            envelope.record.require_available(captured)
            if self.store.get(envelope.record.evidence_id) != envelope:
                _fail("source_envelope_differs")
        manifest = {
            "implementation": VERSION,
            "policy": policy or kind,
            "sources": [
                {
                    "evidence_id": e.record.evidence_id,
                    "envelope_hash": e.content_hash(),
                    "raw_hash": e.record.raw_hash,
                }
                for e in sources
            ],
        }
        value = builder(captured)
        if isinstance(value, dict):
            value = {**value, "source_manifest": manifest}
        else:
            _fail("normalized_object_required")
        return store_observation(
            self.store,
            raw=canonical_bytes(manifest),
            normalized=value,
            provider="operating",
            version=VERSION,
            source_uri="operating://" + kind,
            kind=kind,
            logical_key=logical_key,
            context=context,
            ingested_at=captured,
            instrument=instrument,
        )

    def result(
        self, evidence, *, detail="actual receipt availability; external access not certified"
    ):
        return FetchResult("available", tuple(evidence), detail)

    @_operation
    def current_universe(
        self,
        instruments: tuple[Instrument, ...],
        context,
        *,
        reference_ids: tuple[str, ...],
        universe_version: str,
    ):
        if (
            not instruments
            or len({i.instrument_id for i in instruments}) != len(instruments)
            or not universe_version
        ):
            _fail("explicit_unique_universe_required")
        sources = [self.store.get(key) for key in reference_ids]
        members = {}
        for source in sources:
            value = self.store.normalized(source.record.evidence_id, as_of=self.now())
            if source.kind != "operating_reference" or not value.get("active"):
                _fail("active_current_reference_required")
            item = Instrument.model_validate(value["instrument"])
            if item.instrument_id in members:
                _fail("duplicate_universe_reference")
            members[item.instrument_id] = item
        if members != {i.instrument_id: i for i in instruments}:
            _fail("universe_reference_identity_mismatch")
        from src.domain.portfolio import UniverseEligibility

        source_id = digest(canonical_bytes([e.content_hash() for e in sources]))
        result = self.derived(
            context,
            sources,
            lambda at: dict(
                universe_history=[
                    UniverseEligibility(
                        universe_version=universe_version,
                        data_snapshot_id=source_id,
                        effective_at=at,
                        available_at=at,
                        instrument_ids=tuple(i.instrument_id for i in instruments),
                    ).model_dump(mode="json")
                ]
            ),
            kind="operating_universe",
            logical_key=universe_version,
            policy="explicit-requested-members-observed-active-at-capture-1",
        )
        return self.result([result])


def verify_operating_capture(store: EvidenceStore, evidence_id: str, *, as_of: datetime) -> dict:
    """Verify complete transitive raw/envelope lineage before using array pointers."""
    seen = set()

    def check(key):
        if key in seen:
            return
        seen.add(key)
        envelope = store.get(key, as_of=as_of)
        if envelope.record.available_at < envelope.record.ingested_at:
            _fail("operating_availability_precedes_measured_capture")
        raw = store.raw(key)
        if digest(raw) != envelope.record.raw_hash:
            _fail("source_raw_hash_differs")
        value = store.normalized(key, as_of=as_of)
        if (
            envelope.record.provider != "operating"
            or envelope.record.provider_version not in SUPPORTED_VERSIONS
        ):
            _fail("not_operating_capture")
        if envelope.kind == "operating_raw":
            if value["wire_hash"] != digest(raw):
                _fail("wire_hash_differs")
            return
        if envelope.kind == "bar":
            # MarketBar's canonical normalization is intentionally flat. The
            # complete provider body is retained as this record's raw payload.
            MarketBar(evidence_id=key, **value).require_execution_price()
            return
        manifest = value.get("source_manifest")
        if (
            not manifest
            or canonical_bytes(manifest) != raw
            or manifest["implementation"] != envelope.record.provider_version
            or not manifest["sources"]
        ):
            _fail("source_manifest_differs")
        for source in manifest["sources"]:
            actual = store.get(source["evidence_id"], as_of=as_of)
            actual.record.require_available(envelope.record.ingested_at)
            if (
                actual.content_hash() != source["envelope_hash"]
                or actual.record.raw_hash != source["raw_hash"]
            ):
                _fail("source_envelope_hash_differs")
            # Non-operating archived FX/market records remain full typed evidence.
            if actual.record.provider == "operating":
                check(source["evidence_id"])
            elif digest(store.raw(source["evidence_id"])) != actual.record.raw_hash:
                _fail("external_source_raw_differs")

    check(evidence_id)
    return store.normalized(evidence_id, as_of=as_of)


def _calendar(store, evidence_id, as_of):
    value = verify_operating_capture(store, evidence_id, as_of=as_of)
    if "calendar" not in value:
        _fail("calendar_source_required")
    return {
        row["session"]: (row, _instant(row["session_open"]), _instant(row["session_close"]))
        for row in value["calendar"]
    }


def _daily_adv(store, bar_ids, instrument, cutoff, *, expected_sessions):
    rows, sources = {}, []
    wanted = tuple(sorted(expected_sessions))[-20:]
    for key in bar_ids:
        source = store.get(key, as_of=cutoff)
        if source.kind != "bar" or instrument.instrument_id not in source.record.instrument_ids:
            _fail("adv_bar_source_identity")
        payload = store.normalized(key, as_of=cutoff)
        bar = MarketBar(evidence_id=key, **payload)
        if (
            bar.instrument != instrument
            or bar.price_basis != "raw"
            or bar.closed_at > cutoff
            or bar.vwap is None
        ):
            _fail("actual_completed_raw_vwap_required")
        if bar.session not in wanted:
            continue
        if (
            bar.session not in expected_sessions
            or (bar.opened_at, bar.closed_at) != expected_sessions[bar.session]
        ):
            _fail("complete_daily_adv_interval_required")
        if bar.session in rows and rows[bar.session] != bar.vwap * bar.volume:
            _fail("conflicting_adv_session")
        rows[bar.session] = bar.vwap * bar.volume
        sources.append(source)
    if len(wanted) != 20 or any(session not in rows for session in wanted):
        _fail("twenty_complete_observed_adv_sessions_required")
    return sum((rows[s] for s in wanted), Decimal(0)) / 20, sources


def _market_record(
    capture, context, wire, instrument, *, opened, closed, values, calendar, session
):
    normalized = dict(
        instrument=instrument.model_dump(mode="json"),
        opened_at=opened.isoformat(),
        closed_at=closed.isoformat(),
        **{k: str(values[k]) for k in ("open", "high", "low", "close", "volume", "vwap")},
        price_basis="raw",
        market_type="spot" if instrument.asset_class == "crypto" else "equity",
        calendar=calendar,
        session=session,
    )
    MarketBar(evidence_id="validation", **normalized)
    return store_observation(
        capture.store,
        raw=capture.store.raw(wire.record.evidence_id),
        normalized=normalized,
        provider="operating",
        version=VERSION,
        source_uri=wire.record.source_uri,
        kind="bar",
        logical_key=f"{instrument.instrument_id}:1d:{session}",
        context=context,
        ingested_at=capture.now(),
        instrument=instrument,
        observed_at=closed,
        adjustment_policy="raw",
    )


def _print_capture(
    capture,
    instrument,
    context,
    wire_sources,
    trades,
    *,
    calendar,
    fx_rate=Decimal(1),
    fx_sources=(),
    convention,
):
    """Every chosen row is one actual provider print, never candle volume."""
    from src.domain.portfolio import OpeningEvent

    if not trades:
        _fail("no_observed_print_coverage")
    rows, wire_times = [], []
    identities = set()
    for trade_id, timestamp, price, size in trades:
        opened = _instant(timestamp)
        wire_instant = pd.Timestamp(timestamp).tz_convert("UTC").value
        trade_identity = (str(trade_id), wire_instant)
        if not str(trade_id) or trade_identity in identities:
            _fail("duplicate_or_missing_trade_id")
        identities.add(trade_identity)
        matching = [item for item in calendar.values() if item[1] <= opened < item[2]]
        if len(matching) != 1:
            _fail("print_outside_authoritative_session")
        row, session_open, session_close = matching[0]
        p, volume = _decimal(price), _decimal(size)
        event = digest(
            canonical_bytes(
                [instrument.model_dump(mode="json"), convention, str(trade_id), wire_instant]
            )
        )
        bar = OpeningEvent(
            event_id=event,
            instrument=instrument,
            opened_at=opened,
            price=p,
            opening_volume=volume,
            fx_rate=fx_rate,
            calendar=row["calendar"],
            calendar_version=row["calendar_version"],
            session=row["session"],
            session_open=session_open,
            session_close=session_close,
        )
        rows.append(bar.model_dump(mode="json"))
        wire_times.append(
            dict(event_id=event, trade_id=str(trade_id), event_time_wire=str(timestamp))
        )
    rows.sort(key=lambda r: (r["opened_at"], r["instrument"]["instrument_id"], r["event_id"]))
    result = capture.derived(
        context,
        (*wire_sources, *fx_sources),
        lambda at: dict(
            opening_events=rows,
            opening_liquidity=[
                {
                    k: r[k]
                    for k in ("event_id", "instrument", "opened_at", "opening_volume", "fx_rate")
                }
                for r in rows
            ],
            prints=wire_times,
            convention=convention,
            timestamp_projection="ceil-microsecond-1",
        ),
        kind="operating_prints",
        logical_key=instrument.instrument_id,
        instrument=instrument,
        policy="individual-print-size-1",
    )
    return result


class AlpacaOperatingAdapter(OperatingCapture):
    """Closed Alpaca/FMP GET surface; no account/orders/positions dispatch."""

    def __init__(
        self,
        store,
        api_key: str,
        secret_key: str,
        *,
        fmp_key: str = "",
        data_http=None,
        trading_http=None,
        profile_http=None,
        clock=None,
    ):
        super().__init__(store, clock=clock)
        if not api_key or not secret_key:
            _fail("alpaca_credentials_required")
        self.headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
        self.fmp_key = fmp_key
        self.data = data_http or ReadHTTP("alpaca", "https://data.alpaca.markets")
        self.trading = trading_http or ReadHTTP("alpaca", "https://paper-api.alpaca.markets")
        self.profile = profile_http or ReadHTTP("fmp", "https://financialmodelingprep.com")

    def close(self):
        for http in (self.data, self.trading, self.profile):
            http.close()

    @_operation
    def calendar(self, context, *, start: str, end: str):
        if date.fromisoformat(start) > date.fromisoformat(end):
            _fail("invalid_calendar_window")
        wire, values = self.wire(
            self.trading,
            "/v2/calendar",
            context,
            upstream="alpaca-trading-v2",
            params={"start": start, "end": end},
            headers=self.headers,
        )
        if not isinstance(values, list):
            _fail("calendar_array_required")
        rows = []
        for value in values:
            session = date.fromisoformat(value["date"]).isoformat()
            if not start <= session <= end:
                _fail("calendar_outside_requested_window")
            opened = (
                datetime.fromisoformat(session + "T" + value["open"])
                .replace(tzinfo=ZoneInfo("America/New_York"))
                .astimezone(UTC)
            )
            closed = (
                datetime.fromisoformat(session + "T" + value["close"])
                .replace(tzinfo=ZoneInfo("America/New_York"))
                .astimezone(UTC)
            )
            if opened >= closed:
                _fail("invalid_calendar_session")
            rows.append(
                dict(
                    calendar="XNYS",
                    calendar_version="alpaca-calendar-v2:" + wire.record.raw_hash,
                    session=session,
                    session_open=opened.isoformat().replace("+00:00", "Z"),
                    session_close=closed.isoformat().replace("+00:00", "Z"),
                )
            )
        if len({r["session"] for r in rows}) != len(rows):
            _fail("duplicate_calendar_session")
        from src.data.calendars import sessions

        expected = sessions(start, end, calendar="XNYS")
        actual = {
            r["session"]: (_instant(r["session_open"]), _instant(r["session_close"])) for r in rows
        }
        if actual != expected:
            _fail("provider_calendar_coverage_or_session_conflict")
        proof = self.derived(
            context,
            [wire],
            lambda at: dict(
                calendar=sorted(rows, key=canonical_bytes), requested_start=start, requested_end=end
            ),
            kind="operating_calendar",
            logical_key=start + ":" + end,
        )
        return self.result([wire, proof])

    def _pages(self, path, context, instrument, params, *, max_pages):
        if not 1 <= max_pages <= 1000:
            _fail("invalid_page_budget")
        seen, result = set(), []
        for _ in range(max_pages):
            wire, value = self.wire(
                self.data,
                path,
                context,
                upstream="alpaca-market-data",
                params=params,
                headers=self.headers,
                instrument=instrument,
            )
            result.append((wire, value))
            token = value.get("next_page_token")
            if not token:
                return result
            if token in seen:
                _fail("pagination_did_not_advance")
            seen.add(token)
            params = {**params, "page_token": token}
        _fail("pagination_budget_exhausted")

    @_operation
    def bars(
        self, instrument, context, *, start, end, calendar_id, feed, symbol_asof, max_pages=100
    ):
        _usd(instrument)
        if instrument.asset_class not in ("equity", "etf") or feed not in ("iex", "sip"):
            _fail("explicit_equity_feed_required")
        start, end = _window(start, end, context)
        date.fromisoformat(symbol_asof)
        calendar = _calendar(self.store, calendar_id, self.now())
        pages = self._pages(
            "/v2/stocks/bars",
            context,
            instrument,
            dict(
                symbols=instrument.symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                timeframe="1Day",
                adjustment="raw",
                feed=feed,
                asof=symbol_asof,
                sort="asc",
                limit=10000,
            ),
            max_pages=max_pages,
        )
        evidence = [w for w, _ in pages]
        observed = set()
        for wire, value in pages:
            if set(value.get("bars", {})) - {instrument.symbol}:
                _fail("bar_symbol_identity_mismatch")
            for row in value.get("bars", {}).get(instrument.symbol, []):
                session = (
                    _instant(row["t"]).astimezone(ZoneInfo("America/New_York")).date().isoformat()
                )
                if session not in calendar:
                    _fail("bar_calendar_coverage_missing")
                _, opened, closed = calendar[session]
                if opened < start or closed > end:
                    continue
                if session in observed:
                    _fail("duplicate_daily_bar")
                observed.add(session)
                evidence.append(
                    _market_record(
                        self,
                        context,
                        wire,
                        instrument,
                        opened=opened,
                        closed=closed,
                        values={
                            k: row[v]
                            for k, v in dict(
                                open="o", high="h", low="l", close="c", volume="v", vwap="vw"
                            ).items()
                        },
                        calendar="XNYS",
                        session=session,
                    )
                )
        if not observed:
            _fail("daily_bar_coverage_missing")
        return self.result(evidence)

    @_operation
    def reference(self, instrument, context, *, asset_id, bar_ids, calendar_id):
        _usd(instrument)
        if not self.fmp_key or instrument.asset_class not in ("equity", "etf") or not asset_id:
            _fail("equity_profile_credentials_and_asset_id_required")
        asset_wire, asset = self.wire(
            self.trading,
            "/v2/assets/" + quote(asset_id, safe=""),
            context,
            upstream="alpaca-trading-v2",
            headers=self.headers,
            instrument=instrument,
        )
        profile_wire, profile = self.wire(
            self.profile,
            "/stable/profile",
            context,
            upstream="fmp-stable-profile",
            params={"symbol": instrument.symbol, "apikey": self.fmp_key},
            instrument=instrument,
        )
        if (
            asset.get("id") != asset_id
            or asset.get("symbol") != instrument.symbol
            or asset.get("class") != "us_equity"
        ):
            _fail("asset_identity_mismatch")
        if (
            not isinstance(profile, list)
            or len(profile) != 1
            or profile[0].get("symbol") != instrument.symbol
        ):
            _fail("profile_identity_mismatch")
        profile = profile[0]
        if profile.get("currency") != "USD" or not str(profile.get("sector") or "").strip():
            _fail("sector_or_currency_coverage_missing")
        step = _decimal(asset.get("min_trade_increment"))
        minimum = _decimal(asset.get("min_order_size"))
        tick = _decimal(asset.get("price_increment"))
        at = self.now()
        calendar = _calendar(self.store, calendar_id, at)
        from src.data.calendars import sessions

        schedule = sessions(
            (context.as_of.date() - timedelta(days=90)).isoformat(),
            context.as_of.date().isoformat(),
            calendar="XNYS",
        )
        completed = [(s, times) for s, times in schedule.items() if times[1] <= context.as_of][-20:]
        wanted = dict(completed)
        if len(wanted) != 20 or any(
            s not in calendar or calendar[s][1:] != times for s, times in completed
        ):
            _fail("current_twenty_session_calendar_coverage_required")
        adv, sources = _daily_adv(self.store, bar_ids, instrument, at, expected_sessions=wanted)
        active = (
            asset.get("status") == "active"
            and asset.get("tradable") is True
            and profile.get("isActivelyTrading") is True
        )
        from src.domain.portfolio import InstrumentRisk
        from src.execution.venue import VenueRules

        def normalized(captured):
            metadata = InstrumentRisk(
                instrument=instrument,
                sector=profile["sector"],
                adv_base=adv,
                as_of=captured,
                lot_size=step,
                min_notional=0,
            )
            rules = VenueRules(
                venue="paper",
                instrument=instrument,
                symbol=instrument.symbol,
                available_at=captured,
                quantity_step=step,
                price_tick=tick,
                minimum_quantity=minimum,
                minimum_notional=0,
                order_types=("market",),
                time_in_force=("DAY",),
                market_type="equity",
                active=active,
            )
            return dict(
                instrument=instrument.model_dump(mode="json"),
                active=active,
                instrument_metadata=[metadata.model_dump(mode="json")],
                venue_rules=[rules.model_dump(mode="json")],
                asset_id=asset_id,
                adv_sessions=sorted(wanted)[-20:],
                adv_convention="mean-of-20-completed-session-vwap-times-volume-USD",
                rule_policy="paper-market-only-native-quantity-and-price-increments-1",
            )

        proof = self.derived(
            context,
            [asset_wire, profile_wire, self.store.get(calendar_id), *sources],
            normalized,
            kind="operating_reference",
            logical_key=instrument.instrument_id,
            instrument=instrument,
        )
        return self.result([asset_wire, profile_wire, proof])

    @_operation
    def trades(
        self, instrument, context, *, start, end, calendar_id, feed, symbol_asof, max_pages=100
    ):
        _usd(instrument)
        start, end = _window(start, end, context)
        if instrument.asset_class not in ("equity", "etf") or feed not in ("iex", "sip"):
            _fail("explicit_equity_feed_required")
        date.fromisoformat(symbol_asof)
        pages = self._pages(
            "/v2/stocks/trades",
            context,
            instrument,
            dict(
                symbols=instrument.symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                feed=feed,
                asof=symbol_asof,
                sort="asc",
                limit=10000,
            ),
            max_pages=max_pages,
        )
        trades = []
        for _, value in pages:
            if set(value.get("trades", {})) - {instrument.symbol}:
                _fail("trade_symbol_identity_mismatch")
            for row in value.get("trades", {}).get(instrument.symbol, []):
                at = _instant(row["t"])
                # Explicit regular-sale policy; auction/odd-lot/correction prints
                # are archived but not silently admitted as regular opening size.
                if start <= at < end and row.get("c") == ["@"]:
                    trades.append(
                        (str(row["x"]) + ":" + str(row["i"]), row["t"], row["p"], row["s"])
                    )
        proof = _print_capture(
            self,
            instrument,
            context,
            [w for w, _ in pages],
            trades,
            calendar=_calendar(self.store, calendar_id, self.now()),
            convention="alpaca-" + feed + "-regular-sale-individual-print-1",
        )
        return self.result([*(w for w, _ in pages), proof])

    @_operation
    def corporate_actions(
        self, instrument, context, *, start: str, end: str, calendar_id, max_pages=100
    ):
        """All provider types requested. Unsupported economics fail explicitly.

        Cash dividends require ex-date entitlement/payable-date accounting absent
        from the simple canonical action; they are archived and rejected here.
        """
        if instrument.asset_class not in ("equity", "etf") or date.fromisoformat(
            start
        ) > date.fromisoformat(end):
            _fail("invalid_action_scope")
        pages = self._pages(
            "/v1/corporate-actions",
            context,
            instrument,
            dict(
                symbols=instrument.symbol,
                start=start,
                end=end,
                region="us",
                data_quality="all",
                sort="asc",
                limit=1000,
            ),
            max_pages=max_pages,
        )
        calendar = _calendar(self.store, calendar_id, self.now())
        from src.domain.portfolio import CorporateAction

        actions, ids = [], set()
        for wire, value in pages:
            groups = value.get("corporate_actions")
            if not isinstance(groups, dict):
                _fail("corporate_action_coverage_unknown")
            for kind, rows in groups.items():
                if rows and kind not in ("forward_splits", "reverse_splits"):
                    _fail("unsupported_action_economics_wire_archived")
                for row in rows:
                    if row["symbol"] != instrument.symbol or not row.get("id") or row["id"] in ids:
                        _fail("action_identity_mismatch")
                    ids.add(row["id"])
                    session = row["ex_date"]
                    if session not in calendar:
                        _fail("action_effective_calendar_missing")
                    actions.append(
                        CorporateAction(
                            action_id=str(row["id"]),
                            instrument=instrument,
                            effective_at=calendar[session][1],
                            kind="split",
                            value=_decimal(row["new_rate"]) / _decimal(row["old_rate"]),
                            source_id=wire.record.evidence_id,
                        ).model_dump(mode="json")
                    )
        actions.sort(key=lambda a: (a["effective_at"], a["action_id"]))
        proof = self.derived(
            context,
            [*(w for w, _ in pages), self.store.get(calendar_id)],
            lambda at: dict(
                corporate_actions=actions,
                coverage=dict(
                    instrument_ids=[instrument.instrument_id],
                    provider_process_start=start,
                    provider_process_end=end,
                    types="all",
                    historical_completeness="not-certified",
                ),
            ),
            kind="operating_actions",
            logical_key=instrument.instrument_id,
            instrument=instrument,
        )
        return self.result([*(w for w, _ in pages), proof])


def _crypto_identity(instrument, venue):
    if instrument.asset_class != "crypto" or instrument.venue != venue or ":" in instrument.symbol:
        _fail("explicit_spot_venue_identity_required")


def _fx(capture, instrument, context, fx_evidence_id, *, at):
    if instrument.quote_currency == "USD":
        if fx_evidence_id is not None:
            _fail("usd_identity_does_not_take_fx_override")
        return Decimal(1), ()
    if not fx_evidence_id:
        _fail("explicit_usd_fx_evidence_required")
    source = capture.store.get(fx_evidence_id, as_of=at)
    value = capture.store.normalized(fx_evidence_id, as_of=at)
    if (
        source.kind != "fx"
        or value.get("base_currency") != instrument.quote_currency
        or value.get("quote_currency") != "USD"
    ):
        _fail("fx_currency_identity_mismatch")
    observed = _instant(value["observed_at"])
    if observed > at or (at - observed).total_seconds() > 300 or source.record.available_at > at:
        _fail("fx_missing_stale_or_future")
    return _decimal(value["rate"]), (source,)


class _CryptoOperatingAdapter(OperatingCapture):
    venue: str

    def close(self):
        self.http.close()

    @_operation
    def calendar(self, instrument, context, *, start: str, end: str, reference_id: str):
        """Explicit UTC continuous-market policy grounded in actual spot identity."""
        _crypto_identity(instrument, self.venue)
        source = self.store.get(reference_id, as_of=self.now())
        ref = verify_operating_capture(self.store, reference_id, as_of=self.now())
        if (
            source.kind != "operating_reference"
            or ref["instrument"] != instrument.model_dump(mode="json")
            or not ref["active"]
        ):
            _fail("active_spot_reference_required")
        first, last = date.fromisoformat(start), date.fromisoformat(end)
        if first > last or (last - first).days > 10000:
            _fail("invalid_calendar_window")
        rows = []
        for offset in range((last - first).days + 1):
            opened = datetime.combine(
                first + timedelta(days=offset), datetime.min.time(), tzinfo=UTC
            )
            rows.append(
                dict(
                    calendar="24/7",
                    calendar_version="UTC-continuous-paper-1",
                    session=opened.date().isoformat(),
                    session_open=opened.isoformat().replace("+00:00", "Z"),
                    session_close=(opened + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
                )
            )
        proof = self.derived(
            context,
            [source],
            lambda at: dict(
                calendar=sorted(rows, key=canonical_bytes),
                convention="declared-UTC-continuous-paper-sessions-not-exchange-uptime",
            ),
            kind="operating_calendar",
            logical_key=instrument.instrument_id + ":" + start + ":" + end,
            instrument=instrument,
        )
        return self.result([proof])

    def _reference(
        self, instrument, context, wire, market, bar_ids, *, fx_evidence_id=None, extra_sources=()
    ):
        from src.domain.portfolio import InstrumentRisk
        from src.execution.venue import VenueRules

        at = self.now()
        fx, fx_sources = _fx(self, instrument, context, fx_evidence_id, at=at)
        last = context.as_of.date() - timedelta(days=1)
        wanted = {}
        for i in range(20):
            opened = datetime.combine(last - timedelta(days=i), datetime.min.time(), tzinfo=UTC)
            wanted[opened.date().isoformat()] = (opened, opened + timedelta(days=1))
        adv, bar_sources = _daily_adv(self.store, bar_ids, instrument, at, expected_sessions=wanted)

        def normalized(captured):
            risk = InstrumentRisk(
                instrument=instrument,
                sector="crypto",
                adv_base=adv * fx,
                as_of=captured,
                lot_size=market.get("risk_step", market["step"]),
                min_notional=market["minimum_notional"] * fx,
            )
            rules = VenueRules(
                venue="paper",
                instrument=instrument,
                symbol=instrument.symbol,
                available_at=captured,
                quantity_step=market["step"],
                price_tick=market["tick"],
                minimum_quantity=market["minimum_quantity"],
                minimum_notional=market["minimum_notional"],
                maximum_quantity=market.get("maximum_quantity"),
                order_types=("market",),
                time_in_force=("GTC",),
                market_type="spot",
                active=market["active"],
                **market.get("rules_extra", {}),
            )
            return dict(
                instrument=instrument.model_dump(mode="json"),
                active=market["active"],
                instrument_metadata=[risk.model_dump(mode="json")],
                venue_rules=[rules.model_dump(mode="json")],
                market_id=market["id"],
                adv_sessions=sorted(wanted),
                adv_convention="mean-of-20-completed-UTC-day-quote-turnover-times-observed-FX",
                sector_convention="declared-crypto-asset-class-risk-group",
                rule_policy=market.get("rule_policy", "paper-market-only-native-spot-increments-1"),
                **(
                    {"native_filters": market["native_filters"]}
                    if "native_filters" in market
                    else {}
                ),
            )

        proof = self.derived(
            context,
            [wire, *extra_sources, *bar_sources, *fx_sources],
            normalized,
            kind="operating_reference",
            logical_key=instrument.instrument_id,
            instrument=instrument,
        )
        return self.result([wire, *extra_sources, proof])


class KrakenOperatingAdapter(_CryptoOperatingAdapter):
    """Exact public REST fields, including VWAP and individual trade IDs."""

    venue = "kraken"

    def __init__(self, store, *, http=None, clock=None):
        super().__init__(store, clock=clock)
        self.http = http or ReadHTTP("kraken", "https://api.kraken.com")

    def _get(self, operation, context, instrument, params):
        if operation not in ("AssetPairs", "OHLC", "Trades"):
            _fail("unsupported_kraken_operation")
        wire, value = self.wire(
            self.http,
            "/0/public/" + operation,
            context,
            upstream="kraken-spot-rest-v0",
            params=params,
            instrument=instrument,
        )
        if value.get("error") or not isinstance(value.get("result"), dict):
            _fail("kraken_provider_error_wire_archived")
        return wire, value["result"]

    @_operation
    def reference(self, instrument, context, *, market_id, bar_ids, fx_evidence_id=None):
        _crypto_identity(instrument, self.venue)
        wire, values = self._get("AssetPairs", context, instrument, dict(pair=market_id))
        if set(values) != {market_id}:
            _fail("kraken_market_identity_mismatch")
        row = values[market_id]
        if (
            row.get("wsname") != instrument.symbol
            or row.get("aclass_base") != "currency"
            or row.get("lot") != "unit"
            or row.get("lot_multiplier") != 1
        ):
            _fail("kraken_spot_identity_mismatch")
        if instrument.symbol.split("/")[-1] != instrument.quote_currency:
            _fail("quote_currency_identity_mismatch")
        decimals = int(row["lot_decimals"])
        if not 0 <= decimals <= 18:
            _fail("unsupported_quantity_precision")
        market = dict(
            id=market_id,
            step=Decimal(10) ** (-decimals),
            tick=_decimal(row["tick_size"]),
            minimum_quantity=_decimal(row["ordermin"]),
            minimum_notional=_decimal(row["costmin"]),
            active=row.get("status") == "online",
        )
        return self._reference(
            instrument, context, wire, market, bar_ids, fx_evidence_id=fx_evidence_id
        )

    @_operation
    def bars(self, instrument, context, *, market_id, start, end):
        _crypto_identity(instrument, self.venue)
        start, end = _window(start, end, context)
        wire, values = self._get(
            "OHLC",
            context,
            instrument,
            dict(pair=market_id, interval=1440, since=pd.Timestamp(start).value // 1_000_000_000),
        )
        if set(values) - {"last"} != {market_id}:
            _fail("kraken_market_identity_mismatch")
        evidence = [wire]
        rows = values[market_id]
        if not isinstance(rows, list) or not rows:
            _fail("daily_bar_coverage_missing")
        # Kraken always includes a final uncommitted candle. Never use it.
        for row in rows[:-1]:
            opened = _epoch(row[0], 1_000_000_000)
            closed = opened + timedelta(days=1)
            if opened < start or closed > end:
                continue
            if opened.time() != datetime.min.time():
                _fail("daily_bar_boundary_mismatch")
            evidence.append(
                _market_record(
                    self,
                    context,
                    wire,
                    instrument,
                    opened=opened,
                    closed=closed,
                    values=dict(
                        zip(
                            ("open", "high", "low", "close", "vwap", "volume"),
                            row[1:7],
                            strict=True,
                        )
                    ),
                    calendar="24/7",
                    session=opened.date().isoformat(),
                )
            )
        expected = (end.date() - start.date()).days
        if len(evidence) - 1 != expected:
            _fail("kraken_history_truncated_or_missing_sessions")
        return self.result(evidence)

    @_operation
    def trades(
        self,
        instrument,
        context,
        *,
        market_id,
        start,
        end,
        calendar_id,
        max_pages=100,
        fx_evidence_id=None,
    ):
        _crypto_identity(instrument, self.venue)
        start, end = _window(start, end, context)
        if not 1 <= max_pages <= 1000:
            _fail("invalid_page_budget")
        cursor = str(pd.Timestamp(start).value)
        sources, trades, seen = [], [], set()
        reached = False
        for _ in range(max_pages):
            wire, values = self._get(
                "Trades", context, instrument, dict(pair=market_id, since=cursor, count=1000)
            )
            sources.append(wire)
            if set(values) - {"last"} != {market_id}:
                _fail("kraken_market_identity_mismatch")
            rows = values[market_id]
            for row in rows:
                if len(row) < 7:
                    _fail("individual_trade_id_required")
                at = _epoch(row[2], 1_000_000_000)
                if at >= end:
                    reached = True
                if start <= at < end and row[6] not in seen:
                    seen.add(row[6])
                    trades.append(
                        (
                            row[6],
                            pd.Timestamp(
                                int(Decimal(str(row[2])) * 1_000_000_000), unit="ns", tz="UTC"
                            ).isoformat(),
                            row[0],
                            row[1],
                        )
                    )
            next_cursor = str(values["last"])
            if reached:
                break
            if int(next_cursor) <= int(cursor) or not rows:
                _fail("trade_window_not_fully_observed")
            cursor = next_cursor
        if not reached:
            _fail("pagination_budget_exhausted")
        earliest = min((_instant(t[1]) for t in trades), default=start)
        fx, fx_sources = _fx(self, instrument, context, fx_evidence_id, at=earliest)
        proof = _print_capture(
            self,
            instrument,
            context,
            sources,
            trades,
            calendar=_calendar(self.store, calendar_id, self.now()),
            fx_rate=fx,
            fx_sources=fx_sources,
            convention="kraken-individual-spot-print-1",
        )
        return self.result([*sources, proof])


class BinanceOperatingAdapter(_CryptoOperatingAdapter):
    """Binance public spot market data. Aggregate trades are never opening size."""

    venue = "binance"

    def __init__(self, store, *, http=None, clock=None, network="mainnet"):
        from src.data.binance_reference import BINANCE_REFERENCE_ORIGINS

        if network not in BINANCE_REFERENCE_ORIGINS:
            _fail("explicit_binance_network_required")
        super().__init__(store, clock=clock)
        self.network = network
        self.http = http or ReadHTTP("binance", BINANCE_REFERENCE_ORIGINS[network][0])

    def _get(self, operation, context, instrument, params):
        if operation not in ("exchangeInfo", "klines", "trades"):
            _fail("unsupported_binance_operation")
        wire, value = self.wire(
            self.http,
            "/api/v3/" + operation,
            context,
            upstream="binance-spot-v3",
            params=params,
            instrument=instrument,
        )
        if isinstance(value, dict) and "code" in value:
            _fail("binance_provider_error_wire_archived")
        return wire, value

    @staticmethod
    def _symbol(instrument):
        if (
            not re.fullmatch(r"[A-Z0-9]+/[A-Z0-9]+", instrument.symbol)
            or instrument.symbol.split("/")[-1] != instrument.quote_currency
        ):
            _fail("canonical_binance_spot_symbol_required")
        return instrument.symbol.replace("/", "")

    def _notional_sources(self, instrument, context, average_minutes):
        from src.data.binance_reference import (
            parse_binance_notional_payload,
            parse_binance_notional_reference,
        )

        symbol = self._symbol(instrument)

        def fetch(operation, params):
            return self.wire(
                self.http,
                "/api/v3/" + operation,
                context,
                upstream="binance-spot-v3",
                params=params,
                instrument=instrument,
                timestamp_unit="MILLISECOND",
                allowed_error_statuses=(400,) if operation == "referencePrice" else (),
            )

        source, _ = fetch("referencePrice", {"symbol": symbol})
        sources = [source]
        ids = {}
        native = parse_binance_notional_payload(
            self.store.raw(source.record.evidence_id),
            operation="referencePrice",
            symbol=symbol,
            timestamp_unit="MILLISECOND",
        )
        if native.kind == "reference-absent":
            operation = "avgPrice" if average_minutes else "trades"
            params = {"symbol": symbol, **({} if average_minutes else {"limit": 1})}
            fallback, _ = fetch(operation, params)
            sources.append(fallback)
            ids["average_evidence_id" if average_minutes else "last_trade_evidence_id"] = (
                fallback.record.evidence_id
            )
        actual = context.model_copy(update={"as_of": self.now()})
        reference = parse_binance_notional_reference(
            self.store,
            instrument,
            actual,
            reference_evidence_id=source.record.evidence_id,
            average_minutes=average_minutes,
            network=self.network,
            **ids,
        )
        return reference, sources

    @_operation
    def notional_reference(self, instrument, context, *, average_minutes):
        """Capture only native reference/window evidence, without bar/FX inputs.

        Returned raw envelopes are the exact IDs for freeze_binance_reference;
        the caller's subsequent actual cutoff must include these receipts.
        """
        _crypto_identity(instrument, self.venue)
        if type(average_minutes) is not int or average_minutes < 0:
            _fail("native_notional_window_invalid")
        _, sources = self._notional_sources(instrument, context, average_minutes)
        return self.result(sources)

    @_operation
    def reference(self, instrument, context, *, bar_ids, fx_evidence_id=None):
        _crypto_identity(instrument, self.venue)
        symbol = self._symbol(instrument)
        wire, value = self._get("exchangeInfo", context, instrument, dict(symbol=symbol))
        rows = value.get("symbols")
        if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("symbol") != symbol:
            _fail("binance_market_identity_mismatch")
        row = rows[0]
        base, quote_currency = instrument.symbol.split("/")
        if (
            row.get("baseAsset") != base
            or row.get("quoteAsset") != quote_currency
            or row.get("isSpotTradingAllowed") is not True
            or "MARKET" not in row.get("orderTypes", [])
        ):
            _fail("binance_spot_identity_mismatch")
        filters = {f["filterType"]: f for f in row["filters"]}
        if len(filters) != len(row["filters"]):
            _fail("conflicting_duplicate_binance_filters")
        lot = filters["LOT_SIZE"]
        market_lot = filters.get("MARKET_LOT_SIZE", dict(stepSize="0", minQty="0", maxQty="0"))
        step = _decimal(lot["stepSize"])
        extra = _decimal(market_lot["stepSize"], positive=False)
        from decimal import localcontext
        from math import lcm

        exponent = (
            min(step.as_tuple().exponent, extra.as_tuple().exponent)
            if extra
            else step.as_tuple().exponent
        )
        with localcontext() as arithmetic:
            arithmetic.prec = max(
                40, len(step.as_tuple().digits) + len(extra.as_tuple().digits) + abs(exponent) + 4
            )
            unit = Decimal(10) ** exponent
            risk_step = Decimal(lcm(int(step / unit), int(extra / unit))) * unit if extra else step
        notionals, windows, maxima = [], set(), []
        for name in ("MIN_NOTIONAL", "NOTIONAL"):
            if name not in filters:
                continue
            native = filters[name]
            flags = (
                ("applyToMarket",)
                if name == "MIN_NOTIONAL"
                else ("applyMinToMarket", "applyMaxToMarket")
            )
            if (
                any(type(native.get(flag)) is not bool for flag in flags)
                or type(native.get("avgPriceMins")) is not int
                or native["avgPriceMins"] < 0
            ):
                _fail("native_notional_applicability_or_window_missing")
            if native[flags[0]]:
                minimum = _decimal(native["minNotional"], positive=False)
                if minimum:
                    notionals.append(minimum)
                    windows.add(native["avgPriceMins"])
            if name == "NOTIONAL" and native["applyMaxToMarket"]:
                maxima.append(_decimal(native["maxNotional"]))
                windows.add(native["avgPriceMins"])
        if len(windows) > 1:
            _fail("independent_native_notional_windows_not_representable")
        reference, sources = (None, [])
        if windows:
            reference, sources = self._notional_sources(instrument, context, next(iter(windows)))
        price_filter = filters["PRICE_FILTER"]

        def optional(value):
            return _decimal(value, positive=False) or None

        extras = dict(
            maximum_notional=min(maxima) if maxima else None,
            market_notional_average_minutes=next(iter(windows)) if windows else None,
            market_notional_reference=reference,
            minimum_price=optional(price_filter["minPrice"]),
            maximum_price=optional(price_filter["maxPrice"]),
            market_quantity_step=extra or None,
            market_minimum_quantity=optional(market_lot["minQty"]),
            market_maximum_quantity=optional(market_lot["maxQty"]),
        )
        market = dict(
            id=symbol,
            step=step,
            risk_step=risk_step,
            tick=_decimal(price_filter["tickSize"]),
            minimum_quantity=_decimal(lot["minQty"], positive=False),
            maximum_quantity=optional(lot["maxQty"]),
            minimum_notional=max(notionals) if notionals else Decimal(0),
            active=row.get("status") == "TRADING",
            rules_extra=extras,
            native_filters=row["filters"],
            rule_policy="paper-market-only-lot-price-notional-intersection-1",
        )
        return self._reference(
            instrument,
            context,
            wire,
            market,
            bar_ids,
            fx_evidence_id=fx_evidence_id,
            extra_sources=sources,
        )

    @_operation
    def bars(self, instrument, context, *, start, end, max_pages=100):
        _crypto_identity(instrument, self.venue)
        start, end = _window(start, end, context)
        cursor = pd.Timestamp(start).value // 1_000_000
        end_ms = pd.Timestamp(end).value // 1_000_000
        if (
            not 1 <= max_pages <= 1000
            or start.time() != datetime.min.time()
            or end.time() != datetime.min.time()
        ):
            _fail("bounded_complete_UTC_days_required")
        evidence, observed = [], set()
        for _ in range(max_pages):
            wire, rows = self._get(
                "klines",
                context,
                instrument,
                dict(
                    symbol=self._symbol(instrument),
                    interval="1d",
                    startTime=cursor,
                    endTime=end_ms - 1,
                    limit=1000,
                    timeZone="0",
                ),
            )
            evidence.append(wire)
            if not isinstance(rows, list) or not rows:
                _fail("daily_bar_coverage_missing")
            newest = cursor
            for row in rows:
                opened = _epoch(row[0], 1_000_000)
                closed = opened + timedelta(days=1)
                newest = max(newest, int(row[0]) + 86400000)
                if opened < start or closed > end:
                    continue
                if (
                    int(row[6]) != pd.Timestamp(closed).value // 1_000_000 - 1
                    or opened.time() != datetime.min.time()
                ):
                    _fail("daily_bar_boundary_mismatch")
                if row[0] in observed:
                    _fail("duplicate_daily_bar")
                observed.add(row[0])
                volume = _decimal(row[5])
                vwap = _decimal(row[7]) / volume  # Actual quote turnover, not close*volume.
                evidence.append(
                    _market_record(
                        self,
                        context,
                        wire,
                        instrument,
                        opened=opened,
                        closed=closed,
                        values=dict(
                            open=row[1],
                            high=row[2],
                            low=row[3],
                            close=row[4],
                            volume=volume,
                            vwap=vwap,
                        ),
                        calendar="24/7",
                        session=opened.date().isoformat(),
                    )
                )
            if newest <= cursor:
                _fail("pagination_did_not_advance")
            cursor = newest
            if cursor >= end_ms:
                break
        else:
            _fail("pagination_budget_exhausted")
        if len(observed) != (end - start).days:
            _fail("missing_daily_sessions")
        return self.result(evidence)

    @_operation
    def trades(self, instrument, context, *, start, end, calendar_id, fx_evidence_id=None):
        _crypto_identity(instrument, self.venue)
        start, end = _window(start, end, context)
        wire, rows = self._get(
            "trades", context, instrument, dict(symbol=self._symbol(instrument), limit=1000)
        )
        if not isinstance(rows, list) or not rows:
            _fail("individual_trade_coverage_missing")
        times = [_epoch(r["time"], 1_000_000) for r in rows]
        # Public recent-trades has no cursor. Require an actual bracket for the
        # requested window rather than pretending its latest 1000 are history.
        if min(times) > start or max(times) < end:
            _fail("recent_trade_window_not_fully_observed")
        trades = [
            (r["id"], _epoch(r["time"], 1_000_000).isoformat(), r["price"], r["qty"])
            for r, at in zip(rows, times, strict=True)
            if start <= at < end and r.get("isBestMatch") is True
        ]
        earliest = min((_instant(t[1]) for t in trades), default=start)
        fx, fx_sources = _fx(self, instrument, context, fx_evidence_id, at=earliest)
        proof = _print_capture(
            self,
            instrument,
            context,
            [wire],
            trades,
            calendar=_calendar(self.store, calendar_id, self.now()),
            fx_rate=fx,
            fx_sources=fx_sources,
            convention="binance-individual-spot-print-1",
        )
        return self.result([wire, proof])


def economic_sources(
    store: EvidenceStore,
    snapshot: MarketSnapshot,
    *,
    reference_ids: tuple[str, ...],
    universe_ids: tuple[str, ...],
    calendar_ids: tuple[str, ...],
    print_ids: tuple[str, ...],
    action_ids: tuple[str, ...],
    context: RunContext,
    clock: Callable[[], datetime] | None = None,
):
    """Return exact eight `(kind, tuple[SourceBinding])` arrays for H.prepare.

    No synthetic actions/calendar/liquidity/metadata are supplied. Selections
    retain original source availability, including prior captured calendars.
    Current metadata/universe cannot authorize an earlier execution admission.
    """
    from src.domain.portfolio import (
        CorporateAction,
        ExecutionBar,
        InstrumentRisk,
        OpeningEvent,
        UniverseEligibility,
    )
    from src.execution.venue import VenueRules
    from src.research.evidence import SourceBinding

    capture = OperatingCapture(store, clock=clock)
    at = capture.now()
    if snapshot.context.as_of > at or context.as_of > at:
        _fail("bundle_clock_precedes_source_context")
    identities = {b.instrument.instrument_id: b.instrument for b in snapshot.bars}
    if not print_ids:
        _fail("explicit_opening_event_sources_required")
    print_values = [verify_operating_capture(store, key, as_of=at) for key in print_ids]
    current = all(
        "opening_events" in value and "execution_bars" not in value for value in print_values
    )
    legacy = all(
        "execution_bars" in value
        and "opening_events" not in value
        and store.get(key).record.provider_version == "operating-wire-1"
        for key, value in zip(print_ids, print_values, strict=True)
    )
    if not (current or legacy):
        _fail("mixed_or_unsupported_opening_source_versions")
    execution_kind = "opening_events" if current else "execution_bars"
    kinds_schema = KINDS if current else LEGACY_KINDS
    bindings = {kind: [] for kind in kinds_schema}
    selected = {kind: [] for kind in kinds_schema}
    groups = [
        (reference_ids, ("instrument_metadata", "venue_rules")),
        (universe_ids, ("universe_history",)),
        (calendar_ids, ("calendar",)),
        (print_ids, (execution_kind, "opening_liquidity")),
        (action_ids, ("corporate_actions",)),
    ]
    all_sources = []
    for ids, kinds in groups:
        if not ids:
            _fail("explicit_" + kinds[0] + "_sources_required")
        for key in ids:
            value = verify_operating_capture(store, key, as_of=at)
            envelope = store.get(key, as_of=at)
            if kinds == ("corporate_actions",) and (
                envelope.kind != "operating_actions"
                or not isinstance(value.get("coverage"), dict)
                or value["coverage"].get("types") != "all"
            ):
                _fail("explicit_observed_action_scope_required")
            all_sources.append(envelope)
            for kind in kinds:
                if kind not in value or not isinstance(value[kind], list):
                    _fail("source_array_missing_" + kind)
                for i, row in enumerate(value[kind]):
                    # Calendar arrays may include future/past sessions. Bind only
                    # exact sessions used by actual selected execution prints.
                    selected[kind].append(
                        (
                            row,
                            SourceBinding(
                                evidence_id=key,
                                pointer=f"/{kind}/{i}",
                                available_at=envelope.record.available_at,
                            ),
                        )
                    )
                if not value[kind]:
                    bindings[kind].append(
                        SourceBinding(
                            evidence_id=key,
                            pointer="/" + kind,
                            available_at=envelope.record.available_at,
                        )
                    )
    contract = OpeningEvent if current else ExecutionBar
    bars = [contract.model_validate(row) for row, _ in selected[execution_kind]]
    if not bars or len({b.event_id for b in bars}) != len(bars):
        _fail("unique_actual_execution_events_required")
    for b in bars:
        if identities.get(b.instrument.instrument_id) != b.instrument:
            _fail("execution_market_identity_mismatch")

    def ordering(pair):
        return (pair[0]["opened_at"], pair[0]["instrument"]["instrument_id"], pair[0]["event_id"])

    selected[execution_kind].sort(key=ordering)
    selected["opening_liquidity"].sort(key=ordering)
    expected = [
        {k: row[k] for k in ("event_id", "instrument", "opened_at", "opening_volume", "fx_rate")}
        for row, _ in selected[execution_kind]
    ]
    if [r for r, _ in selected["opening_liquidity"]] != expected:
        _fail("opening_liquidity_differs_from_actual_prints")
    calendar_needed = {
        canonical_bytes(
            {
                k: b.model_dump(mode="json")[k]
                for k in (
                    "calendar",
                    "calendar_version",
                    "session",
                    "session_open",
                    "session_close",
                )
            }
        )
        for b in bars
    }
    calendar_rows = {}
    for row, binding in selected["calendar"]:
        encoded = canonical_bytes(row)
        if encoded in calendar_needed:
            calendar_rows.setdefault(encoded, (row, binding))
    if calendar_rows.keys() != calendar_needed:
        _fail("execution_calendar_source_mismatch")
    selected["calendar"] = [calendar_rows[key] for key in sorted(calendar_rows)]
    metadata = [InstrumentRisk.model_validate(r) for r, _ in selected["instrument_metadata"]]
    rules = [VenueRules.model_validate(r) for r, _ in selected["venue_rules"]]
    if {m.instrument.instrument_id for m in metadata} != set(identities) or len(metadata) != len(
        identities
    ):
        _fail("one_current_metadata_per_market_instrument_required")
    if {r.instrument.instrument_id for r in rules} != set(identities) or len(rules) != len(
        identities
    ):
        _fail("one_current_rule_per_market_instrument_required")
    for row, binding in selected["instrument_metadata"]:
        if binding.available_at > _instant(row["as_of"]):
            _fail("metadata_backdated_before_capture")
    for row, binding in selected["universe_history"]:
        universe = UniverseEligibility.model_validate(row)
        if (
            set(universe.instrument_ids) - identities.keys()
            or binding.available_at > universe.available_at
        ):
            _fail("universe_backdated_or_unknown_identity")
    for row, _ in selected["corporate_actions"]:
        CorporateAction.model_validate(row)
    action_scope = set()
    for key in action_ids:
        action_scope.update(store.normalized(key, as_of=at)["coverage"].get("instrument_ids", ()))
    if action_scope != set(identities):
        _fail("corporate_action_scope_missing_or_unrelated_instruments")
    # Snapshot evidence order is authoritative; there is no independent sort.
    audit = capture.derived(
        context,
        [*all_sources, *(store.get(r.evidence_id) for r in snapshot.evidence)],
        lambda now: dict(availability_audit=[r.model_dump(mode="json") for r in snapshot.evidence]),
        kind="operating_audit",
        logical_key=snapshot.content_hash(),
    )
    bindings["availability_audit"] = [
        SourceBinding(
            evidence_id=audit.record.evidence_id,
            pointer="/availability_audit",
            available_at=audit.record.available_at,
        )
    ]
    for kind in kinds_schema:
        if kind != "availability_audit":
            bindings[kind].extend(binding for _, binding in selected[kind])
    return tuple((kind, tuple(bindings[kind])) for kind in kinds_schema)
