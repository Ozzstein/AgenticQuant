"""FMP stable API and vintage-aware FRED/ALFRED observations.

FMP call/statement dates alone do not attest when this revision was published.
Those responses are usable from capture time. ALFRED's day-resolution vintage
starts use the END of that US/Eastern day as a conservative availability bound.
"""
from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from src.data.adapters.base import FetchResult, ProviderError, ReadHTTP, store_observation
from src.data.evidence.store import EvidenceStore, canonical_bytes
from src.domain.common import Instrument, RunContext, _utc


def json_payload(raw: bytes, provider: str):
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise ProviderError(provider, "malformed_json") from None
    if isinstance(payload, dict) and any(k in payload for k in ("Error Message", "error", "error_message")):
        raise ProviderError(provider, "provider_error_payload")
    return payload


class FMPAdapter:
    VERSION = "fmp-stable-capture-v1"
    PATHS = {
        "income_statement": "/income-statement", "balance_sheet": "/balance-sheet-statement",
        "cash_flow": "/cash-flow-statement", "ratios": "/ratios",
        "analyst_estimates": "/analyst-estimates", "insider": "/insider-trading/search",
        "institutional": "/institutional-ownership/latest", "earnings": "/earnings",
        "filings": "/sec-filings-search/symbol", "transcript": "/earning-call-transcript",
    }

    def __init__(self, store: EvidenceStore, api_key: str | None, *, http: ReadHTTP | None = None):
        self.store, self._key = store, api_key
        self.http = http or ReadHTTP("fmp", "https://financialmodelingprep.com/stable")

    def fetch(self, capability: str, instrument: Instrument, context: RunContext, *,
              ingested_at: datetime, year: int | None = None, quarter: int | None = None,
              limit: int = 100) -> FetchResult:
        if not self._key:
            raise ProviderError("fmp", "credentials_missing")
        if capability not in self.PATHS or not 1 <= limit <= 1000:
            raise ProviderError("fmp", "unsupported_capability_or_limit")
        params = {"symbol": instrument.symbol, "apikey": self._key, "limit": limit}
        if capability == "transcript":
            if year is None or quarter not in (1, 2, 3, 4):
                raise ProviderError("fmp", "explicit_fiscal_year_and_quarter_required")
            params.update(year=year, quarter=quarter)
        path = self.PATHS[capability]
        raw = self.http.get(path, params=params)
        payload = json_payload(raw, "fmp")
        if not isinstance(payload, list):
            raise ProviderError("fmp", "unexpected_response_shape")
        records = []
        for item in payload:
            if not isinstance(item, dict):
                raise ProviderError("fmp", "malformed_item")
            if item.get("symbol", instrument.symbol) != instrument.symbol:
                raise ProviderError("fmp", "instrument_mismatch")
            document = item.get("content") if capability == "transcript" else canonical_bytes(item).decode()
            if capability == "transcript" and (not isinstance(document, str) or not document):
                raise ProviderError("fmp", "missing_transcript_content")
            key = f"{instrument.instrument_id}:{capability}:{year or item.get('date', '')}:{quarter or item.get('period', '')}"
            records.append(store_observation(self.store, raw=raw, normalized=item, provider="fmp",
                           version=self.VERSION, source_uri=self.http.base_url + path, kind=capability,
                           logical_key=key, context=context, ingested_at=ingested_at,
                           instrument=instrument, document=document, revision_id=item.get("accessionNumber")))
        return FetchResult("available" if records else "missing", tuple(records),
                           "capture-time availability; FMP event dates do not establish historical revision availability")


class FredAdapter:
    VERSION = "fred-alfred-vintage-v1"

    def __init__(self, store: EvidenceStore, api_key: str | None, *, http: ReadHTTP | None = None):
        self.store, self._key = store, api_key
        self.http = http or ReadHTTP("fred", "https://api.stlouisfed.org/fred")

    def observations(self, series_id: str, context: RunContext, *, ingested_at: datetime,
                     start: str, end: str, units: str = "lin") -> FetchResult:
        if not self._key:
            raise ProviderError("fred", "credentials_missing")
        if units != "lin":
            raise ProviderError("fred", "ingest_native_units_before_transforming")
        records, offset = [], 0
        while True:
            raw = self.http.get("/series/observations", params={"api_key": self._key, "file_type": "json",
                                "series_id": series_id, "observation_start": start, "observation_end": end,
                                "realtime_start": "1776-07-04", "realtime_end": context.as_of.date().isoformat(),
                                "output_type": 1, "units": units, "offset": offset, "limit": 100000})
            payload = json_payload(raw, "fred")
            if not isinstance(payload, dict) or not isinstance(payload.get("observations"), list):
                raise ProviderError("fred", "unexpected_response_shape")
            items = payload["observations"]
            for item in items:
                try:
                    vintage = datetime.fromisoformat(item["realtime_start"]).date()
                    available = datetime.combine(vintage + timedelta(days=1), time.min, ZoneInfo("America/New_York"))
                    observed = datetime.fromisoformat(item["date"]).replace(tzinfo=ZoneInfo("America/New_York"))
                    value = None if item["value"] == "." else float(item["value"])
                    normalized = dict(series_id=series_id, date=item["date"], value=value,
                                      realtime_start=item["realtime_start"], realtime_end=item["realtime_end"])
                except (KeyError, TypeError, ValueError):
                    raise ProviderError("fred", "malformed_observation") from None
                records.append(store_observation(self.store, raw=raw, normalized=normalized, provider="fred",
                               version=self.VERSION, source_uri=self.http.base_url + "/series/observations",
                               kind="macro", logical_key=f"{series_id}:{item['date']}", context=context,
                               ingested_at=ingested_at, observed_at=_utc(observed), available_at=_utc(available),
                               revision_id=item["realtime_start"], quality="missing" if value is None else "usable",
                               units="native-series-units"))
            offset += len(items)
            if offset >= payload.get("count", offset):
                break
            if not items:
                raise ProviderError("fred", "incomplete_pagination")
        return FetchResult("available" if records else "missing", tuple(records))
