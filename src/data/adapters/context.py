"""Finnhub news, SEC immutable filings, CoinGecko context and Alpaca GET-only data."""
from __future__ import annotations

import re
from datetime import datetime

from src.data.adapters.base import FetchResult, ProviderError, ReadHTTP, store_observation
from src.data.adapters.fundamentals import json_payload
from src.data.evidence.store import EvidenceStore
from src.domain.common import Instrument, RunContext


class ContextAdapter:
    def __init__(self, store: EvidenceStore, provider: str, http: ReadHTTP):
        self.store, self.provider, self.http = store, provider, http

    def capture(self, path: str, context: RunContext, *, ingested_at: datetime, capability: str,
                instrument: Instrument | None = None, params: dict | None = None,
                headers: dict | None = None) -> FetchResult:
        raw = self.http.get(path, params=params, headers=headers)
        payload = json_payload(raw, self.provider)
        if payload is None or payload == [] or payload == {}:
            return FetchResult("missing", ())
        item = store_observation(self.store, raw=raw, normalized=payload, provider=self.provider,
                                 version="direct-capture-v1", source_uri=self.http.base_url + path,
                                 kind=capability, logical_key=f"{instrument.instrument_id if instrument else self.provider}:{capability}",
                                 context=context, ingested_at=ingested_at, instrument=instrument,
                                 document=raw.decode("utf-8"))
        return FetchResult("available", (item,), "capture-time context; no historical publication claim")


class FinnhubAdapter(ContextAdapter):
    def __init__(self, store: EvidenceStore, api_key: str | None, *, http: ReadHTTP | None = None):
        super().__init__(store, "finnhub", http or ReadHTTP("finnhub", "https://finnhub.io/api/v1"))
        self._key = api_key

    def news(self, instrument: Instrument, context: RunContext, *, start: str, end: str,
             ingested_at: datetime) -> FetchResult:
        if not self._key:
            raise ProviderError("finnhub", "credentials_missing")
        return self.capture("/company-news", context, ingested_at=ingested_at, capability="news",
                            instrument=instrument, params={"symbol": instrument.symbol, "from": start, "to": end},
                            headers={"X-Finnhub-Token": self._key})


class CoinGeckoAdapter(ContextAdapter):
    def __init__(self, store: EvidenceStore, *, http: ReadHTTP | None = None,
                 api_key: str | None = None, tier: str = "demo", clock=None):
        from src.data.adapters.coingecko_context import ORIGINS
        if tier not in ORIGINS:
            raise ProviderError("coingecko", "explicit_demo_or_pro_tier_required")
        super().__init__(store, "coingecko", http or ReadHTTP("coingecko", ORIGINS[tier]))
        self.tier, self.clock = tier, clock
        self._cg_headers = {"x-cg-pro-api-key" if tier == "pro" else "x-cg-demo-api-key": api_key} if api_key else {}

    def current_dominance(self, context: RunContext) -> FetchResult:
        """Native dated BTC percentage, with measured capture availability."""
        from src.data.adapters.coingecko_context import current_dominance
        return current_dominance(self, context)

    def dominance_history(self, context: RunContext, *, days: str = "30") -> FetchResult:
        """Pro global/BTC caps joined only at equal native timestamps."""
        from src.data.adapters.coingecko_context import dominance_history
        return dominance_history(self, context, days=days)

    def global_context(self, context: RunContext, *, ingested_at: datetime) -> FetchResult:
        return self.capture("/global", context, ingested_at=ingested_at, capability="crypto_global",
                            headers=self._cg_headers)

    def token(self, coin_id: str, instrument: Instrument, context: RunContext, *, ingested_at: datetime) -> FetchResult:
        if not re.fullmatch(r"[a-z0-9-]+", coin_id):
            raise ProviderError("coingecko", "invalid_canonical_coin_id")
        return self.capture(f"/coins/{coin_id}", context, ingested_at=ingested_at,
                            capability="token_context", instrument=instrument,
                            params={"localization": "false", "tickers": "false"}, headers=self._cg_headers)


class SECAdapter(ContextAdapter):
    def __init__(self, store: EvidenceStore, user_agent: str, *, http: ReadHTTP | None = None):
        if "@" not in user_agent:
            raise ProviderError("sec", "declared_contact_user_agent_required")
        super().__init__(store, "sec", http or ReadHTTP("sec", "https://data.sec.gov", min_interval=0.2))
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

    def submissions(self, cik: str, instrument: Instrument, context: RunContext, *, ingested_at: datetime) -> FetchResult:
        if not cik.isdigit() or len(cik) > 10:
            raise ProviderError("sec", "invalid_cik")
        return self.capture(f"/submissions/CIK{int(cik):010d}.json", context, ingested_at=ingested_at,
                            instrument=instrument, capability="filing_index", headers=self._headers)

    def filing(self, cik: str, accession: str, document: str, instrument: Instrument,
               context: RunContext, *, ingested_at: datetime, index_evidence_id: str,
               archives: ReadHTTP | None = None) -> FetchResult:
        """Publication comes from a stored SEC accession/acceptance entry, not caller guess."""
        if not cik.isdigit() or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) or not re.fullmatch(r"[\w.-]+", document):
            raise ProviderError("sec", "invalid_accession_or_document")
        # Index itself may be captured later; it is used only to attest the SEC
        # immutable accession. Its index availability is not assigned to filings.
        envelope = self.store.get(index_evidence_id)
        if envelope.record.provider != "sec" or envelope.kind != "filing_index" or instrument.instrument_id not in envelope.record.instrument_ids:
            raise ProviderError("sec", "invalid_index_provenance")
        import json
        index = json.loads(self.store.blob(envelope.normalized_hash))
        recent = index.get("filings", {}).get("recent", {})
        try:
            position = recent["accessionNumber"].index(accession)
            if recent["primaryDocument"][position] != document or int(index["cik"]) != int(cik):
                raise ValueError()
            accepted = datetime.fromisoformat(recent["acceptanceDateTime"][position].replace("Z", "+00:00"))
            if accepted.tzinfo is None:
                raise ValueError()
        except (KeyError, ValueError, IndexError, TypeError):
            raise ProviderError("sec", "missing_documented_acceptance") from None
        transport = archives or ReadHTTP("sec", "https://www.sec.gov", min_interval=0.2)
        path = f"/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}"
        try:
            raw = transport.get(path, headers=self._headers)
        finally:
            if archives is None:
                transport.close()
        # Preserve the complete HTML document, including tables; extraction may
        # be a separate versioned transformation, never destructive truncation.
        item = store_observation(self.store, raw=raw, normalized={"accession": accession, "document": document},
                                 document=raw.decode("utf-8"), provider="sec", version="accession-v1",
                                 source_uri=transport.base_url + path, kind="filing", logical_key=accession,
                                 context=context, ingested_at=ingested_at, available_at=accepted,
                                 published_at=accepted, observed_at=accepted, instrument=instrument, revision_id=accession)
        return FetchResult("available", (item,))


class AlpacaReadAdapter(ContextAdapter):
    PATHS = {"account": "/v2/account", "positions": "/v2/positions", "orders": "/v2/orders"}

    def __init__(self, store: EvidenceStore, api_key: str | None, secret_key: str | None,
                 *, paper: bool = True, http: ReadHTTP | None = None):
        super().__init__(store, "alpaca", http or ReadHTTP("alpaca", "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"))
        self._key, self._secret = api_key, secret_key

    def inspect(self, operation: str, context: RunContext, *, ingested_at: datetime) -> FetchResult:
        if operation not in self.PATHS:
            raise ProviderError("alpaca", "operation_not_permitted")
        if not self._key or not self._secret:
            raise ProviderError("alpaca", "credentials_missing")
        return self.capture(self.PATHS[operation], context, ingested_at=ingested_at,
                            capability=f"broker_{operation}", headers={"APCA-API-KEY-ID": self._key,
                                                                       "APCA-API-SECRET-KEY": self._secret})
