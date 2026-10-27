"""Bounded HTTP observation transport and uniform provenance ingestion."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import httpx

from src.data.evidence.store import EvidenceEnvelope, EvidenceStore, canonical_bytes, digest
from src.domain.common import EvidenceRecord, Instrument, RunContext, _utc


class ProviderError(RuntimeError):
    """Redacted, explicit failure; never implies empty/neutral successful data."""

    def __init__(self, provider: str, code: str, status: int | None = None):
        self.provider, self.code, self.status = provider, code, status
        super().__init__(f"{provider}: {code}" + (f" (HTTP {status})" if status else ""))


@dataclass(frozen=True)
class FetchResult:
    status: str
    evidence: tuple[EvidenceEnvelope, ...]
    detail: str = ""


class ReadHTTP:
    """GET-only transport with bounded decoded responses, retries and rate."""

    def __init__(self, provider: str, base_url: str, *, client: httpx.Client | None = None,
                 max_requests: int = 100, attempts: int = 3, timeout: float = 20,
                 min_interval: float = 0.12, sleep: Callable = time.sleep,
                 maximum_response_bytes: int = 8 * 1024 * 1024):
        if max_requests < 1 or not 1 <= attempts <= 5 or timeout <= 0 or min_interval < 0:
            raise ValueError("invalid HTTP limits")
        if type(maximum_response_bytes) is not int or maximum_response_bytes < 1:
            raise ValueError("maximum_response_bytes must be a positive integer")
        self.provider, self.base_url = provider, base_url.rstrip("/")
        self._owned_client = client is None
        self.client = client or httpx.Client(follow_redirects=False)
        self.max_requests, self.attempts, self.timeout = max_requests, attempts, timeout
        self.min_interval, self.sleep = min_interval, sleep
        self.maximum_response_bytes = maximum_response_bytes
        self.requests = 0
        self._last = 0.0

    def close(self):
        if self._owned_client:
            self.client.close()

    def get(self, path: str, *, params: dict | None = None, headers: dict | None = None,
            allowed_error_statuses: tuple[int, ...] = ()) -> bytes:
        # A closed adapter may archive a documented native error as evidence
        # (e.g. Binance -2043). Authentication/entitlement/rate failures cannot
        # be turned into successful source reads by this opt-in.
        if any(type(status) is not int or status not in (400, 404) for status in allowed_error_statuses):
            raise ValueError("only explicit native client-error bodies may be archived")
        if not path.startswith("/") or path.startswith("//") or ".." in path or "?" in path:
            raise ProviderError(self.provider, "invalid_path")
        for attempt in range(self.attempts):
            if self.requests >= self.max_requests:
                raise ProviderError(self.provider, "request_budget_exhausted")
            delay = max(0, self.min_interval - (time.monotonic() - self._last))
            if delay:
                self.sleep(delay)
            self.requests += 1
            self._last = time.monotonic()
            try:
                with self.client.stream("GET", self.base_url + path, params=params,
                                        headers=headers, timeout=self.timeout) as response:
                    status = response.status_code
                    if 200 <= status < 300 or status in allowed_error_statuses:
                        body = bytearray()
                        # Content-Length may describe compressed bytes or be
                        # absent. Count actual decoded bytes before retaining
                        # each chunk; never archive a truncated success body.
                        for chunk in response.iter_bytes():
                            if len(chunk) > self.maximum_response_bytes - len(body):
                                raise ProviderError(self.provider, "response_bytes_exceeded")
                            body.extend(chunk)
                        return bytes(body)
                    if status not in (429, 500, 502, 503, 504):
                        code = "credentials_missing_or_rejected" if status == 401 else "entitlement_or_access_denied" if status == 403 else "http_failure"
                        raise ProviderError(self.provider, code, status)
                    if attempt + 1 == self.attempts:
                        raise ProviderError(self.provider, "rate_limited" if status == 429 else "provider_unavailable", status)
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt + 1 == self.attempts:
                    raise ProviderError(self.provider, "transport_unavailable") from None
            self.sleep(min(2 ** attempt, 4))
        raise ProviderError(self.provider, "provider_unavailable")


def store_observation(store: EvidenceStore, *, raw: bytes, normalized: object,
                      provider: str, version: str, source_uri: str, kind: str,
                      logical_key: str, context: RunContext, ingested_at: datetime,
                      instrument: Instrument | None = None, observed_at: datetime | None = None,
                      available_at: datetime | None = None, published_at: datetime | None = None,
                      revised_at: datetime | None = None, revision_id: str | None = None,
                      document: str | None = None, quality: str = "usable", units: str | None = None,
                      adjustment_policy: str | None = None, tool_name: str | None = None,
                      tool_version: str | None = None, supersedes_id: str | None = None) -> EvidenceEnvelope:
    """Capture availability defaults to receipt, NEVER to the event's date.

    An earlier ``available_at`` is only for an adapter with documented vintage
    metadata or an archival manifest. Present-day historical prices/statements
    are not automatically point-in-time data for earlier decisions.
    """
    ingested_at = _utc(ingested_at)
    available_at = _utc(available_at) if available_at else ingested_at
    identity = dict(provider=provider, version=version, raw=digest(raw), key=logical_key,
                    kind=kind, normalized=normalized, cutoff=context.as_of.isoformat(),
                    capture=ingested_at.isoformat(), available=available_at.isoformat())
    record = EvidenceRecord(evidence_id=digest(canonical_bytes(identity)), provider=provider,
                            provider_version=version, source_uri=source_uri, raw_hash=digest(raw),
                            ingested_at=ingested_at, available_at=available_at, observed_at=observed_at,
                            requested_as_of=context.as_of, availability_basis="documented", quality=quality,
                            instrument_ids=(instrument.instrument_id,) if instrument else (),
                            revision_id=revision_id, units=units,
                            currency=instrument.quote_currency if instrument else None,
                            adjustment_policy=adjustment_policy, tool_name=tool_name, tool_version=tool_version)
    return store.put(record, raw, kind=kind, logical_key=logical_key, document=document,
                     normalized=normalized, published_at=published_at, revised_at=revised_at,
                     supersedes_id=supersedes_id)
