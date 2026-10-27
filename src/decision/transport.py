"""Official System One HTTP contract, verified against OpenAPI 0.2.0.

No SDK defaults, alias resolution, broker credentials or document-directed URLs.
Retries and the outer deadline belong to DecisionService, not httpx.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

ENDPOINT = "https://api.typesafe.ai/v1/systemone"


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: str
    retry_after: str | None = None


class JevHTTPTransport:
    def __init__(self, api_key: str | None = None, *, transport=None):
        self._key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")
        self._transport = transport

    @property
    def available(self) -> bool:
        return bool(self._key.strip())

    async def post(self, payload: dict, timeout: float) -> TransportResponse:
        # No redirects: never forward Authorization to an endpoint from a source.
        async with httpx.AsyncClient(
            transport=self._transport, follow_redirects=False, trust_env=False, timeout=timeout
        ) as client:
            async with client.stream(
                "POST",
                ENDPOINT,
                json=payload,
                headers={
                    "Authorization": "Bearer " + self._key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            ) as response:
                parts, length = [], 0
                async for part in response.aiter_bytes():
                    length += len(part)
                    if length > 2_000_000:
                        raise ValueError("provider response exceeds byte limit")
                    parts.append(part)
                return TransportResponse(
                    response.status_code,
                    b"".join(parts).decode("utf-8"),
                    response.headers.get("retry-after"),
                )
