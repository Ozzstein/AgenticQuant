"""Application-owned MCP evidence connections with an exact read-tool boundary.

Deployment manifests pin server version AND each observed input schema hash.
Discovery does not authorize a new capability. Generic dispatch is unwrapped
and rechecked locally; the generic dispatch tool is never sent to a server.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, model_validator

from src.data.adapters.base import ProviderError, store_observation
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import Instrument, NonEmptyString, RunContext, Sha256, VersionedContract

# Source-reviewed exact names. No substring/prefix read heuristics. New names
# require code review even if a remote server claims readOnlyHint=true.
APPROVED_TOOLS = {
    "alpaca": frozenset({"get_account_info", "get_account_config", "get_all_positions", "get_open_position",
                         "get_orders", "get_order_by_id", "get_order_by_client_id", "get_portfolio_history"}),
    "fmp": frozenset({"getEarningsTranscript", "getIncomeStatement", "getBalanceSheetStatement",
                      "getCashFlowStatement", "getAnalystEstimates", "searchInsiderTrades",
                      "getLatestInstitutionalFilings", "getFilingsBySymbol", "getEarningsCalendar"}),
    "openbb": frozenset({"equity_price_historical", "economy_fred_series", "news_company",
                         "equity_fundamental_ratios", "equity_fundamental_metrics"}),
    "valyu": frozenset({"valyu_search", "valyu_datasources", "valyu_contents"}),
}


class ToolPin(VersionedContract):
    name: NonEmptyString
    capability: NonEmptyString
    schema_hash: Sha256
    input_schema: dict[str, Any]
    fixed_arguments: dict[str, Any] = Field(default_factory=dict)
    max_cost_usd: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def coherent(self):
        if digest(canonical_bytes(self.input_schema)) != self.schema_hash:
            raise ValueError("tool schema pin mismatch")
        if self.input_schema.get("type") != "object":
            raise ValueError("tool arguments must have an object schema")
        return self


class ServerManifest(VersionedContract):
    provider: Literal["openbb", "fmp", "alpaca", "valyu"]
    server_name: NonEmptyString
    server_version: NonEmptyString
    source_revision: NonEmptyString
    url: NonEmptyString
    tools: tuple[ToolPin, ...]
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_requests: int = Field(default=100, gt=0)
    budget_usd: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def closed_permissions(self):
        from urllib.parse import urlsplit
        url = urlsplit(self.url)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("MCP URL must not contain credentials/query/fragment")
        if url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")):
            raise ValueError("MCP requires HTTPS except local server")
        if not self.tools or len({t.name for t in self.tools}) != len(self.tools):
            raise ValueError("unique nonempty tools required")
        for tool in self.tools:
            if tool.name not in APPROVED_TOOLS[self.provider]:
                raise ValueError(f"tool is outside application read allowlist: {tool.name}")
        return self


class MCPManager:
    """Reuse one context-managed connection per server; cache immutable captures.

    No automatic retries of chargeable tool calls: a timeout can have incurred a
    charge. Reserved cost remains charged to the local budget on failure. Actual
    provider-reported spend stays in the raw result and may be unknown.
    """

    def __init__(self, store: EvidenceStore, manifests: tuple[ServerManifest, ...]):
        self.store = store
        self.manifests = {m.provider: m for m in manifests}
        if len(self.manifests) != len(manifests):
            raise ValueError("duplicate provider manifest")
        capabilities = [t.capability for m in manifests for t in m.tools]
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("duplicate capability must have a single provider route")
        self._stack = AsyncExitStack()
        self._sessions = {}
        self._calls = {m.provider: 0 for m in manifests}
        self._reserved = {m.provider: Decimal(0) for m in manifests}
        self._lock = asyncio.Lock()
        self._cache = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()
        self._sessions.clear()

    async def connect(self, provider: str, *, bearer_token: str | None = None):
        if provider in self._sessions:
            return
        # MCP 2.3 uses httpx2 internally. Credentials stay in request headers.
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        manifest = self.manifests[provider]
        headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
        client = await self._stack.enter_async_context(httpx2.AsyncClient(headers=headers, timeout=manifest.timeout_seconds))
        streams = await self._stack.enter_async_context(streamable_http_client(manifest.url, http_client=client))
        session = await self._stack.enter_async_context(ClientSession(streams[0], streams[1], read_timeout_seconds=manifest.timeout_seconds))
        await self.verify_session(provider, session)

    async def verify_session(self, provider: str, session):
        """Handshake helper also exercised with a protocol fixture offline."""
        self._sessions.pop(provider, None)
        manifest = self.manifests[provider]
        try:
            result = await asyncio.wait_for(session.initialize(), manifest.timeout_seconds)
            info = result.server_info
            if (info.name, info.version) != (manifest.server_name, manifest.server_version):
                raise ProviderError(provider, "server_version_mismatch")
            catalog = await asyncio.wait_for(session.list_tools(), manifest.timeout_seconds)
            if catalog.next_cursor:
                raise ProviderError(provider, "filtered_complete_tool_catalog_required")
            tools = {tool.name: tool for tool in catalog.tools}
            for pin in manifest.tools:
                if pin.name not in tools or digest(canonical_bytes(tools[pin.name].input_schema)) != pin.schema_hash:
                    raise ProviderError(provider, "tool_schema_mismatch")
        except ProviderError:
            raise
        except Exception:
            raise ProviderError(provider, "connection_or_handshake_failed") from None
        self._sessions[provider] = session

    def authorize(self, provider: str, name: str, arguments: dict) -> tuple[ToolPin, dict]:
        if provider not in self.manifests:
            raise ProviderError(provider, "unconfigured_provider")
        if name == "call_tool":
            if set(arguments) != {"name", "arguments"} or not isinstance(arguments["arguments"], dict):
                raise ProviderError(provider, "invalid_generic_dispatch")
            name, arguments = arguments["name"], arguments["arguments"]
        if name not in APPROVED_TOOLS[provider]:
            raise ProviderError(provider, "operation_not_permitted")
        pin = next((t for t in self.manifests[provider].tools if t.name == name), None)
        if pin is None:
            raise ProviderError(provider, "tool_not_pinned")
        from jsonschema import Draft202012Validator
        # Reject undeclared args even when a provider's permissive schema would
        # accept them. Fixed params lock provider routes and spending controls.
        allowed = set(pin.input_schema.get("properties", {}))
        if set(arguments) - allowed:
            raise ProviderError(provider, "unexpected_arguments")
        for key, value in pin.fixed_arguments.items():
            if key in arguments and arguments[key] != value:
                raise ProviderError(provider, "fixed_argument_override")
        effective = {**arguments, **pin.fixed_arguments}
        if list(Draft202012Validator(pin.input_schema).iter_errors(effective)):
            raise ProviderError(provider, "invalid_arguments")
        return pin, effective

    async def call(self, provider: str, name: str, arguments: dict, context: RunContext, *,
                   ingested_at: datetime, instrument: Instrument | None = None):
        pin, effective = self.authorize(provider, name, arguments)
        manifest = self.manifests[provider]
        cache_key = digest(canonical_bytes({"manifest": manifest.content_hash(), "tool": pin.name,
                                           "arguments": effective, "context": context.model_dump(mode="json"),
                                           "instrument": instrument.model_dump(mode="json") if instrument else None}))
        async with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
            # No LLM/tool-controlled connection or endpoint selection.
            if provider not in self._sessions:
                raise ProviderError(provider, "connection_unavailable")
            if self._calls[provider] >= manifest.max_requests or self._reserved[provider] + pin.max_cost_usd > manifest.budget_usd:
                raise ProviderError(provider, "request_or_spend_budget_exhausted")
            self._calls[provider] += 1
            self._reserved[provider] += pin.max_cost_usd
            try:
                result = await asyncio.wait_for(self._sessions[provider].call_tool(pin.name, effective), manifest.timeout_seconds)
            except asyncio.TimeoutError:
                raise ProviderError(provider, "tool_timeout_cost_unknown") from None
            except Exception:
                raise ProviderError(provider, "tool_unavailable_cost_unknown") from None
            if getattr(result, "is_error", False):
                raise ProviderError(provider, "tool_error")
            payload = result.model_dump(mode="json", by_alias=True)
            raw = canonical_bytes(payload)
            texts = [block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"]
            document = "\n".join(texts)
            envelope = store_observation(self.store, raw=raw, normalized=payload, document=document,
                                        provider=provider, version=manifest.source_revision, source_uri=manifest.url,
                                        kind=pin.capability, logical_key=cache_key, context=context,
                                        ingested_at=ingested_at, instrument=instrument,
                                        tool_name=pin.name, tool_version=manifest.server_version)
            self._cache[cache_key] = envelope
            return envelope

    def health(self) -> dict:
        return {provider: {"connected": provider in self._sessions, "requests": self._calls[provider],
                           "reserved_cost_usd": str(self._reserved[provider]), "actual_cost": "provider_response_or_unknown",
                           "manifest_hash": manifest.content_hash()} for provider, manifest in self.manifests.items()}


def load_manifests(path: str) -> tuple[ServerManifest, ...]:
    with open(path) as source:
        return tuple(ServerManifest.model_validate(item) for item in json.load(source))
