"""Run-scoped, bounded OpenAI-compatible research gateway.

Only this process receives the upstream credential. Generated research code gets a
run bearer and two fixed endpoints; networking/container isolation belongs to the
supervisor. No streaming, arbitrary destinations, tool execution or free pricing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
import struct
import sys
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field, field_validator, model_validator

from src.agents.budget import AgentBudgetError, AgentCallBudget
from src.agents.receipts import ACCOUNTING_POLICY, capture_openai_usage, read_usage_summary
from src.domain.common import NonEmptyString, UtcTimestamp, VersionedContract

CHAT_PATH = "/v1/chat/completions"
EMBEDDING_PATH = "/v1/embeddings"
RECEIPT_PATH = "/v1/budget"
UPSTREAM = "https://api.openai.com"
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Money = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class GatewayModel(VersionedContract):
    provider: Literal["openai"] = "openai"
    model: NonEmptyString
    price_version: NonEmptyString
    source_uri: NonEmptyString
    effective_at: UtcTimestamp
    input_usd_per_million: Money
    output_usd_per_million: Money

    @field_validator("model")
    @classmethod
    def fixed_model(cls, value):
        if (
            not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value)
            or value.lower()
            in {
                "auto",
                "default",
                "latest",
            }
            or value.lower().endswith("-latest")
        ):
            raise ValueError("an explicit provider model ID is required")
        return value

    @field_validator("source_uri")
    @classmethod
    def public_pricing_source(cls, value):
        uri = urlsplit(value)
        if (
            uri.scheme != "https"
            or not uri.hostname
            or uri.username
            or uri.password
            or uri.query
            or uri.fragment
        ):
            raise ValueError("pricing provenance must be a public HTTPS URL without credentials")
        return value


class GatewayConfig(VersionedContract):
    run_id: NonEmptyString
    expires_at: UtcTimestamp
    budget_store_path: NonEmptyString
    chat: GatewayModel
    embedding: GatewayModel
    max_requests: PositiveInt
    max_tokens: PositiveInt
    max_cost_usd: Money
    max_requests_per_day: PositiveInt
    max_tokens_per_day: PositiveInt
    max_cost_per_day_usd: Money
    request_timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 120
    max_request_bytes: Annotated[int, Field(strict=True, ge=1024, le=4_000_000)] = 1_000_000
    max_response_bytes: Annotated[int, Field(strict=True, ge=1024, le=32_000_000)] = 16_000_000
    max_input_tokens: PositiveInt = 100000
    max_output_tokens: PositiveInt = 8192
    max_concurrent_requests: Annotated[int, Field(strict=True, ge=1, le=32)] = 4

    @model_validator(mode="after")
    def distinct_models(self):
        if self.chat.model == self.embedding.model:
            raise ValueError("chat and embedding model identities must be distinct")
        if self.embedding.output_usd_per_million != 0:
            raise ValueError("embedding pricing must explicitly declare zero output token charge")
        return self


class GatewayError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message
        super().__init__(code)


def _json_bytes(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode()


def _strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("nonfinite number")

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    # Reject overflowed JSON exponents and unpaired surrogate escapes anywhere,
    # including extra response metadata, before admission or response serialization.
    _json_bytes(value)
    return value


def _positive(value, name, maximum):
    if type(value) is not int or not 0 < value <= maximum:
        raise GatewayError(400, "invalid_request", name + " is outside its configured bound")
    return value


def _number(value, minimum, maximum):
    # Bound arbitrary-precision integers before converting them to a float.
    return type(value) in (int, float) and minimum <= value <= maximum and math.isfinite(value)


class ResearchGateway:
    def __init__(self, config: GatewayConfig, *, bearer: str, provider_key: str, transport=None):
        if len(bearer) < 32 or not bearer.isascii() or any(c.isspace() for c in bearer):
            raise ValueError(
                "gateway bearer must have at least 32 ASCII characters without whitespace"
            )
        if bearer == provider_key:
            raise ValueError("gateway bearer must never equal the provider credential")
        self.config, self._provider_key, self._transport = config, provider_key, transport
        self._bearer_hash = hashlib.sha256(bearer.encode()).hexdigest()
        self._budget_config = SimpleNamespace(
            budget_store_path=config.budget_store_path,
            llm_provider="openai",
            pricing=(config.chat, config.embedding),
            max_requests_per_analysis=config.max_requests,
            max_tokens_per_analysis=config.max_tokens,
            max_cost_per_analysis=config.max_cost_usd,
            max_requests_per_day=config.max_requests_per_day,
            max_tokens_per_day=config.max_tokens_per_day,
            max_cost_per_day_usd=config.max_cost_per_day_usd,
        )
        self._budget = AgentCallBudget(self._budget_config, config.run_id, deadline=math.inf)
        self._initialize_run()

    def _connect(self):
        return sqlite3.connect(self.config.budget_store_path, timeout=10)

    def _initialize_run(self):
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS gateway_runs (
                run_id TEXT PRIMARY KEY, config_hash TEXT NOT NULL, config_json TEXT NOT NULL,
                bearer_hash TEXT UNIQUE NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS gateway_audit (
                call_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, endpoint TEXT NOT NULL,
                request_hash TEXT NOT NULL, response_hash TEXT, status TEXT NOT NULL)""")
            for table in ("gateway_runs", "gateway_audit"):
                for operation in ("UPDATE", "DELETE"):
                    db.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()}
                        BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'immutable'); END""")
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT config_hash,bearer_hash FROM gateway_runs WHERE run_id=?",
                (self.config.run_id,),
            ).fetchone()
            if row and row != (self.config.content_hash(), self._bearer_hash):
                raise ValueError("run configuration and bearer identity cannot change on restart")
            if not row:
                if db.execute(
                    "SELECT 1 FROM calls WHERE run_id=?", (self.config.run_id,)
                ).fetchone():
                    raise ValueError("run ID already belongs to another budget owner")
                db.execute(
                    "INSERT INTO gateway_runs VALUES(?,?,?,?)",
                    (
                        self.config.run_id,
                        self.config.content_hash(),
                        self.config.model_dump_json(),
                        self._bearer_hash,
                    ),
                )

    def authenticate(self, authorization):
        if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
            raise GatewayError(401, "unauthorized", "A run-scoped bearer is required")
        supplied = hashlib.sha256(authorization[7:].encode()).hexdigest()
        if not hmac.compare_digest(supplied, self._bearer_hash):
            raise GatewayError(401, "unauthorized", "Invalid run-scoped bearer")

    def receipt(self):
        """No keys, bearer hashes, prompts, generated content or filesystem paths."""
        now = datetime.now(timezone.utc)
        summary = read_usage_summary(
            self.config.budget_store_path,
            day=now.date().isoformat(),
            run_id=self.config.run_id,
            as_of=now,
        )
        if summary["status"] != "available":
            raise ValueError("verified gateway accounting unavailable")
        run, day = summary["run_usage"], summary["day_usage"]
        charged_cost = Decimal(run["conservative_cost_usd"])
        charged_tokens = run["conservative_tokens"]
        expired = now >= self.config.expires_at
        price_ready = all(p.effective_at <= now for p in (self.config.chat, self.config.embedding))
        blocked = run["breached"] or day["breached"]
        remaining = {
            "requests": max(0, self.config.max_requests - run["requests"]),
            "tokens": max(0, self.config.max_tokens - charged_tokens),
            "cost_usd": str(max(Decimal(0), self.config.max_cost_usd - charged_cost)),
        }
        day_remaining = {
            "requests": max(0, self.config.max_requests_per_day - day["requests"]),
            "tokens": max(0, self.config.max_tokens_per_day - day["conservative_tokens"]),
            "cost_usd": str(
                max(
                    Decimal(0),
                    self.config.max_cost_per_day_usd - Decimal(day["conservative_cost_usd"]),
                )
            ),
        }

        def can_admit_minimum(price, output):
            cost = (
                8192 * price.input_usd_per_million + output * price.output_usd_per_million
            ) / Decimal(1_000_000)
            return all(
                r["requests"] > 0
                and r["tokens"] >= 8192 + output
                and Decimal(r["cost_usd"]) >= cost
                for r in (remaining, day_remaining)
            )

        admission_available = {
            "chat": can_admit_minimum(self.config.chat, 1),
            "embedding": can_admit_minimum(self.config.embedding, 0),
        }
        return {
            "schema_version": "1.0",
            "run_id": self.config.run_id,
            "config_hash": self.config.content_hash(),
            "expires_at": self.config.expires_at.isoformat(),
            "ready": bool(self._provider_key.strip())
            and not expired
            and price_ready
            and not blocked
            and all(admission_available.values()),
            "admission_available": admission_available,
            "provider_configured": bool(self._provider_key.strip()),
            "authenticated_provider_validation": "unverified",
            "cost_semantics": "verified_usage_with_reservation_floor",
            "accounting": {
                "schema_version": "2.0",
                "policy": ACCOUNTING_POLICY,
                "run_id": self.config.run_id,
                "as_of": summary["as_of"],
                "day_id": summary["day"],
                "run": run,
                "day": day,
            },
            "expired": expired,
            "blocked": blocked,
            "models": {
                "chat": self.config.chat.model_dump(mode="json"),
                "embedding": self.config.embedding.model_dump(mode="json"),
            },
            "limits": {
                "requests": self.config.max_requests,
                "tokens": self.config.max_tokens,
                "cost_usd": str(self.config.max_cost_usd),
            },
            "day_limits": {
                "requests": self.config.max_requests_per_day,
                "tokens": self.config.max_tokens_per_day,
                "cost_usd": str(self.config.max_cost_per_day_usd),
            },
            "remaining": remaining,
            "day_remaining": day_remaining,
            "charged": {
                "requests": run["requests"],
                "tokens": charged_tokens,
                "cost_usd": str(charged_cost),
            },
            "actual": {
                "requests_with_usage": run["requests"] - run["unpriced_requests"],
                "input_tokens": run["known_input_tokens"],
                "output_tokens": run["known_output_tokens"],
                "cost_usd": run["actual_cost_usd"],
            },
            "uncertain": {
                "requests": run["unpriced_requests"],
                "reserved_tokens": run["uncertain_tokens"],
                "reserved_cost_usd": run["uncertain_cost_usd"],
            },
            "observed_models": run["observed_models"],
            "statuses": run["statuses"],
            "streaming": False,
        }

    def dispatch(self, method, path, authorization, body=b""):
        """HTTP-neutral entry used by the server and isolated transport tests."""
        try:
            self.authenticate(authorization)
            if method == "GET" and path == RECEIPT_PATH:
                return 200, self.receipt()
            if method != "POST" or path not in (CHAT_PATH, EMBEDDING_PATH):
                raise GatewayError(404, "not_found", "Unsupported gateway endpoint")
            if len(body) > self.config.max_request_bytes:
                raise GatewayError(
                    413, "request_too_large", "Request exceeds its configured byte limit"
                )
            try:
                payload = _strict_json(body)
                if not isinstance(payload, dict):
                    raise ValueError("not object")
            except (ValueError, UnicodeError, RecursionError):
                raise GatewayError(400, "invalid_json", "Expected a finite JSON object") from None
            model, payload, output_limit, count = self._validate_request(path, payload)
            receipt = self.receipt()
            if receipt["expired"]:
                raise GatewayError(403, "run_expired", "Research run admission has expired")
            if not receipt["provider_configured"]:
                raise GatewayError(
                    503, "provider_unavailable", "Provider credential is not configured"
                )
            if receipt["blocked"]:
                raise GatewayError(503, "run_blocked", "A prior model or usage invariant failed")
            input_bound = len(_json_bytes(payload)) + 8192
            if input_bound > self.config.max_input_tokens:
                raise GatewayError(400, "input_limit", "Input exceeds its conservative token bound")
            seconds = min(
                self.config.request_timeout_seconds,
                (self.config.expires_at - datetime.now(timezone.utc)).total_seconds(),
            )
            budget = AgentCallBudget(
                self._budget_config, self.config.run_id, deadline=time.monotonic() + seconds
            )
            try:
                call_id, price, remaining = budget.reserve_bound(
                    model.model,
                    input_bound,
                    output_limit,
                    request_hash=hashlib.sha256(_json_bytes(payload)).hexdigest(),
                )
            except AgentBudgetError:
                raise GatewayError(
                    429,
                    "budget_exhausted",
                    "Request cannot be admitted within verified pricing and caps",
                ) from None
            return self._call(path, payload, count, call_id, price, budget, min(seconds, remaining))
        except GatewayError as exc:
            return exc.status, {
                "error": {"message": exc.message, "type": "gateway_error", "code": exc.code}
            }
        except (sqlite3.Error, OSError, ValueError):
            return 503, {
                "error": {
                    "message": "Budget storage unavailable",
                    "type": "gateway_error",
                    "code": "storage_unavailable",
                }
            }

    def _validate_request(self, path, body):
        chat = path == CHAT_PATH
        model = self.config.chat if chat else self.config.embedding
        if body.get("model") != model.model:
            raise GatewayError(
                400, "model_not_allowed", "Model must equal the configured endpoint model ID"
            )
        common = {"model", "user"}
        allowed = common | (
            {
                "messages",
                "stream",
                "n",
                "max_tokens",
                "max_completion_tokens",
                "temperature",
                "top_p",
                "seed",
                "stop",
                "frequency_penalty",
                "presence_penalty",
                "response_format",
                "reasoning_effort",
            }
            if chat
            else {"input", "encoding_format", "dimensions"}
        )
        if set(body) - allowed:
            raise GatewayError(
                400, "unsupported_parameter", "Request contains unsupported parameters"
            )
        if "user" in body and (not isinstance(body["user"], str) or len(body["user"]) > 256):
            raise GatewayError(400, "invalid_request", "Invalid user identifier")
        if not chat:
            inputs = body.get("input")

            def tokens(x):
                return (
                    isinstance(x, list)
                    and bool(x)
                    and all(type(v) is int and 0 <= v < 2**31 for v in x)
                )

            def text(x):
                return isinstance(x, str) and bool(x.strip())

            if text(inputs) or tokens(inputs):
                count = 1
            elif (
                isinstance(inputs, list)
                and 0 < len(inputs) <= 2048
                and (all(text(x) for x in inputs) or all(tokens(x) for x in inputs))
            ):
                count = len(inputs)
            else:
                raise GatewayError(
                    400,
                    "invalid_request",
                    "Embedding input must contain nonempty text or token arrays",
                )
            if body.get("encoding_format", "float") not in ("float", "base64"):
                raise GatewayError(400, "invalid_request", "Unsupported embedding encoding")
            if "dimensions" in body:
                _positive(body["dimensions"], "dimensions", 65536)
            return model, body, 0, count
        if body.get("stream", False) is not False:
            raise GatewayError(400, "streaming_unsupported", "Streaming is not supported")
        if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
            raise GatewayError(400, "invalid_request", "Exactly one completion is supported")
        messages = body.get("messages")
        if not isinstance(messages, list) or not 0 < len(messages) <= 1024:
            raise GatewayError(
                400, "invalid_request", "A bounded nonempty messages array is required"
            )
        for message in messages:
            if not isinstance(message, dict) or set(message) - {"role", "content", "name"}:
                raise GatewayError(400, "invalid_request", "Only text messages are supported")
            if message.get("role") not in (
                "system",
                "developer",
                "user",
                "assistant",
            ) or not isinstance(message.get("content"), str):
                raise GatewayError(
                    400, "invalid_request", "Only text messages with known roles are supported"
                )
            if "name" in message and (
                not isinstance(message["name"], str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", message["name"])
            ):
                raise GatewayError(400, "invalid_request", "Invalid message name")
        if "max_tokens" in body and "max_completion_tokens" in body:
            raise GatewayError(400, "invalid_request", "Supply only one output token limit")
        output = _positive(
            body.get(
                "max_completion_tokens", body.get("max_tokens", self.config.max_output_tokens)
            ),
            "max_completion_tokens",
            self.config.max_output_tokens,
        )
        for key, low, high in (
            ("temperature", 0, 2),
            ("top_p", 0, 1),
            ("frequency_penalty", -2, 2),
            ("presence_penalty", -2, 2),
        ):
            if key in body and not _number(body[key], low, high):
                raise GatewayError(400, "invalid_request", "Invalid sampling parameter")
        if "seed" in body and type(body["seed"]) is not int:
            raise GatewayError(400, "invalid_request", "Seed must be an integer")
        if "stop" in body and not (
            isinstance(body["stop"], str)
            or isinstance(body["stop"], list)
            and len(body["stop"]) <= 4
            and all(isinstance(s, str) for s in body["stop"])
        ):
            raise GatewayError(400, "invalid_request", "Invalid stop sequence")
        if "reasoning_effort" in body and body["reasoning_effort"] not in (
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        ):
            raise GatewayError(400, "invalid_request", "Invalid reasoning effort")
        if "response_format" in body:
            form = body["response_format"]
            if (
                not isinstance(form, dict)
                or form.get("type") not in ("text", "json_object", "json_schema")
                or set(form) - {"type", "json_schema"}
            ):
                raise GatewayError(400, "invalid_request", "Unsupported response format")
            if form["type"] == "json_schema" and not isinstance(form.get("json_schema"), dict):
                raise GatewayError(400, "invalid_request", "JSON schema is required")
        payload = {
            **body,
            "max_completion_tokens": output,
            "stream": False,
            "n": 1,
            "store": False,
            "service_tier": "default",
        }
        payload.pop("max_tokens", None)
        return model, payload, output, 1

    async def _post(self, path, payload, timeout):
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(
                transport=self._transport, timeout=timeout, follow_redirects=False, trust_env=False
            ) as client:
                async with client.stream(
                    "POST",
                    UPSTREAM + path,
                    json=payload,
                    headers={
                        "Authorization": "Bearer " + self._provider_key,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                ) as response:
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.config.max_response_bytes:
                            raise GatewayError(
                                502, "invalid_response", "Provider response exceeds byte bound"
                            )
                        chunks.append(chunk)
                    return response.status_code, b"".join(chunks)

    def _call(self, path, payload, count, call_id, price, budget, timeout):
        started, response, evidence, status = time.monotonic(), None, None, "failed"
        failure = None
        try:
            code, response = asyncio.run(self._post(path, payload, timeout))
            evidence = capture_openai_usage(response, embedding=path == EMBEDDING_PATH)
            if code != 200:
                status = "upstream_error"
                raise GatewayError(
                    502, status, "Provider rejected the request; reservation retained"
                )
            try:
                result = _strict_json(response)
                if not isinstance(result, dict):
                    raise ValueError("not object")
            except (ValueError, UnicodeError, RecursionError):
                raise GatewayError(
                    502, "invalid_response", "Malformed provider response; reservation retained"
                ) from None
            actual_model = result.get("model")
            if actual_model != payload["model"]:
                status = "model_mismatch"
                raise GatewayError(
                    502, status, "Provider model differs from the configured model; run blocked"
                )
            usage = result.get("usage", {})
            chat = path == CHAT_PATH
            input_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            output_tokens = (
                usage.get("completion_tokens") if isinstance(usage, dict) and chat else 0
            )
            total = usage.get("total_tokens") if isinstance(usage, dict) else None
            if (
                not all(
                    type(x) is int and 0 <= x < 2**62 for x in (input_tokens, output_tokens, total)
                )
                or total != input_tokens + output_tokens
            ):
                status = "usage_unknown"
                raise GatewayError(
                    502, status, "Provider usage is unavailable or invalid; reservation retained"
                )
            if chat and output_tokens > payload["max_completion_tokens"]:
                status = "usage_overrun"
                raise GatewayError(502, status, "Provider exceeded the output limit; run blocked")
            self._validate_response(path, payload, result, count)
            status = "ok"
        except (TimeoutError, httpx.TimeoutException):
            status = "timeout"
            failure = GatewayError(504, status, "Provider deadline expired; reservation retained")
        except httpx.HTTPError:
            status = "transport_error"
            failure = GatewayError(502, status, "Provider transport failed; reservation retained")
        except GatewayError as exc:
            if status == "failed":
                status = exc.code
            failure = exc
        except (ValueError, TypeError, KeyError, OverflowError):
            status = "invalid_response"
            failure = GatewayError(502, status, "Malformed provider response")
        try:
            budget.finish(
                call_id,
                price,
                evidence=evidence,
                status=status,
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
        except AgentBudgetError as exc:
            status = exc.status or "accounting_unavailable"
            failure = GatewayError(
                502,
                status,
                "Provider accounting failed verification; available usage and conservative charge retained",
            )
        with self._connect() as db:
            db.execute(
                "INSERT INTO gateway_audit VALUES(?,?,?,?,?,?)",
                (
                    call_id,
                    self.config.run_id,
                    path,
                    hashlib.sha256(_json_bytes(payload)).hexdigest(),
                    hashlib.sha256(response).hexdigest() if response else None,
                    status,
                ),
            )
        if failure:
            raise failure
        return 200, result

    def _validate_response(self, path, payload, result, count):
        if path == CHAT_PATH:
            choices = result.get("choices")
            if (
                result.get("object") != "chat.completion"
                or not isinstance(result.get("id"), str)
                or type(result.get("created")) is not int
                or not isinstance(choices, list)
                or len(choices) != 1
            ):
                raise GatewayError(502, "invalid_response", "Invalid completion envelope")
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, dict) else None
            if (
                not isinstance(message, dict)
                or message.get("role") != "assistant"
                or message.get("tool_calls")
                or not (
                    isinstance(message.get("content"), str)
                    or isinstance(message.get("refusal"), str)
                )
            ):
                raise GatewayError(
                    502, "invalid_response", "Expected a text completion or explicit refusal"
                )
            if (
                choice.get("finish_reason") not in ("stop", "length", "content_filter")
                or type(choice.get("index")) is not int
                or choice.get("index") != 0
            ):
                raise GatewayError(502, "invalid_response", "Invalid completion termination")
            return
        data = result.get("data")
        if result.get("object") != "list" or not isinstance(data, list) or len(data) != count:
            raise GatewayError(502, "invalid_response", "Invalid embedding envelope")
        dimensions = set()
        indices = set()
        for item in data:
            if (
                not isinstance(item, dict)
                or item.get("object") != "embedding"
                or type(item.get("index")) is not int
            ):
                raise GatewayError(502, "invalid_response", "Invalid embedding item")
            indices.add(item["index"])
            vector = item.get("embedding")
            if payload.get("encoding_format", "float") == "base64":
                try:
                    decoded = base64.b64decode(vector, validate=True)
                    vector = struct.unpack("<" + "f" * (len(decoded) // 4), decoded)
                except (ValueError, TypeError, struct.error):
                    raise GatewayError(
                        502, "invalid_response", "Invalid encoded embedding"
                    ) from None
            if (
                not isinstance(vector, (list, tuple))
                or not vector
                or not all(_number(v, -1e38, 1e38) for v in vector)
            ):
                raise GatewayError(502, "invalid_response", "Invalid or nonfinite embedding")
            dimensions.add(len(vector))
        if (
            indices != set(range(count))
            or len(dimensions) != 1
            or ("dimensions" in payload and dimensions != {payload["dimensions"]})
        ):
            raise GatewayError(502, "invalid_response", "Embedding dimensions or indices disagree")


class GatewayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, gateway):
        self.gateway = gateway
        self._slots = threading.BoundedSemaphore(gateway.config.max_concurrent_requests)
        super().__init__(address, GatewayHandler)

    def process_request(self, request, client_address):
        # Bound thread creation and socket resources as well as paid requests.
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            request.settimeout(min(10, self.gateway.config.request_timeout_seconds))
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request, client_address):
        # Do not dump request payloads or authorization in diagnostics.
        pass


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _handle(self):
        gateway = self.server.gateway
        try:
            auth = self.headers.get_all("Authorization", [])
            if len(auth) != 1:
                raise GatewayError(401, "unauthorized", "One run-scoped bearer is required")
            gateway.authenticate(auth[0])
            body = b""
            if self.command == "POST":
                lengths = self.headers.get_all("Content-Length", [])
                if (
                    self.headers.get("Transfer-Encoding")
                    or len(lengths) != 1
                    or len(lengths[0]) > 10
                    or not lengths[0].isdigit()
                ):
                    raise GatewayError(
                        400,
                        "invalid_request",
                        "One Content-Length and no transfer encoding are required",
                    )
                length = int(lengths[0])
                if length > gateway.config.max_request_bytes:
                    raise GatewayError(
                        413, "request_too_large", "Request exceeds configured byte limit"
                    )
                if self.headers.get_content_type() != "application/json" or self.headers.get(
                    "Content-Encoding"
                ):
                    raise GatewayError(
                        415, "unsupported_media_type", "Uncompressed JSON is required"
                    )
                body = self.rfile.read(length)
                if len(body) != length:
                    raise GatewayError(400, "invalid_request", "Incomplete request body")
            code, result = gateway.dispatch(self.command, self.path, auth[0], body)
        except GatewayError as exc:
            code, result = (
                exc.status,
                {"error": {"message": exc.message, "type": "gateway_error", "code": exc.code}},
            )
        self.close_connection = True
        payload = _json_bytes(result)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle
    do_CONNECT = _handle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = GatewayConfig.model_validate_json(args.config.read_text())
        if args.check_config:
            print(json.dumps({"run_id": config.run_id, "config_hash": config.content_hash()}))
            return 0
        if not 0 <= args.port <= 65535:
            raise ValueError("invalid port")
        gateway = ResearchGateway(
            config,
            bearer=os.environ.get("RESEARCH_GATEWAY_BEARER", ""),
            provider_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        with GatewayHTTPServer((args.host, args.port), gateway) as server:
            print(
                json.dumps(
                    {
                        "status": "listening",
                        "run_id": config.run_id,
                        "config_hash": config.content_hash(),
                        "port": server.server_port,
                    }
                ),
                flush=True,
            )
            server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        return 0
    except (ValueError, OSError, sqlite3.Error):
        print(
            "Research gateway configuration, credentials, address or store is invalid",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
