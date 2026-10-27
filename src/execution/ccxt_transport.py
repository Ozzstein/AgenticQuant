"""Bounded native HTTP transport using CCXT signing, without unified float parsing.

The only CCXT operation is local request signing. Every response is archived
before native JSON validation; no network retry or order submission is inferred.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from threading import RLock
from urllib.parse import urlsplit

import ccxt
import httpx
from pydantic import SecretStr

from src.domain.common import _utc
from src.execution.ledger import LedgerError


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def decimal(value):
    if isinstance(value, (bool, float)) or value is None:
        raise LedgerError("native monetary values require exact decimal text/integer boundaries")
    result = Decimal(value)
    if not result.is_finite():
        raise LedgerError("nonfinite native monetary value")
    return result


def milliseconds(at):
    delta = _utc(at) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


class CCXTTransport:
    """One serialized, bounded operation with durable native nonce and receipts."""

    def __init__(
        self,
        ledger,
        account_id,
        *,
        venue,
        api_key: str | SecretStr,
        api_secret: str | SecretStr,
        sandbox=False,
        allow_live=False,
        client: httpx.AsyncClient | None = None,
        clock=None,
        operation_timeout_seconds=30.0,
        max_requests_per_operation=128,
        maximum_response_bytes=8 * 1024 * 1024,
    ):
        spec = ledger.spec(account_id)
        if venue not in ("binance", "kraken"):
            raise LedgerError("explicit Binance/Kraken spot transport required")
        if spec.mode not in ("paper", "live"):
            raise LedgerError("remote spot execution requires an operational account")
        if sandbox != (spec.mode == "paper"):
            raise LedgerError("native sandbox must match the actual paper account mode")
        if venue == "kraken" and sandbox:
            raise LedgerError("Kraken spot has no verified paper endpoint; use local PaperVenue")
        if not sandbox and not allow_live:
            raise LedgerError("live spot transport requires independent explicit authorization")
        if (
            not math.isfinite(operation_timeout_seconds)
            or operation_timeout_seconds <= 0
            or type(max_requests_per_operation) is not int
            or max_requests_per_operation < 1
            or type(maximum_response_bytes) is not int
            or maximum_response_bytes < 1
        ):
            raise LedgerError("positive finite native transport limits required")
        if client is not None and not isinstance(client, httpx.AsyncClient):
            raise LedgerError("injected native transport must be httpx.AsyncClient")
        key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        secret = api_secret if isinstance(api_secret, SecretStr) else SecretStr(api_secret)
        if not key.get_secret_value() or not secret.get_secret_value():
            raise LedgerError("native spot credentials unavailable")
        self.ledger, self.account_id, self.venue = ledger, account_id, venue
        self._signer = getattr(ccxt, venue)(
            {
                "apiKey": key.get_secret_value(),
                "secret": secret.get_secret_value(),
                "enableRateLimit": False,
            }
        )
        if sandbox:
            self._signer.set_sandbox_mode(True)
        self._signer.incrementing_nonce = self._nonce
        self._signer.nonce = self._nonce
        self._client = client or httpx.AsyncClient(
            timeout=15, trust_env=False, follow_redirects=False
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._runner, self._lock, self._budget = asyncio.Runner(), RLock(), None
        self.timeout, self.request_limit, self.byte_limit = (
            operation_timeout_seconds,
            max_requests_per_operation,
            maximum_response_bytes,
        )
        self._hosts = (
            {"testnet.binance.vision"}
            if sandbox
            else {"api.binance.com"}
            if venue == "binance"
            else {"api.kraken.com"}
        )
        with ledger.atomic():
            ledger.db.execute(
                "CREATE TABLE IF NOT EXISTS native_nonces(account TEXT NOT NULL,venue TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(account,venue))"
            )

    def _nonce(self):
        # One account/credential binding must be exclusive. A restarted process
        # cannot reuse a Kraken nonce merely because its wall clock is equal.
        with self.ledger.atomic():
            row = self.ledger.db.execute(
                "SELECT value FROM native_nonces WHERE account=? AND venue=?",
                (self.account_id, self.venue),
            ).fetchone()
            value = max(milliseconds(self.now()), int(row[0]) + 1 if row else 0)
            self.ledger.db.execute(
                "INSERT INTO native_nonces VALUES (?,?,?) ON CONFLICT(account,venue) DO UPDATE SET value=excluded.value",
                (self.account_id, self.venue, str(value)),
            )
            return value

    def now(self):
        at = _utc(self._clock())
        row = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id LIKE ? ORDER BY rowid DESC LIMIT 1",
            (self.account_id, self.venue + "-http:%"),
        ).fetchone()
        if row and datetime.fromisoformat(json.loads(row[0])["observed_at"]) > at:
            raise LedgerError("native receipt clock regressed")
        return at

    @contextmanager
    def operation(self):
        with self._lock:
            outer = self._budget is None
            if outer:
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    pass
                else:
                    raise LedgerError(
                        "run synchronous native venue operations outside an event loop"
                    )
                self._budget = [time.monotonic() + self.timeout, 0]
            try:
                yield
            finally:
                if outer:
                    self._budget = None

    async def _stream(self, signed, remaining):
        body, status, encoding, complete, error = bytearray(), None, "identity", False, None
        try:
            async with asyncio.timeout(remaining):
                async with self._client.stream(
                    signed["method"],
                    signed["url"],
                    content=signed.get("body"),
                    headers={**(signed.get("headers") or {}), "Accept-Encoding": "identity"},
                    timeout=httpx.Timeout(min(15, remaining)),
                    follow_redirects=False,
                ) as response:
                    status = response.status_code
                    encoding = response.headers.get("content-encoding", "identity")
                    async for chunk in response.aiter_raw():
                        available = self.byte_limit - len(body)
                        body.extend(chunk[:available])
                        if len(chunk) > available:
                            error = "ResponseByteLimit"
                            break
                    else:
                        complete = True
        except (TimeoutError, httpx.HTTPError) as exc:
            error = type(exc).__name__
        return status, bytes(body), encoding, complete, error

    def request(self, path, *, api="private", method=None, params=None):
        """Archive one request/response and return native JSON plus receipt identity.

        The concrete venue adapter owns order claims and error classification.
        HTTP/native error payloads remain available to that adapter as data.
        """
        if self._budget is None:
            raise LedgerError("native request requires a bounded operation")
        remaining = self._budget[0] - time.monotonic()
        if remaining <= 0 or self._budget[1] >= self.request_limit:
            raise LedgerError("native operation deadline/request allowance exhausted")
        if api not in ("public", "private") or not path or "/" in path or "?" in path:
            raise LedgerError("explicit native spot endpoint required")
        parameters = dict(params or {})
        if any(
            key.lower() in {"apikey", "api_key", "secret", "signature", "password", "nonce"}
            for key in parameters
        ):
            raise LedgerError("credentials/nonces cannot be supplied as native parameters")
        method = method or ("POST" if api == "private" and self.venue == "kraken" else "GET")
        if self.venue == "binance" and api == "private":
            parameters["timestamp"] = milliseconds(self.now())
            parameters["recvWindow"] = 5000
        self._budget[1] += 1
        try:
            signed = self._signer.sign(path, [api], method, parameters)
        except Exception as exc:
            raise LedgerError("native CCXT signing failed: " + type(exc).__name__) from None
        url = urlsplit(signed["url"])
        if url.scheme != "https" or url.hostname not in self._hosts or url.username or url.password:
            raise LedgerError("native signing produced an unapproved endpoint")
        status, body, encoding, complete, error = self._runner.run(self._stream(signed, remaining))
        observed_at = self.now()
        evidence = {
            "venue": self.venue,
            "endpoint": url.scheme + "://" + url.netloc + url.path,
            "api": api,
            "method": method,
            "parameters": parameters,
            "status": status,
            "content_encoding": encoding,
            "observed_at": observed_at.isoformat(),
            "complete": complete,
            "error": error,
            "response_base64": base64.b64encode(body).decode("ascii"),
            "ccxt_version": ccxt.__version__,
        }
        evidence_id = self.venue + "-http:" + digest(evidence)
        with self.ledger.atomic():
            self.ledger.record_event(self.account_id, evidence_id, canonical(evidence))
        if not complete or error:
            raise LedgerError(
                "native response incomplete: " + str(error) + "; receipt " + evidence_id
            )
        if encoding not in ("", "identity"):
            raise LedgerError("unsupported native response encoding; receipt " + evidence_id)
        if time.monotonic() > self._budget[0]:
            raise LedgerError("native operation deadline exhausted; receipt " + evidence_id)
        try:
            value = json.loads(
                body,
                parse_float=Decimal,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except (ValueError, UnicodeError):
            raise LedgerError("invalid native JSON retained; receipt " + evidence_id) from None
        return value, status, observed_at, evidence_id

    def read(self, evidence_id):
        row = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?", (self.account_id, evidence_id)
        ).fetchone()
        if row is None:
            raise LedgerError("native source receipt unavailable")
        evidence = json.loads(row[0])
        if (
            evidence_id != self.venue + "-http:" + digest(evidence)
            or not evidence["complete"]
            or evidence["error"]
        ):
            raise LedgerError("native source receipt changed or incomplete")
        return evidence, json.loads(
            base64.b64decode(evidence["response_base64"], validate=True), parse_float=Decimal
        )

    def close(self):
        with self._lock:
            if not self._client.is_closed:
                self._runner.run(self._client.aclose())
            self._runner.close()
