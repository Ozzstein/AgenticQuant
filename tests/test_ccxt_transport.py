"""Real local CCXT signing and bounded offline HTTP protocol tests."""

import asyncio
import base64
import hashlib
import hmac
import json
from urllib.parse import parse_qs

import httpx
import pytest

from src.domain.portfolio import AccountSpec
from src.execution.ccxt_transport import CCXTTransport, milliseconds
from src.execution.ledger import LedgerError, PortfolioLedger
from tests.test_alpaca_venue import Chunks
from tests.test_durable_accounting import T


def transport(tmp_path, handler, *, venue="binance", **limits):
    ledger = PortfolioLedger(tmp_path / "native.sqlite")
    mode = "paper" if venue == "binance" else "live"
    ledger.create_account(
        AccountSpec(
            account_id=mode + ":native",
            mode=mode,
            base_currency="USD",
            initial_cash=10000,
            opened_at=T,
        )
    )
    adapter = CCXTTransport(
        ledger,
        mode + ":native",
        venue=venue,
        api_key="fixture-key",
        api_secret=base64.b64encode(b"fixture-secret").decode(),
        sandbox=mode == "paper",
        allow_live=mode == "live",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        clock=lambda: T,
        **limits,
    )
    return ledger, adapter


@pytest.mark.parametrize(
    "body,limit,error",
    [
        (b'{"payload":"xxxxxxxxxxxxxxxxx"}', 12, "incomplete"),
        (b"not JSON", 100, "invalid native JSON"),
    ],
)
def test_bounded_raw_body_is_retained_before_parse_error(tmp_path, body, limit, error):
    async def handler(request):
        return httpx.Response(200, stream=Chunks([body[:5], body[5:]]))

    ledger, adapter = transport(tmp_path, handler, maximum_response_bytes=limit)
    try:
        with adapter.operation(), pytest.raises(LedgerError, match=error):
            adapter.request("exchangeInfo", api="public")
        rows = list(ledger.db.execute("SELECT payload FROM events"))
        assert len(rows) == 1
        receipt = json.loads(rows[0][0])
        assert base64.b64decode(receipt["response_base64"]) == body[:limit]
        assert receipt["complete"] is (len(body) <= limit)
    finally:
        adapter.close()
        ledger.close()


def test_total_operation_deadline_stops_a_dripping_response_without_retry(tmp_path):
    class Drip(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"["
            for _ in range(100):
                await asyncio.sleep(0.02)
                yield b"0,"

    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=Drip())

    ledger, adapter = transport(tmp_path, handler, operation_timeout_seconds=0.08)
    try:
        with adapter.operation(), pytest.raises(LedgerError, match="incomplete"):
            adapter.request("exchangeInfo", api="public")
        receipt = json.loads(ledger.db.execute("SELECT payload FROM events").fetchone()[0])
        assert receipt["error"] == "TimeoutError" and receipt["complete"] is False
        assert base64.b64decode(receipt["response_base64"]).startswith(b"[")
        assert len(calls) == 1
    finally:
        adapter.close()
        ledger.close()


def test_request_allowance_covers_all_nested_operations(tmp_path):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=Chunks([b"{}"]))

    ledger, adapter = transport(tmp_path, handler, max_requests_per_operation=2)
    try:
        with adapter.operation():
            adapter.request("exchangeInfo", api="public")
            with adapter.operation():
                adapter.request("exchangeInfo", api="public")
            with pytest.raises(LedgerError, match="allowance exhausted"):
                adapter.request("exchangeInfo", api="public")
        assert len(calls) == 2
    finally:
        adapter.close()
        ledger.close()


def test_kraken_local_signing_uses_same_nonce_in_body_digest_and_restart(tmp_path):
    nonces = []

    async def handler(request):
        assert request.url.host == "api.kraken.com" and request.url.path == "/0/private/BalanceEx"
        raw = request.content
        nonce = parse_qs(raw.decode())["nonce"][0]
        nonces.append(int(nonce))
        encoded = hashlib.sha256(nonce.encode() + raw).digest()
        signature = base64.b64encode(
            hmac.new(
                b"fixture-secret", request.url.path.encode() + encoded, hashlib.sha512
            ).digest()
        ).decode()
        assert request.headers["API-Sign"] == signature
        return httpx.Response(200, stream=Chunks([b'{"error":[],"result":{}}']))

    for _ in range(2):
        ledger, adapter = transport(tmp_path, handler, venue="kraken")
        try:
            with adapter.operation():
                adapter.request("BalanceEx")
        finally:
            adapter.close()
            ledger.close()
    assert nonces == [milliseconds(T), milliseconds(T) + 1]
