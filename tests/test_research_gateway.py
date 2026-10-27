"""Offline provider fixtures exercise real gateway transport, admission and HTTP parsing."""

import base64
import copy
import json
import socket
import sqlite3
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from src.agents.budget import AgentBudgetError
from src.decision.research_gateway import (
    CHAT_PATH,
    EMBEDDING_PATH,
    RECEIPT_PATH,
    GatewayConfig,
    GatewayHandler,
    ResearchGateway,
)

BEARER = "test-only-run-scoped-bearer-123456789012345"
PROVIDER = "test-only-provider-credential-never-for-research-container"
CHAT = "test-chat-2026-01-01"
EMBED = "test-embedding-version-one"


@pytest.fixture
def gateway_config(tmp_path):
    pricing = {
        "price_version": "fixture-prices-v1",
        "source_uri": "https://example.test/pricing",
        "effective_at": datetime(2020, 1, 1, tzinfo=timezone.utc),
        "input_usd_per_million": "1",
        "output_usd_per_million": "2",
    }
    return GatewayConfig(
        run_id="research-run-one",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        budget_store_path=str(tmp_path / "budget.sqlite"),
        chat={**pricing, "model": CHAT},
        embedding={**pricing, "model": EMBED, "output_usd_per_million": "0"},
        max_requests=30,
        max_tokens=500000,
        max_cost_usd=5,
        max_requests_per_day=100,
        max_tokens_per_day=1000000,
        max_cost_per_day_usd=10,
        max_output_tokens=100,
    )


def chat_body(**kwargs):
    return {
        "model": CHAT,
        "messages": [{"role": "user", "content": "test-only source text"}],
        **kwargs,
    }


def chat_result(**kwargs):
    return {
        "id": "fixture-chat",
        "object": "chat.completion",
        "created": 1234,
        "model": CHAT,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "fixture result"},
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        **kwargs,
    }


def embedding_result(**kwargs):
    return {
        "object": "list",
        "model": EMBED,
        "data": [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5]}],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
        **kwargs,
    }


def make_gateway(config, response=None, *, handler=None, key=PROVIDER):
    calls = []

    def respond(request):
        calls.append(request)
        if handler:
            return handler(request)
        return httpx.Response(200, json=response or chat_result())

    return ResearchGateway(
        config, bearer=BEARER, provider_key=key, transport=httpx.MockTransport(respond)
    ), calls


def post(gateway, body=None, path=CHAT_PATH, token=BEARER):
    return gateway.dispatch(
        "POST", path, "Bearer " + token, json.dumps(body or chat_body()).encode()
    )


def test_actual_transport_pinned_route_body_limits_and_settlement(gateway_config, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.invalid")
    monkeypatch.setenv("HTTPS_PROXY", "https://attacker.invalid")
    gateway, calls = make_gateway(gateway_config)
    code, result = post(gateway, chat_body(max_tokens=40, stream=False))
    assert code == 200 and result["choices"][0]["message"]["content"] == "fixture result"
    request = calls[0]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer " + PROVIDER
    assert BEARER not in str(request.headers)
    sent = json.loads(request.content)
    assert sent["max_completion_tokens"] == 40 and "max_tokens" not in sent
    assert sent["stream"] is False and sent["store"] is False and sent["n"] == 1
    receipt = gateway.receipt()
    assert receipt["actual"] == {
        "requests_with_usage": 1,
        "input_tokens": 20,
        "output_tokens": 10,
        "cost_usd": "0.00004",
    }
    assert receipt["uncertain"]["requests"] == 0
    assert receipt["observed_models"] == [CHAT]


@pytest.mark.parametrize(
    "changes",
    [
        {"stream": True},
        {"stream": 0},
        {"n": 2},
        {"n": True},
        {"max_tokens": 0},
        {"max_tokens": 101},
        {"max_tokens": True},
        {"max_tokens": 20, "max_completion_tokens": 20},
        {"model": "unknown"},
        {"api_base": "https://attacker.invalid"},
        {"tools": []},
        {"service_tier": "priority"},
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
            ]
        },
        {"temperature": float("nan")},
        {"top_p": float("inf")},
    ],
)
def test_invalid_requests_never_reach_provider_or_consume_budget(gateway_config, changes):
    gateway, calls = make_gateway(gateway_config)
    code, result = post(gateway, chat_body(**changes))
    assert code == 400 and "error" in result
    assert not calls and gateway.receipt()["charged"]["requests"] == 0


@pytest.mark.parametrize(
    "raw",
    [
        b'{"model":"x","model":"y"}',
        b"[]",
        b"null",
        b"\xff",
        b"{x}",
        b'{"x":1e999}',
        b'{"x":"\\ud800"}',
    ],
)
def test_invalid_json_is_explicit(gateway_config, raw):
    gateway, calls = make_gateway(gateway_config)
    code, result = gateway.dispatch("POST", CHAT_PATH, "Bearer " + BEARER, raw)
    assert code == 400 and result["error"]["code"] == "invalid_json"
    assert not calls


def test_expiry_provider_missing_and_future_pricing_fail_without_outbound(gateway_config):
    gateway, calls = make_gateway(gateway_config, key="")
    assert post(gateway)[0] == 503 and not gateway.receipt()["ready"]
    assert not calls
    config = gateway_config.model_copy(
        update={
            "run_id": "expired-run",
            "expires_at": datetime.now(timezone.utc) - timedelta(seconds=1),
        }
    )
    gateway = ResearchGateway(config, bearer=BEARER + "expired", provider_key=PROVIDER)
    assert post(gateway, token=BEARER + "expired")[0] == 403
    future = gateway_config.chat.model_copy(
        update={"effective_at": datetime.now(timezone.utc) + timedelta(days=1)}
    )
    config = gateway_config.model_copy(update={"run_id": "future-pricing", "chat": future})
    gateway = ResearchGateway(config, bearer=BEARER + "future", provider_key=PROVIDER)
    assert post(gateway, token=BEARER + "future")[0] == 429
    assert not gateway.receipt()["ready"]


def test_receipt_requires_scope_and_never_exposes_secrets_or_prompts(gateway_config):
    gateway, _ = make_gateway(gateway_config)
    post(gateway)
    for authorization in (None, "", "Bearer bad", "Basic " + BEARER):
        assert gateway.dispatch("GET", RECEIPT_PATH, authorization)[0] == 401
    code, receipt = gateway.dispatch("GET", RECEIPT_PATH, "Bearer " + BEARER)
    assert code == 200 and receipt["run_id"] == gateway_config.run_id
    assert receipt["config_hash"] == gateway_config.content_hash()
    serialized = json.dumps(receipt)
    for secret in (
        BEARER,
        PROVIDER,
        "test-only source text",
        "fixture result",
        "bearer_hash",
        gateway_config.budget_store_path,
    ):
        assert secret not in serialized
    with sqlite3.connect(gateway_config.budget_store_path) as db:
        audit = db.execute("SELECT * FROM gateway_audit").fetchall()
        assert len(audit) == 1 and "test-only source text" not in str(audit)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("DELETE FROM gateway_audit")


def test_restart_config_bearer_scope_and_durable_caps(gateway_config):
    config = gateway_config.model_copy(update={"max_requests": 1})
    gateway, calls = make_gateway(config)
    assert post(gateway)[0] == 200
    restarted, repeated = make_gateway(config)
    assert post(restarted)[0] == 429 and not repeated and len(calls) == 1
    with pytest.raises(ValueError, match="cannot change"):
        make_gateway(config.model_copy(update={"max_requests": 2}))
    with pytest.raises(ValueError, match="cannot change"):
        ResearchGateway(config, bearer=BEARER + "changed", provider_key=PROVIDER)
    with pytest.raises(sqlite3.IntegrityError):
        make_gateway(config.model_copy(update={"run_id": "different-run"}))


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
        {"prompt_tokens": 1, "completion_tokens": -1, "total_tokens": 0},
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
    ],
)
def test_unknown_usage_retains_reserved_cost(gateway_config, usage):
    gateway, _ = make_gateway(gateway_config, chat_result(usage=usage))
    code, result = post(gateway)
    assert code == 502 and result["error"]["code"] == "usage_unknown"
    receipt = gateway.receipt()
    assert receipt["actual"]["requests_with_usage"] == 0
    assert receipt["uncertain"]["requests"] == 1
    assert Decimal(receipt["charged"]["cost_usd"]) > 0
    assert receipt["charged"]["tokens"] > 8192


def test_wrong_model_blocks_restart_and_preserves_unknown_charge(gateway_config):
    gateway, calls = make_gateway(gateway_config, chat_result(model="test-other-version"))
    code, result = post(gateway)
    assert code == 502 and result["error"]["code"] == "model_mismatch"
    receipt = gateway.receipt()
    assert not receipt["ready"] and receipt["blocked"]
    assert receipt["uncertain"]["requests"] == 1
    restarted, _ = make_gateway(gateway_config)
    assert post(restarted)[0] == 503 and len(calls) == 1


@pytest.mark.parametrize(
    "failure", ["timeout", "transport", "rate_limit", "redirect", "malformed", "huge"]
)
def test_outage_timeout_redirect_malformed_no_retry_or_false_zero(gateway_config, failure):
    config = gateway_config.model_copy(update={"max_response_bytes": 1024})

    def respond(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("must not leak " + PROVIDER)
        if failure == "transport":
            raise httpx.ConnectError("must not leak " + PROVIDER)
        if failure == "rate_limit":
            return httpx.Response(429, text=PROVIDER)
        if failure == "redirect":
            return httpx.Response(307, headers={"Location": "https://attacker.invalid"})
        if failure == "huge":
            return httpx.Response(200, text="x" * 2048)
        return httpx.Response(200, text="{malformed " + PROVIDER)

    gateway, calls = make_gateway(config, handler=respond)
    code, result = post(gateway)
    assert code in (502, 504) and PROVIDER not in json.dumps(result)
    assert len(calls) == 1
    assert gateway.receipt()["uncertain"]["requests"] == 1
    assert Decimal(gateway.receipt()["charged"]["cost_usd"]) > 0


def test_actual_overrun_commits_before_error_and_blocks_further_calls(gateway_config):
    usage = {"prompt_tokens": 20000, "completion_tokens": 10, "total_tokens": 20010}
    gateway, calls = make_gateway(gateway_config, chat_result(usage=usage))
    code, result = post(gateway)
    assert code == 502 and result["error"]["code"] == "usage_overrun"
    receipt = gateway.receipt()
    assert receipt["actual"]["input_tokens"] == 20000
    assert Decimal(receipt["charged"]["cost_usd"]) == Decimal("0.02002")
    assert receipt["blocked"] and not receipt["ready"]
    assert post(gateway)[0] == 503 and len(calls) == 1
    with sqlite3.connect(gateway_config.budget_store_path) as db:
        row = db.execute("SELECT id FROM calls").fetchone()
    with pytest.raises(AgentBudgetError, match="already settled"):
        gateway._budget.finish(row[0], gateway_config.chat)
    assert gateway.receipt()["actual"]["input_tokens"] == 20000


def test_atomic_concurrent_request_cap_and_day_cap_shared_across_runs(gateway_config):
    config = gateway_config.model_copy(update={"max_requests": 3, "max_requests_per_day": 3})
    gateway, calls = make_gateway(config)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: post(gateway)[0], range(8)))
    assert results.count(200) == 3 and results.count(429) == 5 and len(calls) == 3
    second = ResearchGateway(
        config.model_copy(update={"run_id": "second-run"}),
        bearer=BEARER + "second",
        provider_key=PROVIDER,
    )
    assert post(second, token=BEARER + "second")[0] == 429


@pytest.mark.parametrize("encoding", ["float", "base64"])
def test_embeddings_token_arrays_and_actual_usage(gateway_config, encoding):
    result = embedding_result()
    if encoding == "base64":
        result["data"][0]["embedding"] = base64.b64encode(struct.pack("<ff", 0.25, -0.5)).decode()
    gateway, calls = make_gateway(gateway_config, result)
    code, returned = post(
        gateway,
        {"model": EMBED, "input": [1, 2, 3], "encoding_format": encoding, "dimensions": 2},
        EMBEDDING_PATH,
    )
    assert code == 200 and returned == result
    assert str(calls[0].url) == "https://api.openai.com/v1/embeddings"
    assert gateway.receipt()["actual"]["output_tokens"] == 0
    assert gateway.receipt()["actual"]["input_tokens"] == 5


@pytest.mark.parametrize("mutation", ["nan", "dimensions", "index", "count", "base64"])
def test_invalid_embeddings_do_not_masquerade_as_features(gateway_config, mutation):
    result = embedding_result()
    body = {"model": EMBED, "input": "fixture text", "dimensions": 2}
    if mutation == "nan":
        result["data"][0]["embedding"] = [float("nan"), 0]
    elif mutation == "dimensions":
        result["data"][0]["embedding"] = [0.1]
    elif mutation == "index":
        result["data"][0]["index"] = 5
    elif mutation == "count":
        result["data"].append(copy.deepcopy(result["data"][0]))
    else:
        body["encoding_format"] = "base64"
        result["data"][0]["embedding"] = "not base64"
    gateway, _ = make_gateway(
        gateway_config, handler=lambda req: httpx.Response(200, content=json.dumps(result))
    )
    assert post(gateway, body, EMBEDDING_PATH)[0] == 502


def test_configuration_rejects_hidden_auth_rates_alias_and_naive_time(gateway_config):
    for change in (
        {"api_key": PROVIDER},
        {"max_cost_usd": "NaN"},
        {"max_requests": True},
        {"expires_at": "2026-01-01"},
    ):
        with pytest.raises(ValidationError):
            GatewayConfig.model_validate({**gateway_config.model_dump(), **change})
    for change in (
        {"source_uri": "https://secret@example.test/price"},
        {"source_uri": "https://example.test/price?key=secret"},
        {"model": "latest"},
        {"model": "openai/model"},
    ):
        with pytest.raises(ValidationError):
            GatewayConfig.model_validate(
                {
                    **gateway_config.model_dump(),
                    "chat": {**gateway_config.chat.model_dump(), **change},
                }
            )


def test_http_handler_real_framing_and_auth_with_unix_socketpair(gateway_config):
    gateway, calls = make_gateway(gateway_config)

    def exchange(request):
        client, server = socket.socketpair()

        def handle():
            try:
                GatewayHandler(server, ("local", 0), SimpleNamespace(gateway=gateway))
            finally:
                server.close()

        thread = threading.Thread(target=handle)
        thread.start()
        try:
            client.settimeout(5)
            client.sendall(request)
            client.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                part = client.recv(65536)
                if not part:
                    break
                data += part
        finally:
            client.close()
            server.close()
            thread.join(timeout=5)
        return data

    # The real handler sees duplicate Authorization and does not dispatch either.
    request = (
        "POST /v1/chat/completions HTTP/1.1\r\nHost: gateway\r\nAuthorization: Bearer "
        + BEARER
        + "\r\nAuthorization: Bearer bad\r\nContent-Length: 0\r\n\r\n"
    ).encode()
    response = exchange(request)
    assert b" 401 " in response and not calls
    request = (
        "GET /v1/budget HTTP/1.1\r\nHost: gateway\r\nAuthorization: Bearer " + BEARER + "\r\n\r\n"
    ).encode()
    response = exchange(request)
    assert b" 200 " in response and b'"config_hash"' in response and BEARER.encode() not in response
    body = json.dumps(chat_body()).encode()
    headers = (
        "POST /v1/chat/completions HTTP/1.1\r\nHost: gateway\r\nAuthorization: Bearer "
        + BEARER
        + "\r\nContent-Type: application/json\r\nContent-Length: "
        + str(len(body))
        + "\r\n\r\n"
    ).encode()
    response = exchange(headers + body)
    assert b" 200 " in response and len(calls) == 1
    for extra in (
        "Transfer-Encoding: chunked\r\nContent-Length: 0",
        "Content-Length: 0\r\nContent-Length: 0",
    ):
        malformed = (
            "POST /v1/chat/completions HTTP/1.1\r\nHost: gateway\r\nAuthorization: Bearer "
            + BEARER
            + "\r\n"
            + extra
            + "\r\n\r\n"
        ).encode()
        assert b" 400 " in exchange(malformed)
    assert len(calls) == 1


def test_cli_config_validation_and_missing_bearer(gateway_config, tmp_path, monkeypatch):
    config_path = tmp_path / "gateway.json"
    config_path.write_text(gateway_config.model_dump_json())
    monkeypatch.delenv("RESEARCH_GATEWAY_BEARER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    command = [sys.executable, "-m", "src.decision.research_gateway", "--config", str(config_path)]
    validated = subprocess.run(
        command + ["--check-config"], capture_output=True, text=True, check=True
    )
    assert json.loads(validated.stdout)["config_hash"] == gateway_config.content_hash()
    invalid = subprocess.run(command, capture_output=True, text=True)
    assert invalid.returncode == 2 and "invalid" in invalid.stderr


def test_total_deadline_cancels_slow_provider_and_reservation_remains(gateway_config):
    import asyncio

    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json=chat_result())

    config = gateway_config.model_copy(update={"request_timeout_seconds": 0.01})
    gateway = ResearchGateway(
        config, bearer=BEARER, provider_key=PROVIDER, transport=httpx.MockTransport(slow)
    )
    code, result = post(gateway)
    assert code == 504 and result["error"]["code"] == "timeout"
    assert gateway.receipt()["uncertain"]["requests"] == 1


def test_output_limit_breach_records_actual_but_blocks_even_below_total_reservation(gateway_config):
    result = chat_result(usage={"prompt_tokens": 1, "completion_tokens": 101, "total_tokens": 102})
    gateway, _ = make_gateway(gateway_config, result)
    assert post(gateway)[1]["error"]["code"] == "usage_overrun"
    assert gateway.receipt()["actual"]["output_tokens"] == 101
    assert gateway.receipt()["blocked"]


@pytest.mark.parametrize("update", [{"max_cost_usd": Decimal("0.001")}, {"max_tokens": 100}])
def test_spend_and_token_caps_block_before_http_and_readiness_is_false(gateway_config, update):
    gateway, calls = make_gateway(gateway_config.model_copy(update=update))
    assert not gateway.receipt()["ready"]
    assert post(gateway)[0] == 429 and not calls


def test_request_bytes_and_input_token_bounds(gateway_config):
    config = gateway_config.model_copy(update={"max_request_bytes": 1024, "max_input_tokens": 8200})
    gateway, calls = make_gateway(config)
    assert post(gateway)[1]["error"]["code"] == "input_limit"
    body = chat_body(messages=[{"role": "user", "content": "x" * 2048}])
    assert post(gateway, body)[0] == 413 and not calls


def test_huge_usage_retains_reservation_and_invalid_sampling_does_not_crash(gateway_config):
    result = chat_result(
        usage={"prompt_tokens": 10**100, "completion_tokens": 0, "total_tokens": 10**100}
    )
    gateway, _ = make_gateway(gateway_config, result)
    assert post(gateway)[1]["error"]["code"] == "usage_unknown"
    assert post(gateway, chat_body(temperature=10**1000))[0] == 400


def test_semantically_invalid_response_still_settles_verified_usage(gateway_config):
    result = chat_result(choices=[])
    gateway, _ = make_gateway(gateway_config, result)
    assert post(gateway)[1]["error"]["code"] == "invalid_response"
    assert gateway.receipt()["actual"]["input_tokens"] == 20
    assert gateway.receipt()["uncertain"]["requests"] == 0


def test_abrupt_termination_reservation_persists_without_false_actual_usage(gateway_config):
    gateway, _ = make_gateway(gateway_config)
    gateway._budget.reserve_bound(CHAT, 10000, 100)
    restarted, _ = make_gateway(gateway_config)
    before = restarted.receipt()
    assert before["statuses"] == {"reserved": 1}
    assert before["uncertain"]["reserved_tokens"] == 10100
    assert before["actual"]["requests_with_usage"] == 0
    assert restarted.dispatch("GET", RECEIPT_PATH, "Bearer " + BEARER)[0] == 200
    assert restarted.receipt() == before
