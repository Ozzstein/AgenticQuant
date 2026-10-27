"""Actual native dominance parsing through the closed durable refresh boundary."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext

import httpx
import pytest

from src.data.adapters.base import ProviderError, ReadHTTP
from src.data.adapters.coingecko_context import verify_dominance_record
from src.data.evidence.store import EvidenceStore
from src.domain.common import RunContext
from src.operations import refresh
from src.operations.refresh import FetchTask, RefreshPlan, RefreshService
from src.operations.store import OperationsStore, canonical


def task(operation="current_dominance", *, arguments=None, budget=3):
    return FetchTask(task_id="dominance", provider="coingecko", operation=operation,
                     arguments_json=canonical(arguments or {}).decode(), max_http_requests=budget)


def setup(tmp_path, monkeypatch, bodies, *, configured="fixture-configured-key"):
    requests, clients = [], []

    def reply(request):
        requests.append(request)
        return httpx.Response(200, json=bodies[len(requests)-1])

    def transport(provider, base, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(reply))
        clients.append(client)
        return ReadHTTP(provider, base, client=client, min_interval=0, **kwargs)

    monkeypatch.setattr(refresh, "ReadHTTP", transport)
    monkeypatch.setenv("AIQUANT_COINGECKO_API_KEY", configured)
    monkeypatch.setenv("COINGECKO_API_KEY", "fixture-legacy-key" if configured else "")
    store = EvidenceStore(tmp_path / "sources.db")
    operations = OperationsStore(tmp_path / "operations.db")
    return store, operations, requests, clients


def run(store, operations, at, selected):
    context = RunContext(run_id="dominance-refresh", mode="paper", as_of=at,
                         config_hash="a"*64, code_version="fixture")
    return RefreshService(operations, store, fetcher=refresh._fetch).run(
        RefreshPlan(context=context, scheduled_for=at, tasks=(selected,)))


@pytest.mark.parametrize("tier", ["demo", "pro"])
def test_current_refresh_uses_exact_source_clocks_header_key_and_no_repeat(tmp_path, monkeypatch, tier):
    at = datetime.now(timezone.utc)
    body = {"data": {"updated_at": int(at.timestamp()), "market_cap_percentage": {"btc": 60}}}
    store, ops, calls, clients = setup(tmp_path, monkeypatch, [body])
    try:
        result = run(store, ops, at, task(arguments={"tier":tier}))
        assert result["status"] == "completed" and len(calls) == 1
        request = calls[0]
        assert request.url.host == ("api.coingecko.com" if tier=="demo" else "pro-api.coingecko.com")
        assert request.url.path == "/api/v3/global" and not request.url.query
        assert request.headers[f"x-cg-{tier}-api-key"] == "fixture-configured-key"
        report = result["result"]["fetches"]["dominance"]
        cutoff = datetime.fromisoformat(result["result"]["completed_at"])
        assert report["observed_http_requests"] == 1
        for source in report["sources"]:
            envelope = store.get(source["evidence_id"], as_of=cutoff)
            assert source["envelope_hash"] == envelope.content_hash()
            assert at < envelope.record.ingested_at == envelope.record.available_at <= cutoff
            if envelope.kind == "btc_dominance":
                value = verify_dominance_record(store, source["evidence_id"], as_of=cutoff)
                assert Decimal(value["value"]) == Decimal(".6")
            assert "fixture-configured-key" not in envelope.model_dump_json()
        assert "fixture-configured-key" not in json.dumps(result)
        assert run(store, OperationsStore(ops.path), at, task(arguments={"tier":tier})) == result
        assert len(calls) == 1
    finally:
        for client in clients:
            client.close()


def test_pro_history_shares_budget_and_reconstructs_exact_native_cap_ratio(tmp_path, monkeypatch):
    at = datetime.now(timezone.utc)
    stamp = int((at-timedelta(days=1)).timestamp()) * 1000
    bodies = [{"market_cap_chart":{"market_cap":[[stamp,3]]}}, {"market_caps":[[stamp,1]]}]
    store, ops, calls, clients = setup(tmp_path, monkeypatch, bodies)
    try:
        result = run(store, ops, at, task("dominance_history", arguments={"tier":"pro", "days":"30"}, budget=2))
        cutoff = datetime.fromisoformat(result["result"]["completed_at"])
        report = result["result"]["fetches"]["dominance"]
        assert report["observed_http_requests"] == len(calls) == 2
        assert all(request.url.host == "pro-api.coingecko.com" for request in calls)
        assert calls[1].url.path == "/api/v3/coins/bitcoin/market_chart/range"
        assert dict(calls[1].url.params)["precision"] == "full"
        envelope = next(store.get(item["evidence_id"]) for item in report["sources"]
                        if store.get(item["evidence_id"]).kind == "btc_dominance")
        value = verify_dominance_record(store, envelope.record.evidence_id, as_of=cutoff)
        with localcontext() as arithmetic:
            arithmetic.prec = 50
            assert Decimal(value["value"]) == Decimal(1)/Decimal(3)
        assert len(value["source_manifest"]) == 2
    finally:
        for client in clients:
            client.close()


def test_history_second_endpoint_cannot_exceed_shared_request_cap(tmp_path, monkeypatch):
    at = datetime.now(timezone.utc)
    stamp = int((at-timedelta(days=1)).timestamp())*1000
    store, ops, calls, clients = setup(tmp_path, monkeypatch,
        [{"market_cap_chart":{"market_cap":[[stamp,3]]}}])
    try:
        with pytest.raises(ProviderError, match="request_budget_exhausted"):
            run(store, ops, at, task("dominance_history", arguments={"tier":"pro"}, budget=1))
        assert len(calls) == 1 and ops.status("dominance-refresh")["status"] == "failed"
        assert not store.query(kind="btc_dominance", as_of=datetime.now(timezone.utc)).records
        assert len(store.query(kind="crypto_raw", as_of=datetime.now(timezone.utc)).records) == 1
    finally:
        for client in clients:
            client.close()


def test_pro_missing_configured_access_stops_before_first_http_request(tmp_path, monkeypatch):
    at = datetime.now(timezone.utc)
    store, ops, calls, clients = setup(tmp_path, monkeypatch, [], configured="")
    try:
        with pytest.raises(ProviderError, match="pro_entitlement"):
            run(store, ops, at, task("dominance_history", arguments={"tier":"pro"}))
        assert not calls and ops.status("dominance-refresh")["status"] == "failed"
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("operation,arguments", [
    ("current_dominance", {"tier":"custom"}),
    ("current_dominance", {"tier":[]}),
    ("current_dominance", {"api_key":"caller-key"}),
    ("current_dominance", {"url":"https://example.test"}),
    ("current_dominance", {"days":"30"}),
    ("dominance_history", {}),
    ("dominance_history", {"tier":"demo"}),
    ("dominance_history", {"tier":"pro", "days":30}),
    ("dominance_history", {"tier":"pro", "days":"2"}),
])
def test_closed_inputs_reject_before_transport_admission(operation, arguments):
    with pytest.raises(ValueError):
        task(operation, arguments=arguments)
