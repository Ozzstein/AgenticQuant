"""Closed refresh through real native parsing and durable source receipts."""

import json
from datetime import datetime, timedelta, timezone

import ccxt
import pytest

from src.data.adapters.base import ProviderError
from src.data.adapters.crypto_context import verify_derivative_record
from src.domain.common import RunContext
from src.operations import refresh
from src.operations.refresh import FetchTask, RefreshPlan, RefreshService
from src.operations.store import OperationsStore, canonical
from tests.test_crypto_context_sources import funding, native_case


def request(instrument, now, *, operation="funding", **updates):
    arguments = {
        "since": (now - timedelta(days=1)).isoformat(),
        "end": now.isoformat(),
        "limit": 2,
        "max_pages": 10,
        **updates,
    }
    return FetchTask(
        task_id="derivative-history", provider="ccxt", operation=operation,
        instrument=instrument, arguments_json=canonical(arguments).decode(),
        max_http_requests=4,
    )


def boundary(monkeypatch, case):
    exchange = case.adapter.exchange
    native_fetch = exchange.fetch
    requests = []

    def fetch(url, method="GET", headers=None, body=None):
        requests.append((url, method, headers, body))
        assert method == "GET" and body is None
        if url.endswith("/exchangeInfo"):
            exchange.last_http_response = json.dumps({"symbols": [case.market["info"]]})
            return {"symbols": [case.market["info"]]}
        return native_fetch(url, method)

    def markets():
        exchange.fetch("https://fapi.binance.com/fapi/v1/exchangeInfo", "GET")
        return {case.instrument.symbol: case.market}

    exchange.fetch = fetch
    exchange.load_markets = markets
    monkeypatch.setattr(ccxt, exchange.id, lambda settings: exchange)
    return requests


@pytest.mark.parametrize("venue", ["binance", "krakenfutures"])
def test_native_refresh_pins_pages_clocks_and_restarts_without_fetch(tmp_path, monkeypatch, venue):
    now = datetime.now(timezone.utc)
    rows = [funding(now - timedelta(hours=i)) for i in (12, 8, 4)]
    if venue == "krakenfutures":
        rows = [{"timestamp": (now - timedelta(hours=i)).isoformat(),
                 "relativeFundingRate": "-0.000123456789123456789"} for i in (12, 8, 4)]
    case = native_case(tmp_path, rows=rows, venue=venue)
    calls = boundary(monkeypatch, case)
    context = RunContext(run_id="crypto-refresh", mode="paper", as_of=now,
                         config_hash="a" * 64, code_version="fixture")
    task = request(case.instrument, now)
    plan = RefreshPlan(context=context, scheduled_for=now, tasks=(task,))
    operations = OperationsStore(tmp_path / "operations.db")
    receipt = RefreshService(operations, case.store, fetcher=refresh._fetch).run(plan)
    assert receipt["status"] == "completed"
    result = receipt["result"]
    report = result["fetches"][task.task_id]
    cutoff = RunContext.model_validate(result["context"]).as_of
    assert report["observed_http_requests"] == len(calls) == (3 if venue == "binance" else 2)
    values = []
    for source in report["sources"]:
        envelope = case.store.get(source["evidence_id"], as_of=cutoff)
        assert source["envelope_hash"] == envelope.content_hash()
        assert now < envelope.record.ingested_at == envelope.record.available_at <= cutoff
        if envelope.kind == "funding":
            values.append(verify_derivative_record(case.store, source["evidence_id"], as_of=cutoff))
    assert len(values) == 3
    assert {v["units"] for v in values} == {"fraction-per-funding-event"}
    restart = RefreshService(OperationsStore(operations.path), case.store,
                             fetcher=lambda *a: pytest.fail("completed refresh repeated"))
    assert restart.run(plan) == receipt


def test_total_native_budget_includes_contract_metadata_and_blocks_next_page(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    case = native_case(tmp_path, rows=[funding(now - timedelta(hours=i)) for i in (12, 8, 4)])
    calls = boundary(monkeypatch, case)
    task = request(case.instrument, now).model_copy(update={"max_http_requests": 2})
    context = RunContext(run_id="bounded-history", mode="paper", as_of=now,
                         config_hash="a" * 64, code_version="fixture")
    operations = OperationsStore(tmp_path / "operations.db")
    with pytest.raises(ProviderError, match="native_derivative_response_unavailable"):
        RefreshService(operations, case.store, fetcher=refresh._fetch).run(
            RefreshPlan(context=context, scheduled_for=now, tasks=(task,)))
    assert len(calls) == 2
    assert operations.status(context.run_id)["status"] == "failed"


@pytest.mark.parametrize("change", [
    {"end": None}, {"since": "2026-10-26T00:00:00"}, {"max_pages": True},
    {"limit": 0}, {"timeframe": "unknown"}, {"url": "https://example.test"},
])
def test_unsupported_history_inputs_fail_before_provider_admission(tmp_path, change):
    case = native_case(tmp_path)
    with pytest.raises(ValueError):
        request(case.instrument, datetime.now(timezone.utc), **change)


def test_spot_and_unsupported_futures_oi_are_not_admitted(tmp_path):
    case = native_case(tmp_path, venue="krakenfutures")
    with pytest.raises(ValueError, match="supported perpetual"):
        request(case.instrument, datetime.now(timezone.utc), operation="open_interest")
    spot = case.instrument.model_copy(update={"venue": "kraken", "symbol": "BTC/USD"})
    with pytest.raises(ValueError, match="supported perpetual"):
        request(spot, datetime.now(timezone.utc))
