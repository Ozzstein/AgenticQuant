"""Closed scheduled refresh through native raw FX certification; fixture HTTP only."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest

from src.data.adapters.base import ProviderError, ReadHTTP
from src.data.evidence.store import EvidenceStore
from src.data.kraken_fx import parse_kraken_fx_reference
from src.domain.common import RunContext
from src.operations import operating_fetch, refresh
from src.operations.refresh import FetchSourceRef, FetchTask, RefreshPlan, RefreshService
from src.operations.store import OperationsStore, canonical
from tests.test_kraken_fx import FX


def fx_task(**updates):
    values = dict(
        task_id="fx",
        provider="operating",
        operation="kraken_fx_reference",
        instrument=FX,
        arguments_json=canonical({"market_id": "USDTZUSD"}).decode(),
    )
    return FetchTask.model_validate({**values, **updates})


@pytest.mark.parametrize("budget", [2, 3])
def test_native_fx_refresh_certification_budget_and_exact_dependency(tmp_path, monkeypatch, budget):
    now = datetime.now(timezone.utc)
    context = RunContext(
        run_id="fx-refresh",
        mode="paper",
        as_of=now,
        account_id="paper:one",
        config_hash="a" * 64,
        code_version="fixture",
    )
    stamp = format(Decimal(str(now.timestamp())) - Decimal("1.123456789"), "f")
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET" and request.url.host == "api.kraken.com"
        if request.url.path == "/0/public/AssetPairs":
            assert dict(request.url.params) == {"pair": "USDTZUSD", "aclass_base": "currency"}
            body = {
                "USDTZUSD": dict(
                    wsname="USDT/USD",
                    aclass_base="currency",
                    aclass_quote="currency",
                    base="USDT",
                    quote="ZUSD",
                    lot="unit",
                    lot_multiplier=1,
                    status="online",
                )
            }
        elif request.url.path == "/0/public/Assets":
            assert dict(request.url.params) == {"asset": "USDT,ZUSD", "aclass": "currency"}
            body = {
                "USDT": dict(altname="USDT", aclass="currency", status="enabled"),
                "ZUSD": dict(altname="USD", aclass="currency", status="enabled"),
            }
        else:
            assert request.url.path == "/0/public/Trades"
            assert dict(request.url.params) == {"pair": "USDTZUSD", "count": "1000"}
            body = {
                "USDTZUSD": [["0.9987000000000000001", "12", "STAMP", "b", "m", "", 123]],
                "last": "cursor",
            }
        return httpx.Response(
            200,
            content=json.dumps({"error": [], "result": body}).replace('"STAMP"', stamp).encode(),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            operating_fetch,
            "ReadHTTP",
            lambda provider, base, **kwargs: ReadHTTP(
                provider, base, client=client, min_interval=0, **kwargs
            ),
        )
        store = EvidenceStore(tmp_path / "evidence.db")
        operations = OperationsStore(tmp_path / "operations.db")
        service = RefreshService(operations, store, fetcher=refresh._fetch)
        plan = RefreshPlan(
            context=context, scheduled_for=now, tasks=(fx_task(max_http_requests=budget),)
        )
        if budget == 2:
            with pytest.raises(ProviderError, match="request_budget_exhausted"):
                service.run(plan)
            assert len(requests) == 2
            assert operations.status(context.run_id)["stages"]["fetch:fx"]["state"] == "failed"
            with store._connect() as db:
                assert (
                    db.execute("SELECT count(*) FROM evidence WHERE kind='fx'").fetchone()[0] == 0
                )
            return
        receipt = service.run(plan)
        assert receipt["status"] == "completed"
        report = receipt["result"]
        fetch = report["fetches"]["fx"]
        assert fetch["observed_http_requests"] == len(requests) == 3
        sources = [store.get(source["evidence_id"]) for source in fetch["sources"]]
        assert [source.kind for source in sources] == ["operating_raw"] * 3 + ["fx"]
        cutoff = RunContext.model_validate(report["context"])
        certified = parse_kraken_fx_reference(
            store,
            FX,
            cutoff,
            evidence_id=sources[-1].record.evidence_id,
            valuation_at=cutoff.as_of,
            max_age_seconds=30,
        )
        assert certified["rate"] == "0.9987000000000000001"
        assert certified["executable_conversion"] is False
        assert all(now < source.record.available_at <= cutoff.as_of for source in sources)
        assert service.run(plan)["result"] == report and len(requests) == 3
        consumer = FetchTask(
            task_id="reference",
            provider="operating",
            operation="kraken_reference",
            instrument=FX,
            arguments_json=canonical({"market_id": "USDTZUSD", "bar_ids": []}).decode(),
            source_arguments=(FetchSourceRef(argument="fx_evidence_id", task_id="fx", kind="fx"),),
        )
        resolved, dependencies = service._resolve(consumer, report["fetches"])
        assert (
            json.loads(resolved.arguments_json)["fx_evidence_id"] == sources[-1].record.evidence_id
        )
        assert dependencies
        with pytest.raises(ValueError):
            store.get(sources[-1].record.evidence_id, as_of=now - timedelta(seconds=1))


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"market_id": ""},
        {"market_id": "../Trades"},
        {"market_id": 1},
        {"market_id": "USDTZUSD", "assetVersion": 1},
        {"market_id": "USDTZUSD", "count": 1},
        {"market_id": "USDTZUSD", "endpoint": "https://example.test"},
    ],
)
def test_fx_refresh_rejects_missing_or_unapproved_native_arguments(arguments):
    with pytest.raises(ValueError):
        fx_task(arguments_json=canonical(arguments).decode())


@pytest.mark.parametrize(
    "updates",
    [
        {"symbol": "USD/USDT"},
        {"venue": "binance"},
        {"quote_currency": "USDT"},
        {"asset_class": "equity"},
    ],
)
def test_fx_refresh_rejects_unapproved_instrument_identity(updates):
    with pytest.raises(ValueError):
        fx_task(instrument=FX.model_copy(update=updates))
