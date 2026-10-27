"""Closed refresh to certified native execution evidence; HTTP fixtures only."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest

from src.data.adapters.base import ProviderError, ReadHTTP
from src.data.binance_execution import BinanceEvidenceReferenceResolver, freeze_binance_reference
from src.data.evidence.store import EvidenceStore
from src.domain.common import Instrument, RunContext
from src.execution.ledger import LedgerError
from src.operations import operating_fetch, refresh
from src.operations.refresh import FetchTask, RefreshPlan, RefreshService
from src.operations.store import OperationsStore, canonical

BTC = Instrument(
    instrument_id="spot:BTC-USDT",
    symbol="BTC/USDT",
    asset_class="crypto",
    quote_currency="USDT",
    venue="binance",
)


def reference_task(arguments):
    return FetchTask(
        task_id="native-reference",
        provider="operating",
        operation="binance_notional_reference",
        instrument=BTC,
        arguments_json=canonical(arguments).decode(),
    )


@pytest.mark.parametrize("network", ["mainnet", "testnet"])
@pytest.mark.parametrize("native", [True, False])
def test_refresh_freezes_network_specific_raw_reference_without_refetch(
    tmp_path, monkeypatch, network, native
):
    now = datetime.now(timezone.utc)
    context = RunContext(
        run_id="native-reference-refresh",
        mode="paper",
        as_of=now,
        config_hash="a" * 64,
        code_version="fixture",
        account_id="paper:binance",
    )
    plan = RefreshPlan(
        context=context,
        scheduled_for=now,
        tasks=(reference_task({"network": network, "average_minutes": 5}),),
    )
    origin = "testnet.binance.vision" if network == "testnet" else "api.binance.com"
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET" and request.url.host == origin
        assert dict(request.url.params) == {"symbol": "BTCUSDT"}
        timestamp = int((now - timedelta(seconds=1)).timestamp() * 1000)
        if request.url.path == "/api/v3/referencePrice":
            if native:
                return httpx.Response(
                    200,
                    json={
                        "symbol": "BTCUSDT",
                        "referencePrice": "100.000000000000000001",
                        "timestamp": timestamp,
                    },
                )
            return httpx.Response(
                400, json={"code": -2043, "msg": "Reference price not available."}
            )
        assert request.url.path == "/api/v3/avgPrice"
        return httpx.Response(
            200,
            json={
                "mins": 5,
                "price": "101.000000000000000001",
                "closeTime": timestamp,
            },
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
        result = service.run(plan)
        assert result["status"] == "completed"
        report = result["result"]
        receipt = report["fetches"]["native-reference"]
        assert receipt["observed_http_requests"] == len(requests) == (1 if native else 2)
        sources = [store.get(source["evidence_id"]) for source in receipt["sources"]]
        assert len(sources) == len(requests)
        cutoff = RunContext.model_validate(report["context"])
        execution_context = cutoff.model_copy(
            update={
                "mode": "paper" if network == "testnet" else "live",
                "account_id": "paper:binance" if network == "testnet" else "live:binance",
            }
        )
        # Mainnet source certification is read-only; it grants no execution authority.
        binding = freeze_binance_reference(
            store,
            BTC,
            execution_context,
            network=network,
            average_minutes=5,
            reference_evidence_id=sources[0].record.evidence_id,
            average_evidence_id=None if native else sources[1].record.evidence_id,
        )
        for source in binding.sources():
            assert source.record.requested_as_of == context.as_of
            assert context.as_of < source.record.available_at <= cutoff.as_of
            assert source.record.source_uri.startswith("https://" + origin + "/")
        resolver = BinanceEvidenceReferenceResolver(store, execution_context, (binding,))
        certified = resolver(BTC, cutoff.as_of, 5)
        assert certified.price == Decimal(
            "100.000000000000000001" if native else "101.000000000000000001"
        )
        assert certified.kind == ("venue-reference" if native else "weighted-average")
        with pytest.raises(ProviderError, match="network"):
            BinanceEvidenceReferenceResolver(
                store,
                execution_context.model_copy(
                    update={"mode": "live" if network == "testnet" else "paper"}
                ),
                (binding,),
            )
        with pytest.raises(LedgerError, match="denied"):
            resolver(BTC, cutoff.as_of + timedelta(seconds=31), 5)
        restart = RefreshService(
            OperationsStore(operations.path),
            store,
            fetcher=lambda *args: pytest.fail("completed refresh must not repeat HTTP"),
        )
        assert restart.run(plan) == result
        assert len(requests) == (1 if native else 2)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"average_minutes": True},
        {"average_minutes": -1},
        {"average_minutes": "5"},
        {"average_minutes": 5, "network": "custom"},
        {"average_minutes": 5, "url": "https://example.invalid"},
        {"average_minutes": 5, "price": "100"},
    ],
)
def test_reference_refresh_rejects_untyped_or_unapproved_arguments_before_dispatch(arguments):
    with pytest.raises(ValueError):
        reference_task(arguments)


def test_reference_refresh_requires_instrument_and_preserves_default_mainnet():
    task = reference_task({"average_minutes": 0})
    assert json.loads(task.arguments_json) == {"average_minutes": 0}
    with pytest.raises(ValueError, match="explicit instrument"):
        FetchTask.model_validate({**task.model_dump(mode="json"), "instrument": None})
