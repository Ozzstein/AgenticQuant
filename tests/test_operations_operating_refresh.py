"""Operating refresh uses real adapters/stores and synthetic HTTP wires only."""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from src.data.adapters.base import ProviderError, ReadHTTP
from src.data.adapters.operating import verify_operating_capture
from src.data.calendars import sessions
from src.domain.common import RunContext
from src.operations import operating_fetch, refresh
from src.operations.refresh import FetchSourceRef, FetchTask, RefreshPlan, RefreshService
from src.operations.store import OperationError, OperationsStore, canonical
from tests.test_operating_capture import primary

pytest_plugins = ("tests.test_operating_capture",)


def ref(argument, task_id, kind):
    return FetchSourceRef(
        argument=argument,
        task_id=task_id,
        kind=kind,
        selection="all" if argument.endswith("ids") else "one",
    )


def task(name, operation, arguments=None, *, sources=(), instrument=None, **kwargs):
    return FetchTask(
        task_id=name,
        provider="operating",
        operation=operation,
        arguments_json=canonical(arguments or {}).decode(),
        source_arguments=sources,
        instrument=instrument,
        **kwargs,
    )


def current_context(f):
    return f.requested.model_copy(
        update={"run_id": "actual-refresh", "as_of": datetime.now(timezone.utc)}
    )


def wire_boundary(monkeypatch, f):
    for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "FMP_API_KEY"):
        monkeypatch.setenv(name, "offline-boundary-fixture")
    monkeypatch.setattr(
        operating_fetch,
        "ReadHTTP",
        lambda provider, base, **kw: ReadHTTP(
            provider, base, client=f.adapter.data.client, min_interval=0, **kw
        ),
    )
    original = f.adapter.data.client._transport

    def handler(request):
        if request.url.path == "/v2/calendar":
            f.requests.append(request)
            schedule = sessions(
                request.url.params["start"], request.url.params["end"], calendar="XNYS"
            )
            rows = [
                {
                    "date": day,
                    "open": opened.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M"),
                    "close": closed.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M"),
                }
                for day, (opened, closed) in schedule.items()
            ]
            return httpx.Response(200, json=rows)
        if request.url.path == "/v2/stocks/bars":
            f.requests.append(request)
            start = datetime.fromisoformat(request.url.params["start"])
            end = datetime.fromisoformat(request.url.params["end"])
            schedule = sessions(start.date().isoformat(), end.date().isoformat(), calendar="XNYS")
            rows = [
                {
                    "t": opened.isoformat(),
                    "o": "100",
                    "h": "102",
                    "l": "99",
                    "c": "101",
                    "v": "1000",
                    "vw": "100.123456789",
                }
                for opened, closed in schedule.values()
                if start <= opened and closed <= end
            ][-20:]
            return httpx.Response(
                200, json={"bars": {f.instrument.symbol: rows}, "next_page_token": None}
            )
        return original.handle_request(request)

    monkeypatch.setattr(f.adapter.data.client, "_transport", httpx.MockTransport(handler))


def chain(f, as_of=None):
    now = as_of or datetime.now(timezone.utc)
    start = (now - timedelta(days=60)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (
        task(
            "calendar",
            "alpaca_calendar",
            {"start": start.date().isoformat(), "end": now.date().isoformat()},
        ),
        task(
            "bars",
            "alpaca_bars",
            {
                "start": start.isoformat(),
                "end": now.isoformat(),
                "feed": "iex",
                "symbol_asof": now.date().isoformat(),
            },
            instrument=f.instrument,
            sources=(ref("calendar_id", "calendar", "operating_calendar"),),
        ),
        task(
            "reference",
            "alpaca_reference",
            {"asset_id": "asset-123"},
            instrument=f.instrument,
            sources=(
                ref("calendar_id", "calendar", "operating_calendar"),
                ref("bar_ids", "bars", "bar"),
            ),
        ),
        task(
            "universe",
            "current_universe",
            {
                "instruments": [f.instrument.model_dump(mode="json")],
                "universe_version": "captured-current-only",
            },
            sources=(ref("reference_ids", "reference", "operating_reference"),),
        ),
    )


def test_actual_closed_chain_preserves_clock_and_replays_exact_dependency_receipts(
    equity_capture, tmp_path, monkeypatch
):
    f = equity_capture
    wire_boundary(monkeypatch, f)
    ctx = current_context(f)
    plan = RefreshPlan(context=ctx, scheduled_for=ctx.as_of, tasks=chain(f,ctx.as_of))
    ops = OperationsStore(tmp_path / "operations.db")
    service = RefreshService(ops, f.store, fetcher=refresh._fetch)
    before = len(f.requests)
    result = service.run(plan)
    assert result["status"] == "completed" and len(f.requests) - before == 4
    report = result["result"]
    cutoff = RunContext.model_validate(report["context"]).as_of
    receipt = report["fetches"]["reference"]
    assert receipt["observed_http_requests"] == 2
    assert len(receipt["dependency_sources"]) == 21
    reference = next(
        f.store.get(s["evidence_id"])
        for s in receipt["sources"]
        if f.store.get(s["evidence_id"]).kind == "operating_reference"
    )
    normalized = verify_operating_capture(f.store, reference.record.evidence_id, as_of=cutoff)
    assert ctx.as_of < reference.record.ingested_at == reference.record.available_at <= cutoff
    assert (
        datetime.fromisoformat(normalized["instrument_metadata"][0]["as_of"])
        == reference.record.available_at
    )
    assert reference.record.requested_as_of == ctx.as_of
    universe_receipt = report["fetches"]["universe"]
    assert universe_receipt["dependency_sources"][0]["evidence_id"] == reference.record.evidence_id
    assert receipt["task_hash"] == plan.tasks[2].content_hash()
    assert receipt["resolved_task_hash"] != receipt["task_hash"]
    restart = RefreshService(
        OperationsStore(ops.path), f.store, fetcher=lambda *a: pytest.fail("repeated fetch")
    )

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(timezone.utc) + timedelta(days=2)

    monkeypatch.setattr(refresh, "datetime", Later)
    assert restart.run(plan) == result
    assert len(f.requests) - before == 4
    assert "offline-boundary-fixture" not in json.dumps(result)


def test_source_resolution_rejects_changed_identity_missing_kind_and_unavailable(
    equity_capture, tmp_path
):
    f = equity_capture
    service = RefreshService(OperationsStore(tmp_path / "operations.db"), f.store)
    target = chain(f)[3]
    source = {
        "evidence_id": f.reference.record.evidence_id,
        "envelope_hash": f.reference.content_hash(),
    }
    for receipt, error in (
        (
            {"status": "available", "sources": [{**source, "envelope_hash": "f" * 64}]},
            "identity changed",
        ),
        ({"status": "available", "sources": []}, "coverage"),
        ({"status": "missing", "sources": [source]}, "unavailable"),
    ):
        with pytest.raises(OperationError, match=error):
            service._resolve(target, {"reference": receipt})


def test_total_request_cap_spans_multiple_operating_hosts(equity_capture, monkeypatch):
    f = equity_capture
    wire_boundary(monkeypatch, f)
    value = task(
        "reference",
        "alpaca_reference",
        {
            "asset_id": "asset-123",
            "calendar_id": f.calendar.record.evidence_id,
            "bar_ids": list(f.bar_ids),
        },
        instrument=f.instrument,
        max_http_requests=1,
    )
    before = len(f.requests)
    with pytest.raises(ProviderError, match="request_budget_exhausted"):
        refresh._fetch(value, current_context(f), f.store)
    assert len(f.requests) - before == 1
    # First host's actual raw response is retained even though the second was denied.
    assert f.store.query(as_of=datetime.now(timezone.utc), kind="operating_raw").records


def test_bundle_is_archived_before_completed_refresh_and_keeps_exact_eight_sources(
    equity_capture, tmp_path
):
    f = equity_capture
    actions = primary(
        f.adapter.corporate_actions(
            f.instrument,
            f.requested,
            start="2024-03-01",
            end="2024-03-11",
            calendar_id=f.calendar.record.evidence_id,
        ),
        "operating_actions",
    )
    f.clock.at = datetime(2024, 3, 12, 13, 32, tzinfo=timezone.utc)
    print_context = f.requested.model_copy(update={"as_of": f.clock.at})
    prints = primary(
        f.adapter.trades(
            f.instrument,
            print_context,
            start=datetime(2024, 3, 12, 13, 30, tzinfo=timezone.utc),
            end=f.clock.at,
            calendar_id=f.calendar.record.evidence_id,
            feed="iex",
            symbol_asof="2024-03-12",
        ),
        "operating_prints",
    )
    ctx = current_context(f)
    universe = task(
        "universe",
        "current_universe",
        {
            "instruments": [f.instrument.model_dump(mode="json")],
            "reference_ids": [f.reference.record.evidence_id],
            "universe_version": "current-only",
        },
    )
    bundle = task(
        "bundle",
        "economic_bundle",
        {
            "instruments": [f.instrument.model_dump(mode="json")],
            "market_start": "2024-02-01T00:00:00+00:00",
            "reference_ids": [f.reference.record.evidence_id],
            "calendar_ids": [f.calendar.record.evidence_id],
            "print_ids": [prints.record.evidence_id],
            "action_ids": [actions.record.evidence_id],
        },
        sources=(ref("universe_ids", "universe", "operating_universe"),),
    )
    plan = RefreshPlan(context=ctx, scheduled_for=ctx.as_of, tasks=(universe, bundle))
    ops = OperationsStore(tmp_path / "operations.db")
    report = RefreshService(ops, f.store, fetcher=refresh._fetch).run(plan)["result"]
    cutoff = RunContext.model_validate(report["context"]).as_of
    identity = report["fetches"]["bundle"]["sources"][0]["evidence_id"]
    envelope = f.store.get(identity, as_of=cutoff)
    value = verify_operating_capture(f.store, identity, as_of=cutoff)
    assert (
        envelope.kind == "operating_bundle" and ctx.as_of < envelope.record.available_at <= cutoff
    )
    bindings = dict(value["economic_sources"])
    assert set(bindings) == {
        "execution_bars",
        "instrument_metadata",
        "universe_history",
        "corporate_actions",
        "venue_rules",
        "calendar",
        "opening_liquidity",
        "availability_audit",
    }
    assert all(
        datetime.fromisoformat(b["available_at"]) <= envelope.record.available_at
        for rows in bindings.values()
        for b in rows
    )
    audit = f.store.normalized(bindings["availability_audit"][0]["evidence_id"], as_of=cutoff)
    assert [r["evidence_id"] for r in audit["availability_audit"]] == value["snapshot_evidence_ids"]
    assert report["fetches"]["bundle"]["observed_http_requests"] == 0
    loaded = operating_fetch.load_operating_sources(
        OperationsStore(ops.path, read_only=True), f.store, ctx.run_id
    )
    assert [
        [kind, [s.model_dump(mode="json") for s in sources]] for kind, sources in loaded
    ] == value["economic_sources"]
    # Historical fixtures do not gain earlier availability from today's bundle.
    assert f.store.get(prints.record.evidence_id).record == prints.record


@pytest.mark.parametrize(
    "operation,arguments",
    [
        ("submit_order", {}),
        ("alpaca_calendar", {"url": "https://arbitrary.test"}),
        ("economic_bundle", {"ingested_at": "2000-01-01"}),
    ],
)
def test_operation_and_arguments_are_closed(operation, arguments):
    with pytest.raises(ValueError, match="not permitted"):
        task("bad", operation, arguments)


def test_dependencies_are_typed_unique_prior_tasks_and_bundle_is_final(equity_capture):
    f = equity_capture
    ctx = current_context(f)
    with pytest.raises(ValueError, match="earlier tasks"):
        RefreshPlan(context=ctx, scheduled_for=ctx.as_of, tasks=tuple(reversed(chain(f))))
    with pytest.raises(ValueError, match="scalar or array"):
        task(
            "bad",
            "alpaca_bars",
            sources=(
                FetchSourceRef(argument="calendar_id", task_id="x", kind="x", selection="all"),
            ),
        )
    with pytest.raises(ValueError, match="duplicate"):
        task(
            "bad",
            "alpaca_bars",
            sources=(ref("calendar_id", "x", "a"), ref("calendar_id", "y", "a")),
        )
    with pytest.raises(ValueError, match="final refresh"):
        RefreshPlan(
            context=ctx,
            scheduled_for=ctx.as_of,
            tasks=(task("bundle", "economic_bundle"), chain(f)[0]),
        )
