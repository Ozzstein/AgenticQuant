"""Actual refresh -> authoritative receipt reader -> macro/contextual preparation."""

from datetime import datetime, timezone

import httpx
import pytest
from src.domain.portfolio import InstrumentRisk, OpeningEvent
from src.execution.venue import VenueRules
from src.operations.refresh import CaptureStore, FetchTask, RefreshPlan, RefreshService
from src.operations.store import OperationsStore
from src.research.evidence import SourceBinding
from src.research.registry import RegistryError

from src.data.adapters.base import ReadHTTP, store_observation
from src.data.adapters.fundamentals import FredAdapter
from src.data.adapters.operating import KINDS, OperatingCapture
from src.data.operating_plan import prepare_operating_data
from src.research.regime_experiments import FrozenEvidence
from tests.test_operating_plan import NAMES, archive
from tests.test_operating_plan import actual_sources as actual_sources

UTC = timezone.utc


def mock_http(provider, payload):
    return ReadHTTP(
        provider,
        "https://fixture.test",
        min_interval=0,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ),
    )


def test_completed_refresh_uses_exact_actual_task_sources(actual_sources, tmp_path):
    f = actual_sources
    ambient = archive(
        f.store,
        "ambient-context",
        {"pe": 999},
        f.snapshot.bars[-1].closed_at,
        instrument=f.snapshot.bars[-1].instrument,
        kind="fundamentals",
        provider="fmp",
    )
    requested = f.kwargs["context"].model_copy(
        update={"run_id": "actual-preparation-refresh", "as_of": datetime.now(UTC)}
    )
    tasks = tuple(
        FetchTask(task_id=name, provider="fred", operation="observations") for name in NAMES
    )
    tasks += (
        FetchTask(task_id="contextual", provider="fmp", operation="profile"),
        FetchTask(task_id="bundle", provider="operating", operation="economic_bundle"),
    )
    recipe = f.recipe.model_copy(
        update={"series": tuple(s.model_copy(update={"provider": "fred"}) for s in f.recipe.series)}
    )

    def fetcher(task, context, store):
        if task.provider == "fred":
            entries = []
            for source in f.groups[task.task_id]:
                value = store.normalized(source.evidence_id, as_of=requested.as_of)
                entries.append(
                    dict(
                        date=value["date"],
                        value=str(value["value"]),
                        realtime_start=value["date"],
                        realtime_end="9999-12-31",
                    )
                )
            http = mock_http("fred", {"observations": entries})
            result = FredAdapter(CaptureStore(store), "fixture-only", http=http).observations(
                task.task_id.upper(),
                context,
                ingested_at=datetime.now(UTC),
                start=recipe.start_session.isoformat(),
                end=context.as_of.date().isoformat(),
            )
            return {
                "status": result.status,
                "evidence_ids": [e.record.evidence_id for e in result.evidence],
            }
        if task.task_id == "contextual":
            item = store_observation(
                store,
                raw=b'{"pe":"12"}',
                normalized={"pe": "12"},
                provider="fmp",
                version="synthetic-test-1",
                source_uri="test://context",
                kind="fundamentals",
                logical_key="fixture-pe",
                context=context,
                ingested_at=datetime.now(UTC),
                instrument=f.snapshot.bars[-1].instrument,
            )
            return {"status": "available", "evidence_ids": [item.record.evidence_id]}
        capture = OperatingCapture(store)
        bar = f.snapshot.bars[-1]
        opening = OpeningEvent(
            event_id="source-print",
            instrument=bar.instrument,
            opened_at=bar.opened_at,
            price=bar.open,
            opening_volume=7,
            calendar="XNYS",
            calendar_version="test-source-1",
            session=bar.session,
            session_open=bar.opened_at,
            session_close=bar.closed_at,
        )
        metadata = InstrumentRisk(
            instrument=bar.instrument,
            sector="Technology",
            adv_base=100000000,
            as_of=context.as_of,
            lot_size=".001",
        )
        rules = VenueRules(
            venue="paper",
            instrument=bar.instrument,
            symbol=bar.instrument.symbol,
            market_type="equity",
            active=True,
            available_at=context.as_of,
            quantity_step=".001",
            price_tick=".01",
            minimum_quantity=".001",
            minimum_notional=1,
            order_types=("market",),
            time_in_force=("GTC",),
        )
        event = opening.model_dump(mode="json")
        arrays = dict(
            opening_events=[event],
            instrument_metadata=[metadata.model_dump(mode="json")],
            universe_history=[f.universe.model_dump(mode="json")],
            corporate_actions=[],
            venue_rules=[rules.model_dump(mode="json")],
            calendar=[
                {
                    k: event[k]
                    for k in (
                        "calendar",
                        "calendar_version",
                        "session",
                        "session_open",
                        "session_close",
                    )
                }
            ],
            opening_liquidity=[
                {
                    k: event[k]
                    for k in ("event_id", "instrument", "opened_at", "opening_volume", "fx_rate")
                }
            ],
            availability_audit=[r.model_dump(mode="json") for r in f.snapshot.evidence],
        )
        wire, _ = capture.wire(
            mock_http("fixture", arrays),
            "/fixture-economics",
            context,
            upstream="synthetic-test-only",
        )
        economics = capture.derived(
            context,
            [wire],
            lambda at: arrays,
            kind="operating_economics",
            logical_key="test-economics",
        )
        bindings = tuple(
            (
                kind,
                (
                    SourceBinding(
                        evidence_id=economics.record.evidence_id,
                        pointer="/" + kind,
                        available_at=economics.record.available_at,
                    ),
                ),
            )
            for kind in KINDS
        )
        bundle = capture.derived(
            context,
            [economics],
            lambda at: dict(
                economic_sources=[
                    (kind, [s.model_dump(mode="json") for s in values]) for kind, values in bindings
                ],
                snapshot_evidence_ids=[r.evidence_id for r in f.snapshot.evidence],
            ),
            kind="operating_bundle",
            logical_key="final-bundle",
        )
        return {"status": "available", "evidence_ids": [bundle.record.evidence_id]}

    operations = OperationsStore(tmp_path / "operations.sqlite")
    plan = RefreshPlan(context=requested, scheduled_for=requested.as_of, tasks=tasks)
    report = RefreshService(operations, f.store, fetcher=fetcher).run(plan)["result"]
    prepared = prepare_operating_data(
        OperationsStore(operations.path, read_only=True),
        f.store,
        requested.run_id,
        recipe,
        breadth_snapshot=f.snapshot,
        universes=f.kwargs["universes"],
        universe_sources=f.kwargs["universe_sources"],
    )
    assert prepared.context.as_of == datetime.fromisoformat(report["completed_at"])
    assert tuple(kind for kind, _ in prepared.economic_sources) == KINDS
    assert len(prepared.macro_dataset.raw) == 210
    assert prepared.recipe_hash == recipe.content_hash()
    assert prepared.contextual_sources == tuple(
        FrozenEvidence.model_validate(v) for v in report["fetches"]["contextual"]["sources"]
    )
    assert ambient.record.evidence_id not in {s.evidence_id for s in prepared.contextual_sources}
    actual_macro = {v["evidence_id"] for name in NAMES for v in report["fetches"][name]["sources"]}
    used = {source.evidence_id for row in prepared.macro_dataset.raw for _, source in row.values}
    assert used <= actual_macro
    assert not used.intersection(s.evidence_id for values in f.groups.values() for s in values)
    wrong = recipe.model_copy(update={"contextual_task_ids": ("missing-task",)})
    with pytest.raises(RegistryError, match="task missing"):
        prepare_operating_data(
            operations,
            f.store,
            requested.run_id,
            wrong,
            breadth_snapshot=f.snapshot,
            universes=f.kwargs["universes"],
            universe_sources=f.kwargs["universe_sources"],
        )
