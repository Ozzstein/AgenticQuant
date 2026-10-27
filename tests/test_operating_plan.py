"""Real immutable source/materialization boundary; all economic fixtures synthetic."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest
from src.domain.portfolio import UniverseEligibility
from src.research.evidence import SourceBinding
from src.research.registry import RegistryError

from src.data.calendars import sessions
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.data.operating_plan import (
    MacroSeriesRecipe,
    OperatingPreparationRecipe,
    _candidates,
    _select,
    build_operating_macro_dataset,
)
from src.domain.common import EvidenceRecord, Instrument, RunContext
from src.domain.market import MarketBar, MarketSnapshot
from src.research.regime_experiments import FrozenEvidence, materialize_macro_dataset

UTC = timezone.utc
POLICY = "captured-vintage-at-fit-1"
NAMES = ("vix", "dgs10", "dgs2", "sp500", "dxy", "hyg", "tlt")


def archive(
    store,
    key,
    payload,
    observed,
    *,
    instrument=None,
    kind="macro",
    provider="fixture-dated-macro",
    quality="usable",
):
    captured = datetime.now(UTC)
    raw = canonical_bytes(payload)
    record = EvidenceRecord(
        evidence_id=key,
        provider=provider,
        provider_version="synthetic-test-1",
        source_uri="test://" + key,
        raw_hash=digest(raw),
        ingested_at=captured,
        available_at=captured,
        observed_at=observed,
        requested_as_of=captured,
        availability_basis="documented",
        quality=quality,
        instrument_ids=(instrument.instrument_id,) if instrument else (),
    )
    return store.put(record, raw, kind=kind, logical_key=key, normalized=payload)


@pytest.fixture(scope="module")
def actual_sources(tmp_path_factory):
    store = EvidenceStore(tmp_path_factory.mktemp("operating-preparation") / "evidence.sqlite")
    cutoff = datetime.now(UTC)
    schedule = [
        (day, bounds)
        for day, bounds in sessions(
            (cutoff.date() - timedelta(days=400)).isoformat(),
            cutoff.date().isoformat(),
            calendar="XNYS",
        ).items()
        if bounds[1] <= cutoff
    ][-210:]
    instrument = Instrument(
        instrument_id="equity:fixture",
        symbol="FIX",
        asset_class="equity",
        quote_currency="USD",
        venue="XNYS",
    )
    bars, records, groups = [], [], {name: [] for name in NAMES}
    for number, (day, (opened, closed)) in enumerate(schedule):
        price = Decimal(str(100 + number / 10 + 2 * np.sin(number / 10)))
        bar = MarketBar(
            instrument=instrument,
            opened_at=opened,
            closed_at=closed,
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1000,
            evidence_id=f"breadth-{number}",
            price_basis="raw",
            market_type="equity",
            calendar="XNYS",
            session=day,
        )
        envelope = archive(
            store,
            bar.evidence_id,
            bar.model_dump(mode="json"),
            closed,
            instrument=instrument,
            kind="bar",
        )
        bars.append(bar)
        records.append(envelope.record)
        for offset, name in enumerate(NAMES):
            envelope = archive(
                store,
                f"{name}-{number}",
                {
                    "series_id": name.upper(),
                    "date": day,
                    "value": 2 + offset + number / 1000 + np.sin(number / 8) / 10,
                },
                closed,
            )
            groups[name].append(
                FrozenEvidence(
                    evidence_id=envelope.record.evidence_id, envelope_hash=envelope.content_hash()
                )
            )
    effective = schedule[0][1][0] - timedelta(days=1)
    captured = datetime.now(UTC)
    universe = UniverseEligibility(
        universe_version="historical-fixture-1",
        data_snapshot_id="0" * 64,
        effective_at=effective,
        available_at=captured,
        instrument_ids=(instrument.instrument_id,),
    )
    membership = archive(
        store,
        "historical-membership",
        [universe.model_dump(mode="json")],
        effective,
        kind="universe",
    )
    # The source capture is later by microseconds: the explicit membership is
    # available only after that real source exists, never at its economic date.
    universe = universe.model_copy(update={"available_at": membership.record.available_at})
    # Archive the final source-backed observation without mutating either record.
    raw = canonical_bytes([universe.model_dump(mode="json")])
    record = membership.record.model_copy(
        update={"evidence_id": "membership-final", "raw_hash": digest(raw)}
    )
    membership = store.put(
        record,
        raw,
        kind="universe",
        logical_key="history",
        normalized=[universe.model_dump(mode="json")],
    )
    context = RunContext(
        run_id="current-source-build",
        mode="paper",
        as_of=datetime.now(UTC),
        config_hash="0" * 64,
        code_version="fixture",
        account_id="paper:fixture",
    )
    snapshot = MarketSnapshot(context=context, bars=tuple(bars), evidence=tuple(records))
    recipe = OperatingPreparationRecipe(
        start_session=schedule[0][1][1].date(),
        universe_version=universe.universe_version,
        availability_policy=POLICY,
        series=tuple(
            MacroSeriesRecipe(
                name=name,
                task_ids=(name,),
                provider="fixture-dated-macro",
                kind="macro",
                value_pointer="/value",
                observation_pointer="/date",
                filters=(("/series_id", name.upper()),),
            )
            for name in NAMES
        ),
        contextual_task_ids=("contextual",),
    )
    kwargs = dict(
        context=context,
        series_sources=tuple((name, tuple(groups[name])) for name in NAMES),
        breadth_snapshot=snapshot,
        universes=(universe,),
        universe_sources=(
            SourceBinding(
                evidence_id=membership.record.evidence_id,
                pointer="",
                available_at=membership.record.available_at,
            ),
        ),
    )
    return SimpleNamespace(
        store=store,
        recipe=recipe,
        kwargs=kwargs,
        snapshot=snapshot,
        groups=groups,
        universe=universe,
    )


@pytest.fixture(scope="module")
def built(actual_sources):
    f = actual_sources
    return build_operating_macro_dataset(f.store, f.recipe, **f.kwargs)


def test_actual_dated_scalars_and_breadth_materialize_without_restamping(actual_sources, built):
    f = actual_sources
    assert len(built.raw) == 210
    assert built.raw[-1].at == f.snapshot.bars[-1].closed_at
    assert built.breadth_snapshot.bars == f.snapshot.bars
    assert built.universes == (f.universe,)
    assert built.availability_policy == POLICY
    for observation in built.raw:
        for name, source in observation.values:
            envelope = f.store.get(source.evidence_id)
            assert source.pointer == "/value"
            assert envelope.record.observed_at == observation.at
            assert source.available_at > observation.at
            assert envelope.record.available_at <= f.kwargs["context"].as_of
    data = materialize_macro_dataset(built, f.store, availability_policy=POLICY)
    assert len(data.signals) == 11  # 200 actual observations, no fabricated warmup.
    assert np.isfinite(data.signals).all().all()
    with pytest.raises(RegistryError, match="polic"):
        materialize_macro_dataset(built, f.store)


def test_same_capture_fails_historical_dates_and_current_membership_is_not_history(actual_sources):
    f = actual_sources
    historical = f.recipe.model_copy(update={"availability_policy": "decision-time-vintage-1"})
    with pytest.raises(RegistryError, match="coverage"):
        build_operating_macro_dataset(f.store, historical, **f.kwargs)
    current = f.universe.model_copy(update={"effective_at": f.kwargs["context"].as_of})
    with pytest.raises(RegistryError, match="membership"):
        build_operating_macro_dataset(f.store, f.recipe, **{**f.kwargs, "universes": (current,)})


def test_truncated_current_breadth_cannot_redefine_latest_completed_session(actual_sources):
    f = actual_sources
    shortened = f.snapshot.model_copy(update={"bars": f.snapshot.bars[:-1]})
    with pytest.raises(RegistryError, match="coverage gap"):
        build_operating_macro_dataset(
            f.store, f.recipe, **{**f.kwargs, "breadth_snapshot": shortened}
        )


def test_conflicting_sources_and_forged_envelope_hash_are_rejected(actual_sources):
    f = actual_sources
    old = f.groups["vix"][-1]
    observed = f.store.get(old.evidence_id).record.observed_at
    conflict = archive(
        f.store,
        "conflicting-vix",
        {"series_id": "VIX", "date": observed.date().isoformat(), "value": 999},
        observed,
    )
    sources = dict(f.kwargs["series_sources"])
    sources["vix"] = (
        *sources["vix"],
        FrozenEvidence(
            evidence_id=conflict.record.evidence_id, envelope_hash=conflict.content_hash()
        ),
    )
    context = f.kwargs["context"].model_copy(update={"as_of": datetime.now(UTC)})
    with pytest.raises(RegistryError, match="conflicting"):
        build_operating_macro_dataset(
            f.store,
            f.recipe,
            **{**f.kwargs, "context": context, "series_sources": tuple(sources.items())},
        )
    sources["vix"] = (old.model_copy(update={"envelope_hash": "0" * 64}),)
    with pytest.raises(RegistryError, match="envelope differs"):
        build_operating_macro_dataset(
            f.store, f.recipe, **{**f.kwargs, "series_sources": tuple(sources.items())}
        )


def test_training_prefix_retains_exact_old_sources_despite_new_old_date_revision(
    actual_sources, built
):
    f = actual_sources
    prefix = built.model_copy(update={"raw": built.raw[:205]})
    first = built.raw[0].at
    revision = archive(
        f.store,
        "old-date-current-revision",
        {"series_id": "VIX", "date": first.date().isoformat(), "value": 999},
        first,
    )
    sources = dict(f.kwargs["series_sources"])
    sources["vix"] = (
        *sources["vix"],
        FrozenEvidence(
            evidence_id=revision.record.evidence_id, envelope_hash=revision.content_hash()
        ),
    )
    context = f.kwargs["context"].model_copy(update={"as_of": datetime.now(UTC)})
    rebuilt = build_operating_macro_dataset(
        f.store,
        f.recipe,
        **{
            **f.kwargs,
            "context": context,
            "series_sources": tuple(sources.items()),
            "training_prefix": prefix,
        },
    )
    assert rebuilt.raw == built.raw
    assert rebuilt.raw[:205] == prefix.raw
    assert rebuilt.breadth_snapshot.bars == built.breadth_snapshot.bars
    assert revision.record.evidence_id not in {s.evidence_id for s in rebuilt.sources}
    assert set(prefix.sources).issubset(set(rebuilt.sources))


def test_date_pointer_mismatch_cannot_turn_current_scalar_into_history(actual_sources):
    f = actual_sources
    observed = f.snapshot.bars[-1].closed_at
    wrong = archive(
        f.store,
        "wrong-dated-vix",
        {"series_id": "VIX", "date": "2000-01-01", "value": 20},
        observed,
    )
    sources = dict(f.kwargs["series_sources"])
    sources["vix"] = (
        FrozenEvidence(evidence_id=wrong.record.evidence_id, envelope_hash=wrong.content_hash()),
    )
    with pytest.raises(RegistryError, match="economic time"):
        build_operating_macro_dataset(
            f.store, f.recipe, **{**f.kwargs, "series_sources": tuple(sources.items())}
        )


def test_alfred_intervals_use_the_actual_next_day_availability_boundary(tmp_path):
    import httpx

    from src.data.adapters.base import ReadHTTP
    from src.data.adapters.fundamentals import FredAdapter

    store = EvidenceStore(tmp_path / "alfred.sqlite")
    cutoff = datetime(2024, 3, 12, 20, tzinfo=UTC)
    context = RunContext(
        run_id="dated-vintages",
        mode="paper",
        as_of=cutoff,
        config_hash="0" * 64,
        code_version="fixture",
        account_id="paper:fixture",
    )
    rows = [
        dict(date="2024-03-08", value="20", realtime_start="2024-03-08", realtime_end="2024-03-11"),
        dict(date="2024-03-08", value="30", realtime_start="2024-03-12", realtime_end="9999-12-31"),
    ]
    http = ReadHTTP(
        "fred",
        "https://fixture.test",
        min_interval=0,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"observations": rows})
            )
        ),
    )
    result = FredAdapter(store, "fixture", http=http).observations(
        "VIXCLS", context, ingested_at=cutoff, start="2024-03-08", end="2024-03-08"
    )
    recipe = MacroSeriesRecipe(
        name="vix",
        task_ids=("vix",),
        provider="fred",
        kind="macro",
        value_pointer="/value",
        observation_pointer="/date",
        observation_timezone="America/New_York",
        filters=(("/series_id", "VIXCLS"),),
        revision_policy="alfred-intervals-1",
    )
    sources = tuple(
        FrozenEvidence(evidence_id=e.record.evidence_id, envelope_hash=e.content_hash())
        for e in result.evidence
    )
    candidates = _candidates(store, recipe, sources)
    first = _select(recipe, candidates, cutoff, context, POLICY, 7 * 86400)
    assert store.normalized(first.evidence_id, as_of=cutoff)["value"] == 20
    later = context.model_copy(update={"as_of": cutoff + timedelta(days=1)})
    second = _select(recipe, candidates, cutoff, later, POLICY, 7 * 86400)
    assert store.normalized(second.evidence_id, as_of=later.as_of)["value"] == 30
    assert first.evidence_id != second.evidence_id


def test_missing_vendor_observation_retains_prior_actual_date_without_zero_fill(actual_sources):
    f = actual_sources
    observed = f.snapshot.bars[-1].closed_at
    missing = archive(
        f.store,
        "missing-release",
        {"series_id": "VIX", "date": observed.date().isoformat(), "value": None},
        observed,
        quality="missing",
    )
    recipe = next(s for s in f.recipe.series if s.name == "vix")
    sources = (
        *f.groups["vix"][:-1],
        FrozenEvidence(
            evidence_id=missing.record.evidence_id, envelope_hash=missing.content_hash()
        ),
    )
    context = f.kwargs["context"].model_copy(update={"as_of": datetime.now(UTC)})
    selected = _select(
        recipe, _candidates(f.store, recipe, sources), observed, context, POLICY, 7 * 86400
    )
    assert selected.evidence_id == f.groups["vix"][-2].evidence_id
    assert f.store.get(selected.evidence_id).record.observed_at == f.snapshot.bars[-2].closed_at
    assert f.store.get(missing.record.evidence_id).record.quality == "missing"


@pytest.mark.parametrize("current_only", [False, True])
def test_source_backed_membership_conflicts_and_current_only_history_fail(
    actual_sources, current_only
):
    f = actual_sources
    at = datetime.now(UTC)
    first = f.universe.model_copy(update={"available_at": at})
    second = first.model_copy(update={"instrument_ids": ("equity:other",)})
    rows = (first, second)
    if current_only:
        rows = (first.model_copy(update={"effective_at": at}),)
    payload = [u.model_dump(mode="json") for u in rows]
    # A historical source may be archived later; its explicit availability is
    # pinned here to the exact immutable membership declaration's known clock.
    raw = canonical_bytes(payload)
    record = EvidenceRecord(
        evidence_id="membership-case-" + str(current_only),
        provider="synthetic-test",
        provider_version="1",
        source_uri="test://membership",
        raw_hash=digest(raw),
        ingested_at=at,
        available_at=at,
        observed_at=at,
        requested_as_of=at,
        availability_basis="documented",
        quality="usable",
    )
    f.store.put(record, raw, kind="universe", logical_key=record.evidence_id, normalized=payload)
    context = f.kwargs["context"].model_copy(update={"as_of": datetime.now(UTC)})
    with pytest.raises(RegistryError, match="historical membership|historical breadth membership"):
        build_operating_macro_dataset(
            f.store,
            f.recipe,
            **{
                **f.kwargs,
                "context": context,
                "universes": rows,
                "universe_sources": (
                    SourceBinding(evidence_id=record.evidence_id, available_at=at),
                ),
            },
        )


def test_new_tail_membership_and_its_real_price_history_preserve_fitted_prefix(
    actual_sources, built
):
    f = actual_sources
    prefix = built.model_copy(update={"raw": built.raw[:205]})
    newcomer = f.snapshot.bars[0].instrument.model_copy(
        update={"instrument_id": "equity:new-member", "symbol": "NEW"}
    )
    added, records = [], []
    for number, old in enumerate(f.snapshot.bars):
        bar = old.model_copy(
            update={"instrument": newcomer, "evidence_id": "new-member-price-" + str(number)}
        )
        source = archive(
            f.store,
            bar.evidence_id,
            bar.model_dump(mode="json"),
            bar.closed_at,
            instrument=newcomer,
            kind="bar",
        )
        added.append(bar)
        records.append(source.record)
    at = datetime.now(UTC)
    membership = f.universe.model_copy(
        update={
            "available_at": at,
            "effective_at": built.raw[205].at,
            "data_snapshot_id": "1" * 64,
            "instrument_ids": (f.universe.instrument_ids[0], newcomer.instrument_id),
        }
    )
    payload = [membership.model_dump(mode="json")]
    raw = canonical_bytes(payload)
    record = EvidenceRecord(
        evidence_id="new-tail-membership",
        provider="synthetic-test",
        provider_version="1",
        source_uri="test://new-membership",
        raw_hash=digest(raw),
        ingested_at=at,
        available_at=at,
        observed_at=membership.effective_at,
        requested_as_of=at,
        availability_basis="documented",
        quality="usable",
    )
    f.store.put(record, raw, kind="universe", logical_key=record.evidence_id, normalized=payload)
    context = f.kwargs["context"].model_copy(update={"as_of": datetime.now(UTC)})
    snapshot = MarketSnapshot(
        context=context, bars=(*f.snapshot.bars, *added), evidence=(*f.snapshot.evidence, *records)
    )
    updated = build_operating_macro_dataset(
        f.store,
        f.recipe,
        **{
            **f.kwargs,
            "context": context,
            "breadth_snapshot": snapshot,
            "training_prefix": prefix,
            "universes": (f.universe, membership),
            "universe_sources": (
                *f.kwargs["universe_sources"],
                SourceBinding(evidence_id=record.evidence_id, available_at=at),
            ),
        },
    )
    old_data = materialize_macro_dataset(prefix, f.store, availability_policy=POLICY)
    new_data = materialize_macro_dataset(updated, f.store, availability_policy=POLICY)
    assert new_data.signals.loc[old_data.signals.index].equals(old_data.signals)
    assert new_data.source_ids.loc[old_data.signals.index].equals(old_data.source_ids)
    assert new_data.available_at.loc[old_data.signals.index].equals(old_data.available_at)
    assert record.evidence_id not in set().union(*old_data.source_ids)
    assert record.evidence_id in new_data.source_ids.iloc[-1]
    # The new member's actual 200-price MA is source-bound, including dates
    # before it joined; those dates and their late capture clocks stay intact.
    assert set(b.evidence_id for b in added[-200:]) <= set(new_data.source_ids.iloc[-1])
    assert added[0].evidence_id not in set().union(*new_data.source_ids.loc[old_data.signals.index])
