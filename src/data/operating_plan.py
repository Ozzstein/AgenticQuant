"""Prepare source-bound operating data from completed, immutable refresh receipts.

Recipes approve source semantics and task names, never refreshed evidence IDs.
Historical constituent membership is an explicit input. A frozen training prefix
is retained verbatim; today's membership or revised captures cannot replace it.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from src.data.calendars import sessions
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import Instrument, NonEmptyString, RunContext, Sha256, VersionedContract
from src.domain.market import MarketSnapshot
from src.domain.portfolio import UniverseEligibility
from src.research.evidence import SourceBinding, json_pointer, verify_market_snapshot
from src.research.regime_experiments import (
    RAW_MACRO_FIELDS,
    FrozenEvidence,
    MacroDatasetSpec,
    MacroRawObservation,
    materialize_macro_dataset,
    verify_frozen_evidence,
)
from src.research.registry import RegistryError

MacroField = Literal["vix", "dgs10", "dgs2", "sp500", "dxy", "hyg", "tlt"]
VintagePolicy = Literal["decision-time-vintage-1", "captured-vintage-at-fit-1"]


class MacroSeriesRecipe(VersionedContract):
    name: MacroField
    task_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    provider: NonEmptyString
    kind: NonEmptyString
    value_pointer: NonEmptyString
    observation_pointer: NonEmptyString
    observation_timezone: NonEmptyString = "UTC"
    # Exact immutable selectors, e.g. ('/series_id', 'DGS10'). No implicit proxy.
    filters: tuple[tuple[NonEmptyString, str], ...] = ()
    instrument: Instrument | None = None
    revision_policy: Literal["declared-supersession-1", "alfred-intervals-1"] = (
        "declared-supersession-1"
    )

    @model_validator(mode="after")
    def closed_identity(self):
        ZoneInfo(self.observation_timezone)
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("unique source task IDs required")
        if not self.filters and self.instrument is None:
            raise ValueError("explicit series identity or canonical instrument required")
        pointers = (self.value_pointer, self.observation_pointer, *(p for p, _ in self.filters))
        if any(not p.startswith("/") for p in pointers):
            raise ValueError("explicit normalized JSON pointers required")
        if len(dict(self.filters)) != len(self.filters):
            raise ValueError("unique semantic source selectors required")
        if self.revision_policy == "alfred-intervals-1" and self.provider != "fred":
            raise ValueError("ALFRED interval policy requires the pinned Fred adapter")
        return self


class OperatingPreparationRecipe(VersionedContract):
    start_session: date
    universe_version: NonEmptyString
    series: tuple[MacroSeriesRecipe, ...] = Field(min_length=7, max_length=7)
    contextual_task_ids: tuple[NonEmptyString, ...] = ()
    availability_policy: VintagePolicy  # Current-vintage use must be explicit.
    max_source_age_seconds: int = Field(default=7 * 86400, gt=0, le=31 * 86400)
    sampling_policy: Literal["latest-usable-dated-observation-1"] = (
        "latest-usable-dated-observation-1"
    )
    prefix_policy: Literal["immutable-training-prefix-1"] = "immutable-training-prefix-1"

    @model_validator(mode="after")
    def fields(self):
        if {s.name for s in self.series} != set(RAW_MACRO_FIELDS):
            raise ValueError("exactly seven distinct approved macro series required")
        if len(set(self.contextual_task_ids)) != len(self.contextual_task_ids):
            raise ValueError("unique contextual task IDs required")
        return self


class OperatingPreparation(VersionedContract):
    context: RunContext
    refresh_run_id: NonEmptyString
    refresh_manifest_hash: Sha256
    recipe_hash: Sha256
    task_receipts: tuple[tuple[NonEmptyString, Sha256], ...]
    economic_sources: tuple[tuple[str, tuple[SourceBinding, ...]], ...]
    macro_dataset: MacroDatasetSpec
    contextual_sources: tuple[FrozenEvidence, ...]
    training_prefix_hash: Sha256 | None = None
    policy: Literal["completed-refresh-frozen-prefix-1"] = "completed-refresh-frozen-prefix-1"


def _payload(store, source):
    """Verify all receipt contents, including explicitly missing observations."""
    envelope = store.get(source.evidence_id)
    if envelope.content_hash() != source.envelope_hash:
        raise RegistryError("refresh source envelope differs")
    if digest(store.raw(source.evidence_id)) != envelope.record.raw_hash:
        raise RegistryError("refresh source raw payload differs")
    if envelope.normalized_hash is None:
        raise RegistryError("normalized dated source required")
    payload = json.loads(store.blob(envelope.normalized_hash))
    if digest(canonical_bytes(payload)) != envelope.normalized_hash:
        raise RegistryError("refresh source normalization differs")
    return envelope, payload


def _candidates(store, recipe, sources):
    candidates = []
    if len({s.evidence_id for s in sources}) != len(sources):
        raise RegistryError("duplicate macro source identity")
    for source in sources:
        envelope, value = _payload(store, source)
        record = envelope.record
        if record.provider != recipe.provider or envelope.kind != recipe.kind:
            continue  # A task may also return its archived raw response.
        expected = () if recipe.instrument is None else (recipe.instrument.instrument_id,)
        if record.instrument_ids != expected:
            continue
        if recipe.instrument is not None and "instrument" in value:
            if Instrument.model_validate(value["instrument"]) != recipe.instrument:
                raise RegistryError("macro source full instrument differs")
        if any(
            str(json_pointer(value, pointer)) != expected for pointer, expected in recipe.filters
        ):
            continue
        # Missing vendor observations remain archived in the exact task receipt.
        # The named sampling policy selects actual usable dated observations,
        # never manufactures a value for a missing date.
        if record.quality == "missing":
            continue
        observed = record.observed_at
        if observed is None:
            raise RegistryError("macro source has no economic observation time")
        raw_time = json_pointer(value, recipe.observation_pointer)
        if isinstance(raw_time, str) and len(raw_time) == 10:
            if (
                date.fromisoformat(raw_time)
                != observed.astimezone(ZoneInfo(recipe.observation_timezone)).date()
            ):
                raise RegistryError("source date pointer differs from original economic time")
        else:
            stamp = pd.Timestamp(raw_time)
            if stamp.tzinfo is None or stamp != pd.Timestamp(observed):
                raise RegistryError("source timestamp pointer differs from original economic time")
        if recipe.revision_policy == "alfred-intervals-1":
            if record.provider_version != "fred-alfred-vintage-v1":
                raise RegistryError("unpinned ALFRED normalization")
            begin, end = (
                date.fromisoformat(value["realtime_start"]),
                date.fromisoformat(value["realtime_end"]),
            )
            if begin > end:
                raise RegistryError("invalid ALFRED vintage interval")
        candidates.append((envelope, value))
    return candidates


def _select(recipe, candidates, at, context, policy, max_age):
    cutoff = context.as_of if policy == "captured-vintage-at-fit-1" else at
    eligible = []
    for envelope, value in candidates:
        record = envelope.record
        available = (
            max(record.available_at, record.requested_as_of or record.available_at)
            if record.available_at
            else None
        )
        if (
            available is None
            or available > cutoff
            or not 0 <= (at - record.observed_at).total_seconds() <= max_age
        ):
            continue
        record.require_available(cutoff)
        if recipe.revision_policy == "alfred-intervals-1":
            # Use the documented native vintage interval at the real selection
            # cutoff, never ingestion order or an invented revision flag.
            # FredAdapter makes a dated vintage available at NY midnight on
            # the FOLLOWING day. Apply the same conservative boundary to both
            # ends of an interval; a same-day revision must not hide the prior
            # usable vintage before that revision is actually available.
            vintage = cutoff.astimezone(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            if (
                not date.fromisoformat(value["realtime_start"])
                <= vintage
                <= date.fromisoformat(value["realtime_end"])
            ):
                continue
        eligible.append((envelope, value, available))
    superseded = {e.supersedes_id for e, _, _ in eligible if e.supersedes_id}
    eligible = [row for row in eligible if row[0].record.evidence_id not in superseded]
    if not eligible:
        raise RegistryError(f"missing dated macro coverage: {recipe.name} at {at.isoformat()}")
    latest = max(e.record.observed_at for e, _, _ in eligible)
    selected = [row for row in eligible if row[0].record.observed_at == latest]
    if len(selected) != 1:
        raise RegistryError(f"conflicting macro source vintages: {recipe.name} at {at.isoformat()}")
    envelope, value, available = selected[0]
    number = json_pointer(value, recipe.value_pointer)
    try:
        finite = not isinstance(number, bool) and np.isfinite(float(number))
    except (TypeError, ValueError, OverflowError):
        finite = False
    if not finite:
        raise RegistryError("macro source scalar is not finite numeric data")
    return SourceBinding(
        evidence_id=envelope.record.evidence_id,
        pointer=recipe.value_pointer,
        available_at=available,
    )


def _membership_rows(store, sources, cutoff):
    result = []
    for source in sources:
        envelope = store.get(source.evidence_id, as_of=source.available_at)
        if source.available_at > cutoff:
            raise RegistryError("historical membership exceeds actual preparation cutoff")
        payload = json_pointer(
            store.normalized(source.evidence_id, as_of=source.available_at), source.pointer
        )
        rows = payload if isinstance(payload, list) else [payload]
        for index, value in enumerate(rows):
            universe = UniverseEligibility.model_validate(value)
            if (
                envelope.record.available_at > universe.available_at
                or source.available_at > universe.available_at
            ):
                raise RegistryError("historical membership availability predates its source")
            pointer = (
                source.pointer + "/" + str(index) if isinstance(payload, list) else source.pointer
            )
            result.append((universe, source.model_copy(update={"pointer": pointer})))
    return result


def build_operating_macro_dataset(
    store: EvidenceStore,
    recipe: OperatingPreparationRecipe,
    *,
    context: RunContext,
    series_sources: tuple[tuple[str, tuple[FrozenEvidence, ...]], ...],
    breadth_snapshot: MarketSnapshot,
    universes: tuple[UniverseEligibility, ...],
    universe_sources: tuple[SourceBinding, ...],
    training_prefix: MacroDatasetSpec | None = None,
) -> MacroDatasetSpec:
    """Build and actually materialize macro inputs from frozen dated sources."""
    if len(series_sources) != 7 or set(dict(series_sources)) != set(RAW_MACRO_FIELDS):
        raise RegistryError("exact approved seven series source groups required")
    if breadth_snapshot.context.as_of > context.as_of or breadth_snapshot.gaps:
        raise RegistryError("actual complete breadth snapshot required at the preparation cutoff")
    verify_market_snapshot(store, breadth_snapshot)
    schedule = sessions(
        recipe.start_session.isoformat(), context.as_of.date().isoformat(), calendar="XNYS"
    )
    dates = tuple(closed for _, closed in schedule.values() if closed <= context.as_of)
    if len(dates) < 200:
        raise RegistryError("at least 200 actual completed XNYS sessions required")
    prefix_raw, prefix_bars, prefix_records, frozen = (), (), (), {}
    prefix_data = None
    memberships = []
    if training_prefix is not None:
        if (
            training_prefix.availability_policy != recipe.availability_policy
            or training_prefix.universe_version != recipe.universe_version
            or training_prefix.max_source_age_seconds != recipe.max_source_age_seconds
            or training_prefix.raw[0].at != dates[0]
            or training_prefix.raw[-1].at > dates[-1]
            or training_prefix.breadth_snapshot.context.as_of > context.as_of
        ):
            raise RegistryError("frozen training prefix recipe or cutoff differs")
        prefix_data = materialize_macro_dataset(
            training_prefix, store, availability_policy=recipe.availability_policy
        )
        prefix_raw = training_prefix.raw
        if tuple(o.at for o in prefix_raw) != dates[: len(prefix_raw)]:
            raise RegistryError("frozen prefix session schedule differs")
        prefix_bars = tuple(
            b for b in training_prefix.breadth_snapshot.bars if b.closed_at <= prefix_raw[-1].at
        )
        ids = {b.evidence_id for b in prefix_bars}
        prefix_records = tuple(
            r for r in training_prefix.breadth_snapshot.evidence if r.evidence_id in ids
        )
        frozen.update((s.evidence_id, s) for s in training_prefix.sources)
        memberships.extend(_membership_rows(store, training_prefix.universe_sources, context.as_of))
    incoming = _membership_rows(store, universe_sources, context.as_of)
    if tuple(u for u, _ in incoming) != universes:
        raise RegistryError("declared historical memberships differ from immutable source rows")
    known = {u.content_hash() for u, _ in memberships}
    for universe, source in incoming:
        if universe.content_hash() in known:
            continue
        if prefix_raw and universe.effective_at <= prefix_raw[-1].at:
            raise RegistryError("new membership cannot rewrite a frozen training prefix")
        memberships.append((universe, source))
        known.add(universe.content_hash())
    if not memberships:
        raise RegistryError("actual historical membership coverage required")
    identities = {}
    for universe, _ in memberships:
        key = (universe.universe_version, universe.effective_at, universe.available_at)
        previous = identities.setdefault(key, universe)
        if previous != universe:
            raise RegistryError(
                "conflicting historical membership at the same effective/available cutoff"
            )
    prefix_keys = {
        (b.instrument.instrument_id, b.closed_at, b.price_basis, b.market_type) for b in prefix_bars
    }
    tail = tuple(
        b
        for b in breadth_snapshot.bars
        if dates[0] <= b.closed_at <= dates[-1]
        and (b.instrument.instrument_id, b.closed_at, b.price_basis, b.market_type)
        not in prefix_keys
    )
    ids = {b.evidence_id for b in tail}
    records = {
        r.evidence_id: r
        for r in (*prefix_records, *(r for r in breadth_snapshot.evidence if r.evidence_id in ids))
    }
    snapshot = MarketSnapshot(
        context=context, bars=(*prefix_bars, *tail), evidence=tuple(records.values())
    )
    candidates = {
        s.name: _candidates(store, s, dict(series_sources)[s.name]) for s in recipe.series
    }
    observations = list(prefix_raw)
    for at in dates[len(prefix_raw) :]:
        observations.append(
            MacroRawObservation(
                at=at,
                availability_policy=recipe.availability_policy,
                values=tuple(
                    (
                        s.name,
                        _select(
                            s,
                            candidates[s.name],
                            at,
                            context,
                            recipe.availability_policy,
                            recipe.max_source_age_seconds,
                        ),
                    )
                    for s in recipe.series
                ),
            )
        )
    required = set(records)
    required.update(s.evidence_id for _, s in memberships)
    required.update(s.evidence_id for row in observations for _, s in row.values)
    for key in sorted(required):
        envelope = store.get(key, as_of=context.as_of)
        source = FrozenEvidence(evidence_id=key, envelope_hash=envelope.content_hash())
        if key in frozen and frozen[key] != source:
            raise RegistryError("frozen training source envelope changed")
        frozen[key] = source
    dataset = MacroDatasetSpec(
        breadth_snapshot=snapshot,
        universe_version=recipe.universe_version,
        universes=tuple(u for u, _ in memberships),
        universe_sources=tuple(s for _, s in memberships),
        raw=tuple(observations),
        sources=tuple(frozen[key] for key in sorted(frozen)),
        max_source_age_seconds=recipe.max_source_age_seconds,
        availability_policy=recipe.availability_policy,
    )
    materialized = materialize_macro_dataset(
        dataset, store, availability_policy=recipe.availability_policy
    )
    if prefix_data is not None:
        index = prefix_data.signals.index
        if (
            not materialized.signals.loc[index].equals(prefix_data.signals)
            or not materialized.available_at.loc[index].equals(prefix_data.available_at)
            or not materialized.source_ids.loc[index].equals(prefix_data.source_ids)
        ):
            raise RegistryError(
                "new operating sources rewrite the actual frozen macro training prefix"
            )
    return dataset


def prepare_operating_data(
    operations,
    store: EvidenceStore,
    refresh_run_id: str,
    recipe: OperatingPreparationRecipe,
    *,
    breadth_snapshot: MarketSnapshot,
    universes: tuple[UniverseEligibility, ...],
    universe_sources: tuple[SourceBinding, ...],
    training_prefix: MacroDatasetSpec | None = None,
) -> OperatingPreparation:
    """Resolve exact same-refresh source receipts, then prepare real macro data.

    The authoritative root reader verifies refresh completion and the final
    eight-kind economic bundle. No ambient EvidenceStore discovery is performed.
    """
    from src.operations.operating_fetch import load_operating_sources
    from src.operations.refresh import RefreshPlan

    economic_sources = load_operating_sources(operations, store, refresh_run_id)
    manifest, status = operations.manifest(refresh_run_id), operations.status(refresh_run_id)
    plan = RefreshPlan.model_validate(operations.artifact(manifest.input_hash))
    report = operations.artifact(status["stages"]["report"]["output_hash"])
    context = RunContext.model_validate(report["context"])
    tasks = {t.task_id: t for t in plan.tasks}
    required = sorted(set(recipe.contextual_task_ids).union(*(s.task_ids for s in recipe.series)))
    receipts, sources = [], {}
    for key in required:
        if key not in tasks:
            raise RegistryError("preparation source task missing from completed refresh")
        stage = status["stages"]["fetch:" + key]
        receipt = operations.artifact(stage["output_hash"])
        if (
            receipt != report["fetches"][key]
            or receipt["task_hash"] != tasks[key].content_hash()
            or receipt["status"] != "available"
        ):
            raise RegistryError("preparation task receipt differs or lacks coverage")
        selected = tuple(FrozenEvidence.model_validate(v) for v in receipt["sources"])
        if not selected or len({s.evidence_id for s in selected}) != len(selected):
            raise RegistryError("preparation task lacks unique actual source receipts")
        for source in selected:
            envelope, _ = _payload(store, source)
            if (
                envelope.record.requested_as_of != plan.context.as_of
                or not datetime.fromisoformat(stage["started_at"])
                <= envelope.record.ingested_at
                <= context.as_of
            ):
                raise RegistryError("preparation source capture/request clocks differ")
        receipts.append((key, stage["output_hash"]))
        sources[key] = selected
    groups = tuple(
        (s.name, tuple({v.evidence_id: v for key in s.task_ids for v in sources[key]}.values()))
        for s in recipe.series
    )
    dataset = build_operating_macro_dataset(
        store,
        recipe,
        context=context,
        series_sources=groups,
        breadth_snapshot=breadth_snapshot,
        universes=universes,
        universe_sources=universe_sources,
        training_prefix=training_prefix,
    )
    contextual = tuple(
        sorted(
            {s.evidence_id: s for key in recipe.contextual_task_ids for s in sources[key]}.values(),
            key=lambda s: s.evidence_id,
        )
    )
    verify_frozen_evidence(store, contextual, as_of=context.as_of)
    return OperatingPreparation(
        context=context,
        refresh_run_id=refresh_run_id,
        refresh_manifest_hash=manifest.content_hash(),
        recipe_hash=recipe.content_hash(),
        task_receipts=tuple(receipts),
        economic_sources=economic_sources,
        macro_dataset=dataset,
        contextual_sources=contextual,
        training_prefix_hash=training_prefix.content_hash()
        if training_prefix is not None
        else None,
    )
