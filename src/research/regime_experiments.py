"""Registered causal HMM variants and independently reproducible source proofs.

This module has no activation or promotion operation. Research outcomes belong
to the coordinator's shared evaluator and registry lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from src.domain.portfolio import UniverseEligibility
from src.research.datasets import ResearchInputs
from src.research.evidence import SourceBinding, json_pointer, verify_market_snapshot
from src.research.registry import RegistryError, ResearchRegistry

from src.core.macro_regime import (
    MACRO_SIGNALS,
    CausalHMMRegimeDetector,
    CausalRegimeResult,
    macro_signals,
)
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import EvidenceRecord, Sha256, UtcTimestamp, VersionedContract
from src.domain.market import MarketSnapshot


class HMMVariantSpec(VersionedContract):
    implementation: Literal['causal-gaussian-hmm-1'] = 'causal-gaussian-hmm-1'
    states: Literal[4] = 4
    covariance_type: Literal['full'] = 'full'
    normalization_window: int = Field(default=252, ge=20, le=756)
    min_regime_days: int = Field(default=5, ge=1, le=60)
    confidence_threshold: float = Field(default=.6, ge=.25, le=1)
    n_init: int = Field(default=10, ge=1, le=20)
    n_iter: int = Field(default=100, ge=10, le=500)
    risk_weights: tuple[tuple[str, float], ...] = (
        ('vix_level', .35), ('vix_roc_10d', .2), ('sp500_breadth', -.25), ('sp500_momentum_20d', -.2),
    )

    @model_validator(mode='after')
    def valid_weights(self):
        weights = dict(self.risk_weights)
        if len(weights) != len(self.risk_weights) or not weights or set(weights)-set(MACRO_SIGNALS):
            raise ValueError('unique exact macro signal risk weights required')
        if not any(weights.values()) or any(abs(value)>10 for value in weights.values()):
            raise ValueError('risk weights must be nonzero and bounded')
        return self

    def detector_kwargs(self):
        return dict(normalization_window=self.normalization_window, min_regime_days=self.min_regime_days,
                    confidence_threshold=self.confidence_threshold, n_init=self.n_init, n_iter=self.n_iter,
                    risk_weights=dict(self.risk_weights))


RAW_MACRO_FIELDS = ('vix', 'dgs10', 'dgs2', 'sp500', 'dxy', 'hyg', 'tlt')


class FrozenEvidence(VersionedContract):
    evidence_id: str
    envelope_hash: Sha256


def verify_frozen_evidence(store: EvidenceStore, sources: tuple[FrozenEvidence, ...], *, as_of):
    """Compare frozen envelopes AND stored contents, without trusting hash-shaped input."""
    if len({s.evidence_id for s in sources}) != len(sources):
        raise RegistryError('duplicate frozen evidence identity')
    records = {}
    for source in sources:
        envelope = store.get(source.evidence_id, as_of=as_of)
        if envelope.content_hash() != source.envelope_hash:
            raise RegistryError('frozen evidence envelope differs from stored contents')
        if digest(store.raw(source.evidence_id)) != envelope.record.raw_hash:
            raise RegistryError('frozen raw payload differs')
        if envelope.normalized_hash is not None:
            normalized = store.normalized(source.evidence_id, as_of=as_of)
            if digest(canonical_bytes(normalized)) != envelope.normalized_hash:
                raise RegistryError('frozen normalized payload differs')
        if envelope.document_hash is not None:
            document = store.document(source.evidence_id, as_of=as_of)
            if digest(document.text.encode()) != envelope.document_hash:
                raise RegistryError('frozen document differs')
        records[source.evidence_id] = envelope.record
    return records


class MacroRawObservation(VersionedContract):
    at: UtcTimestamp
    # Exact scalar locations in immutable normalized source data, never supplied numbers.
    values: tuple[tuple[str, SourceBinding], ...]
    availability_policy: Literal['decision-time-vintage-1','captured-vintage-at-fit-1'] = 'decision-time-vintage-1'

    @model_validator(mode='after')
    def fields(self):
        if len(self.values) != 7 or set(dict(self.values)) != set(RAW_MACRO_FIELDS):
            raise ValueError('all seven distinct raw macro fields required')
        if self.availability_policy == 'decision-time-vintage-1' and any(source.available_at > self.at for _, source in self.values):
            raise ValueError('future macro source binding')
        return self


class MacroDatasetSpec(VersionedContract):
    breadth_snapshot: MarketSnapshot
    calendar: Literal["XNYS"] = "XNYS"
    universe_version: str
    universes: tuple[UniverseEligibility, ...] = Field(min_length=1)
    universe_sources: tuple[SourceBinding, ...] = Field(min_length=1)
    raw: tuple[MacroRawObservation, ...] = Field(min_length=200)
    sources: tuple[FrozenEvidence, ...] = Field(min_length=1)
    max_source_age_seconds: int = Field(default=7*86400, gt=0, le=31*86400)
    availability_policy: Literal['decision-time-vintage-1','captured-vintage-at-fit-1'] = 'decision-time-vintage-1'

    @model_validator(mode='after')
    def chronological(self):
        dates = [o.at for o in self.raw]
        if dates != sorted(set(dates)):
            raise ValueError('unique chronological macro session cutoffs required')
        if dates[-1] > self.breadth_snapshot.context.as_of:
            raise ValueError('macro date exceeds snapshot cutoff')
        if any(o.availability_policy != self.availability_policy for o in self.raw):
            raise ValueError('macro row/dataset availability policies differ')
        if any(source.available_at>self.breadth_snapshot.context.as_of for observation in self.raw for _,source in observation.values):
            raise ValueError('macro source binding exceeds the actual dataset cutoff')
        return self


@dataclass(frozen=True)
class CausalMacroData:
    signals: pd.DataFrame
    available_at: pd.Series
    source_ids: pd.Series
    evidence: tuple[EvidenceRecord, ...]
    availability_policy: str = 'decision-time-vintage-1'


def materialize_macro_dataset(spec: MacroDatasetSpec, store: EvidenceStore, *,
                              availability_policy: str = 'decision-time-vintage-1') -> CausalMacroData:
    """Rebuild the exact eight signals, including real historical constituent breadth."""
    from src.data.calendars import sessions
    if availability_policy != spec.availability_policy:
        raise RegistryError('explicit macro materialization/dataset availability policies differ')
    captured = availability_policy == 'captured-vintage-at-fit-1'
    cutoff = spec.breadth_snapshot.context.as_of
    calendar = sessions(spec.raw[0].at.date().isoformat(),spec.raw[-1].at.date().isoformat(),calendar=spec.calendar)
    if [o.at.date().isoformat() for o in spec.raw] != list(calendar):
        raise RegistryError('macro observation schedule has missing or nonexchange sessions')
    if any(o.at < calendar[o.at.date().isoformat()][1] for o in spec.raw):
        raise RegistryError('complete daily macro/breadth cutoff precedes the exchange close')
    records = verify_frozen_evidence(store, spec.sources, as_of=spec.breadth_snapshot.context.as_of)
    verify_market_snapshot(store, spec.breadth_snapshot)
    if {r.evidence_id for r in spec.breadth_snapshot.evidence} - records.keys():
        raise RegistryError('breadth snapshot evidence is not frozen')
    archived_universes, membership_sources = [], []
    for source in spec.universe_sources:
        if source.evidence_id not in records:
            raise RegistryError('unfrozen historical membership source')
        records[source.evidence_id].require_available(source.available_at)
        value = json_pointer(store.normalized(source.evidence_id, as_of=source.available_at), source.pointer)
        for row in value if isinstance(value, list) else [value]:
            universe = UniverseEligibility.model_validate(row)
            if source.available_at > universe.available_at:
                raise RegistryError('membership source is later than declared availability')
            archived_universes.append(universe)
            membership_sources.append((universe, source.evidence_id))
    if tuple(archived_universes) != spec.universes:
        raise RegistryError('historical membership differs from stored source')
    dates = pd.DatetimeIndex([o.at for o in spec.raw], name='datetime')
    instruments = sorted({b.instrument.instrument_id for b in spec.breadth_snapshot.bars})
    prices = pd.DataFrame(np.nan, index=dates, columns=instruments)
    membership = pd.DataFrame(False, index=dates, columns=instruments)
    bar_sources = {}
    for bar in spec.breadth_snapshot.bars:
        if bar.closed_at not in dates:
            continue
        key = (pd.Timestamp(bar.closed_at), bar.instrument.instrument_id)
        if key in bar_sources:
            raise RegistryError('ambiguous breadth price convention')
        records[bar.evidence_id].require_available(cutoff if captured else bar.closed_at)
        if bar.market_type != 'equity':
            raise RegistryError('US breadth requires equity constituents')
        prices.loc[key] = float(bar.close)
        bar_sources[key] = bar.evidence_id
    if len({b.price_basis for b in spec.breadth_snapshot.bars}) != 1:
        raise RegistryError('mixed breadth adjustment conventions')
    raw = pd.DataFrame(index=dates, columns=RAW_MACRO_FIELDS, dtype=float)
    row_ids = []
    for observation in spec.raw:
        ids = set()
        for name, source in observation.values:
            if source.evidence_id not in records:
                raise RegistryError('unfrozen macro value source')
            record = records[source.evidence_id]
            record.require_available(source.available_at)
            if record.observed_at is None or not 0 <= (observation.at-record.observed_at).total_seconds() <= spec.max_source_age_seconds:
                raise RegistryError('unknown, future or stale macro observation time')
            value = json_pointer(store.normalized(source.evidence_id, as_of=source.available_at), source.pointer)
            if isinstance(value, bool):
                raise RegistryError('macro scalar must be numeric')
            value = float(value)
            if not np.isfinite(value) or (name not in ('dgs10', 'dgs2') and value <= 0):
                raise RegistryError('invalid raw macro scalar')
            raw.loc[observation.at, name] = value
            ids.add(source.evidence_id)
        eligible = [u for u in spec.universes if u.universe_version == spec.universe_version
                    and u.effective_at <= observation.at and u.available_at <= (cutoff if captured else observation.at)]
        if not eligible:
            raise RegistryError('historical breadth membership missing')
        selected = max(eligible, key=lambda u: (u.effective_at, u.available_at))
        if any(u != selected and (u.effective_at,u.available_at)==(selected.effective_at,selected.available_at) for u in eligible):
            raise RegistryError('conflicting historical breadth membership')
        if set(selected.instrument_ids)-set(instruments):
            raise RegistryError('historical constituent price coverage missing')
        membership.loc[observation.at, list(selected.instrument_ids)] = True
        # A later, genuinely new membership must not rewrite source lineage of
        # earlier fitted observations under the captured-vintage policy.
        ids.update(source_id for universe,source_id in membership_sources if universe==selected)
        row_ids.append(ids)
    signals = macro_signals(raw, prices, membership)
    finite = np.isfinite(signals).all(axis=1)
    if not finite.any():
        raise RegistryError('insufficient real constituent/macro warmup coverage')
    start = int(np.flatnonzero(finite)[0])
    if not finite.iloc[start:].all():
        raise RegistryError('macro or constituent coverage gap after warmup')
    source_ids, availability = [], []
    # Breadth uses 200 prices; all rolling inputs retain actual source lineage.
    for offset in range(start, len(dates)):
        window=dates[max(0,offset-199):offset+1]
        ids=set().union(*row_ids[max(0,offset-199):offset+1])
        # Current constituent breadth really uses each current member's 200
        # original prices, including its pre-membership history. Unused newly
        # captured nonmember history cannot alter the old training prefix.
        for instrument in membership.columns[membership.iloc[offset]]:
            ids.update(bar_sources[(at,instrument)] for at in window if (at,instrument) in bar_sources)
        ids=tuple(sorted(ids))
        for key in ids:
            records[key].require_available(cutoff if captured else dates[offset].to_pydatetime())
        source_ids.append(ids)
        availability.append(max(max(records[key].available_at, records[key].requested_as_of or records[key].available_at) for key in ids))
    signals = signals.iloc[start:]
    used = set().union(*map(set, source_ids))
    return CausalMacroData(signals, pd.Series(availability,index=signals.index),
                           pd.Series(source_ids,index=signals.index), tuple(records[key] for key in sorted(used)),availability_policy)


class CausalRegimeArtifact(VersionedContract):
    inputs_hash: Sha256
    dataset: MacroDatasetSpec
    variant: HMMVariantSpec
    fitted_at: UtcTimestamp
    inference_times: tuple[UtcTimestamp, ...] = Field(min_length=1)
    observations: tuple[CausalRegimeResult, ...] = Field(min_length=1)
    model_hash: Sha256
    model_payload_hash: Sha256
    model_manifest_hash: Sha256

    @model_validator(mode='after')
    def causal(self):
        if tuple(sorted(set(self.inference_times))) != self.inference_times or self.inference_times[0] <= self.fitted_at:
            raise ValueError('ordered strictly out-of-sample inference times required')
        if tuple(o.context.as_of for o in self.observations) != self.inference_times:
            raise ValueError('observations must match exact inference schedule')
        if any(o.model_hash != self.model_hash or o.trained_through > self.fitted_at for o in self.observations):
            raise ValueError('observation model/training identity differs')
        return self


def _run_regimes(store, inputs, dataset, variant, fitted_at, inference_times, work_dir):
    fitted_at = pd.Timestamp(fitted_at).tz_convert("UTC").to_pydatetime().astimezone(timezone.utc)
    inference_times = tuple(pd.Timestamp(at).tz_convert("UTC").to_pydatetime().astimezone(timezone.utc) for at in inference_times)
    if not inference_times or inference_times != tuple(sorted(set(inference_times))) or fitted_at >= inference_times[0]:
        raise RegistryError('ordered strictly out-of-sample inference times required')
    data = materialize_macro_dataset(dataset, store)
    if dataset.breadth_snapshot.context.as_of > inputs.context.as_of or dataset.raw[-1].at > inputs.context.as_of or inference_times[-1] > inputs.context.as_of:
        raise RegistryError('macro experiment exceeds requested cutoff')
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    state = work_dir/'regime-state.sqlite3'
    if state.exists():
        raise RegistryError('regime reconstruction requires a fresh isolated state namespace')
    detector = CausalHMMRegimeDetector(state, **variant.detector_kwargs())

    def prefix(at):
        raw = data.signals.loc[:at]
        ids = set(key for values in data.source_ids.loc[raw.index] for key in values)
        return raw, {'available_at':data.available_at.loc[raw.index], 'source_ids':data.source_ids.loc[raw.index],
                     'evidence':tuple(r for r in data.evidence if r.evidence_id in ids)}
    raw, kwargs = prefix(fitted_at)
    detector.fit(raw, inputs.context.model_copy(update={'as_of':fitted_at}), **kwargs)
    detector.save(work_dir/'hmm.joblib')
    observation_times = []
    for at in inference_times:
        eligible = data.signals.index[data.signals.index <= at]
        if eligible.empty or eligible[-1] <= fitted_at:
            raise RegistryError('no observed out-of-sample macro session at inference cutoff')
        observed_at = eligible[-1]
        if (pd.Timestamp(at)-observed_at).total_seconds() > dataset.max_source_age_seconds:
            raise RegistryError('latest actual macro session is stale at inference cutoff')
        observation_times.append(observed_at)
    # One forward pass persists every causal intermediate state. Repeated full
    # prefixes would re-verify O(n²) identical frozen observations unnecessarily.
    raw, kwargs = prefix(inference_times[-1])
    detector.predict(raw, inputs.context.model_copy(update={'as_of':inference_times[-1]}), **kwargs)
    with detector._db() as db:
        rows = db.execute('SELECT timestamp,result FROM regime_events WHERE model_hash=?', (detector.model_hash,)).fetchall()
    observed = {pd.Timestamp(at):CausalRegimeResult.model_validate_json(value) for at,value in rows}
    # A later decision clock may consume the latest known session. Reusing a
    # session never adds an emission or advances the minimum-duration counter.
    return detector, tuple(observed[source_at].model_copy(update={
        'context':inputs.context.model_copy(update={'as_of':at})})
        for at,source_at in zip(inference_times,observation_times,strict=True))


def build_regime_artifact(registry: ResearchRegistry, store: EvidenceStore, inputs: ResearchInputs, *,
                          dataset: MacroDatasetSpec, variant: HMMVariantSpec, fitted_at,
                          inference_times, work_dir: str | Path) -> CausalRegimeArtifact:
    """Freeze causal input evidence before running P&L; never evaluate/promote a candidate."""
    detector, observations = _run_regimes(store, inputs, dataset, variant, fitted_at, inference_times, work_dir)
    path = Path(work_dir)/'hmm.joblib'
    return CausalRegimeArtifact(inputs_hash=inputs.content_hash(),dataset=dataset,variant=variant,
        fitted_at=fitted_at,inference_times=tuple(inference_times),observations=observations,
        model_hash=detector.model_hash, model_payload_hash=registry.archive(path.read_bytes()),
        model_manifest_hash=registry.archive(Path(str(path)+'.json').read_bytes()))


def recompute_regime_artifact(registry: ResearchRegistry, store: EvidenceStore, artifact: CausalRegimeArtifact,
                              inputs: ResearchInputs, *, work_dir: str | Path) -> tuple[CausalRegimeResult, ...]:
    if artifact.inputs_hash != inputs.content_hash():
        raise RegistryError('regime artifact is bound to different research inputs')
    # Archived pickles are never executed: refit the real implementation from typed source data.
    payload = registry.blob(artifact.model_payload_hash)
    manifest = registry.blob(artifact.model_manifest_hash)
    if digest(payload) != artifact.model_payload_hash or digest(manifest) != artifact.model_manifest_hash:
        raise RegistryError('regime model artifact contents differ')
    detector, observations = _run_regimes(store, inputs, artifact.dataset, artifact.variant,
        artifact.fitted_at, artifact.inference_times, work_dir)
    path = Path(work_dir)/'hmm.joblib'
    if detector.model_hash != artifact.model_hash or observations != artifact.observations:
        raise RegistryError('causal regime observations/model do not reproduce')
    if path.read_bytes() != payload or Path(str(path)+'.json').read_bytes() != manifest:
        raise RegistryError('fitted HMM parameters/training/environment artifacts do not reproduce')
    return observations


# Imported lazily to avoid a competing experiment lifecycle or an import cycle.
def execute_registered_regime(registry: ResearchRegistry, trial_id: str, *, store: EvidenceStore,
                              ledger_path: str | Path, artifact_dir: str | Path):
    from dataclasses import replace

    from src.research.strategy_experiments import _execute_registered
    plan_type = __getattr__('RegimeExperimentPlan')
    candidate = _execute_registered(registry,trial_id,plan_type=plan_type,kind='regime',store=store,
                                    ledger_path=ledger_path,artifact_dir=artifact_dir)
    baseline = _execute_registered(registry,trial_id,plan_type=plan_type,kind='regime',store=store,
        ledger_path=Path(artifact_dir)/'baseline-ledger.sqlite3',artifact_dir=Path(artifact_dir)/'baseline',comparison=True)
    return replace(candidate,comparisons=(('baseline',baseline),))


def __getattr__(name):
    if name != 'RegimeExperimentPlan':
        raise AttributeError(name)
    if name in globals():
        return globals()[name]
    from src.research.strategy_experiments import StrategyExperimentPlan
    from src.strategies.specification import baseline_specs

    class RegimeExperimentPlan(StrategyExperimentPlan):
        variant: HMMVariantSpec
        baseline_variant: HMMVariantSpec = Field(default_factory=HMMVariantSpec)
        baseline_regime_artifact_hash: Sha256

        @model_validator(mode='after')
        def fixed_baseline(self):
            baseline = baseline_specs().get(self.strategy.name)
            if baseline is None or self.strategy != baseline or self.strategy.model_name is None:
                raise ValueError('regime trials require an exact fixed fitted-model code baseline strategy')
            return self

        def candidate_payload(self):
            return self.variant.model_dump(mode='json')
    globals()[name] = RegimeExperimentPlan
    return RegimeExperimentPlan
