"""Instrument-preserving feature/label materialization with availability masks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.evidence.store import canonical_bytes, digest
from src.data.features.contextual import EvidenceFields
from src.data.features.expressions import evaluate_expression
from src.domain.market import FeatureSpec, MarketSnapshot


@dataclass(frozen=True)
class FeatureDataset:
    features: pd.DataFrame
    labels: pd.Series
    feature_available_at: pd.Series
    label_available_at: pd.Series
    snapshot_hash: str
    feature_spec_hash: str
    label_end_at: pd.Series | None = None

    def eligible_training(self, as_of) -> "FeatureDataset":
        """Only rows whose inputs and realized outcomes were known by the fit cutoff."""
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            raise ValueError("training cutoff requires a timezone")
        mask = (self.feature_available_at <= cutoff) & (self.label_available_at <= cutoff) & self.labels.notna()
        return FeatureDataset(self.features.loc[mask], self.labels.loc[mask], self.feature_available_at.loc[mask],
                              self.label_available_at.loc[mask], self.snapshot_hash, self.feature_spec_hash,
                              self.label_end_at.loc[mask] if self.label_end_at is not None else None)


def feature_dataset(snapshot: MarketSnapshot, specs: tuple[FeatureSpec, ...], *, horizon: int = 1,
                    allow_gaps: bool = False, evidence_fields: EvidenceFields | None = None) -> FeatureDataset:
    if horizon < 1 or horizon > 252:
        raise ValueError("label horizon must be 1..252 bars")
    if snapshot.gaps and not allow_gaps:
        raise ValueError("snapshot has unresolved coverage gaps")
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("unique nonempty feature specifications required")
    evidence = {e.evidence_id: e for e in snapshot.evidence}
    rows = []
    identities = set()
    for bar in snapshot.bars:
        identity = (bar.closed_at, bar.instrument.instrument_id)
        if identity in identities:
            raise ValueError("one selected market type and price basis required per instrument/date")
        identities.add(identity)
        rows.append({"datetime": bar.closed_at, "instrument": bar.instrument.instrument_id,
                     **{k: float(getattr(bar, k)) for k in ("open", "high", "low", "close", "volume")},
                     "vwap": float(bar.vwap) if bar.vwap is not None else np.nan,
                     "available_at": max(evidence[bar.evidence_id].available_at,
                                         evidence[bar.evidence_id].requested_as_of or evidence[bar.evidence_id].available_at),
                     "market_type": bar.market_type})
    if not rows:
        raise ValueError("no eligible bars")
    frame = pd.DataFrame(rows).set_index(["datetime", "instrument"]).sort_index()
    if evidence_fields is not None:
        if not frame.index.equals(evidence_fields.values.index) or not frame.index.equals(evidence_fields.available_at.index):
            raise ValueError("evidence feature identities must match market history exactly")
        if set(frame.columns) & set(evidence_fields.values.columns):
            raise ValueError("evidence fields cannot override market observations")
        if (evidence_fields.available_at.dropna() > snapshot.context.as_of).any():
            raise ValueError("future contextual evidence")
        frame = frame.join(evidence_fields.values, validate="one_to_one")
        frame["available_at"] = pd.concat([frame.available_at, evidence_fields.available_at], axis=1).max(axis=1).where(evidence_fields.available_at.notna())
    result = {}
    for spec in specs:
        if spec.engine_version != "bounded-ast-1":
            raise ValueError("feature engine version mismatch")
        if spec.market_type != "any" and not frame.market_type.eq(spec.market_type).all():
            raise ValueError(f"feature {spec.name} requires {spec.market_type}")
        missing = set(spec.required_fields) - set(frame)
        if missing:
            raise ValueError(f"feature {spec.name} missing fields: {sorted(missing)}")
        result[spec.name] = evaluate_expression(spec.expression, frame)
    features = pd.DataFrame(result, index=frame.index)
    # Conservative lineage: each trailing feature may depend on all preceding
    # observations (EMA/OBV). Later revisions never receive an earlier timestamp.
    available = frame.available_at.groupby(level="instrument").cummax()
    future_price = frame.close.groupby(level="instrument").shift(-horizon)
    label_end = pd.Series(frame.index.get_level_values('datetime'),index=frame.index).groupby(level='instrument').shift(-horizon)
    label_available = frame.available_at.groupby(level="instrument").shift(-horizon)
    labels = (future_price / frame.close - 1).rename("forward_return")
    label_available = pd.concat([label_available.rename("future"), available.rename("current")], axis=1).max(axis=1)
    label_available = label_available.where(labels.notna())
    snapshot_hash = snapshot.content_hash() if evidence_fields is None else digest(canonical_bytes({"market": snapshot.content_hash(), "contextual": evidence_fields.snapshot_hash}))
    return FeatureDataset(features, labels, available, label_available,
                          snapshot_hash, digest(canonical_bytes([s.model_dump(mode="json") for s in specs])),label_end)


def join_features(base: pd.DataFrame, additional: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(base.index, pd.MultiIndex) or not isinstance(additional.index, pd.MultiIndex):
        raise ValueError("features require a named datetime/instrument MultiIndex")
    if base.index.names != ["datetime", "instrument"] or additional.index.names != base.index.names:
        raise ValueError("feature index names must be datetime/instrument")
    if not base.index.is_unique or not additional.index.is_unique:
        raise ValueError("duplicate feature identity")
    if not additional.index.isin(base.index).all():
        raise ValueError("additional feature observations are outside base coverage")
    return base.join(additional, how="left", validate="one_to_one")
