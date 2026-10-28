"""Versioned market data contracts shared by research and execution adapters.

Prices are Decimal values in the instrument's quote currency. Bars denote
half-open intervals [opened_at, closed_at); closed_at is not publication time.
Only raw spot/equity bars may be adapted to execution prices.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_serializer, model_validator

from src.domain.common import (
    EvidenceRecord,
    Instrument,
    NonEmptyString,
    RunContext,
    Sha256,
    UtcTimestamp,
    VersionedContract,
)


class MarketBar(VersionedContract):
    instrument: Instrument
    opened_at: UtcTimestamp
    closed_at: UtcTimestamp
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)
    evidence_id: NonEmptyString
    price_basis: Literal["raw", "split_adjusted", "total_return"]
    market_type: Literal["equity", "spot", "perpetual"]
    calendar: NonEmptyString
    session: NonEmptyString
    vwap: Decimal | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def coherent_bar(self) -> "MarketBar":
        if self.opened_at >= self.closed_at:
            raise ValueError("bar interval must be positive")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high is below another bar price")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low is above another bar price")
        if (self.instrument.asset_class == "crypto") != (self.market_type != "equity"):
            raise ValueError("market type disagrees with instrument asset class")
        return self

    def require_execution_price(self) -> None:
        if self.price_basis != "raw" or self.market_type == "perpetual":
            raise ValueError("execution requires raw long-only spot/equity bars")


class CoverageGap(VersionedContract):
    instrument_id: NonEmptyString
    capability: NonEmptyString
    reason: Literal["missing", "unknown_availability", "future", "invalid", "stale", "conflict"]
    detail: NonEmptyString
    expected_at: UtcTimestamp | None = None


class MarketSnapshot(VersionedContract):
    context: RunContext
    bars: tuple[MarketBar, ...]
    evidence: tuple[EvidenceRecord, ...]
    gaps: tuple[CoverageGap, ...] = ()

    @model_validator(mode="after")
    def eligible(self) -> "MarketSnapshot":
        records = {e.evidence_id: e for e in self.evidence}
        if len(records) != len(self.evidence):
            raise ValueError("duplicate evidence IDs")
        for record in self.evidence:
            record.require_available(self.context.as_of)
        keys = set()
        for bar in self.bars:
            key = (bar.instrument.instrument_id, bar.closed_at, bar.price_basis, bar.market_type)
            if key in keys:
                raise ValueError("unresolved duplicate/conflicting bar")
            keys.add(key)
            if bar.closed_at > self.context.as_of:
                raise ValueError("unclosed future bar")
            record = records.get(bar.evidence_id)
            if record is None or bar.instrument.instrument_id not in record.instrument_ids:
                raise ValueError("bar is not linked to instrument evidence")
            record.require_available(self.context.as_of)
            if record.available_at < bar.closed_at:
                raise ValueError("complete bar cannot be available before it closes")
        return self


class FeatureSpec(VersionedContract):
    name: NonEmptyString
    expression: NonEmptyString
    engine_version: NonEmptyString = "bounded-ast-1"
    required_fields: tuple[NonEmptyString, ...] = ()
    market_type: Literal["equity", "spot", "perpetual", "any"] = "any"


class ModelArtifact(VersionedContract):
    model_name: NonEmptyString
    payload_hash: Sha256
    training_data_hash: Sha256
    feature_spec_hash: Sha256
    feature_names: tuple[NonEmptyString, ...]
    fitted_at: UtcTimestamp
    training_start: UtcTimestamp
    training_end: UtcTimestamp
    labels_available_through: UtcTimestamp
    code_version: NonEmptyString
    environment: tuple[tuple[str, str], ...]
    parameters_json: NonEmptyString
    preprocessing: NonEmptyString = "train-median-standard-scaler-v1"
    vintage_policy: Literal["decision-time-vintage-1", "captured-vintage-at-fit-1"] = "decision-time-vintage-1"
    features_available_through: UtcTimestamp | None = None
    label_end_through: UtcTimestamp | None = None
    label_end_policy: Literal["explicit-outcome-end-1", "availability-upper-bound-1"] = "availability-upper-bound-1"

    @model_validator(mode="after")
    def ordered(self) -> "ModelArtifact":
        if not self.training_start <= self.training_end <= self.labels_available_through <= self.fitted_at:
            raise ValueError("model training/label/fit chronology is invalid")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("duplicate model features")
        if self.features_available_through is not None and self.features_available_through > self.fitted_at:
            raise ValueError("features were unavailable at model fit")
        if self.label_end_through is not None and not self.training_end <= self.label_end_through <= self.labels_available_through:
            raise ValueError("outcome end/availability chronology is invalid")
        return self


ModelFamily = Literal["Linear", "LightGBM", "CatBoost", "XGBoost"]
ModelParameter = int | float | str | bool | None


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


class EnsembleChildSpec(VersionedContract):
    """Exact constructor kwargs, with no fit-time default/config expansion."""
    model_name: ModelFamily
    parameters: tuple[tuple[str, ModelParameter], ...] = ()

    @model_validator(mode="after")
    def exact_parameters(self):
        keys = [key for key, _ in self.parameters]
        if keys != sorted(set(keys)) or any(not k or k.strip() != k for k in keys):
            raise ValueError("unique sorted exact parameter names required")
        return self


class EnsembleComposition(VersionedContract):
    implementation: Literal["ensemble-composition-1"] = "ensemble-composition-1"
    children: tuple[EnsembleChildSpec, ...] = Field(min_length=2, max_length=4)
    aggregation: Literal["weighted-arithmetic-mean-1"] = "weighted-arithmetic-mean-1"
    weighting: Literal["ic_weighted", "equal", "custom"]
    weighting_algorithm: Literal["positive-date-spearman-normalization-1", "equal-unit-weights-1", "exact-normalized-custom-1"]
    validation_fraction: float | None = Field(default=None, gt=0, lt=1)
    selection_policy: Literal["retrospective-within-fit-weight-selection-1", "fixed-composition-weights-1"]
    purge_policy: Literal["label-availability-before-validation-1", "outcome-end-before-validation-1"] | None = None
    custom_weights: tuple[tuple[ModelFamily, float], ...] = ()

    @model_validator(mode="after")
    def coherent(self):
        names = [c.model_name for c in self.children]
        if len(set(names)) != len(names):
            raise ValueError("unique ensemble child families required")
        algorithm = {"ic_weighted": "positive-date-spearman-normalization-1", "equal": "equal-unit-weights-1",
                     "custom": "exact-normalized-custom-1"}[self.weighting]
        if self.weighting_algorithm != algorithm:
            raise ValueError("weighting algorithm differs from mode")
        if self.weighting == "ic_weighted":
            if self.validation_fraction is None or self.purge_policy is None or self.selection_policy != "retrospective-within-fit-weight-selection-1":
                raise ValueError("IC composition requires explicit selection/fraction/purge policy")
        elif self.validation_fraction is not None or self.purge_policy is not None or self.selection_policy != "fixed-composition-weights-1":
            raise ValueError("fixed weights cannot claim fitted selection")
        if self.weighting == "custom":
            if [name for name, _ in self.custom_weights] != names or any(w < 0 for _, w in self.custom_weights) or not math.isclose(sum(w for _, w in self.custom_weights), 1., rel_tol=0, abs_tol=1e-12):
                raise ValueError("custom weights must exactly cover ordered children and be normalized")
        elif self.custom_weights:
            raise ValueError("custom weights forbidden for other modes")
        return self


class TrainingFrameIdentity(VersionedContract):
    features: Sha256
    labels: Sha256
    feature_available_at: Sha256
    label_available_at: Sha256
    label_end_at: Sha256

    def aggregate_hash(self) -> str:
        """Preserve the existing H five-frame dictionary hash exactly."""
        return hashlib.sha256(_canonical(self.model_dump(exclude={"schema_version"}))).hexdigest()


class EnsembleWeightEvidence(VersionedContract):
    scope: Literal["retrospective-within-fit-weight-selection-1", "fixed-composition-weights-1"]
    availability_policy: Literal["decision-time-vintage-1", "captured-vintage-at-fit-1"]
    fit_cutoff: UtcTimestamp
    prediction_policy: Literal["non-contextual-internal-validation-1"] | None = None
    boundary: UtcTimestamp | None = None
    selection_fit_cutoff: UtcTimestamp | None = None
    train_frames: TrainingFrameIdentity | None = None
    validation_frames: TrainingFrameIdentity | None = None
    train_rows: int = Field(default=0, ge=0)
    validation_rows: int = Field(default=0, ge=0)
    ic: tuple[tuple[ModelFamily, float], ...] = Field(default=(), max_length=4)
    prediction_hashes: tuple[tuple[ModelFamily, Sha256], ...] = Field(default=(), max_length=4)
    validation_proof_hash: Sha256 | None = None


class EnsembleArtifact(ModelArtifact):
    """Identity of actual serialized components, not approval/reconstructed fitting."""
    model_name: Literal["Ensemble"] = "Ensemble"
    implementation: Literal["ensemble-artifact-1"] = "ensemble-artifact-1"
    preprocessing: Literal["per-child-train-median-standard-scaler-v1"] = "per-child-train-median-standard-scaler-v1"
    composition: EnsembleComposition
    training_frames: TrainingFrameIdentity
    training_frame_hash: Sha256
    children: tuple[ModelArtifact, ...] = Field(min_length=2, max_length=4)
    selection_children: tuple[ModelArtifact, ...] = Field(default=(), max_length=4)
    resolved_weights: tuple[tuple[ModelFamily, float], ...] = Field(min_length=2, max_length=4)
    weight_evidence: EnsembleWeightEvidence
    member_hashes: tuple[tuple[str, Sha256], ...] = Field(max_length=17)
    manifest_hash: Sha256

    @model_validator(mode="after")
    def components(self):
        if not self.feature_names or self.features_available_through is None or self.label_end_through is None or self.label_end_policy != "explicit-outcome-end-1":
            raise ValueError("canonical parent requires actual feature/outcome coverage clocks")
        names = [c.model_name for c in self.composition.children]
        if [a.model_name for a in self.children] != names or [n for n, _ in self.resolved_weights] != names:
            raise ValueError("exact ordered child artifacts and weights required")
        if self.training_frame_hash != self.training_frames.aggregate_hash() or self.parameters_json != _canonical(self.composition.model_dump(mode="json")).decode():
            raise ValueError("parent composition or five-frame identity differs")
        evidence = self.weight_evidence
        if evidence.availability_policy != self.vintage_policy or evidence.fit_cutoff != self.fitted_at:
            raise ValueError("weight evidence outer clock/policy differs")
        common = ("training_data_hash", "feature_spec_hash", "feature_names", "code_version", "vintage_policy", "label_end_policy")
        full_common = common + ("fitted_at", "training_start", "training_end", "labels_available_through", "features_available_through", "label_end_through")
        environment = {}
        for spec, artifact in zip(self.composition.children, self.children):
            if any(getattr(artifact, k) != getattr(self, k) for k in full_common):
                raise ValueError("full child common input/clock identity differs")
            if artifact.label_end_policy != "explicit-outcome-end-1" or artifact.preprocessing != "train-median-standard-scaler-v1" or artifact.parameters_json != _canonical(dict(spec.parameters)).decode():
                raise ValueError("actual child parameters/preprocessing/true ends differ")
            for package, version in artifact.environment:
                if package in environment and environment[package] != version:
                    raise ValueError("inconsistent child environment")
                environment[package] = version
        if self.environment != tuple(environment.items()):
            raise ValueError("parent environment differs from actual children")
        is_ic = self.composition.weighting == "ic_weighted"
        if evidence.scope != self.composition.selection_policy:
            raise ValueError("selection scope differs from composition")
        expected_members = [f"full/{i}{suffix}" for i in range(len(names)) for suffix in (".joblib", ".joblib.json")]
        if is_ic:
            expected_members += [f"selection/{i}{suffix}" for i in range(len(names)) for suffix in (".joblib", ".joblib.json")]
            expected_members += ["selection/validation.json"]
            if ([a.model_name for a in self.selection_children] != names or [n for n, _ in evidence.ic] != names
                    or [n for n, _ in evidence.prediction_hashes] != names or evidence.validation_proof_hash is None
                    or evidence.boundary is None or evidence.selection_fit_cutoff is None
                    or evidence.train_frames is None or evidence.validation_frames is None
                    or evidence.train_rows < 2 or evidence.validation_rows < 20
                    or evidence.prediction_policy != "non-contextual-internal-validation-1"):
                raise ValueError("complete actual IC selection evidence required")
            from datetime import timedelta
            expected_cutoff = self.fitted_at if self.vintage_policy == "captured-vintage-at-fit-1" else evidence.boundary - timedelta(microseconds=1)
            expected_purge = "outcome-end-before-validation-1" if self.vintage_policy == "captured-vintage-at-fit-1" else "label-availability-before-validation-1"
            if evidence.selection_fit_cutoff != expected_cutoff or self.composition.purge_policy != expected_purge or evidence.boundary > self.fitted_at:
                raise ValueError("selection clock or purge policy differs")
            for full, selection in zip(self.children, self.selection_children):
                if any(getattr(selection, k) != getattr(full, k) for k in common + ("environment", "parameters_json", "preprocessing")) or selection.fitted_at != expected_cutoff:
                    raise ValueError("selection child common identity differs")
                purged = selection.label_end_through if self.vintage_policy == "captured-vintage-at-fit-1" else selection.labels_available_through
                if (purged is None or selection.label_end_through is None or selection.features_available_through is None
                        or purged >= evidence.boundary or selection.training_end >= evidence.boundary
                        or selection.training_start < self.training_start or selection.training_end > self.training_end
                        or selection.label_end_through > self.label_end_through
                        or selection.labels_available_through > self.labels_available_through
                        or selection.features_available_through > self.features_available_through):
                    raise ValueError("selection child overlaps validation boundary")
            positive = [max(ic, 0.) for _, ic in evidence.ic]
            if sum(positive) <= 0:
                raise ValueError("no positive selection IC")
            expected_weights = tuple((name, weight / sum(positive)) for name, weight in zip(names, positive))
        else:
            if self.selection_children or evidence != EnsembleWeightEvidence(scope="fixed-composition-weights-1", availability_policy=self.vintage_policy, fit_cutoff=self.fitted_at):
                raise ValueError("fixed modes cannot invent selection evidence")
            expected_weights = self.composition.custom_weights if self.composition.weighting == "custom" else tuple((name, 1 / len(names)) for name in names)
        if self.resolved_weights != expected_weights:
            raise ValueError("stored weights differ from exact algorithm; no renormalization")
        if [name for name, _ in self.member_hashes] != expected_members:
            raise ValueError("closed ordered member inventory required")
        return self


class ModelFileProof(VersionedContract):
    """File identity only; authority/approval must come from its consumer."""
    model_name: ModelFamily | Literal["Ensemble"]
    payload_hash: Sha256
    sidecar_hash: Sha256
    artifact_hash: Sha256
    ensemble_composition_hash: Sha256 | None = None

    @model_serializer(mode="wrap")
    def without_inapplicable_composition(self, handler):
        result = handler(self)
        if self.ensemble_composition_hash is None:
            result.pop("ensemble_composition_hash", None)
        return result

    @model_validator(mode="after")
    def composition_identity(self):
        if (self.model_name == "Ensemble") != (self.ensemble_composition_hash is not None):
            raise ValueError("composition identity required only for ensemble")
        return self
