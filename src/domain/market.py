"""Versioned market data contracts shared by research and execution adapters.

Prices are Decimal values in the instrument's quote currency. Bars denote
half-open intervals [opened_at, closed_at); closed_at is not publication time.
Only raw spot/equity bars may be adapted to execution prices.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

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

    @model_validator(mode="after")
    def ordered(self) -> "ModelArtifact":
        if not self.training_start <= self.training_end <= self.labels_available_through <= self.fitted_at:
            raise ValueError("model training/label/fit chronology is invalid")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("duplicate model features")
        return self
