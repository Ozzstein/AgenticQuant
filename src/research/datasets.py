"""Explicit historical inputs shared by custom research and the replay engine."""
from __future__ import annotations

import numpy as np
from pydantic import Field, model_validator

from src.data.features.dataset import FeatureDataset, feature_dataset
from src.domain.common import RunContext, VersionedContract
from src.domain.market import FeatureSpec, MarketSnapshot
from src.domain.portfolio import (
    AccountSpec,
    CorporateAction,
    ExecutionBar,
    FillPolicy,
    InstrumentRisk,
    RiskPolicy,
    UniverseEligibility,
)


def eligible_features(snapshot: MarketSnapshot, specs: tuple[FeatureSpec, ...],
                      universes: tuple[UniverseEligibility, ...], universe_version: str,
                      *, horizon: int = 1) -> FeatureDataset:
    """Keep finite, known features and membership at each historical decision.

    Unavailable inputs are excluded, never assigned an earlier publication time.
    Label availability remains explicit for training and fold purging.
    """
    data = feature_dataset(snapshot, specs, horizon=horizon)
    dates = data.features.index.get_level_values(0)
    members = {}
    for at in dates.unique():
        choices = [u for u in universes if u.universe_version == universe_version
                   and u.effective_at <= at and u.available_at <= at]
        if not choices:
            raise ValueError(f"documented historical universe missing at {at}")
        selected = max(choices, key=lambda u: (u.effective_at, u.available_at))
        members[at] = set(selected.instrument_ids)
    membership = np.asarray([instrument in members[at] for at, instrument in data.features.index])
    mask = (membership & np.isfinite(data.features.to_numpy(dtype=float)).all(axis=1)
            & (data.feature_available_at <= dates) & data.label_available_at.notna()
            & data.labels.notna())
    return FeatureDataset(data.features.loc[mask], data.labels.loc[mask],
                          data.feature_available_at.loc[mask], data.label_available_at.loc[mask],
                          data.snapshot_hash, data.feature_spec_hash)


class ResearchInputs(VersionedContract):
    """Serializable market/economic inputs; generated candidates cannot replace them."""
    snapshot: MarketSnapshot
    context: RunContext
    account: AccountSpec
    bars: tuple[ExecutionBar, ...]
    metadata: tuple[InstrumentRisk, ...]
    universes: tuple[UniverseEligibility, ...]
    actions: tuple[CorporateAction, ...] = ()
    universe_version: str
    fill_policy: FillPolicy = Field(default_factory=FillPolicy)
    risk_policy: RiskPolicy = Field(default_factory=RiskPolicy)
    topk: int = Field(default=30, ge=1)
    samples_per_year: int = Field(default=252, ge=1)
    valuation_calendar: str
    walk_months: int = Field(default=6, ge=1)
    embargo_days: int = Field(default=5, ge=0)
    validation_months: int = Field(default=1, ge=1)
    initial_train_months: int = Field(default=12, ge=1)
    label_horizon: int = Field(default=1, ge=1, le=252)
    synthetic: bool = False

    @model_validator(mode="after")
    def coherent(self):
        if self.context.mode != "backtest" or self.context.account_id != self.account.account_id:
            raise ValueError("research needs a separate matching backtest account")
        if self.snapshot.context.as_of > self.context.as_of:
            raise ValueError("snapshot exceeds research cutoff")
        if self.snapshot.content_hash() not in self.context.data_snapshot_ids:
            raise ValueError("research context must bind the exact market snapshot hash")
        if not self.bars or not self.metadata or not self.universes:
            raise ValueError("execution bars, instrument metadata and universe history required")
        for u in self.universes:
            if u.data_snapshot_id not in self.context.data_snapshot_ids:
                raise ValueError("universe snapshot is not bound to the run")
        return self

    def materialize(self, specs: tuple[FeatureSpec, ...]) -> FeatureDataset:
        return eligible_features(self.snapshot, specs, self.universes, self.universe_version,
                                 horizon=self.label_horizon)

    def replay_kwargs(self) -> dict:
        return {"context": self.context, "account": self.account, "bars": list(self.bars),
                "metadata": list(self.metadata), "universes": list(self.universes),
                "actions": list(self.actions), "fill_policy": self.fill_policy,
                "risk_policy": self.risk_policy, "topk": self.topk,
                "samples_per_year": self.samples_per_year, "valuation_calendar": self.valuation_calendar}

    def split_manifest(self) -> dict:
        return {"walk_months": self.walk_months, "embargo_days": self.embargo_days,
                "validation_months": self.validation_months, "initial_train_months": self.initial_train_months,
                "label_horizon": self.label_horizon, "universe_version": self.universe_version}
