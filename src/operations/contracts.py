"""Immutable operating inputs and recipes; economic proof references stay outside inputs."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator
from src.data.features.contextual import NumericEvidenceBinding
from src.domain.market import MarketSnapshot
from src.domain.portfolio import (
    AccountSpec,
    CorporateAction,
    ExecutionBar,
    FillPolicy,
    InstrumentRisk,
    OpeningEvent,
    RiskPolicy,
    UniverseEligibility,
)
from src.execution.sleeves import exact_sum
from src.execution.venue import VenueRules
from src.research.evidence import SourceBinding
from src.research.regime_experiments import FrozenEvidence, HMMVariantSpec, MacroDatasetSpec
from src.strategies.specification import StrategySpec

from src.domain.common import (
    Instrument,
    NonEmptyString,
    RunContext,
    Sha256,
    UtcTimestamp,
    VersionedContract,
)


class ObservedOpenReceipt(VersionedContract):
    event_id: NonEmptyString
    evidence_id: NonEmptyString
    evidence_hash: Sha256
    observed_at: UtcTimestamp
    available_at: UtcTimestamp

    @model_validator(mode="after")
    def clocks(self):
        if self.available_at < self.observed_at:
            raise ValueError("opening source eligibility precedes actual receipt observation")
        return self


class PaperCycleInputs(VersionedContract):
    context: RunContext
    snapshot: MarketSnapshot
    account: AccountSpec
    bars: tuple[ExecutionBar, ...] = ()
    opening_events: tuple[OpeningEvent, ...] = ()
    metadata: tuple[InstrumentRisk, ...] = Field(min_length=1)
    universes: tuple[UniverseEligibility, ...] = Field(min_length=1)
    universe_version: NonEmptyString
    actions: tuple[CorporateAction, ...] = ()
    fill_policy: FillPolicy
    risk_policy: RiskPolicy
    rules: tuple[VenueRules, ...] = Field(min_length=1)
    initial_capital: tuple[tuple[NonEmptyString, Decimal], ...]
    receipts: tuple[ObservedOpenReceipt, ...] = ()

    @model_validator(mode="after")
    def causal(self):
        context = self.context
        if (
            context.mode != "paper"
            or self.account.mode != "paper"
            or context.account_id != self.account.account_id
        ):
            raise ValueError("explicit matching paper account/context required")
        if bool(self.bars) == bool(self.opening_events):
            raise ValueError(
                "provide completed execution bars OR actual opening events exclusively"
            )
        events = self.opening_events or self.bars
        if self.opening_events and not self.receipts:
            raise ValueError("actual opening events require exact documentary receipts")
        if self.account.opened_at > min(b.opened_at for b in events):
            raise ValueError("observed events precede paper account inception")
        if (
            self.snapshot.context.as_of > context.as_of
            or self.snapshot.content_hash() not in context.data_snapshot_ids
        ):
            raise ValueError("snapshot cutoff/identity differs from input evidence")
        if self.snapshot.gaps:
            raise ValueError("unresolved market coverage gaps")
        if len({b.event_id for b in self.bars}) != len(self.bars) or any(
            b.closed_at > context.as_of for b in self.bars
        ):
            raise ValueError("unique completed observed bars required")
        if len({e.event_id for e in events}) != len(events) or any(
            e.opened_at > context.as_of for e in events
        ):
            raise ValueError("unique actual nonfuture opening events required")
        if any(a.effective_at > context.as_of for a in self.actions):
            raise ValueError("future corporate action is not an observed account event")
        capital = dict(self.initial_capital)
        if (
            len(capital) != len(self.initial_capital)
            or any(not v.is_finite() or v < 0 for v in capital.values())
            or exact_sum(capital.values()) != self.account.initial_cash
        ):
            raise ValueError(
                "explicit unique sleeve capital must exactly equal initial account cash"
            )
        if len(dict((r.instrument.instrument_id, r) for r in self.rules)) != len(self.rules):
            raise ValueError("unique paper venue rules required")
        if any(r.venue != "paper" or r.available_at > context.as_of for r in self.rules):
            raise ValueError("eligible local paper rules required")
        if self.receipts:
            if len({r.event_id for r in self.receipts}) != len(self.receipts) or {
                r.event_id for r in self.receipts
            } != {b.event_id for b in events}:
                raise ValueError("opening receipts must exactly cover unique economic bar events")
            if any(r.available_at > context.as_of for r in self.receipts):
                raise ValueError("opening receipt is future to the operating cutoff")
        return self


class ModelRecipe(VersionedContract):
    """Approval covers the exact train-only recipe, not ambient estimator settings."""

    strategy: NonEmptyString
    model_name: Literal["Linear", "LightGBM", "CatBoost", "XGBoost"]
    parameters: tuple[tuple[str, int | float | str | bool | None], ...] = ()
    feature_spec_hash: Sha256
    data_snapshot_hash: Sha256
    training_frame_hash: Sha256
    training_snapshot: MarketSnapshot
    training_universe: UniverseEligibility
    training_universe_sources: tuple[SourceBinding, ...] = Field(min_length=1)
    training_numeric_sources: tuple[FrozenEvidence, ...] = ()
    training_universe_policy: Literal["eligible-at-fit-1"] = "eligible-at-fit-1"
    fitted_at: UtcTimestamp
    label_horizon: int = Field(default=1, ge=1, le=252)
    code_version: NonEmptyString
    minimum_rows: int = Field(default=40, ge=10)
    training_availability_policy: Literal["captured-vintage-at-fit-1"] = "captured-vintage-at-fit-1"

    @model_validator(mode="after")
    def parameters_unique(self):
        if len(dict(self.parameters)) != len(self.parameters):
            raise ValueError("duplicate estimator parameter")
        if len({s.evidence_id for s in self.training_numeric_sources}) != len(
            self.training_numeric_sources
        ):
            raise ValueError("duplicate frozen training numeric source")
        if self.training_snapshot.context.as_of > self.fitted_at:
            raise ValueError("training source cutoff exceeds the approved fit cutoff")
        if (
            max(self.training_universe.available_at, self.training_universe.effective_at)
            > self.fitted_at
        ):
            raise ValueError("training universe was not eligible at actual fit cutoff")
        return self


class MacroRecipe(VersionedContract):
    availability_policy: Literal["captured-vintage-at-fit-1"] = "captured-vintage-at-fit-1"
    dataset_hash: Sha256
    training_dataset: MacroDatasetSpec
    variant: HMMVariantSpec
    fitted_at: UtcTimestamp
    code_version: NonEmptyString


class AllocationPolicy(VersionedContract):
    selector: Literal["regime_matched", "performance_weighted", "bandit", "manual"]
    optimizer: Literal[
        "risk_parity", "mean_variance", "black_litterman", "hrp", "max_diversification"
    ]
    manual_strategy: str | None = None
    exploration: float = Field(default=0.1, ge=0, le=1)
    seed: int = 0
    performance_annualization: Literal[252, 365] = 252
    performance_window: int = Field(default=60, ge=20)
    covariance_observations: int = Field(default=120, ge=20, le=1000)
    covariance_sampling_policy: Literal["retrospective-lagged-intervals-1"] = (
        "retrospective-lagged-intervals-1"
    )
    covariance_max_endpoint_age_seconds: int = Field(default=86400 * 4, ge=1, le=86400 * 14)
    # Selection withdraws free cash, not invested capital or passive ownership.
    # Explicit retirement permits an investment exit independent of risk vetoes.
    retiring_sleeves: tuple[str, ...] = ()


class PaperCyclePlan(VersionedContract):
    inputs: PaperCycleInputs
    scheduled_for: UtcTimestamp
    provenance: tuple[tuple[str, Sha256], ...]
    rule_sources: tuple[SourceBinding, ...] = ()
    strategies: tuple[StrategySpec, ...] = Field(min_length=1)
    numeric_bindings: tuple[NumericEvidenceBinding, ...] = ()
    numeric_sources: tuple[FrozenEvidence, ...] = ()
    macro_dataset: MacroDatasetSpec
    macro_recipe: MacroRecipe
    models: tuple[ModelRecipe, ...]
    allocation: AllocationPolicy
    # Only these archived documents can be exposed to the generative desk.
    desk_evidence_ids: tuple[NonEmptyString, ...] = ()
    max_source_chars: int = Field(default=24000, ge=1000, le=120000)

    @model_validator(mode="after")
    def coherent(self):
        if self.scheduled_for > self.inputs.context.as_of:
            raise ValueError("future scheduled cycle")
        specs = {s.name: s for s in self.strategies}
        if len(specs) != len(self.strategies) or set(specs) != set(
            dict(self.inputs.initial_capital)
        ):
            raise ValueError("all initial and retiring sleeves need exact strategy specifications")
        models = {r.strategy: r for r in self.models}
        if len(models) != len(self.models) or set(models) != {
            s.name for s in self.strategies if s.model_name is not None
        }:
            raise ValueError("every numeric strategy requires exactly its model recipe")
        for name, recipe in models.items():
            if (
                recipe.model_name != specs[name].model_name
                or recipe.feature_spec_hash != specs[name].feature_spec_hash
                or recipe.fitted_at >= self.inputs.context.as_of
                or recipe.code_version != self.inputs.context.code_version
            ):
                raise ValueError(
                    "model recipe differs from strategy, causal cutoff or code identity"
                )
        if (
            self.macro_recipe.dataset_hash != self.macro_recipe.training_dataset.content_hash()
            or self.macro_recipe.training_dataset.availability_policy
            != self.macro_recipe.availability_policy
            or self.macro_dataset.availability_policy != self.macro_recipe.availability_policy
            or self.macro_recipe.training_dataset.breadth_snapshot.context.as_of
            > self.macro_recipe.fitted_at
            or self.macro_dataset.breadth_snapshot.context.as_of > self.inputs.context.as_of
            or self.macro_recipe.fitted_at >= self.inputs.context.as_of
            or self.macro_recipe.code_version != self.inputs.context.code_version
        ):
            raise ValueError("macro recipe identity/cutoff differs")
        if len(dict(self.provenance)) != len(self.provenance):
            raise ValueError("duplicate proof reference")
        if len(set(self.desk_evidence_ids)) != len(self.desk_evidence_ids):
            raise ValueError("duplicate desk evidence identity")
        if set(self.allocation.retiring_sleeves) - set(specs):
            raise ValueError("unknown retiring sleeve")
        return self


class PaperCycleTemplate(VersionedContract):
    """Fixed operating policy; every run obtains new source-bound observations."""

    account: AccountSpec
    instruments: tuple[Instrument, ...] = Field(min_length=1)
    market_start: UtcTimestamp
    code_version: NonEmptyString
    universe_version: NonEmptyString
    fill_policy: FillPolicy
    risk_policy: RiskPolicy
    initial_capital: tuple[tuple[NonEmptyString, Decimal], ...]
    strategies: tuple[StrategySpec, ...] = Field(min_length=1)
    models: tuple[ModelRecipe, ...]
    macro_recipe: MacroRecipe
    numeric_bindings: tuple[NumericEvidenceBinding, ...] = ()
    allocation: AllocationPolicy
    max_source_chars: int = Field(default=24000, ge=1000, le=120000)


class CycleResult(VersionedContract):
    run_id: NonEmptyString
    manifest_hash: Sha256
    plan_hash: Sha256
    status: Literal["completed", "blocked", "recovery_required"]
    reason: str
    decision_context: RunContext | None = None
    target_hash: Sha256 | None = None
    prepared_intent_ids: tuple[str, ...] = ()
    snapshot_hash: Sha256 | None = None
    stage_outputs: tuple[tuple[str, Sha256], ...] = ()
