"""Complete declared baseline specifications, separate from promoted candidates."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from src.data.evidence.store import canonical_bytes, digest
from src.domain.common import VersionedContract
from src.domain.market import FeatureSpec


class StrategySpec(VersionedContract):
    name: str
    family: Literal["momentum", "value_quality", "mean_reversion", "crypto_trend", "defensive", "agent_consensus"]
    origin: Literal["declared_baseline", "research_candidate"] = "declared_baseline"
    model_name: Literal["LightGBM", "CatBoost", "XGBoost", "Linear"] | None
    features: tuple[FeatureSpec, ...]
    asset_classes: tuple[Literal["equity", "etf", "crypto"], ...]
    regimes: tuple[Literal["risk_on", "neutral", "risk_off", "crisis"], ...]
    cadence: Literal["daily", "weekly", "monthly"]
    top_k: int = Field(ge=1, le=100)
    capital_fraction: float = Field(gt=0, le=.95)
    position_cap: float = Field(gt=0, le=.1)
    max_signal_age_seconds: int = Field(default=86400, ge=0)
    min_adv: float = Field(default=1_000_000, ge=1_000_000)
    entry_rules: tuple[str, ...]
    exit_rules: tuple[str, ...]
    sizing: Literal["equal", "inverse_volatility"]
    max_holding_sessions: int | None = Field(default=None, ge=1)
    parameters: tuple[tuple[str, float], ...]

    @model_validator(mode="after")
    def coherent_spec(self):
        if not self.features or len({f.name for f in self.features}) != len(self.features):
            raise ValueError("explicit unique feature specifications required")
        if (self.family == "agent_consensus") != (self.model_name is None):
            raise ValueError("agent consensus is the sole model-free baseline")
        if "crypto" in self.asset_classes and self.position_cap > .05:
            raise ValueError("crypto-capable sleeves cannot exceed 5% position cap")
        if self.family == "defensive" and ("crypto" in self.asset_classes or self.capital_fraction > .7):
            raise ValueError("defensive policy requires no crypto and at least 30% cash")
        if self.family == "crypto_trend" and self.capital_fraction > .25:
            raise ValueError("crypto sleeve allocation cap is 25%")
        parameters = dict(self.parameters)
        required = {
            "momentum": {"minimum_prediction", "minimum_momentum"},
            "value_quality": {"minimum_prediction", "maximum_pe", "minimum_roic", "minimum_fcf"},
            "mean_reversion": {"minimum_prediction", "maximum_entry_rsi", "maximum_entry_zscore", "exit_zscore"},
            "crypto_trend": {"minimum_prediction", "minimum_volume_ratio"},
            "defensive": {"minimum_prediction", "minimum_beta", "maximum_beta", "minimum_roic"},
            "agent_consensus": {"reduction_fraction"},
        }[self.family]
        if len(parameters) != len(self.parameters) or set(parameters) != required:
            raise ValueError("exact executable family parameters required")
        if self.family == "agent_consensus" and not 0 < parameters["reduction_fraction"] < 1:
            raise ValueError("REDUCE must be an explicit partial reduction")
        if self.family == "mean_reversion" and self.max_holding_sessions is None:
            raise ValueError("mean reversion requires an explicit session holding limit")
        return self

    @property
    def feature_spec_hash(self):
        return digest(canonical_bytes([feature.model_dump(mode="json") for feature in self.features]))


_BUILTIN_NAMES = ("momentum_topk", "value_quality", "mean_reversion", "crypto_trend", "defensive", "agent_consensus")


def baseline_specs() -> dict[str, StrategySpec]:
    """Load only shipped declarations, never a research-library validated flag."""
    directory = Path(__file__).parent / "specifications"
    result = {}
    for name in _BUILTIN_NAMES:
        spec = StrategySpec.model_validate_json((directory / f"{name}.json").read_text())
        if spec.name != name or spec.origin != "declared_baseline":
            raise ValueError("baseline specification identity mismatch")
        result[name] = spec
    return result
