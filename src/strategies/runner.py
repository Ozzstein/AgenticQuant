"""Actual baseline rules over eligible features/models; no orders or portfolio schema.

The returned dataframe is a sleeve sizing input for the accounting-owned target
adapter. Weights are fractions of sleeve capital. HOLD/ABSTAIN/VETO preserve the
supplied holding; REDUCE/EXIT are explicit. Execution/risk remain downstream.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping

import numpy as np
import pandas as pd

from src.core.macro_regime import CausalRegimeResult
from src.core.model_zoo import ModelWrapper
from src.data.evidence.store import canonical_bytes
from src.data.features.dataset import FeatureDataset
from src.domain.common import Instrument, RunContext
from src.strategies.specification import StrategySpec, baseline_specs

if TYPE_CHECKING:
    from src.agents.consensus import ConsensusResult
    from src.domain.portfolio import UniverseEligibility


class StrategyInputError(ValueError):
    pass


def require_strategy_eligible(spec: StrategySpec, context: RunContext, *, resolver: Callable | None = None,
                              approved_baseline_hashes: frozenset[str] = frozenset(),
                              research_registry=None, trial_id: str | None = None):
    if trial_id is not None:
        if context.mode != "backtest" or research_registry is None:
            raise StrategyInputError("registered candidate experiments require an isolated backtest")
        trial = research_registry.trial(trial_id)
        candidate = research_registry.candidate(trial.candidate_id)
        if candidate.kind != "strategy" or candidate.payload_hash != spec.content_hash():
            raise StrategyInputError("trial candidate differs from executable strategy")
        if not trial.evaluation_start <= context.as_of < trial.evaluation_end:
            raise StrategyInputError("candidate execution is outside its preregistered trial")
        if research_registry.blob(candidate.payload_hash) != canonical_bytes(spec.model_dump(mode="json")):
            raise StrategyInputError("registered candidate content differs")
        return "registered_trial:"+trial_id
    shipped = baseline_specs().get(spec.name)
    declared = shipped is not None and spec.origin == "declared_baseline" and shipped.content_hash() == spec.content_hash()
    if declared and (context.mode in ("research", "backtest") or spec.content_hash() in approved_baseline_hashes):
        return "declared_code_baseline"
    if resolver is None:
        raise StrategyInputError("strategy requires explicit code-baseline approval or verified research activation")
    activation = resolver("strategy:"+spec.name, payload_hash=spec.content_hash(), as_of=context.as_of)
    if activation is None or activation.payload_hash != spec.content_hash() or activation.activated_at > context.as_of:
        raise StrategyInputError("strategy is not active at the requested cutoff")
    return activation.candidate_hash


def rebalance_due(spec: StrategySpec, signal_at, last_rebalanced_at):
    if last_rebalanced_at is None:
        return True
    previous = pd.Timestamp(last_rebalanced_at)
    current = pd.Timestamp(signal_at)
    if previous.tzinfo is None or previous > current:
        raise StrategyInputError("last rebalance must be an explicit past timestamp")
    if spec.cadence == "daily":
        return previous.date() < current.date()
    if spec.cadence == "weekly":
        return previous.isocalendar()[:2] != current.isocalendar()[:2]
    return (previous.year, previous.month) != (current.year, current.month)


def evaluate_strategy(spec: StrategySpec, dataset: FeatureDataset, context: RunContext, *,
                      universe: UniverseEligibility, instruments: tuple[Instrument, ...],
                      holdings: pd.Series, signal_at, regime: CausalRegimeResult,
                      model: ModelWrapper | None = None, holding_sessions: pd.Series | None = None,
                      last_rebalanced_at=None, force_rebalance: bool = False,
                      consensus: Mapping[str, ConsensusResult] | None = None,
                      resolver: Callable | None = None,
                      approved_baseline_hashes: frozenset[str] = frozenset(),
                      research_registry=None, trial_id: str | None = None,
                      market_snapshot_hash: str | None = None) -> pd.DataFrame:
    activation = require_strategy_eligible(spec, context, resolver=resolver, approved_baseline_hashes=approved_baseline_hashes,
                                          research_registry=research_registry, trial_id=trial_id)
    parameters = dict(spec.parameters)
    if market_snapshot_hash is not None and not {market_snapshot_hash, dataset.snapshot_hash}.issubset(context.data_snapshot_ids):
        raise StrategyInputError("common market and actual feature snapshots must be approved in RunContext")
    date = pd.Timestamp(signal_at)
    if date.tzinfo is None or date > context.as_of or (pd.Timestamp(context.as_of)-date).total_seconds() > spec.max_signal_age_seconds:
        raise StrategyInputError("signal is future, stale or lacks timezone")
    if universe.effective_at > date or universe.available_at > date or not universe.universe_version:
        raise StrategyInputError("universe was not effective and available at signal cutoff")
    if universe.data_snapshot_id not in {dataset.snapshot_hash, *context.data_snapshot_ids}:
        raise StrategyInputError("universe/data snapshot mismatch")
    if regime.status != "available" or regime.context.as_of != context.as_of or regime.trained_through >= date:
        raise StrategyInputError("an eligible authoritative HMM result is required")
    if not holdings.index.is_unique or not np.isfinite(holdings).all() or (holdings < 0).any() or holdings.sum() > 1+1e-10:
        raise StrategyInputError("finite, unique, long-only sleeve holdings required")
    identities = {item.instrument_id: item for item in instruments}
    eligible = set(universe.instrument_ids)
    if len(identities) != len(instruments) or (eligible | set(holdings.index)) - identities.keys():
        raise StrategyInputError("canonical identities required for universe and holdings")
    if dataset.feature_spec_hash != spec.feature_spec_hash or tuple(dataset.features.columns) != tuple(f.name for f in spec.features):
        raise StrategyInputError("strategy feature specification/schema mismatch")
    if not dataset.features.index.is_unique or dataset.features.index.names != ["datetime", "instrument"]:
        raise StrategyInputError("unique date/instrument feature identity required")
    eligible = {key for key in eligible if identities[key].asset_class in spec.asset_classes}
    keys = sorted(eligible | set(holdings.index))
    result = pd.DataFrame(index=pd.Index(keys, name="instrument"))
    result["target_weight"] = holdings.reindex(keys).fillna(0).astype(float)
    result["action"], result["reason"], result["score"] = "HOLD", "unchanged", np.nan
    result.attrs.update(strategy=spec.name, strategy_hash=spec.content_hash(), activation=activation,
                        context=context.model_dump(mode="json"), signal_at=date.isoformat(),
                        data_snapshot_hash=dataset.snapshot_hash, market_snapshot_hash=market_snapshot_hash or dataset.snapshot_hash,
                        feature_spec_hash=dataset.feature_spec_hash, model_training_data_hash=None,
                        universe_version=universe.universe_version, regime_model_hash=regime.model_hash,
                        model_hash=None, cadence_due=False)
    removed = result.index.difference(sorted(eligible))
    result.loc[removed, ["target_weight", "action", "reason"]] = [0., "EXIT", "removed_from_eligible_universe"]
    if regime.regime not in spec.regimes:
        result.loc[:, ["target_weight", "action", "reason"]] = [0., "EXIT", "regime_ineligible"]
        return result
    if not (force_rebalance or rebalance_due(spec, date, last_rebalanced_at)):
        return result
    result.attrs["cadence_due"] = True
    rows = dataset.features.loc[dataset.features.index.get_level_values("datetime") == date]
    ready = dataset.feature_available_at.reindex(rows.index)
    valid = rows.apply(lambda row: np.isfinite(row.to_numpy(dtype=float)).all(), axis=1) & ready.notna() & (ready <= date)
    rows = rows.loc[valid & rows.index.get_level_values("instrument").isin(eligible)]
    rows = rows.loc[rows.adv_20 >= spec.min_adv]
    available = set(rows.index.get_level_values("instrument"))
    missing = result.index.intersection(sorted(eligible - available))
    result.loc[missing, ["action", "reason"]] = ["ABSTAIN", "missing_stale_or_illiquid_essential_features"]
    if rows.empty:
        return result
    x = rows.droplevel("datetime")
    if spec.family == "agent_consensus":
        if model is not None:
            raise StrategyInputError("agent consensus must be model-free")
        candidates = []
        for instrument_id in x.index:
            vote = (consensus or {}).get(instrument_id)
            action = getattr(getattr(vote, "action", None), "value", getattr(vote, "action", "ABSTAIN"))
            action = str(action).upper()
            if action not in {"BUY", "HOLD", "REDUCE", "EXIT", "ABSTAIN", "VETO"}:
                raise StrategyInputError("unknown consensus action")
            # Consume the agent lane's consensus result, preserving its explicit
            # named/derived vote provenance rather than recounting votes here.
            if action == "BUY" and len(set(vote.agreeing_roles)) < 4:
                raise StrategyInputError("BUY requires four explicit named agreeing roles")
            result.loc[instrument_id, ["action", "reason"]] = [action, getattr(vote, "reason", "consensus_unavailable")]
            if action == "BUY":
                candidates.append(instrument_id)
            elif action == "REDUCE":
                result.loc[instrument_id, "target_weight"] *= 1-parameters["reduction_fraction"]
            elif action == "EXIT":
                result.loc[instrument_id, "target_weight"] = 0
        result.attrs["consensus_votes"] = {key: {"agreeing_roles": value.agreeing_roles, "derived_votes": value.derived_votes} for key, value in (consensus or {}).items() if key in available}
        score = pd.Series(1., index=candidates)
    else:
        if model is None or model.model_name != spec.model_name or model.artifact is None:
            raise StrategyInputError("the specification requires its actual versioned saved model")
        if model.artifact.feature_spec_hash != spec.feature_spec_hash:
            raise StrategyInputError("model was trained with a different feature specification")
        prediction = model.predict(rows, context=context, feature_available_at=ready.loc[rows.index]).droplevel("datetime")
        result.attrs["model_hash"] = model.artifact.content_hash()
        result.attrs["model_training_data_hash"] = model.artifact.training_data_hash
        screen = prediction > parameters["minimum_prediction"]
        if spec.family == "momentum":
            screen &= x.composite_momentum > parameters["minimum_momentum"]
            score = x.composite_momentum
        elif spec.family == "value_quality":
            screen &= (x.pe > 0) & (x.pe <= parameters["maximum_pe"]) & (x.roic >= parameters["minimum_roic"]) & (x.free_cash_flow > parameters["minimum_fcf"])
            score = x.roic / x.pe
        elif spec.family == "mean_reversion":
            screen &= (x.rsi_14 < parameters["maximum_entry_rsi"]) & (x.price_zscore < parameters["maximum_entry_zscore"])
            score = -x.price_zscore
        elif spec.family == "crypto_trend":
            screen &= (x.sma_20 > x.sma_50) & (x.volume_ratio >= parameters["minimum_volume_ratio"]) & (x.crypto_volatility_20 > 0)
            score = prediction
        else:
            screen &= (x.beta_60 >= parameters["minimum_beta"]) & (x.beta_60 <= parameters["maximum_beta"]) & (x.roic >= parameters["minimum_roic"]) & (x.volatility_20 > 0)
            score = -x.beta_60 - x.volatility_20
        selected = pd.DataFrame({"score": score[screen], "prediction": prediction[screen]})
        selected = selected.sort_values(["score", "prediction"], ascending=False, kind="stable").head(spec.top_k)
        score = selected.score
        # Mean-reversion exits are explicit even between new entry signals.
        if spec.family == "mean_reversion":
            if holding_sessions is None or not holdings[holdings > 0].index.isin(holding_sessions.index).all():
                raise StrategyInputError("actual eligible-session holding ages required")
            if (holding_sessions < 0).any() or not np.isfinite(holding_sessions).all():
                raise StrategyInputError("invalid holding ages")
            exits = x.index[(x.price_zscore >= parameters["exit_zscore"]) | (holding_sessions.reindex(x.index).fillna(0) >= spec.max_holding_sessions)]
            result.loc[exits, ["target_weight", "action", "reason"]] = [0., "EXIT", "reversion_or_maximum_holding_period"]
            score = score.drop(exits, errors="ignore")
        else:
            exits = x.index.difference(score.index)
            result.loc[exits, ["target_weight", "action", "reason"]] = [0., "EXIT", "entry_screen_or_rank_no_longer_satisfied"]
    score = score.sort_index().sort_values(ascending=False, kind="stable").head(spec.top_k)
    fixed = float(result.loc[result.index.difference(score.index), "target_weight"].sum())
    budget = max(0., spec.capital_fraction-fixed)
    if not score.empty:
        sizing = pd.Series(1., index=score.index)
        if spec.sizing == "inverse_volatility":
            sizing = 1 / x.loc[score.index, "crypto_volatility_20" if spec.family == "crypto_trend" else "volatility_20"]
        targets = (sizing / sizing.sum() * budget).clip(upper=spec.position_cap)
        crypto_keys = [key for key in targets.index if identities[key].asset_class == "crypto"]
        fixed_crypto = sum(float(result.loc[key, "target_weight"]) for key in result.index.difference(targets.index) if identities[key].asset_class == "crypto")
        crypto_budget = max(0., .25-fixed_crypto)
        if crypto_keys and targets.loc[crypto_keys].sum() > crypto_budget:
            targets.loc[crypto_keys] *= crypto_budget / targets.loc[crypto_keys].sum()
        result.loc[score.index, "target_weight"] = targets
        result.loc[score.index, "score"] = score
        for key in score.index:
            previous = float(holdings.get(key, 0))
            target = float(targets[key])
            result.loc[key, "action"] = "BUY" if target > previous else "REDUCE" if target < previous else "HOLD"
            result.loc[key, "reason"] = "selected_by_exact_strategy_rules"
    return result
