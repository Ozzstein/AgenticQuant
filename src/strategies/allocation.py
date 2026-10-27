"""Traceable capital allocation over eligible strategy outputs and ledger returns."""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.core.macro_regime import CausalRegimeResult
from src.domain.common import RunContext
from src.strategies.runner import StrategyInputError, require_strategy_eligible
from src.strategies.specification import StrategySpec


def select_allocations(specs: tuple[StrategySpec, ...], context: RunContext, regime: CausalRegimeResult, *,
                       mode: str, performance: pd.DataFrame | None = None, manual: str | None = None,
                       exploration: float = .1, seed: int = 0, resolver: Callable | None = None,
                       approved_baseline_hashes: frozenset[str] = frozenset()) -> dict[str, float]:
    """No performance substitutes: missing metrics never imply equal-weight success.

    Performance rows are produced by AttributedPerformanceStore.metrics() from
    ledger NAV, with cutoff, sample count and source-event lineage.
    """
    if len({s.name for s in specs}) != len(specs) or not specs:
        raise StrategyInputError("unique nonempty strategy universe required")
    if regime.status != "available" or regime.context.as_of != context.as_of:
        raise StrategyInputError("authoritative available HMM result required")
    eligible = {}
    for spec in specs:
        try:
            require_strategy_eligible(spec, context, resolver=resolver, approved_baseline_hashes=approved_baseline_hashes)
        except StrategyInputError:
            continue
        eligible[spec.name] = spec
    if not eligible:
        raise StrategyInputError("no eligible strategies")
    if performance is not None:
        required = {"sharpe", "observations", "available_at", "source_event_ids", "annualization", "complete"}
        if required - set(performance) or not performance.index.is_unique:
            raise StrategyInputError("attributed performance with time and ledger lineage required")
        if performance.available_at.isna().any() or any(pd.Timestamp(t).tzinfo is None for t in performance.available_at) or (performance.available_at > context.as_of).any():
            raise StrategyInputError("performance contains unknown/future availability")
        if any(not isinstance(ids, tuple) or not ids for ids in performance.source_event_ids):
            raise StrategyInputError("performance requires actual source event identities")
        performance = performance.loc[performance.complete == True]  # noqa: E712
    if mode == "manual":
        if manual not in eligible:
            raise StrategyInputError("manual strategy is unknown or not eligible")
        if regime.regime not in eligible[manual].regimes:
            raise StrategyInputError("manual selection cannot bypass regime risk constraints")
        return {manual: 1.}
    matching = {name: spec for name, spec in eligible.items() if regime.regime in spec.regimes}
    if not matching:
        raise StrategyInputError("no strategies match the effective regime")
    metrics = performance.reindex(matching) if performance is not None else pd.DataFrame()
    if mode == "regime_matched":
        known = metrics.loc[(metrics.observations >= 20) & np.isfinite(metrics.sharpe)] if not metrics.empty else metrics
        name = known.sharpe.idxmax() if not known.empty else next(iter(matching))
        return {name: 1.}
    if mode == "performance_weighted":
        if metrics.empty:
            raise StrategyInputError("attributed performance is unavailable")
        usable = metrics.loc[(metrics.observations >= 20) & np.isfinite(metrics.sharpe) & (metrics.sharpe > 0)]
        if usable.empty:
            raise StrategyInputError("no positive sufficiently observed attributed performance")
        if usable.annualization.nunique() != 1:
            raise StrategyInputError("performance must use a common declared sampling calendar")
        return (usable.sharpe / usable.sharpe.sum()).to_dict()
    if mode == "bandit":
        if not 0 <= exploration <= 1:
            raise StrategyInputError("exploration probability must be between zero and one")
        # Explicit reproducible seed; only already eligible candidates enter the
        # exploration pool. Unknown research files never receive trial capital.
        rng = np.random.default_rng(seed)
        names = sorted(matching)
        if rng.random() < exploration:
            return {str(rng.choice(names)): 1.}
        if metrics.empty:
            raise StrategyInputError("bandit exploitation requires attributed observations")
        usable = metrics.loc[(metrics.observations >= 20) & np.isfinite(metrics.sharpe)]
        if usable.empty:
            raise StrategyInputError("no observed eligible bandit candidate")
        return {str(usable.sharpe.idxmax()): 1.}
    raise StrategyInputError("unknown allocation mode")


def blend_sleeves(outputs: Mapping[str, pd.DataFrame], allocations: Mapping[str, float],
                  context: RunContext, *, current_weights: pd.Series,
                  current_contributions: Mapping[str, pd.Series] | None = None) -> pd.DataFrame:
    """Net sleeve targets; canonical PortfolioTarget/risk/execution stay accounting-owned.

    Caller must explicitly include any retiring sleeve output. Dropping a sleeve
    while it still owns holdings is an orchestration error, not a liquidation
    instruction. Downstream accounting records resulting capital transfers.
    """
    if set(outputs) != set(allocations) or not allocations:
        raise StrategyInputError("each allocated sleeve requires its actual output")
    weights = np.asarray(list(allocations.values()), dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() > 1+1e-10:
        raise StrategyInputError("long-only unlevered capital allocations required")
    if not current_weights.index.is_unique or (current_weights < 0).any() or not np.isfinite(current_weights).all():
        raise StrategyInputError("actual current portfolio weights required")
    if current_contributions is None:
        if len(outputs) > 1 and (current_weights > 0).any():
            raise StrategyInputError("multiple held sleeves require actual current attributed contributions")
        current_contributions = {name: current_weights if len(outputs) == 1 else pd.Series(dtype=float) for name in outputs}
    if set(current_contributions) != set(outputs):
        raise StrategyInputError("current contribution ownership must match every sleeve")
    for values in current_contributions.values():
        if not values.index.is_unique or not np.isfinite(values).all() or (values < 0).any():
            raise StrategyInputError("invalid attributed current holdings")
    attributed = pd.DataFrame(current_contributions).fillna(0.).sum(axis=1)
    all_current = attributed.index.union(current_weights.index)
    if not np.allclose(attributed.reindex(all_current, fill_value=0), current_weights.reindex(all_current, fill_value=0), rtol=0, atol=1e-12):
        raise StrategyInputError("attributed current holdings do not reconcile to portfolio")
    contributions, vetoed = {}, set()
    identities, signal_ats = [], {}
    for name, frame in outputs.items():
        if not frame.index.is_unique or frame.attrs.get("strategy") != name or frame.attrs.get("context") != context.model_dump(mode="json"):
            raise StrategyInputError("sleeve provenance/identity mismatch")
        if not np.isfinite(frame.target_weight).all() or (frame.target_weight < 0).any():
            raise StrategyInputError("invalid sleeve targets")
        required = ("data_snapshot_hash", "universe_version", "signal_at", "strategy_hash", "feature_spec_hash", "model_hash")
        if any(key not in frame.attrs for key in required):
            raise StrategyInputError("sleeve source/model/spec provenance is incomplete")
        market = frame.attrs.get("market_snapshot_hash", frame.attrs["data_snapshot_hash"])
        if market != frame.attrs["data_snapshot_hash"] and not {market, frame.attrs["data_snapshot_hash"]}.issubset(context.data_snapshot_ids):
            raise StrategyInputError("common market and actual feature snapshots must be approved in RunContext")
        signal = pd.Timestamp(frame.attrs['signal_at'])
        if signal.tzinfo is None or signal > context.as_of:
            raise StrategyInputError("sleeve signal must precede the shared actual decision cutoff")
        signal_ats[name] = frame.attrs['signal_at']
        identities.append((market, frame.attrs["universe_version"]))
        contributions[name] = frame.target_weight * allocations[name]
        previous = current_contributions[name].reindex(frame.index, fill_value=0)
        omitted = current_contributions[name][current_contributions[name] > 0].index.difference(frame.index)
        if len(omitted):
            raise StrategyInputError("held sleeve instrument omitted without an explicit transition")
        passive = frame.action.isin(["HOLD", "ABSTAIN", "VETO"])
        contributions[name].loc[passive] = previous.loc[passive]
        if (contributions[name].loc[frame.action == "REDUCE"] > previous.loc[frame.action == "REDUCE"]+1e-12).any():
            raise StrategyInputError("REDUCE cannot increase attributed exposure")
        if (contributions[name].loc[frame.action == "EXIT"] != 0).any():
            raise StrategyInputError("EXIT must explicitly close its sleeve contribution")
        vetoed.update(frame.index[frame.action == "VETO"])
    if len(set(identities)) != 1:
        raise StrategyInputError("cannot net mismatched snapshot or universe")
    contribution = pd.DataFrame(contributions).fillna(0.)
    missing = current_weights[current_weights > 0].index.difference(contribution.index)
    if len(missing):
        raise StrategyInputError("existing holding omitted; supply explicit transition/retirement output")
    target = contribution.sum(axis=1)
    for key in vetoed:
        for name in contributions:
            contribution.loc[key, name] = min(contribution.loc[key, name], float(current_contributions[name].get(key, 0)))
        target[key] = contribution.loc[key].sum()
    if target.sum() > 1+1e-10:
        raise StrategyInputError("preserved holdings plus new sleeve targets exceed available capital")
    result = pd.DataFrame({"target_weight": target})
    result.attrs.update(context=context.model_dump(mode="json"), allocations=dict(allocations),
                        data_snapshot_hash=identities[0][0], market_snapshot_hash=identities[0][0],
                        data_snapshot_hashes={name: frame.attrs["data_snapshot_hash"] for name, frame in outputs.items()},
                        model_training_data_hashes={name: frame.attrs.get("model_training_data_hash") for name, frame in outputs.items()},
                        universe_version=identities[0][1],
                        signal_at=max(pd.Timestamp(t) for t in signal_ats.values()).isoformat(),
                        signal_ats=signal_ats,
                        strategy_hashes={name: frame.attrs["strategy_hash"] for name, frame in outputs.items()},
                        model_artifact_hashes={name: frame.attrs["model_hash"] for name, frame in outputs.items()},
                        sleeve_provenance={name: dict(frame.attrs) for name, frame in outputs.items()},
                        contributions={name: contribution.loc[frame.index, name].to_dict()
                                       for name, frame in outputs.items()},
                        vetoed=tuple(sorted(vetoed)))
    return result
