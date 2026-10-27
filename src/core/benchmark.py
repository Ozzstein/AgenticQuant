"""Matched benchmark and cost reruns using the authoritative portfolio ledger."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal as D
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.core.backtester import (
    PortfolioReplay,
    ReplayArtifact,
    ReplayResult,
    canonical_hash,
    file_hash,
    load_replay_artifact,
    persist_replay_artifact,
    recorded_decisions,
    recorded_replay_inputs,
    require_matching_inputs,
    write_immutable,
)
from src.domain.common import Sha256, UtcTimestamp
from src.domain.portfolio import FillPolicy, PortfolioTarget, RankedSignal, RiskPolicy, TargetWeight
from src.execution.ledger import PortfolioLedger
from src.utils.exceptions import BacktestError


class MatchedComparison(BaseModel):
    schema_version: str = "2.0"
    strategy_run_id: str
    strategy_manifest_hash: str = ""
    strategy_result_hash: str = ""
    signal_artifact: dict = Field(default_factory=dict)
    opportunity_artifact: dict = Field(default_factory=dict)
    strategy_artifact: ReplayArtifact | None = None
    random_artifacts: list[ReplayArtifact] = Field(default_factory=list)
    benchmark_artifacts: dict[str, ReplayArtifact] = Field(default_factory=dict)
    strategy_sharpe: float | None
    benchmark_metrics: dict[str, dict] = Field(default_factory=dict)
    unavailable: dict[str, str] = Field(default_factory=dict)
    random_sharpes: list[float | None] = Field(default_factory=list)
    random_nav: list[list[float]] = Field(default_factory=list)
    monte_carlo_percentile: float | None = None
    simulations: int
    seed: int
    match: dict


class OpportunityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    instrument_id: str
    prediction: float


class OpportunityRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    at: UtcTimestamp
    universe_hash: Sha256
    universe_version: str
    model_payload_hash: Sha256
    model_artifact_hash: Sha256
    regime_observation_hash: Sha256
    rebalance_due: bool
    eligible: tuple[OpportunityCandidate, ...]
    trace_link: dict[str, str]


class CallbackOpportunities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["1.0"]
    implementation: Literal["registered-strategy-opportunities-1"]
    inputs_hash: Sha256
    plan_hash: Sha256
    feature_data_hash: Sha256
    feature_spec_hash: Sha256
    cadence: str
    decision_times: tuple[UtcTimestamp, ...]
    rows: tuple[OpportunityRow, ...]


def _callback_source(baseline, strategy_artifact, opportunity_artifact):
    """Load actual baseline economics and frozen opportunity bytes independently."""
    if strategy_artifact is None or load_replay_artifact(strategy_artifact) != baseline:
        raise BacktestError("callback benchmark requires the exact durable baseline replay")
    source = baseline.manifest.get("provenance", {}).get("opportunities_artifact")
    if (
        not source
        or opportunity_artifact.get("sha256") != source["sha256"]
        or file_hash(opportunity_artifact["path"]) != source["sha256"]
    ):
        raise BacktestError("opportunity artifact differs from frozen baseline provenance")
    opportunities = CallbackOpportunities.model_validate_json(
        Path(opportunity_artifact["path"]).read_text()
    )
    ledger = PortfolioLedger(strategy_artifact.ledger_path, read_only=True)
    try:
        inputs = recorded_replay_inputs(ledger, baseline)
        decisions = recorded_decisions(ledger, baseline)
    finally:
        ledger.close()
    schedule = tuple(datetime.fromisoformat(t) for t in baseline.manifest["decision_times"])
    if (
        opportunities.decision_times != schedule
        or tuple(r.at for r in opportunities.rows) != schedule
        or opportunities.feature_data_hash != baseline.manifest["provenance"].get("features_hash")
        or not opportunities.cadence
        or not decisions
    ):
        raise BacktestError("opportunity schedule/features differ from baseline decision trace")
    identities = {b.instrument.instrument_id: b.instrument for b in inputs["bars"]}
    for row, decision in zip(opportunities.rows, decisions, strict=True):
        link = row.trace_link
        if (
            set(link) != {"account_id", "run_id", "at"}
            or link["account_id"] != baseline.account_id
            or link["run_id"] != baseline.run_id
            or datetime.fromisoformat(link["at"]) != decision.at
            or row.at != decision.at
        ):
            raise BacktestError("opportunity trace link differs from actual baseline decision")
        choices = [
            u
            for u in inputs["universes"]
            if u.universe_version == row.universe_version
            and u.effective_at <= row.at
            and u.available_at <= row.at
        ]
        if not choices:
            raise BacktestError("opportunity lacks a causal universe")
        universe = max(choices, key=lambda u: (u.effective_at, u.available_at))
        ids = [c.instrument_id for c in row.eligible]
        if (
            universe.content_hash() != row.universe_hash
            or len(set(ids)) != len(ids)
            or not set(ids) <= set(universe.instrument_ids)
            or not set(ids) <= identities.keys()
        ):
            raise BacktestError(
                "opportunity eligibility conflicts with actual PIT universe/identities"
            )
        for key in ids:
            if not any(
                b.instrument.instrument_id == key and b.closed_at <= row.at for b in inputs["bars"]
            ):
                raise BacktestError("eligible opportunity lacks a causal executable price")
            metadata = [
                m
                for m in inputs["metadata"]
                if m.instrument.instrument_id == key and m.as_of <= row.at
            ]
            if not metadata:
                raise BacktestError("eligible opportunity lacks causal liquidity metadata")
        folds = [
            f
            for f in baseline.walk_forward_results
            if datetime.fromisoformat(f.test_start) <= row.at < datetime.fromisoformat(f.test_end)
        ]
        if baseline.walk_forward_results and (
            len(folds) != 1 or folds[0].model_version != row.model_payload_hash
        ):
            raise BacktestError("opportunity model payload differs from actual OOS fold")
        if decision.target is not None:
            if decision.target.sleeve_contributions:
                raise BacktestError(
                    "callback opportunity benchmarks require explicit per-sleeve opportunity sets for multi-sleeve strategies"
                )
            if decision.target.universe_version != row.universe_version:
                raise BacktestError("target and opportunity universe versions differ")
            active = {
                w.instrument.instrument_id
                for w in decision.target.weights
                if w.weight > 0
                and w.instrument.instrument_id not in decision.target.preserve_quantities
            }
            if not active <= set(ids):
                raise BacktestError("baseline active target is outside its frozen opportunity set")
    return opportunities, inputs, decisions, identities


def _callback_policy(baseline, opportunities, decisions, identities, *, name, seed, index=None):
    """A pure selection control conditioned on observed strategy actions.

    Same active slots, asset-class budgets and decision times; this tests
    selection conditional on the original decisions, not a reimplementation of
    the strategy rules. Passive quantities come from each control's own fills.
    """
    recipe = canonical_hash(
        {
            "implementation": "callback-selection-control-1",
            "parent": canonical_hash(baseline.manifest),
            "opportunities": opportunities.model_dump(mode="json"),
            "name": name,
            "seed": seed,
            "index": index,
        }
    )
    mapping, plans, nondegenerate = {}, {}, False
    risk = RiskPolicy.model_validate(baseline.manifest["risk_policy"])
    symbols = {i.symbol: i for i in identities.values()}
    for row, decision in zip(opportunities.rows, decisions, strict=True):
        target = decision.target
        if target is None:
            plans[row.at] = None
            continue
        if name == "random":
            positive = {w.instrument.instrument_id: w for w in target.weights if w.weight > 0}
            passive = set(target.preserve_quantities)
            next_mapping = {k: mapping[k] for k in passive if k in mapping}
            if any(target.preserve_quantities[k] > 0 for k in passive - mapping.keys()):
                raise BacktestError("passive baseline slot has no prior benchmark allocation")
            active = sorted(positive.keys() - passive)
            eligible = {c.instrument_id for c in row.eligible}
            # Preserve off-cadence REDUCE identities. New active slots require
            # an explicit rebalance opportunity, never a hidden daily rotation.
            if not row.rebalance_due:
                if set(active) - mapping.keys():
                    raise BacktestError("new baseline allocation outside frozen rebalance cadence")
                next_mapping.update({k: mapping[k] for k in active})
            else:
                rng = np.random.default_rng(
                    int(
                        canonical_hash({"seed": seed, "index": index, "at": row.at.isoformat()}), 16
                    )
                )
                for asset_class in sorted({identities[k].asset_class for k in active}):
                    slots = [k for k in active if identities[k].asset_class == asset_class]
                    candidates = sorted(
                        k
                        for k in eligible
                        if identities[k].asset_class == asset_class
                        and k not in next_mapping.values()
                    )
                    if len(candidates) < len(slots):
                        raise BacktestError(
                            "insufficient eligible identities for matched active slots"
                        )
                    nondegenerate |= len(candidates) > len(slots) or len(slots) > 1
                    chosen = rng.choice(candidates, size=len(slots), replace=False).tolist()
                    next_mapping.update(zip(slots, chosen, strict=True))
            if any(next_mapping[k] not in eligible for k in active):
                raise BacktestError("off-cadence control holding left its opportunity universe")
            mapping = next_mapping
            plans[row.at] = (target, dict(mapping), passive)
        else:
            plans[row.at] = (target, row, None)

    def provider(at, snapshot):
        plan = plans[at]
        if plan is None:
            return None
        original, allocation, passive = plan
        preserved = {}
        if name == "random":
            held = {p.instrument.instrument_id: p for p in snapshot.positions}
            original_weights = {w.instrument.instrument_id: w.weight for w in original.weights}
            weights = {}
            for slot, key in allocation.items():
                if slot in passive:
                    position = held.get(key)
                    preserved[key] = position.quantity if position else D(0)
                    weights[key] = position.market_value / snapshot.nav if position else D(0)
                else:
                    weights[key] = original_weights[slot]
        elif name == "equal-weight":
            eligible = [c.instrument_id for c in allocation.eligible]
            weights = {
                key: min(
                    (1 - risk.min_cash) / len(eligible),
                    risk.crypto_weight
                    if identities[key].asset_class == "crypto"
                    else risk.equity_weight,
                )
                for key in eligible
            }
        else:
            components = {
                "SPY": [("SPY", D(1))],
                "BTC": [("BTC/USD", D(1))],
                "60/40": [("SPY", D(".6")), ("AGG", D(".4"))],
            }[name]
            if any(s not in symbols for s, _ in components):
                raise BacktestError("actual benchmark bars and instrument identity not supplied")
            scale = min(
                [1 - risk.min_cash]
                + [
                    (
                        risk.crypto_weight
                        if symbols[s].asset_class == "crypto"
                        else risk.equity_weight
                    )
                    / w
                    for s, w in components
                ]
            )
            weights = {symbols[s].instrument_id: w * scale for s, w in components}
        return PortfolioTarget(
            target_id=canonical_hash(
                {"recipe": recipe, "at": at.isoformat(), "snapshot": snapshot.content_hash()}
            ),
            decision_at=at,
            universe_version=original.universe_version,
            strategy_id="benchmark:" + name,
            artifact_version=recipe,
            data_snapshot_id=original.data_snapshot_id,
            weights=tuple(
                TargetWeight(instrument=identities[k], weight=v) for k, v in sorted(weights.items())
            ),
            preserve_quantities=preserved,
        )

    return provider, recipe, nondegenerate


def signal_groups(signals):
    groups = {}
    for signal in sorted(signals, key=lambda s: (s.decision_at, s.instrument.instrument_id)):
        groups.setdefault(signal.decision_at, []).append(signal)
    return groups


def random_signals(groups, rng, index, seed):
    result = []
    for group in groups.values():
        scores = rng.permutation(len(group))
        result.extend(
            s.model_copy(
                update={
                    "signal_id": f"random-{index}:" + s.signal_id,
                    "score": float(score),
                    "model_version": f"random-seed-{seed}",
                    "strategy_id": "random",
                }
            )
            for s, score in zip(group, scores, strict=True)
        )
    return result


def require_matched_replay(baseline: ReplayResult, candidate: ReplayResult, *, cost_variant=False):
    if candidate.status != "complete" or candidate.timestamps != baseline.timestamps:
        raise BacktestError("comparison replay is incomplete or has different valuation timestamps")
    keys = [
        "bars_hash",
        "metadata_hash",
        "universe_hash",
        "actions_hash",
        "risk_policy",
        "topk",
        "samples_per_year",
        "valuation_calendar",
        "valuation_times",
        "cash_return",
        "cost_convention",
    ]
    if cost_variant:
        keys.append("sleeve_capital")
        if "decision_recipe_hash" in baseline.manifest:
            # Holdings change with fees; rerun the same causal recipe, not its
            # old target stream. Each candidate's actual trace is verified.
            keys.extend(["decision_recipe_hash", "decision_times"])
        else:
            keys.extend(["signals_hash", "targets_hash"])
    else:
        keys.append("fill_policy")
    if any(candidate.manifest.get(k) != baseline.manifest.get(k) for k in keys):
        raise BacktestError(
            "comparison manifest is not matched to frozen strategy inputs/conventions"
        )
    for section, omit in (("account", {"account_id"}), ("context", {"account_id", "run_id"})):
        if {k: v for k, v in candidate.manifest[section].items() if k not in omit} != {
            k: v for k, v in baseline.manifest[section].items() if k not in omit
        }:
            raise BacktestError("comparison account capital/clock/context differs")
    if cost_variant:
        omit = {"equity_cost_bps", "crypto_cost_bps"}
        if {k: v for k, v in candidate.manifest["fill_policy"].items() if k not in omit} != {
            k: v for k, v in baseline.manifest["fill_policy"].items() if k not in omit
        }:
            raise BacktestError("cost scenario changes non-cost fill conventions")


def benchmark_targets(name, groups, instruments, risk, snapshot_id):
    definitions = {
        "SPY": [("SPY", D(1))],
        "BTC": [("BTC/USD", D(1))],
        "60/40": [("SPY", D(".6")), ("AGG", D(".4"))],
    }
    targets = []
    fixed_weights = None
    if name != "equal-weight":
        components = definitions[name]
        if any(symbol not in instruments for symbol, _ in components):
            raise BacktestError("actual benchmark bars and instrument identity not supplied")
        scale = min(
            [1 - risk.min_cash]
            + [
                (
                    risk.crypto_weight
                    if instruments[s].asset_class == "crypto"
                    else risk.equity_weight
                )
                / w
                for s, w in components
            ]
        )
        fixed_weights = {s: str(w * scale) for s, w in components}
    for at, group in groups.items():
        if name == "equal-weight":
            weights = tuple(
                TargetWeight(
                    instrument=s.instrument,
                    weight=min(
                        (1 - risk.min_cash) / len(group),
                        risk.crypto_weight
                        if s.instrument.asset_class == "crypto"
                        else risk.equity_weight,
                    ),
                )
                for s in group
            )
            target_id, version, data_id = (
                "equal:" + at.isoformat(),
                "equal-weight-v1",
                group[0].data_snapshot_id,
            )
        else:
            weights = tuple(
                TargetWeight(instrument=instruments[s], weight=w * scale) for s, w in components
            )
            target_id, version, data_id = (
                f"{name}:{at.isoformat()}",
                "fixed-constrained-benchmark-v1",
                snapshot_id,
            )
        targets.append(
            PortfolioTarget(
                universe_version=group[0].universe_version,
                target_id=target_id,
                decision_at=at,
                strategy_id=name,
                artifact_version=version,
                data_snapshot_id=data_id,
                weights=weights,
            )
        )
    return targets, fixed_weights


def verify_comparison(
    baseline: ReplayResult, comparison: MatchedComparison, *, minimum_simulations=1000
):
    """Verify every underlying replay, including seeded ranks and named allocations."""
    if (
        comparison.strategy_run_id != baseline.run_id
        or comparison.strategy_manifest_hash != canonical_hash(baseline.manifest)
        or comparison.strategy_result_hash != canonical_hash(baseline.model_dump(mode="json"))
        or comparison.strategy_sharpe != baseline.metrics.sharpe_ratio
    ):
        raise BacktestError("benchmark strategy binding differs")
    if (
        comparison.simulations < minimum_simulations
        or len(comparison.random_artifacts) != comparison.simulations
    ):
        raise BacktestError("missing durable random replay artifacts")
    if "decision_recipe_hash" in baseline.manifest:
        return _verify_callback_comparison(baseline, comparison)
    reference = comparison.signal_artifact
    if file_hash(reference["path"]) != reference["sha256"]:
        raise BacktestError("benchmark source signal artifact hash differs")
    signals = [
        RankedSignal.model_validate(s) for s in json.loads(Path(reference["path"]).read_text())
    ]
    if (
        canonical_hash(
            [s.model_dump(mode="json") for s in sorted(signals, key=lambda s: s.signal_id)]
        )
        != baseline.manifest["signals_hash"]
    ):
        raise BacktestError("benchmark source signal content differs")
    groups = signal_groups(signals)
    if not groups or any(baseline.manifest["topk"] >= len(g) for g in groups.values()):
        raise BacktestError("random selection is degenerate")
    rng, sharpes, curves, identities = np.random.default_rng(comparison.seed), [], [], set()
    for index, reference in enumerate(comparison.random_artifacts):
        candidate = load_replay_artifact(reference)
        require_matched_replay(baseline, candidate)
        expected = random_signals(groups, rng, index, comparison.seed)
        if candidate.manifest["signals_hash"] != canonical_hash(
            [s.model_dump(mode="json") for s in sorted(expected, key=lambda s: s.signal_id)]
        ) or candidate.manifest["targets_hash"] != canonical_hash([]):
            raise BacktestError("random replay does not match the seeded date-local selection")
        identity = (candidate.run_id, candidate.account_id)
        if identity in identities:
            raise BacktestError("duplicate random replay identity")
        identities.add(identity)
        sharpes.append(candidate.metrics.sharpe_ratio)
        curves.append(candidate.equity_curve)
    if sharpes != comparison.random_sharpes or curves != comparison.random_nav:
        raise BacktestError("random summary differs from persisted actual replay results")
    if (
        any(s is None for s in sharpes)
        or baseline.metrics.sharpe_ratio is None
        or np.std(sharpes) < 1e-12
    ):
        raise BacktestError("random percentile undefined or degenerate")
    value = baseline.metrics.sharpe_ratio
    percentile = float(
        100 * np.mean([1 if s < value else 0.5 if s == value else 0 for s in sharpes])
    )
    if percentile != comparison.monte_carlo_percentile:
        raise BacktestError("reported random percentile does not recompute")
    for key in (
        "timestamps",
        "bars_hash",
        "metadata_hash",
        "signals_hash",
        "topk",
        "fill_policy",
        "risk_policy",
        "cash_return",
    ):
        expected = baseline.timestamps if key == "timestamps" else baseline.manifest[key]
        if comparison.match.get(key) != expected:
            raise BacktestError("benchmark declared match differs from strategy")
    for name, summary_name in (
        ("SPY", "SPY"),
        ("BTC", "BTC"),
        ("60/40", "60/40"),
        ("equal-weight", "equal_weight"),
    ):
        if (
            name not in comparison.benchmark_artifacts
            or summary_name not in comparison.benchmark_metrics
        ):
            raise BacktestError("missing durable named benchmark " + name)
        reference = comparison.benchmark_artifacts[name]
        candidate = load_replay_artifact(reference)
        require_matched_replay(baseline, candidate)
        ledger = PortfolioLedger(reference.ledger_path, read_only=True)
        try:
            inputs = recorded_replay_inputs(ledger, candidate)
        finally:
            ledger.close()
        instruments = {b.instrument.symbol: b.instrument for b in inputs["bars"]}
        expected, weights = benchmark_targets(
            name,
            groups,
            instruments,
            RiskPolicy.model_validate(baseline.manifest["risk_policy"]),
            baseline.manifest["context"]["data_snapshot_ids"][0],
        )
        if candidate.manifest["targets_hash"] != canonical_hash(
            [t.model_dump(mode="json") for t in expected]
        ) or candidate.manifest["signals_hash"] != canonical_hash([]):
            raise BacktestError("named benchmark allocation differs from declared policy")
        summary = {
            **candidate.metrics.model_dump(),
            "nav": candidate.equity_curve,
            "dates": candidate.timestamps,
        }
        if weights is not None:
            summary["weights"] = weights
        if summary != comparison.benchmark_metrics[summary_name]:
            raise BacktestError("named benchmark summary differs from actual persisted result")


class BenchmarkRunner:
    def __init__(self, config=None):
        self.config = config

    def compare_replay(
        self,
        result: ReplayResult,
        *,
        inputs: dict,
        output_dir: str | Path,
        n_simulations=1000,
        seed=42,
        ledger_path: str | Path | None = None,
    ) -> MatchedComparison:
        """Rebalance on the strategy's decision times, with identical costs/risk.

        Random portfolios have the same date-local universe and top-K. If K
        spans the universe, a random selection benchmark is mathematically
        degenerate and is reported unavailable, never as a neutral percentile.
        SPY/BTC and 60/40 are constrained benchmark sleeves, with unused capital
        in cash. They need their own real price/metadata inputs in the bundle.
        """
        if n_simulations < 1:
            raise BacktestError("positive simulation count required")
        if "decision_recipe_hash" in result.manifest:
            return _compare_callbacks(
                result,
                inputs=inputs,
                output_dir=output_dir,
                n_simulations=n_simulations,
                seed=seed,
                ledger_path=ledger_path,
            )
        if not inputs.get("signals"):
            raise BacktestError(
                "matched random selection requires the original point-in-time ranked universe"
            )
        require_matching_inputs(result, inputs)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        groups = signal_groups(inputs["signals"])
        signal_path = (output / "source-signals.json").resolve()
        write_immutable(
            signal_path,
            json.dumps(
                [
                    s.model_dump(mode="json")
                    for s in sorted(inputs["signals"], key=lambda s: s.signal_id)
                ],
                sort_keys=True,
                indent=2,
            ),
        )
        topk = inputs.get("topk", 30)
        rng = np.random.default_rng(seed)
        comparison = MatchedComparison(
            strategy_run_id=result.run_id,
            strategy_manifest_hash=canonical_hash(result.manifest),
            strategy_result_hash=canonical_hash(result.model_dump(mode="json")),
            signal_artifact={"path": str(signal_path), "sha256": file_hash(signal_path)},
            strategy_sharpe=result.metrics.sharpe_ratio,
            simulations=n_simulations,
            seed=seed,
            match={
                "bars_hash": result.manifest["bars_hash"],
                "metadata_hash": result.manifest["metadata_hash"],
                "signals_hash": result.manifest["signals_hash"],
                "timestamps": result.timestamps,
                "topk": topk,
                "fill_policy": result.manifest["fill_policy"],
                "risk_policy": result.manifest["risk_policy"],
                "cash_return": "zero",
                "costs": "one-way fill charges, identical rules",
                "seed": seed,
            },
        )

        def execute(name, *, signals=None, targets=None):
            account = inputs["account"].model_copy(
                update={"account_id": "backtest:benchmark:" + name}
            )
            context = inputs["context"].model_copy(
                update={"account_id": account.account_id, "run_id": "benchmark:" + name}
            )
            args = {
                **inputs,
                "account": account,
                "context": context,
                "signals": signals,
                "targets": targets,
                "provenance": {
                    **inputs.get("provenance", {}),
                    "benchmark_parent_manifest_hash": canonical_hash(result.manifest),
                    "benchmark_name": name,
                    "random_seed": seed,
                },
            }
            slug = name.replace("/", "-")
            ledger_path = output / "replays" / f"{slug}.sqlite3"
            candidate = PortfolioReplay(ledger_path).run(**args)
            reference = persist_replay_artifact(
                candidate, ledger_path, output / "replays" / f"{slug}.json"
            )
            if name.startswith("random-"):
                comparison.random_artifacts.append(reference)
            else:
                comparison.benchmark_artifacts[name] = reference
            if candidate.timestamps != result.timestamps:
                raise BacktestError("benchmark dates do not match strategy ledger dates")
            return candidate

        if any(topk >= len(g) for g in groups.values()):
            comparison.unavailable["random_selection"] = (
                "topk spans the eligible universe on at least one date; no nondegenerate matched selection distribution"
            )
        else:
            for i in range(n_simulations):
                signals = random_signals(groups, rng, i, seed)
                candidate = execute(f"random-{i}", signals=signals)
                comparison.random_sharpes.append(candidate.metrics.sharpe_ratio)
                comparison.random_nav.append(candidate.equity_curve)
            finite = [s for s in comparison.random_sharpes if s is not None]
            if len(finite) != n_simulations or result.metrics.sharpe_ratio is None:
                comparison.unavailable["random_percentile"] = (
                    "undefined strategy or random-portfolio Sharpe"
                )
            elif np.std(finite) < 1e-12:
                comparison.unavailable["random_percentile"] = (
                    "all matched portfolio Sharpes are identical"
                )
            else:
                value = result.metrics.sharpe_ratio
                comparison.monte_carlo_percentile = float(
                    100 * np.mean([1 if s < value else 0.5 if s == value else 0 for s in finite])
                )

        risk = inputs.get("risk_policy") or RiskPolicy()
        instruments = {b.instrument.symbol: b.instrument for b in inputs["bars"]}
        for name, summary_name in (
            ("SPY", "SPY"),
            ("BTC", "BTC"),
            ("60/40", "60/40"),
            ("equal-weight", "equal_weight"),
        ):
            try:
                targets, weights = benchmark_targets(
                    name, groups, instruments, risk, inputs["context"].data_snapshot_ids[0]
                )
            except BacktestError as exc:
                comparison.unavailable[name] = str(exc)
                continue
            candidate = execute(name, targets=targets)
            comparison.benchmark_metrics[summary_name] = {
                **candidate.metrics.model_dump(),
                "nav": candidate.equity_curve,
                "dates": candidate.timestamps,
            }
            if weights is not None:
                comparison.benchmark_metrics[summary_name]["weights"] = weights
        if any(topk != len(group) for group in groups.values()):
            comparison.match["equal_weight_breadth"] = (
                "entire date-local universe; random benchmark alone matches topk"
            )
        write_immutable(output / "benchmarks.json", comparison.model_dump_json(indent=2))
        return comparison

    def cost_sensitivity(
        self,
        *,
        inputs: dict,
        output_dir: str | Path,
        equity_bps=(0, 10, 20),
        crypto_bps=(0, 20, 40),
    ) -> list[ReplayResult]:
        if len(equity_bps) != len(crypto_bps):
            raise BacktestError("paired equity/crypto cost scenarios required")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        results = []
        references = []
        base = inputs.get("fill_policy") or FillPolicy()
        for eq, crypto in zip(equity_bps, crypto_bps, strict=True):
            # Reconstruct/validate rather than using unvalidated model_copy values.
            policy = FillPolicy(
                **{**base.model_dump(), "equity_cost_bps": eq, "crypto_cost_bps": crypto}
            )
            name = f"cost-{eq}-{crypto}"
            account = inputs["account"].model_copy(update={"account_id": "backtest:" + name})
            context = inputs["context"].model_copy(
                update={"account_id": account.account_id, "run_id": name}
            )
            result = PortfolioReplay(output / f"{name}.sqlite3").run(
                **{**inputs, "account": account, "context": context, "fill_policy": policy}
            )
            references.append(
                persist_replay_artifact(result, output / f"{name}.sqlite3", output / f"{name}.json")
            )
            results.append(result)
        write_immutable(
            output / "cost-artifacts.json",
            json.dumps([r.model_dump(mode="json") for r in references], sort_keys=True, indent=2),
        )
        return results

    def compare(self, *args, **kwargs):
        raise BacktestError(
            "unmatched legacy return comparisons are unverified; use compare_replay with actual input events"
        )

    def run_monte_carlo(self, *args, **kwargs):
        raise BacktestError(
            "return-only random sampling cannot match execution costs; use compare_replay"
        )


def _callback_match(baseline, opportunities):
    return {
        **{
            key: baseline.manifest[key]
            for key in (
                "bars_hash",
                "metadata_hash",
                "universe_hash",
                "actions_hash",
                "topk",
                "fill_policy",
                "risk_policy",
                "cash_return",
                "decision_times",
                "decision_recipe_hash",
            )
        },
        "timestamps": baseline.timestamps,
        "opportunities_hash": baseline.manifest["provenance"]["opportunities_artifact"]["sha256"],
        "cadence": opportunities.cadence,
        "selection_policy": "callback-selection-control-1",
        "conditioning": "actual baseline decisions and active asset-class slots/budgets; passive quantities from control fills; off-cadence identity retention",
        "named_policy": "risk-constrained named/equal allocations on actual non-None baseline decisions",
    }


def _set_percentile(baseline, comparison):
    values = comparison.random_sharpes
    if (
        len(values) != comparison.simulations
        or any(s is None for s in values)
        or baseline.metrics.sharpe_ratio is None
    ):
        comparison.unavailable["random_percentile"] = (
            "undefined strategy or random-portfolio Sharpe"
        )
    elif np.std(values) < 1e-12:
        comparison.unavailable["random_percentile"] = "all matched portfolio Sharpes are identical"
    else:
        value = baseline.metrics.sharpe_ratio
        comparison.monte_carlo_percentile = float(
            100 * np.mean([1 if s < value else 0.5 if s == value else 0 for s in values])
        )


def _compare_callbacks(result, *, inputs, output_dir, n_simulations, seed, ledger_path):
    if ledger_path is None:
        raise BacktestError(
            "callback comparison requires baseline ledger_path for verified decision economics"
        )
    require_matching_inputs(result, inputs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = persist_replay_artifact(result, ledger_path, output / "source-replay.json")
    source = result.manifest.get("provenance", {}).get("opportunities_artifact")
    if not source or file_hash(source["path"]) != source["sha256"]:
        raise BacktestError("original bound callback opportunity artifact required")
    path = (output / "source-opportunities.json").resolve()
    write_immutable(path, Path(source["path"]).read_bytes().decode("utf-8"))
    opportunity_ref = {"path": str(path), "sha256": file_hash(path)}
    opportunities, original, decisions, identities = _callback_source(
        result, reference, opportunity_ref
    )
    comparison = MatchedComparison(
        strategy_run_id=result.run_id,
        strategy_manifest_hash=canonical_hash(result.manifest),
        strategy_result_hash=canonical_hash(result.model_dump(mode="json")),
        strategy_artifact=reference,
        opportunity_artifact=opportunity_ref,
        strategy_sharpe=result.metrics.sharpe_ratio,
        simulations=n_simulations,
        seed=seed,
        match=_callback_match(result, opportunities),
    )

    def execute(name, index=None):
        provider, recipe, nondegenerate = _callback_policy(
            result,
            opportunities,
            decisions,
            identities,
            name=name,
            seed=seed,
            index=index,
        )
        if name == "random" and not nondegenerate:
            raise BacktestError("all active selection opportunities are degenerate")
        slug = f"random-{index}" if name == "random" else name.replace("/", "-")
        account = original["account"].model_copy(
            update={"account_id": "backtest:benchmark:" + slug}
        )
        context = original["context"].model_copy(
            update={"account_id": account.account_id, "run_id": "benchmark:" + slug}
        )
        args = {
            **original,
            "context": context,
            "account": account,
            "signals": [],
            "targets": [],
            "decision_provider": provider,
            "decision_recipe_hash": recipe,
            "provenance": {
                **original["provenance"],
                "benchmark_parent_manifest_hash": canonical_hash(result.manifest),
                "benchmark_policy": "callback-selection-control-1",
                "benchmark_name": name,
                "random_seed": seed,
                "random_index": index,
            },
        }
        candidate_ledger = output / "replays" / (slug + ".sqlite3")
        candidate = PortfolioReplay(candidate_ledger).run(**args)
        require_matched_replay(result, candidate)
        artifact = persist_replay_artifact(
            candidate, candidate_ledger, output / "replays" / (slug + ".json")
        )
        return candidate, artifact

    for index in range(n_simulations):
        candidate, artifact = execute("random", index)
        comparison.random_artifacts.append(artifact)
        comparison.random_sharpes.append(candidate.metrics.sharpe_ratio)
        comparison.random_nav.append(candidate.equity_curve)
    _set_percentile(result, comparison)
    for name, key in (
        ("SPY", "SPY"),
        ("BTC", "BTC"),
        ("60/40", "60/40"),
        ("equal-weight", "equal_weight"),
    ):
        # Missing actual named components are explicit unavailable evidence.
        try:
            candidate, artifact = execute(name)
        except BacktestError as exc:
            comparison.unavailable[name] = str(exc)
            continue
        comparison.benchmark_artifacts[name] = artifact
        comparison.benchmark_metrics[key] = {
            **candidate.metrics.model_dump(),
            "nav": candidate.equity_curve,
            "dates": candidate.timestamps,
        }
    write_immutable(output / "benchmarks.json", comparison.model_dump_json(indent=2))
    return comparison


def _verify_callback_comparison(baseline, comparison):
    opportunities, _, decisions, identities = _callback_source(
        baseline,
        comparison.strategy_artifact,
        comparison.opportunity_artifact,
    )
    if comparison.signal_artifact or comparison.match != _callback_match(baseline, opportunities):
        raise BacktestError("callback opportunity/cadence match differs from frozen source")
    seen = set()

    def verify(reference, name, index=None):
        candidate = load_replay_artifact(reference)
        require_matched_replay(baseline, candidate)
        provider, recipe, nondegenerate = _callback_policy(
            baseline,
            opportunities,
            decisions,
            identities,
            name=name,
            seed=comparison.seed,
            index=index,
        )
        if name == "random" and not nondegenerate:
            raise BacktestError("callback random selection is degenerate")
        if (
            candidate.manifest.get("decision_recipe_hash") != recipe
            or candidate.manifest.get("decision_times") != baseline.manifest["decision_times"]
            or candidate.manifest["signals_hash"] != canonical_hash([])
        ):
            raise BacktestError("callback comparison recipe/schedule differs")
        identity = (candidate.run_id, candidate.account_id)
        if identity in seen:
            raise BacktestError("duplicate callback comparison identity")
        seen.add(identity)
        ledger = PortfolioLedger(reference.ledger_path, read_only=True)
        try:
            records = recorded_decisions(ledger, candidate)
        finally:
            ledger.close()
        for record in records:
            if provider(record.at, record.snapshot) != record.target:
                raise BacktestError(
                    "callback target differs from seeded opportunity policy and actual holdings"
                )
        return candidate

    candidates = [
        verify(ref, "random", index) for index, ref in enumerate(comparison.random_artifacts)
    ]
    if [c.metrics.sharpe_ratio for c in candidates] != comparison.random_sharpes or [
        c.equity_curve for c in candidates
    ] != comparison.random_nav:
        raise BacktestError("callback random summaries differ from actual durable replays")
    expected = comparison.model_copy(update={"monte_carlo_percentile": None, "unavailable": {}})
    _set_percentile(baseline, expected)
    if (
        expected.monte_carlo_percentile is None
        or expected.monte_carlo_percentile != comparison.monte_carlo_percentile
    ):
        raise BacktestError(
            "callback random percentile is undefined/degenerate or does not recompute"
        )
    for name, key in (
        ("SPY", "SPY"),
        ("BTC", "BTC"),
        ("60/40", "60/40"),
        ("equal-weight", "equal_weight"),
    ):
        if name not in comparison.benchmark_artifacts:
            raise BacktestError("missing durable callback named benchmark " + name)
        candidate = verify(comparison.benchmark_artifacts[name], name)
        summary = {
            **candidate.metrics.model_dump(),
            "nav": candidate.equity_curve,
            "dates": candidate.timestamps,
        }
        if comparison.benchmark_metrics.get(key) != summary:
            raise BacktestError("callback named summary differs from actual durable replay")
