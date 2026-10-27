"""Matched benchmark and cost reruns using the authoritative portfolio ledger."""

from __future__ import annotations

import json
from decimal import Decimal as D
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from src.core.backtester import (
    PortfolioReplay,
    ReplayArtifact,
    ReplayResult,
    canonical_hash,
    file_hash,
    load_replay_artifact,
    persist_replay_artifact,
    recorded_replay_inputs,
    require_matching_inputs,
    write_immutable,
)
from src.domain.portfolio import FillPolicy, PortfolioTarget, RankedSignal, RiskPolicy, TargetWeight
from src.execution.ledger import PortfolioLedger
from src.utils.exceptions import BacktestError


class MatchedComparison(BaseModel):
    schema_version: str = "2.0"
    strategy_run_id: str
    strategy_manifest_hash: str = ""
    strategy_result_hash: str = ""
    signal_artifact: dict = Field(default_factory=dict)
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
