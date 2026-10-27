"""Matched benchmark and cost reruns using the authoritative portfolio ledger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from src.core.backtester import PortfolioReplay, ReplayResult, canonical_hash
from src.domain.portfolio import FillPolicy, PortfolioTarget, TargetWeight
from src.utils.exceptions import BacktestError


class MatchedComparison(BaseModel):
    schema_version: str = "2.0"
    strategy_run_id: str
    strategy_sharpe: float | None
    benchmark_metrics: dict[str, dict] = Field(default_factory=dict)
    unavailable: dict[str, str] = Field(default_factory=dict)
    random_sharpes: list[float | None] = Field(default_factory=list)
    random_nav: list[list[float]] = Field(default_factory=list)
    monte_carlo_percentile: float | None = None
    simulations: int
    seed: int
    match: dict


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
        expected = {
            "bars_hash": canonical_hash(
                [
                    b.model_dump(mode="json")
                    for b in sorted(
                        inputs["bars"], key=lambda b: (b.opened_at, b.instrument.instrument_id)
                    )
                ]
            ),
            "metadata_hash": canonical_hash(
                [m.model_dump(mode="json") for m in inputs["metadata"]]
            ),
            "signals_hash": canonical_hash(
                [
                    s.model_dump(mode="json")
                    for s in sorted(inputs["signals"], key=lambda s: s.signal_id)
                ]
            ),
            "universe_hash": canonical_hash(
                [u.model_dump(mode="json") for u in inputs["universes"]]
            ),
        }
        if any(result.manifest.get(key) != value for key, value in expected.items()):
            raise BacktestError("benchmark inputs differ from the strategy's frozen data/signals")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        groups = {}
        for s in inputs["signals"]:
            groups.setdefault(s.decision_at, []).append(s)
        topk = inputs.get("topk", 30)
        rng = np.random.default_rng(seed)
        comparison = MatchedComparison(
            strategy_run_id=result.run_id,
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
            }
            candidate = PortfolioReplay(":memory:").run(**args)
            if candidate.timestamps != result.timestamps:
                raise BacktestError("benchmark dates do not match strategy ledger dates")
            return candidate

        if any(topk >= len(g) for g in groups.values()):
            comparison.unavailable["random_selection"] = (
                "topk spans the eligible universe on at least one date; no nondegenerate matched selection distribution"
            )
        else:
            for i in range(n_simulations):
                signals = []
                for group in groups.values():
                    scores = rng.permutation(len(group))
                    signals.extend(
                        s.model_copy(
                            update={
                                "signal_id": f"random-{i}:" + s.signal_id,
                                "score": float(score),
                                "model_version": f"random-seed-{seed}",
                                "strategy_id": "random",
                            }
                        )
                        for s, score in zip(group, scores, strict=True)
                    )
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

        from decimal import Decimal as D

        from src.domain.portfolio import RiskPolicy

        risk = inputs.get("risk_policy") or RiskPolicy()
        instruments = {b.instrument.symbol: b.instrument for b in inputs["bars"]}
        definitions = {
            "SPY": [("SPY", D(1))],
            "BTC": [("BTC/USD", D(1))],
            "60/40": [("SPY", D("0.6")), ("AGG", D("0.4"))],
        }
        for name, components in definitions.items():
            if any(symbol not in instruments for symbol, _ in components):
                comparison.unavailable[name] = (
                    "actual benchmark price bars and risk metadata not supplied"
                )
                continue
            scale = min(
                [1 - risk.min_cash]
                + [
                    (
                        risk.crypto_weight
                        if instruments[s].asset_class == "crypto"
                        else risk.equity_weight
                    )
                    / weight
                    for s, weight in components
                ]
            )
            weights = tuple(
                TargetWeight(instrument=instruments[s], weight=w * scale) for s, w in components
            )
            targets = [
                PortfolioTarget(
                    universe_version=groups[at][0].universe_version,
                    target_id=f"{name}:{at.isoformat()}",
                    decision_at=at,
                    strategy_id=name,
                    artifact_version="fixed-constrained-benchmark-v1",
                    data_snapshot_id=inputs["context"].data_snapshot_ids[0],
                    weights=weights,
                )
                for at in groups
            ]
            candidate = execute(name, targets=targets)
            comparison.benchmark_metrics[name] = {
                **candidate.metrics.model_dump(),
                "weights": {s: str(w * scale) for s, w in components},
                "nav": candidate.equity_curve,
                "dates": candidate.timestamps,
            }
        equal_targets = []
        for at, group in groups.items():
            if topk != len(group):
                # Equal weight whole-universe baseline deliberately has a different
                # breadth; disclose it instead of claiming matched portfolio size.
                comparison.match["equal_weight_breadth"] = (
                    "entire date-local universe; random benchmark alone matches topk"
                )
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
            equal_targets.append(
                PortfolioTarget(
                    universe_version=group[0].universe_version,
                    target_id="equal:" + at.isoformat(),
                    decision_at=at,
                    strategy_id="equal-weight",
                    artifact_version="equal-weight-v1",
                    data_snapshot_id=group[0].data_snapshot_id,
                    weights=weights,
                )
            )
        equal = execute("equal-weight", targets=equal_targets)
        comparison.benchmark_metrics["equal_weight"] = {
            **equal.metrics.model_dump(),
            "nav": equal.equity_curve,
            "dates": equal.timestamps,
        }
        (output / "benchmarks.json").write_text(comparison.model_dump_json(indent=2))
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
            (output / f"{name}.json").write_text(result.model_dump_json(indent=2))
            results.append(result)
        return results

    def compare(self, *args, **kwargs):
        raise BacktestError(
            "unmatched legacy return comparisons are unverified; use compare_replay with actual input events"
        )

    def run_monte_carlo(self, *args, **kwargs):
        raise BacktestError(
            "return-only random sampling cannot match execution costs; use compare_replay"
        )
