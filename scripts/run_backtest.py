#!/usr/bin/env python3
"""Replay, train/evaluate, compare and validate actual versioned input bundles.

Input JSON contains explicit contracts; this CLI never downloads a convenient
replacement universe, manufactures fills from labels, or substitutes success.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import typer

from src.core.backtester import PortfolioReplay, ReplayArtifact, ReplayResult
from src.core.benchmark import BenchmarkRunner, MatchedComparison
from src.core.validation import ValidationSuite
from src.core.walk_forward import WalkForwardRunner, reconstruct_oos_inputs
from src.domain.common import RunContext
from src.domain.portfolio import (
    AccountSpec,
    CashDividend,
    CorporateAction,
    DividendDistribution,
    ExecutionBar,
    FillPolicy,
    InstrumentRisk,
    PortfolioTarget,
    RankedSignal,
    RiskPolicy,
    UniverseEligibility,
)
from src.utils.exceptions import BacktestError

app = typer.Typer(
    help="Chronological ledger backtests from explicit versioned data inputs.", add_completion=False
)


def load_inputs(path: Path) -> tuple[dict, dict]:
    raw = json.loads(path.read_text())
    required = ("context", "account", "bars", "metadata", "universes")
    missing = [k for k in required if k not in raw]
    if missing:
        raise BacktestError("input bundle missing: " + ", ".join(missing))
    args = {
        "context": RunContext.model_validate(raw["context"]),
        "account": AccountSpec.model_validate(raw["account"]),
        "bars": [ExecutionBar.model_validate(b) for b in raw["bars"]],
        "metadata": [InstrumentRisk.model_validate(m) for m in raw["metadata"]],
        "universes": [UniverseEligibility.model_validate(u) for u in raw["universes"]],
        "signals": [RankedSignal.model_validate(s) for s in raw.get("signals", [])],
        "targets": [PortfolioTarget.model_validate(t) for t in raw.get("targets", [])],
        "actions": [CorporateAction.model_validate(a) for a in raw.get("actions", [])],
        "cash_dividends": [CashDividend.model_validate(t) for t in raw.get("cash_dividends", [])],
        "dividend_distributions": [
            DividendDistribution.model_validate(t) for t in raw.get("dividend_distributions", [])
        ],
        "fill_policy": FillPolicy.model_validate(raw.get("fill_policy", {})),
        "risk_policy": RiskPolicy.model_validate(raw.get("risk_policy", {})),
        "topk": raw.get("topk", 30),
        "samples_per_year": raw.get("samples_per_year", 252),
        "valuation_calendar": raw.get("valuation_calendar", "declared-equity-sessions"),
        "valuation_times": [pd.Timestamp(t).to_pydatetime() for t in raw["valuation_times"]]
        if raw.get("valuation_times")
        else None,
        "provenance": raw.get("provenance", {}),
    }
    return args, raw


def _run_pipeline(input_path: Path, ledger: Path, artifact_dir: Path) -> ReplayResult:
    args, raw = load_inputs(input_path)
    if "walk_forward" not in raw:
        return PortfolioReplay(ledger).run(**args)
    wf = raw["walk_forward"]
    rows = wf.get("rows", [])
    if not rows:
        raise BacktestError("walk-forward contains no feature observations")
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(r["decision_at"]), r["instrument_id"]) for r in rows],
        names=["decision_at", "instrument_id"],
    )
    features = pd.DataFrame([r["features"] for r in rows], index=index)
    labels = pd.Series([r["label"] for r in rows], index=index)
    ends = pd.Series([pd.Timestamp(r["label_end_at"]) for r in rows], index=index)
    available = pd.Series([pd.Timestamp(r["feature_available_at"]) for r in rows], index=index)
    if args["signals"] or args["targets"]:
        raise BacktestError(
            "walk-forward generates OOS signals; precomputed signal/target stream is incompatible"
        )
    selected = {
        k: args[k]
        for k in (
            "context",
            "account",
            "bars",
            "metadata",
            "universes",
            "actions",
            "cash_dividends",
            "dividend_distributions",
            "topk",
            "fill_policy",
            "risk_policy",
            "samples_per_year",
            "valuation_calendar",
            "valuation_times",
            "provenance",
        )
    }
    return WalkForwardRunner().run(
        features,
        labels,
        label_end_times=ends,
        feature_available_at=available,
        feature_spec_hash=wf.get("feature_spec_hash"),
        data_snapshot_hash=wf.get("data_snapshot_hash"),
        model_name=wf.get("model_name", "LightGBM"),
        walk_months=wf.get("walk_months", 6),
        embargo_days=wf.get("embargo_days", 5),
        validation_months=wf.get("validation_months", 1),
        initial_train_months=wf.get("initial_train_months", 12),
        universe_version=wf["universe_version"],
        ledger_path=ledger,
        artifact_dir=artifact_dir,
        **selected,
    )


def _failure(output: Path, exc: Exception):
    output.parent.mkdir(parents=True, exist_ok=True)
    failure = output.with_suffix(output.suffix + ".failure.json")
    failure.write_text(
        json.dumps(
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, indent=2
        )
    )
    typer.echo(f"FAILED: {exc}\nFailure record: {failure}", err=True)
    raise typer.Exit(1) from exc


@app.command()
def run(
    input: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    ledger: Path = typer.Option(..., "--ledger", dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
):
    """Run actual predictions/targets or purged walk-forward feature rows."""
    try:
        result = _run_pipeline(input, ledger, output.parent / (output.stem + "-artifacts"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.model_dump_json(indent=2))
    except Exception as exc:
        _failure(output, exc)
    typer.echo(
        f"{result.status.upper()}: {result.run_id}; return={result.metrics.total_return:.6f}; fills={result.metrics.total_trades}; result={output}"
    )
    if result.status != "complete":
        raise typer.Exit(2)


@app.command()
def compare(
    input: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    result: Path = typer.Option(..., "--result", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output"),
    simulations: int = typer.Option(1000, min=1),
):
    """Run matched benchmarks and cost reruns, retaining every random NAV path."""
    try:
        args, raw = load_inputs(input)
        baseline = ReplayResult.model_validate_json(result.read_text())
        if "walk_forward" in raw:
            args = reconstruct_oos_inputs(baseline, args)
        runner = BenchmarkRunner()
        comparison = runner.compare_replay(
            baseline, inputs=args, output_dir=output, n_simulations=simulations
        )
        runner.cost_sensitivity(inputs=args, output_dir=output / "costs")
    except Exception as exc:
        _failure(output / "comparison.json", exc)
    typer.echo(f"Comparison saved: {output / 'benchmarks.json'}")
    if comparison.unavailable:
        typer.echo(
            json.dumps({"status": "incomplete", "unavailable": comparison.unavailable}), err=True
        )
        raise typer.Exit(2)


@app.command()
def validate(
    result: Path = typer.Option(..., "--result", exists=True, dir_okay=False),
    ledger: Path = typer.Option(..., "--ledger", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    costs: Path | None = typer.Option(None, "--costs", exists=True, file_okay=False),
    benchmarks: Path | None = typer.Option(None, "--benchmarks", exists=True, dir_okay=False),
):
    """Consolidated promotion evidence checks; rejected evidence exits nonzero."""
    try:
        data = ReplayResult.model_validate_json(result.read_text())
        cost_results = (
            [
                ReplayResult.model_validate_json(p.read_text())
                for p in sorted(costs.glob("cost-*.json"))
                if p.name != "cost-artifacts.json"
            ]
            if costs
            else []
        )
        cost_artifacts = (
            [
                ReplayArtifact.model_validate(item)
                for item in json.loads((costs / "cost-artifacts.json").read_text())
            ]
            if costs and (costs / "cost-artifacts.json").exists()
            else None
        )
        benchmark = (
            MatchedComparison.model_validate_json(benchmarks.read_text()) if benchmarks else None
        )
        report = ValidationSuite().validate(
            data,
            ledger_path=ledger,
            cost_results=cost_results,
            cost_artifacts=cost_artifacts,
            benchmark=benchmark,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2))
    except Exception as exc:
        _failure(output, exc)
    typer.echo(f"{report.overall}: {report.summary}; report={output}")
    if report.overall != "RELIABLE":
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
