#!/usr/bin/env python3
"""Evidence-grounded agent desk from explicit identities, clocks and archives."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer

from src.agents.contracts import PortfolioReviewContext, hmm_context_from_result
from src.core.macro_regime import CausalRegimeResult
from src.data.evidence.service import EvidenceService
from src.data.evidence.store import EvidenceStore
from src.decision.evidence import load_desk_sources
from src.domain.common import Instrument, RunContext, VersionedContract
from src.domain.decisions import AgentAction, AgentDecision
from src.execution.ledger import PortfolioLedger

app = typer.Typer(help="Full agent desk over archived evidence and authoritative quantitative context.", add_completion=False)


class DeskAnalysisInput(VersionedContract):
    instrument: Instrument
    context: RunContext
    hmm_result: CausalRegimeResult | None = None
    correlations_json: str | None = None
    deterministic_vetoes: tuple[str, ...] = ()
    qlib_context: str = ""


def analyze_input(request: DeskAnalysisInput, *, evidence_path: Path, ledger_path: Path | None = None,
                  config=None, observation_mode="cache") -> tuple[AgentDecision, dict]:
    from src.agents.graph import analyze_instrument

    store = EvidenceStore(evidence_path)
    sources, coverage = load_desk_sources(EvidenceService(store), request.instrument, request.context)
    if request.hmm_result is not None and request.hmm_result.context.as_of > request.context.as_of:
        raise ValueError("authoritative HMM result exceeds the decision cutoff")
    hmm = hmm_context_from_result(request.hmm_result) if request.hmm_result else None
    portfolio = None
    if ledger_path is not None:
        if request.context.account_id is None:
            raise ValueError("ledger analysis requires an account identity")
        ledger = PortfolioLedger(ledger_path, read_only=True)
        try:
            ledger.reconcile(request.context.account_id)
            snapshots = [s for s in ledger.history(request.context.account_id) if s.as_of <= request.context.as_of]
        finally:
            ledger.close()
        if snapshots and request.correlations_json:
            snapshot = snapshots[-1]
            portfolio = PortfolioReviewContext(as_of=snapshot.as_of, snapshot_id=snapshot.content_hash(),
                portfolio_json=snapshot.model_dump_json(), correlation_json=request.correlations_json,
                deterministic_vetoes=request.deterministic_vetoes)
    delivered = {role: {kind: value.model_dump(mode="json") for kind, value in kinds.items()}
                 for role, kinds in coverage.items()}
    decision = analyze_instrument(request.instrument, request.context, sources=sources,
        source_coverage=delivered,
        hmm_context=hmm, portfolio_context=portfolio, qlib_context=request.qlib_context,
        config=config, observation_mode=observation_mode, evidence_store=store)
    return decision, delivered


def _write(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("analysis output is immutable; use a new run/output path")
        return
    with path.open("x") as output:
        output.write(payload)


def _execute(input: Path, evidence: Path, ledger: Path | None, output: Path,
             mode: Literal["cache", "refresh", "replay"], *, transcript: bool = False):
    try:
        request = DeskAnalysisInput.model_validate_json(input.read_text())
        decision, coverage = analyze_input(request, evidence_path=evidence, ledger_path=ledger, observation_mode=mode)
        artifact = {"schema_version": "2.0", "input_hash": request.content_hash(),
                    "decision": decision.model_dump(mode="json"), "coverage": coverage}
        _write(output, artifact)
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"Analysis failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    if transcript:
        for entry in decision.debate_transcript:
            typer.echo(entry)
    typer.echo(f"{decision.instrument.symbol}: {decision.action.value}\n{decision.reasoning}\n{output}")
    if decision.action in (AgentAction.ABSTAIN, AgentAction.VETO):
        raise typer.Exit(2)


@app.command()
def analyze(input: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
            evidence: Path = typer.Option(..., "--evidence", exists=True, dir_okay=False),
            output: Path = typer.Option(..., "--output", dir_okay=False),
            ledger: Path | None = typer.Option(None, "--ledger", exists=True, dir_okay=False),
            mode: Literal["cache", "refresh", "replay"] = typer.Option("cache", "--mode")):
    """Analyze exact historical context; ABSTAIN/VETO remain distinct and exit 2."""
    _execute(input, evidence, ledger, output, mode)


@app.command()
def debate(input: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
           evidence: Path = typer.Option(..., "--evidence", exists=True, dir_okay=False),
           output: Path = typer.Option(..., "--output", dir_okay=False),
           ledger: Path | None = typer.Option(None, "--ledger", exists=True, dir_okay=False),
           mode: Literal["cache", "refresh", "replay"] = typer.Option("cache", "--mode")):
    """Run the configured full desk and print its recorded challenge rounds."""
    _execute(input, evidence, ledger, output, mode, transcript=True)


@app.command()
def batch(inputs: list[Path] = typer.Argument(..., exists=True, dir_okay=False),
          evidence: Path = typer.Option(..., "--evidence", exists=True, dir_okay=False),
          output_dir: Path = typer.Option(..., "--output-dir"),
          ledger: Path | None = typer.Option(None, "--ledger", exists=True, dir_okay=False)):
    """Analyze explicit per-instrument requests; any failure blocks batch success."""
    exit_code = 0
    for path in inputs:
        try:
            request = DeskAnalysisInput.model_validate_json(path.read_text())
            output = output_dir / (request.content_hash()+".json")
            _execute(path, evidence, ledger, output, "cache")
        except typer.Exit as exc:
            exit_code = max(exit_code, exc.exit_code)
        except (ValueError, OSError) as exc:
            typer.echo(f"{path}: {exc}", err=True)
            exit_code = max(exit_code, 1)
    if exit_code:
        raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
