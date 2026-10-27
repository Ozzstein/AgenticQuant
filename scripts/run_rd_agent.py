#!/usr/bin/env python3
"""Native RD-Agent and registered research lifecycle. No simulated performance."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import typer

from src.data.evidence.store import EvidenceStore
from src.domain.market import MarketSnapshot
from src.research.evaluator import ExperimentPlan
from src.research.native import NativeRequest, NativeSupervisor
from src.research.registry import Holdout, PromotionPolicy, ResearchRegistry
from src.research.runner import ResearchRunner

app=typer.Typer(help='Native RD-Agent plus quarantined, preregistered ledger experiments.',add_completion=False)


def fail(exc):
    typer.echo(f'Research failed: {exc}',err=True)
    raise typer.Exit(1) from exc


@app.command()
def register(plan: Path=typer.Option(...,exists=True,dir_okay=False),
             registry: Path=typer.Option(...),evidence: Path=typer.Option(...),
             code: Path=typer.Option(...,exists=True,dir_okay=False),
             environment: Path=typer.Option(...,exists=True,dir_okay=False),
             candidate: str=typer.Option(...),trial: str=typer.Option(...),family: str=typer.Option(...),
             kind: Literal['factor','model','semantic_question']=typer.Option(...),
             start: str=typer.Option(...),end: str=typer.Option(...),
             holdout: str | None=typer.Option(None),policy: Path | None=typer.Option(None,exists=True,dir_okay=False)):
    """Freeze candidate, inputs, splits and policy before any trial executes."""
    try:
        service=ResearchRunner(ResearchRegistry(registry),EvidenceStore(evidence))
        record=service.register(plan=ExperimentPlan.model_validate_json(plan.read_text()),candidate_id=candidate,
            trial_id=trial,family_id=family,kind=kind,origin='custom',code=code.read_bytes(),environment=environment.read_bytes(),
            at=datetime.now(timezone.utc),evaluation_start=datetime.fromisoformat(start),evaluation_end=datetime.fromisoformat(end),
            holdout_id=holdout,policy=PromotionPolicy.model_validate_json(policy.read_text()) if policy else None)
        typer.echo(record.model_dump_json(indent=2))
    except (ValueError,OSError) as exc:
        fail(exc)


@app.command('reserve-holdout')
def reserve_holdout(registry: Path=typer.Option(...),snapshot: Path=typer.Option(...,exists=True,dir_okay=False),
                    identity: str=typer.Option(...),start: str=typer.Option(...),end: str=typer.Option(...)):
    """Reserve an exact dataset/interval; later development trials cannot touch it."""
    try:
        service=ResearchRegistry(registry)
        data=MarketSnapshot.model_validate_json(snapshot.read_text())
        record=Holdout(holdout_id=identity,dataset_hash=service.archive_contract(data),starts_at=datetime.fromisoformat(start),
                       ends_at=datetime.fromisoformat(end),registered_at=datetime.now(timezone.utc))
        service.register_holdout(record)
        typer.echo(record.model_dump_json(indent=2))
    except (ValueError,OSError) as exc:
        fail(exc)


@app.command()
def run(trial: str=typer.Option(...),registry: Path=typer.Option(...,exists=True),evidence: Path=typer.Option(...,exists=True),
        output: Path=typer.Option(...),simulations: int=typer.Option(1000,min=1000)):
    """Train, replay, run matched cost/benchmark portfolios and record an outcome."""
    try:
        outcome=ResearchRunner(ResearchRegistry(registry),EvidenceStore(evidence)).run(trial,output_dir=output,simulations=simulations)
        typer.echo(outcome.model_dump_json(indent=2))
    except Exception as exc:
        fail(exc)
    if not outcome.validation_reliable:
        raise typer.Exit(2)


@app.command()
def promote(registry: Path=typer.Option(...,exists=True),trial: str=typer.Option(...),slot: str=typer.Option(...),
            policy: Path=typer.Option(...,exists=True),generation: int=typer.Option(...,min=0)):
    """Atomically activate a fully verified holdout outcome under its frozen policy."""
    try:
        result=ResearchRegistry(registry).activate(slot,trial,PromotionPolicy.model_validate_json(policy.read_text()),
                                                   expected_generation=generation,at=datetime.now(timezone.utc))
        typer.echo(result.model_dump_json(indent=2))
    except (ValueError,OSError) as exc:
        fail(exc)


@app.command()
def rollback(registry: Path=typer.Option(...,exists=True),slot: str=typer.Option(...),
             to_generation: int=typer.Option(...,min=1),generation: int=typer.Option(...,min=1)):
    """Restore an earlier verified activation as a new immutable generation."""
    try:
        result=ResearchRegistry(registry).rollback(slot,to_generation,expected_generation=generation,at=datetime.now(timezone.utc))
        typer.echo(result.model_dump_json(indent=2))
    except (ValueError,OSError) as exc:
        fail(exc)


@app.command('library-status')
def library_status(registry: Path=typer.Option(...,exists=True)):
    typer.echo(json.dumps(ResearchRegistry(registry).status(),indent=2))


def _native_command(name):
    def execute(request: Path=typer.Option(...,exists=True,dir_okay=False),
                gateway_token_file: Path=typer.Option(...,exists=True,dir_okay=False),workdir: Path=typer.Option(...)):
        try:
            value=NativeRequest.model_validate_json(request.read_text())
            if value.command!=name:
                raise ValueError('native command differs from request manifest')
            result=NativeSupervisor(workdir).run(value,gateway_token_path=gateway_token_file)
            typer.echo(json.dumps(result,indent=2))
        except Exception as exc:
            fail(exc)
        if result['status']!='discovery_complete':
            raise typer.Exit(2)
    return execute


for _command in ('fin_factor','fin_model','fin_quant','fin_factor_report','general_model','ui','health_check'):
    app.command(_command)(_native_command(_command))


if __name__=='__main__':
    app()
