"""Grounded proposals and development memos; every candidate stays quarantined."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
from pydantic import Field

from src.core.backtester import write_immutable
from src.data.evidence.store import EvidenceStore
from src.domain.common import NonEmptyString, RunContext, Sha256, VersionedContract
from src.domain.decisions import DecisionSource
from src.research.copilot import CopilotLimits, ResearchCopilot
from src.research.registry import ResearchRegistry
from src.utils.config import get_config

app = typer.Typer(help='Budgeted source-grounded research proposals and immutable development memos.',add_completion=False)


class ProposalInput(VersionedContract):
    context: RunContext
    kind: Literal['factor','model','strategy','regime','semantic_question']
    description: NonEmptyString
    sources: tuple[DecisionSource,...] = ()
    paper_evidence_ids: tuple[NonEmptyString,...] = ()
    n: int = Field(default=1,ge=1,le=20)
    tested: tuple[str,...] = ()
    parent_ids: tuple[str,...] = ()
    allowed_fields: tuple[str,...] = ('open','high','low','close','volume')
    selected_model: Literal['Linear','LightGBM','CatBoost','XGBoost','Ensemble'] = 'Linear'
    prior_memo_hash: Sha256 | None = None
    limits: CopilotLimits = Field(default_factory=CopilotLimits)


class MemoInput(VersionedContract):
    context: RunContext
    sources: tuple[DecisionSource,...] = ()
    development_trial_ids: tuple[str,...] = ()
    prior_memo_hash: Sha256 | None = None
    limits: CopilotLimits = Field(default_factory=CopilotLimits)


def _service(registry,evidence,code,environment,limits):
    store = ResearchRegistry(registry)
    payloads = []
    for path in (code,environment):
        if path.stat().st_size > 16*1024*1024:
            raise ValueError('execution code/environment artifact exceeds bounded input size')
        payloads.append(path.read_bytes())
    return ResearchCopilot(get_config(),registry=store,evidence_store=EvidenceStore(evidence),
        code_hash=store.archive(payloads[0]),environment_hash=store.archive(payloads[1]),limits=limits)


def _finish(outcome,output):
    write_immutable(output,outcome.model_dump_json(indent=2))
    typer.echo(f'{outcome.status}: {outcome.reason}\n{output}')
    if outcome.status != 'available':
        raise typer.Exit(2)


@app.command()
def propose(request: Path=typer.Option(...,exists=True,dir_okay=False),
            registry: Path=typer.Option(...),evidence: Path=typer.Option(...,exists=True,dir_okay=False),
            code: Path=typer.Option(...,exists=True,dir_okay=False),environment: Path=typer.Option(...,exists=True,dir_okay=False),
            output: Path=typer.Option(...)):
    """Read exact sources or every archived paper chunk, then quarantine proposals."""
    try:
        value=ProposalInput.model_validate_json(request.read_text())
        service=_service(registry,evidence,code,environment,value.limits)
        options=dict(context=value.context,n=value.n,tested=value.tested,parent_ids=value.parent_ids,
                     allowed_fields=value.allowed_fields,selected_model=value.selected_model,prior_memo_hash=value.prior_memo_hash)
        if value.paper_evidence_ids:
            if value.sources:
                raise ValueError('paper workflow reads complete stored documents; do not substitute supplied passages')
            outcome=service.propose_from_papers(value.paper_evidence_ids,kind=value.kind,
                                               description=value.description,**options)
        else:
            outcome=service.propose(value.kind,value.description,sources=value.sources,**options)
        _finish(outcome,output)
    except (ValueError,OSError) as exc:
        typer.echo(f'Research request failed: {type(exc).__name__}: {exc}',err=True)
        raise typer.Exit(1) from exc


@app.command()
def memo(request: Path=typer.Option(...,exists=True,dir_okay=False),
         registry: Path=typer.Option(...,exists=True,dir_okay=False),evidence: Path=typer.Option(...,exists=True,dir_okay=False),
         code: Path=typer.Option(...,exists=True,dir_okay=False),environment: Path=typer.Option(...,exists=True,dir_okay=False),
         output: Path=typer.Option(...)):
    """Use only verified recorded development outcomes; holdout reads are rejected."""
    try:
        value=MemoInput.model_validate_json(request.read_text())
        service=_service(registry,evidence,code,environment,value.limits)
        outcome=service.write_memo(context=value.context,sources=value.sources,
            development_trial_ids=value.development_trial_ids,prior_memo_hash=value.prior_memo_hash)
        _finish(outcome,output)
    except (ValueError,OSError) as exc:
        typer.echo(f'Research memo failed: {type(exc).__name__}: {exc}',err=True)
        raise typer.Exit(1) from exc


if __name__ == '__main__':
    app()
