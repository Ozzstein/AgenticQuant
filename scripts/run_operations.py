"""Actual operating stores, bounded data refresh and durable notification outbox."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from src.data.evidence.store import EvidenceStore
from src.operations.notifications import NotificationPlan, dispatch_notifications
from src.operations.refresh import RefreshPlan, RefreshService
from src.operations.store import OperationsStore, canonical

app=typer.Typer(help='Measured operating receipts and recovery.',add_completion=False)


def emit(value):
    typer.echo(canonical(value).decode())


def fail(exc):
    typer.echo('Operating action failed: '+str(exc),err=True)
    raise typer.Exit(1) from exc


@app.command()
def refresh(input: Path=typer.Option(...,exists=True,dir_okay=False),
            store: Path=typer.Option(...),evidence: Path=typer.Option(...)):
    """Execute explicit read-only fetches with measured receipt clocks and durable caps."""
    try:
        plan=RefreshPlan.model_validate_json(input.read_text())
        emit(RefreshService(OperationsStore(store),EvidenceStore(evidence)).run(plan))
    except Exception as exc:
        fail(exc)


@app.command()
def status():
    """Read actual configured stores without creating empty accounts or data."""
    from src.operations.reporting import system_view
    from src.utils.config import get_config
    emit(system_view(get_config()))


@app.command('notify')
def notify(plan: Path=typer.Option(...,exists=True,dir_okay=False),
           store: Path=typer.Option(...,exists=True,dir_okay=False)):
    """Send pending durable alerts to configured destinations, with no blind retries."""
    try:
        emit(dispatch_notifications(OperationsStore(store),NotificationPlan.model_validate_json(plan.read_text())))
    except Exception as exc:
        fail(exc)


@app.command('resolve-delivery')
def resolve_delivery(store: Path=typer.Option(...,exists=True,dir_okay=False),
                     alert: str=typer.Option(...),channel: str=typer.Option(...),
                     attempt: int=typer.Option(...,min=1),
                     outcome: Literal['confirmed','abandoned','retry_authorized']=typer.Option(...),
                     evidence: Path=typer.Option(...,exists=True,dir_okay=False)):
    """Archive explicit operator evidence resolving one uncertain delivery attempt."""
    import json
    try:
        proof=json.loads(evidence.read_text())
        if not isinstance(proof,dict) or not proof:
            raise ValueError('nonempty operator review evidence object required')
        emit(OperationsStore(store).resolve_delivery(alert,channel,expected_attempt=attempt,
            outcome=outcome,evidence=proof))
    except Exception as exc:
        fail(exc)


if __name__=='__main__':
    app()
