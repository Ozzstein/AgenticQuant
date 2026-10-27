"""Actual operating stores, bounded data refresh and durable notification outbox."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import typer

from src.data.evidence.store import EvidenceStore
from src.operations.notifications import NotificationPlan, dispatch_notifications
from src.operations.refresh import RefreshPlan, RefreshService
from src.operations.store import OperationsStore, canonical

app=typer.Typer(help='Measured operating receipts and recovery.',add_completion=False)
schedule_app=typer.Typer(help='Durable operating and independent research schedules.',add_completion=False)
app.add_typer(schedule_app,name='schedule')


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


@app.command('prepare-cycle')
def prepare_cycle(template:Path=typer.Option(...,exists=True,dir_okay=False),
                  macro_dataset:Path=typer.Option(...,exists=True,dir_okay=False),
                  refresh_run_id:str=typer.Option(...),run_id:str=typer.Option(...),
                  scheduled_for:datetime=typer.Option(...),output:Path=typer.Option(...),
                  contextual_sources:Path|None=typer.Option(None,exists=True,dir_okay=False),
                  income_authorities:Path|None=typer.Option(None,exists=True,dir_okay=False)):
    """Freeze a paper plan from authoritative completed refresh sources and recipes."""
    import json

    from src.core.backtester import write_immutable
    from src.operations.application import prepare_cycle as prepare
    from src.utils.config import get_config
    try:
        plan=prepare(json.loads(template.read_text()),json.loads(macro_dataset.read_text()),get_config(),
            refresh_run_id=refresh_run_id,run_id=run_id,scheduled_for=scheduled_for,
            contextual_sources=json.loads(contextual_sources.read_text()) if contextual_sources else (),
            income_authorities=json.loads(income_authorities.read_text()) if income_authorities else ())
        write_immutable(output,canonical(plan.model_dump(mode='json')).decode())
        emit({'run_id':run_id,'plan_hash':plan.content_hash(),'path':str(output)})
    except Exception as exc:
        fail(exc)


@app.command('cycle')
def cycle(input:Path=typer.Option(...,exists=True,dir_okay=False)):
    """Run the complete configured paper cycle from an immutable source-bound plan."""
    import json

    from src.operations.application import execute_cycle
    from src.utils.config import get_config
    try:
        result=execute_cycle(json.loads(input.read_text()),get_config())
        emit(result)
    except Exception as exc:
        fail(exc)
    if result['status']!='completed':
        raise typer.Exit(2)


@app.command('build-cycle')
def build_cycle(input:Path=typer.Option(...,exists=True,dir_okay=False),output:Path=typer.Option(...)):
    """Resolve approved macro/contextual task sources from completed refresh, then freeze a plan."""
    import json

    from src.core.backtester import write_immutable
    from src.operations.application import build_cycle as build
    from src.utils.config import get_config
    try:
        plan=build(json.loads(input.read_text()),get_config())
        write_immutable(output,canonical(plan.model_dump(mode='json')).decode())
        emit({'run_id':plan.inputs.context.run_id,'plan_hash':plan.content_hash(),'path':str(output)})
    except Exception as exc:
        fail(exc)


@app.command('dispatch-pending')
def dispatch_pending(input:Path=typer.Option(...,exists=True,dir_okay=False)):
    """Dispatch only actually eligible prepared paper intents at the measured clock."""
    import json

    from src.operations.application import execute_cycle
    from src.utils.config import get_config
    try:
        emit(execute_cycle(json.loads(input.read_text()),get_config(),dispatch_only=True))
    except Exception as exc:
        fail(exc)


def _supervisor(lane, agent_authorities=None):
    from src.operations.application import schedule_supervisor
    from src.research.authorities import AgentAuthorityPaths
    from src.utils.config import get_config
    authorities = AgentAuthorityPaths.model_validate_json(agent_authorities.read_text()).open() if agent_authorities else None
    return schedule_supervisor(get_config(),lane=lane,authorities=authorities)


@schedule_app.command('register')
def register_schedule(input:Path=typer.Option(...,exists=True,dir_okay=False)):
    """Register an immutable explicit schedule without resetting prior progress."""
    from src.operations.scheduler import ScheduleSpec
    try:
        spec=ScheduleSpec.model_validate_json(input.read_text())
        _supervisor(spec.execution_lane).register(spec)
        emit({'schedule_id':spec.schedule_id,'spec_hash':spec.content_hash(),'status':'registered'})
    except Exception as exc:
        fail(exc)


@schedule_app.command('tick')
def tick_schedule(lane:Literal['operating','research']=typer.Option('operating'),
                  agent_authorities:Path|None=typer.Option(None,exists=True,dir_okay=False)):
    """Claim due frozen jobs in one lane at the actual UTC clock."""
    try:
        values=_supervisor(lane,agent_authorities).tick(datetime.now(timezone.utc),lane=lane)
        emit(values)
    except Exception as exc:
        fail(exc)
    if any(r['status'] in ('failed','uncertain','recovery_required','missed') for r in values):
        raise typer.Exit(2)


@schedule_app.command('serve')
def serve_schedule(lane:Literal['operating','research']=typer.Option('operating'),
                   agent_authorities:Path|None=typer.Option(None,exists=True,dir_okay=False)):
    """Poll one durable lane; run research in a separate worker process."""
    from threading import Event

    from src.utils.config import get_config
    try:
        supervisor=_supervisor(lane,agent_authorities)
        stop=Event()
        while not stop.is_set():
            emit(supervisor.tick(datetime.now(timezone.utc),lane=lane))
            stop.wait(get_config().operations.poll_seconds)
    except KeyboardInterrupt:
        return
    except Exception as exc:
        fail(exc)


@schedule_app.command('resume')
def resume_schedule(identity:str=typer.Option(...),input_hash:str=typer.Option(...)):
    """Authorize recovery of the exact interrupted immutable scheduled input."""
    try:
        _supervisor('operating').resume(identity,expected_input_hash=input_hash)
        emit({'schedule_id':identity,'input_hash':input_hash,'status':'recovery_requested'})
    except Exception as exc:
        fail(exc)


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
