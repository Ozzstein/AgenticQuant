"""Read-only operational views of authoritative stores; missing stays unknown."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.execution.ledger import PortfolioLedger
from src.operations.store import OperationsStore
from src.research.monitoring import research_snapshot
from src.research.registry import ResearchRegistry


@contextmanager
def read_database(path: str | Path):
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    db = sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=10)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def portfolio_view(path: str | Path, account_id: str) -> dict:
    if not Path(path).is_file():
        return {'status':'missing','account_id':account_id,'snapshot':None,'history':[],'fills':[],'pending':[]}
    ledger = PortfolioLedger(path,read_only=True)
    try:
        reconciliation = ledger.reconcile(account_id)
        history = ledger.history(account_id)
        return {'status':'available' if history else 'no_valuations','account_id':account_id,
                'base_currency':ledger.spec(account_id).base_currency,
                'snapshot':history[-1].model_dump(mode='json') if history else None,
                'history':[s.model_dump(mode='json') for s in history],
                'fills':[f.model_dump(mode='json') for f in ledger.fills(account_id)],
                'pending':[{key:value.model_dump(mode='json') if hasattr(value,'model_dump') else
                            str(value) if isinstance(value,Decimal) else value
                            for key,value in order.items()} for order in ledger.pending(account_id)],
                'reconciliation':reconciliation}
    finally:
        ledger.close()


def budget_view(path: str | Path, *, day: str | None = None) -> dict:
    day = day or datetime.now(timezone.utc).date().isoformat()
    if not Path(path).is_file():
        return {'status':'missing','day':day,'actual_cost_usd':None,'charged_cost_usd':None,'calls':[]}
    with read_database(path) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='calls' AND type='table'").fetchone():
            return {'status':'missing','day':day,'actual_cost_usd':None,'charged_cost_usd':None,'calls':[]}
        rows = [dict(r) for r in db.execute('''SELECT id,run_id,day,provider,model,pricing_version,
                     reserved_tokens,charged_tokens,reserved_cost,charged_cost,input_tokens,output_tokens,
                     status,latency_ms,actual_model FROM calls WHERE day=? ORDER BY rowid''',(day,))]
    known = [r for r in rows if r['input_tokens'] is not None and r['output_tokens'] is not None]
    unknown = [r for r in rows if r not in known]
    return {'status':'available','day':day,'cost_semantics':'reported_tokens_at_configured_rates',
            'actual_cost_usd':str(sum((Decimal(r['charged_cost']) for r in known),Decimal(0))),
            'charged_cost_usd':str(sum((Decimal(r['charged_cost']) for r in rows),Decimal(0))),
            'uncertain_reserved_cost_usd':str(sum((Decimal(r['charged_cost']) for r in unknown),Decimal(0))),
            'calls_with_unknown_usage':len(unknown),'calls':rows}


def research_view(path: str | Path) -> dict:
    if not Path(path).is_file():
        return {'status':'missing','candidates':[],'activations':[], 'active':[],
                'blocked_activations':[], 'development_outcomes':[], 'holdouts':[], 'native_runs':[]}
    return research_snapshot(ResearchRegistry(path, read_only=True))


def evidence_view(path: str | Path) -> dict:
    if not Path(path).is_file():
        return {'status':'missing','sources':[]}
    with read_database(path) as db:
        counts = [dict(row) for row in db.execute('SELECT kind,count(*) AS observations FROM evidence GROUP BY kind ORDER BY kind')]
        sources = [json.loads(row[0]) for row in db.execute('SELECT envelope FROM evidence ORDER BY rowid DESC LIMIT 100')]
    return {'status':'available','kinds':counts,'sources':sources}


def operating_view(path: str | Path) -> dict:
    if not Path(path).is_file():
        return {'status':'missing','runs':[],'alerts':[],'schedule_ticks':[]}
    store = OperationsStore(path,read_only=True)
    with store.connection() as db:
        alerts = [dict(row) for row in db.execute('''SELECT a.id,a.created_at,a.run_id,a.severity,a.code,
                   d.channel,d.state AS delivery_state FROM alerts a LEFT JOIN alert_delivery d ON a.id=d.id
                   ORDER BY a.created_at DESC LIMIT 200''')]
        ticks = [dict(row) for row in db.execute('SELECT * FROM schedule_ticks ORDER BY at DESC LIMIT 200')] if db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='schedule_ticks'").fetchone() else []
    return {'status':'available','runs':store.recent(),'alerts':alerts,'schedule_ticks':ticks}


def system_view(config) -> dict:
    sections = {'operations':lambda:operating_view(config.operations.store_path),
                'portfolio':lambda:portfolio_view(config.paper_ledger_path,config.paper_account_id),
                'budgets':lambda:budget_view(config.agents.budget_store_path),
                'research':lambda:research_view(config.operations.research_registry_path),
                'evidence':lambda:evidence_view(config.operations.evidence_store_path)}
    result = {}
    for name,read in sections.items():
        try:
            result[name] = read()
        except (ValueError,OSError,sqlite3.Error) as exc:
            result[name] = {'status':'error','error_type':type(exc).__name__}
    return result
