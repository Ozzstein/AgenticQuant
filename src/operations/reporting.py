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


def budget_view(path: str | Path, *, day: str | None = None,
                run_id: str | None = None, as_of: datetime | None = None) -> dict:
    """Project the sole verified generative usage authority, never mutable SQL totals."""
    from src.agents.receipts import read_usage_summary

    cutoff = as_of or datetime.now(timezone.utc)
    try:
        return read_usage_summary(path, day=day or cutoff.astimezone(timezone.utc).date().isoformat(),
                                  run_id=run_id, as_of=cutoff)
    except Exception as exc:
        return {'status':'unavailable','reason':'generative_usage_authority_invalid',
                'error_type':type(exc).__name__}


def decision_budget_view(path: str | Path, *, day: str | None = None,
                         run_id: str | None = None, as_of: datetime | None = None) -> dict:
    """Use the decision store's verified charge authority without copying its math."""
    from src.decision.store import read_usage_summary

    cutoff = as_of or datetime.now(timezone.utc)
    try:
        return read_usage_summary(path, day=day or cutoff.astimezone(timezone.utc).date().isoformat(),
                                  run_id=run_id, as_of=cutoff)
    except Exception as exc:
        # A monitoring failure must never turn corrupt/missing charge evidence
        # into a measured zero or expose raw requests through an exception.
        return {'status':'unavailable','reason':'decision_usage_authority_invalid',
                'error_type':type(exc).__name__}


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
                'decision_budgets':lambda:decision_budget_view(Path(config.output_dir)/'decisions.sqlite'),
                'research':lambda:research_view(config.operations.research_registry_path),
                'evidence':lambda:evidence_view(config.operations.evidence_store_path)}
    result = {}
    for name,read in sections.items():
        try:
            result[name] = read()
        except (ValueError,OSError,sqlite3.Error) as exc:
            result[name] = {'status':'error','error_type':type(exc).__name__}
    return result
