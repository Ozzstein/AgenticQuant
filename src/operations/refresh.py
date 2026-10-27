"""Durable bounded refresh of the selected read-only data sources.

Each admitted fetch runs in a bounded worker process. Current retrieval becomes
available at measured receipt; only pinned vintage/accession adapters may attest
earlier availability. A new decision context is returned after all observations.
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from src.data.adapters.base import FetchResult, ProviderError, ReadHTTP
from src.data.evidence.mcp import MCPManager, ServerManifest
from src.data.evidence.store import EvidenceStore
from src.domain.common import (
    Instrument,
    NonEmptyString,
    RunContext,
    UtcTimestamp,
    VersionedContract,
)
from src.operations.store import OperationError, OperationsStore, RunManifest, canonical, digest


class FetchTask(VersionedContract):
    task_id: NonEmptyString
    provider: Literal['fmp','fred','finnhub','sec','coingecko','alpaca','ccxt','yfinance','mcp']
    operation: NonEmptyString
    instrument: Instrument | None = None
    arguments_json: str = '{}'
    required: bool = True
    deadline_seconds: int = Field(default=120,ge=1,le=600)
    max_http_requests: int = Field(default=30,ge=1,le=1000)
    mcp_manifest: ServerManifest | None = None
    bearer_env: str | None = None

    @model_validator(mode='after')
    def closed_input(self):
        arguments=json.loads(self.arguments_json)
        if not isinstance(arguments,dict) or canonical(arguments).decode()!=self.arguments_json:
            raise ValueError('fetch arguments must use canonical object JSON')
        if self.provider=='mcp':
            if self.mcp_manifest is None:
                raise ValueError('MCP fetch needs an exact deployed manifest')
            # Authorize locally before reserving money or opening a connection.
            manager=MCPManager(None,(self.mcp_manifest,))
            manager.authorize(self.mcp_manifest.provider,self.operation,arguments)
        elif self.mcp_manifest is not None or self.bearer_env is not None:
            raise ValueError('MCP transport settings belong only to MCP tasks')
        if self.bearer_env:
            from src.operations.notifications import _environment
            _environment(self.bearer_env)
        return self

    @property
    def reserved_cost(self):
        if self.mcp_manifest is None:
            return Decimal(0)
        pin, _ = MCPManager(None, (self.mcp_manifest,)).authorize(
            self.mcp_manifest.provider, self.operation, json.loads(self.arguments_json))
        return pin.max_cost_usd


class RefreshPlan(VersionedContract):
    context: RunContext
    scheduled_for: UtcTimestamp
    tasks: tuple[FetchTask,...] = Field(min_length=1,max_length=100)
    max_age_seconds: int = Field(default=3600,ge=1,le=86400)
    max_fetches_per_day: int = Field(default=500,ge=1,le=10000)
    max_mcp_reserved_usd_per_day: Decimal = Field(default=Decimal(10),ge=0)

    @model_validator(mode='after')
    def valid_schedule(self):
        if self.context.mode not in ('paper','research') or self.scheduled_for>self.context.as_of:
            raise ValueError('refresh requires a current paper/research context and due schedule')
        if len({t.task_id for t in self.tasks})!=len(self.tasks):
            raise ValueError('unique refresh task IDs required')
        manifests={}
        for task in self.tasks:
            if task.mcp_manifest:
                key=task.mcp_manifest.provider
                manifests.setdefault(key,[]).append(task)
        for tasks in manifests.values():
            manifest=tasks[0].mcp_manifest
            if any(t.mcp_manifest != manifest for t in tasks):
                raise ValueError('one immutable MCP manifest per provider/run required')
            if len(tasks)>manifest.max_requests or sum((t.reserved_cost for t in tasks),Decimal(0))>manifest.budget_usd:
                raise ValueError('planned MCP calls exceed pinned provider request/spend limits')
        return self


class CaptureStore:
    """Application receipt clock around existing adapters, before archive insertion."""
    def __init__(self,store:EvidenceStore,*,clock=None):
        self.store=store
        self.clock=clock or (lambda:datetime.now(timezone.utc))

    def __getattr__(self,name):
        return getattr(self.store,name)

    def put(self,record,raw,**kwargs):
        measured=self.clock()
        if measured.tzinfo is None:
            raise OperationError('receipt clock requires timezone')
        historical=(record.provider,record.provider_version) in {
            ('fred','fred-alfred-vintage-v1'),('sec','accession-v1')}
        available=record.available_at if historical else max(record.available_at or measured,measured)
        identity=digest(canonical({'adapter_record':record.content_hash(),'received_at':measured.isoformat(),
                                   'availability_policy':'operational-receipt-clock-1'}))
        updated=type(record).model_validate({**record.model_dump(), 'evidence_id':identity,
                                             'ingested_at':measured,'available_at':available})
        return self.store.put(updated,raw,**kwargs)


def _arguments(task,allowed,required=()):
    value=json.loads(task.arguments_json)
    if set(value)-set(allowed) or set(required)-set(value):
        raise ProviderError(task.provider,'unsupported_or_missing_arguments')
    return value


def _fetch(task:FetchTask,context:RunContext,store:EvidenceStore)->dict:
    """Closed dispatch; neither models nor MCP servers choose endpoints/methods."""
    capture=CaptureStore(store)
    now=datetime.now(timezone.utc)
    transport=None
    adapter=None
    counts=None
    def http(provider,base,**kwargs):
        nonlocal transport
        transport=ReadHTTP(provider,base,max_requests=task.max_http_requests,attempts=2,
                           timeout=min(20,task.deadline_seconds),**kwargs)
        return transport
    def instrument():
        if task.instrument is None:
            raise ProviderError(task.provider,'explicit_instrument_required')
        return task.instrument
    try:
        if task.provider=='fmp':
            from src.data.adapters.fundamentals import FMPAdapter
            args=_arguments(task,('year','quarter','limit'))
            adapter=FMPAdapter(capture,os.getenv('FMP_API_KEY'),http=http('fmp','https://financialmodelingprep.com/stable'))
            result=adapter.fetch(task.operation,instrument(),context,ingested_at=now,**args)
        elif task.provider=='fred' and task.operation=='observations':
            from src.data.adapters.fundamentals import FredAdapter
            args=_arguments(task,('series_id','start','end','units'),('series_id','start','end'))
            adapter=FredAdapter(capture,os.getenv('FRED_API_KEY'),http=http('fred','https://api.stlouisfed.org/fred'))
            result=adapter.observations(context=context,ingested_at=now,**args)
        elif task.provider=='finnhub' and task.operation=='news':
            from src.data.adapters.context import FinnhubAdapter
            args=_arguments(task,('start','end'),('start','end'))
            adapter=FinnhubAdapter(capture,os.getenv('FINNHUB_API_KEY'),http=http('finnhub','https://finnhub.io/api/v1'))
            result=adapter.news(instrument(),context,ingested_at=now,**args)
        elif task.provider=='coingecko' and task.operation in ('global','token'):
            from src.data.adapters.context import CoinGeckoAdapter
            adapter=CoinGeckoAdapter(capture,http=http('coingecko','https://api.coingecko.com/api/v3'))
            if task.operation=='global':
                _arguments(task,())
                result=adapter.global_context(context,ingested_at=now)
            else:
                args=_arguments(task,('coin_id',),('coin_id',))
                result=adapter.token(args['coin_id'],instrument(),context,ingested_at=now)
        elif task.provider=='sec' and task.operation in ('submissions','filing'):
            from src.data.adapters.context import SECAdapter
            adapter=SECAdapter(capture,os.getenv('SEC_USER_AGENT',''),http=http('sec','https://data.sec.gov',min_interval=.2))
            if task.operation=='submissions':
                args=_arguments(task,('cik',),('cik',))
                result=adapter.submissions(instrument=instrument(),context=context,ingested_at=now,**args)
            else:
                fields=('cik','accession','document','index_evidence_id')
                args=_arguments(task,fields,fields)
                # A separate fixed host is required for actual archived filings.
                archives=ReadHTTP('sec','https://www.sec.gov',max_requests=task.max_http_requests,attempts=2,
                                  timeout=min(20,task.deadline_seconds),min_interval=.2)
                try:
                    result=adapter.filing(instrument=instrument(),context=context,ingested_at=now,archives=archives,**args)
                    counts=archives.requests
                finally:
                    archives.close()
        elif task.provider=='alpaca':
            from src.data.adapters.context import AlpacaReadAdapter
            args=_arguments(task,('paper',))
            paper=args.get('paper',True)
            if not isinstance(paper,bool):
                raise ProviderError('alpaca','paper_flag_must_be_boolean')
            base='https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
            adapter=AlpacaReadAdapter(capture,os.getenv('ALPACA_API_KEY'),os.getenv('ALPACA_API_SECRET'),paper=paper,http=http('alpaca',base))
            result=adapter.inspect(task.operation,context,ingested_at=now)
        elif task.provider=='yfinance' and task.operation=='bars':
            from src.data.adapters.market import YFinanceAdapter
            args=_arguments(task,('start','end'),('start','end'))
            result=YFinanceAdapter(capture).bars(instrument(),context,ingested_at=now,**args)
        elif task.provider=='ccxt' and task.operation in ('bars','funding','open_interest'):
            import ccxt

            from src.data.adapters.market import CCXTAdapter
            if instrument().venue not in ('binance','kraken'):
                raise ProviderError('ccxt','unsupported_explicit_exchange')
            exchange=getattr(ccxt,instrument().venue)({'enableRateLimit':True,'timeout':min(20000,task.deadline_seconds*1000)})
            wire_fetch=exchange.fetch
            wire_count=[0]
            def bounded_fetch(url,method='GET',headers=None,body=None):
                if method!='GET' or wire_count[0]>=task.max_http_requests:
                    raise ProviderError('ccxt','readonly_request_budget_exhausted')
                wire_count[0]+=1
                return wire_fetch(url,method,headers,body)
            exchange.fetch=bounded_fetch
            adapter=CCXTAdapter(capture,exchange)
            if task.operation=='bars':
                args=_arguments(task,('start','end','timeframe','market_type','max_pages'),('start','end'))
                for key in ('start','end'):
                    args[key]=datetime.fromisoformat(args[key])
                args['max_pages']=min(args.get('max_pages',task.max_http_requests),task.max_http_requests)
                result=adapter.bars(instrument(),context,ingested_at=now,**args)
            else:
                args=_arguments(task,('since','limit'),('since',))
                args['since']=datetime.fromisoformat(args['since'])
                result=adapter.derivatives(instrument(),context,ingested_at=now,capability=task.operation,**args)
            counts=wire_count[0]
        elif task.provider=='mcp':
            async def remote():
                async with MCPManager(capture,(task.mcp_manifest,)) as manager:
                    provider=task.mcp_manifest.provider
                    await manager.connect(provider,bearer_token=os.getenv(task.bearer_env) if task.bearer_env else None)
                    envelope=await manager.call(provider,task.operation,json.loads(task.arguments_json),context,
                                                ingested_at=now,instrument=task.instrument)
                    return FetchResult('available',(envelope,))
            result=asyncio.run(remote())
        else:
            raise ProviderError(task.provider,'operation_not_permitted')
        if transport:
            counts=(counts or 0)+transport.requests
        return {'status':result.status,'evidence_ids':[e.record.evidence_id for e in result.evidence],
                'detail':result.detail,'observed_http_requests':counts,
                'received_at':datetime.now(timezone.utc).isoformat()}
    finally:
        if transport:
            transport.close()


def _worker(connection,task,context,path):
    try:
        result=_fetch(FetchTask.model_validate(task),RunContext.model_validate(context),EvidenceStore(path))
        connection.send(result)
    except BaseException as exc:
        connection.send({'status':'unavailable','error_type':type(exc).__name__,
                         'code':exc.code if isinstance(exc,ProviderError) else 'fetch_failed','evidence_ids':[]})
    finally:
        connection.close()


def bounded_fetch(task:FetchTask,context:RunContext,store:EvidenceStore)->dict:
    """Hard wall deadline also covers SDK pagination and asynchronous MCP work."""
    runtime=multiprocessing.get_context('spawn')
    receive,send=runtime.Pipe(duplex=False)
    child=runtime.Process(target=_worker,args=(send,task.model_dump(),context.model_dump(),str(store.path)),daemon=True)
    child.start()
    send.close()
    try:
        if not receive.poll(task.deadline_seconds):
            return {'status':'unavailable','error_type':'TimeoutError','code':'deadline_cost_unknown','evidence_ids':[]}
        try:
            result=receive.recv()
        except EOFError:
            return {'status':'unavailable','error_type':'WorkerFailure','code':'worker_exit_cost_unknown','evidence_ids':[]}
        child.join(timeout=1)
        return result
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=1)
            if child.is_alive():
                child.kill()
                child.join(timeout=1)
        receive.close()


class RefreshService:
    def __init__(self,operations:OperationsStore,evidence:EvidenceStore,*,fetcher=None):
        self.operations,self.evidence=operations,evidence
        self.fetcher=fetcher or bounded_fetch
        with operations.connection() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS refresh_limits(day TEXT PRIMARY KEY,caps TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS refresh_calls(run_id TEXT,task_id TEXT,day TEXT,task_hash TEXT,
                    reserved_usd TEXT,receipt_hash TEXT,PRIMARY KEY(run_id,task_id));''')

    def _call(self,plan,task):
        now=datetime.now(timezone.utc)
        day=now.date().isoformat()
        caps=canonical({'fetches':plan.max_fetches_per_day,'mcp_reserved_usd':str(plan.max_mcp_reserved_usd_per_day)}).decode()
        with self.operations.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT caps FROM refresh_limits WHERE day=?',(day,)).fetchone()
            if old and old[0]!=caps:
                raise OperationError('refresh day caps are immutable once admitted')
            db.execute('INSERT OR IGNORE INTO refresh_limits VALUES (?,?)',(day,caps))
            spent=[Decimal(row[0]) for row in db.execute('SELECT reserved_usd FROM refresh_calls WHERE day=?',(day,))]
            if len(spent)>=plan.max_fetches_per_day or sum(spent,Decimal(0))+task.reserved_cost>plan.max_mcp_reserved_usd_per_day:
                raise OperationError('durable refresh request/MCP spend limit exhausted')
            db.execute('INSERT INTO refresh_calls VALUES (?,?,?,?,?,NULL)',
                       (plan.context.run_id,task.task_id,day,task.content_hash(),str(task.reserved_cost)))
        result=self.fetcher(task,plan.context,self.evidence)
        # Store actual envelope identities, not a provider's success boolean.
        sources=[]
        identities=result.get('evidence_ids',[])
        if not isinstance(identities,list) or len(identities)!=len(set(identities)):
            raise OperationError('fetch returned invalid/duplicate source identities')
        provider=task.mcp_manifest.provider if task.mcp_manifest else (
            'ccxt:'+task.instrument.venue if task.provider=='ccxt' else task.provider)
        for identity in identities:
            envelope=self.evidence.get(identity)
            record=envelope.record
            if record.provider!=provider or record.requested_as_of!=plan.context.as_of:
                raise OperationError('fetch source provider/request cutoff mismatch')
            if not now<=record.ingested_at<=datetime.now(timezone.utc):
                raise OperationError('fetch source lacks a receipt within this admission')
            if task.instrument and task.instrument.instrument_id not in record.instrument_ids:
                raise OperationError('fetch source instrument mismatch')
            self.evidence.raw(identity)
            for key in (envelope.document_hash,envelope.normalized_hash):
                if key:
                    self.evidence.blob(key)
            sources.append({'evidence_id':identity,'envelope_hash':envelope.content_hash()})
        receipt={**result,'sources':sources,'reserved_mcp_cost_usd':str(task.reserved_cost),
                 'cost_semantics':'MCP maximum reserved; direct subscription/entitlement costs not measured',
                 'task_hash':task.content_hash()}
        key=self.operations.archive(receipt)
        with self.operations.connection() as db:
            db.execute('UPDATE refresh_calls SET receipt_hash=? WHERE run_id=? AND task_id=?',
                       (key,plan.context.run_id,task.task_id))
        if receipt.get('status')!='available' or not sources:
            self.operations.alert(run_id=plan.context.run_id,severity='critical' if task.required else 'warning',
                                  code='data_fetch_unavailable',payload={'task':task.task_id,'receipt_hash':key})
            if task.required:
                raise OperationError('required data fetch unavailable; exact failure receipt archived')
        return receipt

    def run(self,plan:RefreshPlan)->dict:
        now=datetime.now(timezone.utc)
        if not 0 <= (now-plan.context.as_of).total_seconds() <= plan.max_age_seconds:
            raise OperationError('refresh context is future or stale')
        stages=tuple('fetch:'+task.task_id for task in plan.tasks)+('report',)
        manifest=RunManifest(context=plan.context,kind='data',input_hash=plan.content_hash(),
                             scheduled_for=plan.scheduled_for,stages=stages)
        self.operations.admit(manifest,plan.model_dump(mode='json'))
        with self.operations.exclusive('refresh:'+plan.context.run_id):
            receipts={task.task_id:self.operations.stage(plan.context.run_id,'fetch:'+task.task_id,
                task.model_dump(mode='json'),lambda task=task:self._call(plan,task)) for task in plan.tasks}
            def report():
                completed=datetime.now(timezone.utc)
                return {'context':plan.context.model_copy(update={'as_of':completed}).model_dump(mode='json'),
                        'requested_context':plan.context.model_dump(mode='json'),'plan_hash':plan.content_hash(),
                        'fetches':receipts,'completed_at':completed.isoformat()}
            receipt=self.operations.stage(plan.context.run_id,'report',receipts,report,recoverable=True)
            return {**self.operations.status(plan.context.run_id),'result':receipt}
