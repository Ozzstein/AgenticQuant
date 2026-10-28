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
import re
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
    _utc,
)
from src.operations.credentials import provider_credential
from src.operations.store import OperationError, OperationsStore, RunManifest, canonical, digest


class FetchSourceRef(VersionedContract):
    argument: Literal['calendar_id','calendar_ids','bar_ids','reference_id','reference_ids',
                      'universe_ids','print_ids','action_ids','fx_evidence_id']
    task_id: NonEmptyString
    kind: NonEmptyString
    selection: Literal['one','all'] = 'one'


class FetchTask(VersionedContract):
    task_id: NonEmptyString
    provider: Literal['fmp','fred','finnhub','sec','coingecko','alpaca','ccxt','yfinance','mcp','operating']
    operation: NonEmptyString
    instrument: Instrument | None = None
    arguments_json: str = '{}'
    required: bool = True
    deadline_seconds: int = Field(default=120,ge=1,le=600)
    max_http_requests: int = Field(default=30,ge=1,le=1000)
    mcp_manifest: ServerManifest | None = None
    bearer_env: str | None = None
    source_arguments: tuple[FetchSourceRef,...] = ()

    @model_validator(mode='after')
    def closed_input(self):
        arguments=json.loads(self.arguments_json)
        if not isinstance(arguments,dict) or canonical(arguments).decode()!=self.arguments_json:
            raise ValueError('fetch arguments must use canonical object JSON')
        if self.source_arguments and self.provider!='operating':
            raise ValueError('source argument dependencies belong to closed operating tasks')
        scalar_arguments={'calendar_id','reference_id','fx_evidence_id'}
        scalar_seen=set()
        references=set()
        for ref in self.source_arguments:
            if (ref.argument in scalar_arguments)!=(ref.selection=='one'):
                raise ValueError('source selection must match scalar or array argument')
            identity=(ref.argument,ref.task_id,ref.kind)
            if identity in references or (ref.selection=='one' and ref.argument in scalar_seen):
                raise ValueError('duplicate source argument dependency')
            references.add(identity)
            if ref.selection=='one':
                scalar_seen.add(ref.argument)
        if any(ref.argument in arguments for ref in self.source_arguments):
            raise ValueError('source reference cannot override a literal argument')
        if self.provider=='operating':
            from src.operations.operating_fetch import OPERATIONS
            if self.operation not in OPERATIONS or (set(arguments)|{r.argument for r in self.source_arguments})-set(OPERATIONS[self.operation]):
                raise ValueError('operating operation/arguments are not permitted')
            if self.operation=='kraken_fx_reference' and (
                self.instrument is None or self.source_arguments
                or self.instrument.asset_class!='crypto'
                or self.instrument.symbol!='USDT/USD'
                or self.instrument.quote_currency!='USD'
                or self.instrument.venue!='kraken'
                or not isinstance(arguments.get('market_id'),str)
                or re.fullmatch(r'[A-Z0-9]+',arguments['market_id']) is None
            ):
                raise ValueError('Kraken FX needs exact USDT/USD identity and explicit native market ID')
            if self.operation=='binance_notional_reference' and (
                self.instrument is None or self.source_arguments
                or arguments.get('network','mainnet') not in ('mainnet','testnet')
                or type(arguments.get('average_minutes')) is not int
                or arguments['average_minutes']<0
            ):
                raise ValueError('Binance reference needs explicit instrument, native network and integer window')
        if self.provider=='ccxt' and self.operation in ('funding','open_interest'):
            _derivative_arguments(self,arguments)
        if self.provider=='coingecko' and self.operation in ('current_dominance','dominance_history'):
            _dominance_arguments(self,arguments)
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
        seen=set()
        for index, task in enumerate(self.tasks):
            if any(ref.task_id not in seen for ref in task.source_arguments):
                raise ValueError('source dependencies must name earlier tasks in the same immutable plan')
            if task.provider=='operating' and task.operation=='economic_bundle' and index!=len(self.tasks)-1:
                raise ValueError('economic bundle must be the final refresh capture')
            seen.add(task.task_id)
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


def _derivative_arguments(task,arguments):
    """Validate the closed native history request before any SDK/network work."""
    allowed={'since','end','timeframe','limit','max_pages'}
    maximum=1000 if task.operation=='funding' else 500
    if (set(arguments)-allowed or not {'since','end'}<=set(arguments)
        or task.instrument is None or task.instrument.asset_class!='crypto'
        or ':' not in task.instrument.symbol
        or task.instrument.venue not in ('binance','krakenfutures')
        or (task.instrument.venue=='krakenfutures' and task.operation!='funding')):
        raise ValueError('explicit supported perpetual instrument and closed history window required')
    result=dict(arguments)
    for field in ('since','end'):
        if not isinstance(result[field],str):
            raise ValueError('derivative history needs timezone-aware ISO timestamps')
        result[field]=_utc(datetime.fromisoformat(result[field]))
    if result['since']>=result['end']:
        raise ValueError('derivative history interval must be positive')
    for field,upper in (('limit',maximum),('max_pages',1000)):
        if field in result and (type(result[field]) is not int or not 1<=result[field]<=upper):
            raise ValueError('derivative history pagination limit invalid')
    if 'timeframe' in result:
        from src.data.adapters.crypto_context import OI_PERIODS
        if result['timeframe'] not in OI_PERIODS:
            raise ValueError('unsupported derivative sampling period')
    result['max_pages']=min(result.get('max_pages',task.max_http_requests),task.max_http_requests)
    return result


def _dominance_arguments(task,arguments):
    from src.data.adapters.coingecko_context import DAYS, ORIGINS

    allowed={'tier','days'} if task.operation=='dominance_history' else {'tier'}
    tier=arguments.get('tier','demo')
    if task.instrument is not None or set(arguments)-allowed or not isinstance(tier,str) or tier not in ORIGINS:
        raise ValueError('dominance requires closed global scope and explicit supported tier')
    if task.operation=='dominance_history' and (
        tier!='pro' or not isinstance(arguments.get('days','30'),str)
        or arguments.get('days','30') not in DAYS
    ):
        raise ValueError('dominance history requires Pro tier and a documented days string')
    return dict(arguments)


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
        if task.provider=='operating':
            from src.operations.operating_fetch import fetch_operating
            # This adapter owns its measured clock inside normalized economics.
            result,counts=fetch_operating(task,context,store)
        elif task.provider=='fmp':
            from src.data.adapters.fundamentals import FMPAdapter
            args=_arguments(task,('year','quarter','limit'))
            adapter=FMPAdapter(capture,provider_credential('fmp'),http=http('fmp','https://financialmodelingprep.com/stable'))
            result=adapter.fetch(task.operation,instrument(),context,ingested_at=now,**args)
        elif task.provider=='fred' and task.operation=='observations':
            from src.data.adapters.fundamentals import FredAdapter
            args=_arguments(task,('series_id','start','end','units'),('series_id','start','end'))
            adapter=FredAdapter(capture,provider_credential('fred'),http=http('fred','https://api.stlouisfed.org/fred'))
            result=adapter.observations(context=context,ingested_at=now,**args)
        elif task.provider=='finnhub' and task.operation=='news':
            from src.data.adapters.context import FinnhubAdapter
            args=_arguments(task,('start','end'),('start','end'))
            adapter=FinnhubAdapter(capture,provider_credential('finnhub'),http=http('finnhub','https://finnhub.io/api/v1'))
            result=adapter.news(instrument(),context,ingested_at=now,**args)
        elif task.provider=='coingecko' and task.operation in ('global','token'):
            from src.data.adapters.context import CoinGeckoAdapter
            adapter=CoinGeckoAdapter(capture,api_key=provider_credential('coingecko'),http=http('coingecko','https://api.coingecko.com/api/v3'))
            if task.operation=='global':
                _arguments(task,())
                result=adapter.global_context(context,ingested_at=now)
            else:
                args=_arguments(task,('coin_id',),('coin_id',))
                result=adapter.token(args['coin_id'],instrument(),context,ingested_at=now)
        elif task.provider=='coingecko' and task.operation in ('current_dominance','dominance_history'):
            from src.data.adapters.coingecko_context import ORIGINS
            from src.data.adapters.context import CoinGeckoAdapter

            args=_dominance_arguments(task,json.loads(task.arguments_json))
            tier=args.pop('tier','demo')
            # Native source manifests bind the adapter's own measured clocks.
            # CaptureStore would rewrite the transitive envelope identities.
            adapter=CoinGeckoAdapter(store,tier=tier,api_key=provider_credential('coingecko'),
                                    http=http('coingecko',ORIGINS[tier]))
            result=(adapter.current_dominance(context) if task.operation=='current_dominance'
                    else adapter.dominance_history(context,**args))
        elif task.provider=='sec' and task.operation in ('submissions','filing'):
            from src.data.adapters.context import SECAdapter
            adapter=SECAdapter(capture,provider_credential('sec'),http=http('sec','https://data.sec.gov',min_interval=.2))
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
            adapter=AlpacaReadAdapter(capture,provider_credential('alpaca_key'),provider_credential('alpaca_secret'),paper=paper,http=http('alpaca',base))
            result=adapter.inspect(task.operation,context,ingested_at=now)
        elif task.provider=='yfinance' and task.operation=='bars':
            from src.data.adapters.market import YFinanceAdapter
            args=_arguments(task,('start','end'),('start','end'))
            result=YFinanceAdapter(capture).bars(instrument(),context,ingested_at=now,**args)
        elif task.provider=='ccxt' and task.operation in ('bars','funding','open_interest'):
            import ccxt

            from src.data.adapters.market import CCXTAdapter
            allowed_venues=('binance','kraken') if task.operation=='bars' else ('binance','krakenfutures')
            if instrument().venue not in allowed_venues:
                raise ProviderError('ccxt','unsupported_explicit_exchange')
            args=None
            if task.operation!='bars':
                args=_derivative_arguments(task,json.loads(task.arguments_json))
                if args['end']>context.as_of:
                    raise ProviderError('ccxt','history_end_after_request_cutoff')
            exchange=getattr(ccxt,instrument().venue)({'enableRateLimit':True,'timeout':min(20000,task.deadline_seconds*1000)})
            wire_fetch=exchange.fetch
            wire_count=[0]
            def bounded_fetch(url,method='GET',headers=None,body=None):
                if method!='GET' or wire_count[0]>=task.max_http_requests:
                    raise ProviderError('ccxt','readonly_request_budget_exhausted')
                wire_count[0]+=1
                return wire_fetch(url,method,headers,body)
            exchange.fetch=bounded_fetch
            # Native derivative records contain their own measured receipt and
            # source links. Re-clocking would change those authoritative IDs.
            adapter=CCXTAdapter(capture if task.operation=='bars' else store,exchange)
            if task.operation=='bars':
                args=_arguments(task,('start','end','timeframe','market_type','max_pages'),('start','end'))
                for key in ('start','end'):
                    args[key]=datetime.fromisoformat(args[key])
                args['max_pages']=min(args.get('max_pages',task.max_http_requests),task.max_http_requests)
                result=adapter.bars(instrument(),context,ingested_at=now,**args)
            else:
                result=adapter.derivatives(instrument(),context,capability=task.operation,**args)
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

    def _resolve(self,task,receipts):
        if not task.source_arguments:
            return task,[]
        arguments=json.loads(task.arguments_json)
        bindings=[]
        for reference in task.source_arguments:
            receipt=receipts[reference.task_id]
            if receipt['status']!='available':
                raise OperationError('required dependency source task is unavailable')
            selected=[]
            for source in receipt['sources']:
                envelope=self.evidence.get(source['evidence_id'])
                if envelope.content_hash()!=source['envelope_hash']:
                    raise OperationError('dependency source identity changed')
                if envelope.kind==reference.kind:
                    selected.append(envelope.record.evidence_id)
                    bindings.append({'argument':reference.argument,'source_task':reference.task_id,**source})
            if not selected or (reference.selection=='one' and len(selected)!=1):
                raise OperationError('dependency kind lacks exact required source coverage')
            if reference.selection=='one':
                if reference.argument in arguments:
                    raise OperationError('duplicate scalar source dependency')
                arguments[reference.argument]=selected[0]
            else:
                target=arguments.setdefault(reference.argument,[])
                if set(target)&set(selected):
                    raise OperationError('duplicate source identity in dependency array')
                target.extend(selected)
        resolved=FetchTask.model_validate({**task.model_dump(), 'source_arguments':(),
                                          'arguments_json':canonical(arguments).decode()})
        return resolved,bindings

    def _call(self,plan,task,resolved=None,dependencies=()):
        now=datetime.now(timezone.utc)
        if not 0 <= (now-plan.context.as_of).total_seconds() <= plan.max_age_seconds:
            raise OperationError('refresh context is future or stale')
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
        resolved=resolved or task
        result=self.fetcher(resolved,plan.context,self.evidence)
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
                 'task_hash':task.content_hash(),'resolved_task_hash':resolved.content_hash(),
                 'dependency_sources':list(dependencies)}
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
        stages=tuple('fetch:'+task.task_id for task in plan.tasks)+('report',)
        manifest=RunManifest(context=plan.context,kind='data',input_hash=plan.content_hash(),
                             scheduled_for=plan.scheduled_for,stages=stages)
        self.operations.admit(manifest,plan.model_dump(mode='json'))
        with self.operations.exclusive('refresh:'+plan.context.run_id):
            receipts={}
            for task in plan.tasks:
                resolved,dependencies=self._resolve(task,receipts)
                stage_input=task.model_dump(mode='json') if not dependencies else {
                    'task':task.model_dump(mode='json'),'resolved_task':resolved.model_dump(mode='json'),
                    'dependency_sources':dependencies}
                receipts[task.task_id]=self.operations.stage(plan.context.run_id,'fetch:'+task.task_id,
                    stage_input,lambda task=task,resolved=resolved,dependencies=dependencies:
                    self._call(plan,task,resolved,dependencies))
            def report():
                completed=datetime.now(timezone.utc)
                return {'context':plan.context.model_copy(update={'as_of':completed}).model_dump(mode='json'),
                        'requested_context':plan.context.model_dump(mode='json'),'plan_hash':plan.content_hash(),
                        'fetches':receipts,'completed_at':completed.isoformat()}
            receipt=self.operations.stage(plan.context.run_id,'report',receipts,report,recoverable=True)
            return {**self.operations.status(plan.context.run_id),'result':receipt}
