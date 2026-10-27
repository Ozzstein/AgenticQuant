"""Real receipts/stores/adapters with only provider I/O replaced offline."""
import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest

from src.data.adapters.base import ReadHTTP, store_observation
from src.data.evidence.store import EvidenceStore
from src.domain.common import Instrument, RunContext
from src.operations import refresh
from src.operations.refresh import CaptureStore, FetchTask, RefreshPlan, RefreshService
from src.operations.store import OperationError, OperationsStore, RecoveryRequired, canonical


def context():
    return RunContext(run_id='refresh-one',mode='paper',as_of=datetime.now(timezone.utc),
                      account_id='paper:test',config_hash='a'*64,code_version='fixture')


def task(**updates):
    return FetchTask(task_id='financials',provider='fmp',operation='income_statement',
        instrument=Instrument(instrument_id='us:test',symbol='TEST',asset_class='equity',quote_currency='USD'),
        **updates)


def observation(store, ctx, *, at=None, provider='fmp', version='fmp-stable-capture-v1', instrument=None):
    return store_observation(store,raw=b'{"value":42}',normalized={'value':42},document='Complete source: 42.',
        provider=provider,version=version,source_uri='https://example.test/observation',kind='income_statement',
        logical_key='test:42',context=ctx,ingested_at=at or ctx.as_of,available_at=ctx.as_of-timedelta(days=5),
        instrument=instrument)


def test_measured_receipt_never_backdates_current_fetch_and_preserves_full_blobs(tmp_path):
    store=EvidenceStore(tmp_path/'data.db')
    ctx=context()
    measured=ctx.as_of+timedelta(seconds=3)
    item=observation(CaptureStore(store,clock=lambda:measured),ctx)
    assert item.record.ingested_at==item.record.available_at==measured
    assert item.record.requested_as_of==ctx.as_of
    with pytest.raises(ValueError,match='unavailable'):
        store.get(item.record.evidence_id,as_of=ctx.as_of)
    assert store.raw(item.record.evidence_id)==b'{"value":42}'
    assert store.document(item.record.evidence_id,as_of=measured).text=='Complete source: 42.'
    assert store.normalized(item.record.evidence_id,as_of=measured)=={'value':42}


@pytest.mark.parametrize('provider,version,historical',[('fred','fred-alfred-vintage-v1',True),
    ('sec','accession-v1',True),('sec','direct-capture-v1',False),('fred','unknown-version',False)])
def test_only_pinned_vintage_adapters_preserve_historical_availability(tmp_path,provider,version,historical):
    ctx=context()
    measured=ctx.as_of+timedelta(seconds=3)
    item=observation(CaptureStore(EvidenceStore(tmp_path/'data.db'),clock=lambda:measured),
                     ctx,provider=provider,version=version)
    assert item.record.available_at==(ctx.as_of-timedelta(days=5) if historical else measured)
    assert item.record.ingested_at==measured


def real_adapter_fetcher(monkeypatch):
    calls=[]
    def response(request):
        calls.append(request)
        return httpx.Response(200,json=[{'symbol':'TEST','date':'2020-01-01','revenue':42}])
    client=httpx.Client(transport=httpx.MockTransport(response))
    monkeypatch.setenv('FMP_API_KEY','fixture-not-real')
    monkeypatch.setattr(refresh,'ReadHTTP',lambda provider,base,**kw:ReadHTTP(provider,base,client=client,**kw))
    return calls,client


def test_actual_adapter_refresh_restarts_without_refetch_and_advances_clock(tmp_path,monkeypatch):
    calls,client=real_adapter_fetcher(monkeypatch)
    ops=OperationsStore(tmp_path/'ops.db')
    store=EvidenceStore(tmp_path/'sources.db')
    ctx=context()
    plan=RefreshPlan(context=ctx,scheduled_for=ctx.as_of,tasks=(task(),))
    service=RefreshService(ops,store,fetcher=refresh._fetch)
    try:
        receipt=service.run(plan)
        assert receipt['status']=='completed' and len(calls)==1
        later=RunContext.model_validate(receipt['result']['context'])
        fetched=receipt['result']['fetches']['financials']
        record=store.get(fetched['sources'][0]['evidence_id']).record
        assert ctx.as_of<record.available_at<=later.as_of
        assert fetched['observed_http_requests']==1
        assert fetched['sources'][0]['envelope_hash']==store.get(record.evidence_id).content_hash()
        restart=RefreshService(OperationsStore(ops.path),EvidenceStore(store.path),
            fetcher=lambda *a:pytest.fail('duplicate fetch'))
        assert restart.run(plan)['result']==receipt['result']
        assert len(calls)==1 and 'fixture-not-real' not in json.dumps(receipt)
    finally:
        client.close()


def test_configured_key_reaches_fmp_without_entering_persisted_receipts(tmp_path, monkeypatch):
    calls, client = real_adapter_fetcher(monkeypatch)
    monkeypatch.setenv('AIQUANT_FMP_API_KEY', 'configured-read-only-fixture')
    ops = OperationsStore(tmp_path/'ops.db')
    store = EvidenceStore(tmp_path/'sources.db')
    ctx = context()
    try:
        receipt = RefreshService(ops, store, fetcher=refresh._fetch).run(
            RefreshPlan(context=ctx, scheduled_for=ctx.as_of, tasks=(task(),)))
        assert len(calls) == 1
        assert calls[0].url.params['apikey'] == 'configured-read-only-fixture'
        assert 'fixture-not-real' not in str(calls[0].url)
        source = receipt['result']['fetches']['financials']['sources'][0]
        envelope = store.get(source['evidence_id'])
        assert 'configured-read-only-fixture' not in envelope.model_dump_json()
        assert 'configured-read-only-fixture' not in json.dumps(receipt)
    finally:
        client.close()


def test_nested_alpaca_credentials_reach_actual_read_transport_only(tmp_path, monkeypatch):
    calls = []

    def response(request):
        calls.append(request)
        return httpx.Response(200, json={'id': 'fixture-account', 'cash': '125.50'})

    client = httpx.Client(transport=httpx.MockTransport(response))
    monkeypatch.setattr(refresh, 'ReadHTTP',
        lambda provider, base, **kw: ReadHTTP(provider, base, client=client, **kw))
    monkeypatch.setenv('AIQUANT_ALPACA__API_KEY', 'configured-fixture-id')
    monkeypatch.setenv('AIQUANT_ALPACA__API_SECRET', 'configured-fixture-secret')
    monkeypatch.setenv('ALPACA_API_KEY', 'legacy-fixture-id')
    monkeypatch.setenv('ALPACA_API_SECRET', 'legacy-fixture-secret')
    ctx = context()
    store = EvidenceStore(tmp_path/'sources.db')
    selected = FetchTask(task_id='account', provider='alpaca', operation='account')
    try:
        receipt = RefreshService(OperationsStore(tmp_path/'ops.db'), store,
                                 fetcher=refresh._fetch).run(
            RefreshPlan(context=ctx, scheduled_for=ctx.as_of, tasks=(selected,)))
        assert len(calls) == 1 and calls[0].method == 'GET'
        assert calls[0].url.host == 'paper-api.alpaca.markets'
        assert calls[0].url.path == '/v2/account' and not calls[0].url.query
        assert calls[0].headers['APCA-API-KEY-ID'] == 'configured-fixture-id'
        assert calls[0].headers['APCA-API-SECRET-KEY'] == 'configured-fixture-secret'
        for source in receipt['result']['fetches']['account']['sources']:
            envelope = store.get(source['evidence_id'])
            assert 'configured-fixture-secret' not in envelope.model_dump_json()
        assert 'configured-fixture-secret' not in json.dumps(receipt)
    finally:
        client.close()


def test_unavailable_required_data_cannot_complete_or_silently_retry(tmp_path):
    ops=OperationsStore(tmp_path/'ops.db')
    store=EvidenceStore(tmp_path/'sources.db')
    ctx=context()
    plan=RefreshPlan(context=ctx,scheduled_for=ctx.as_of,tasks=(task(),))
    fetches=[]
    def failed(*args):
        fetches.append(1)
        return {'status':'unavailable','code':'deadline_cost_unknown','evidence_ids':[]}
    service=RefreshService(ops,store,fetcher=failed)
    with pytest.raises(OperationError,match='required data'):
        service.run(plan)
    with pytest.raises(RecoveryRequired):
        service.run(plan)
    assert len(fetches)==1 and ops.status(ctx.run_id)['status']=='failed'
    with ops.connection() as db:
        row=db.execute('SELECT receipt_hash FROM refresh_calls').fetchone()
    assert ops.artifact(row[0])['code']=='deadline_cost_unknown'


def test_optional_gaps_are_visible_and_attempt_budget_is_durable(tmp_path):
    ops=OperationsStore(tmp_path/'ops.db')
    store=EvidenceStore(tmp_path/'sources.db')
    ctx=context()
    plan=RefreshPlan(context=ctx,scheduled_for=ctx.as_of,tasks=(task(required=False),),max_fetches_per_day=1)
    service=RefreshService(ops,store,fetcher=lambda *a:{'status':'missing','evidence_ids':[]})
    first=service.run(plan)
    assert first['result']['fetches']['financials']['status']=='missing'
    assert ops.pending_alerts('local')
    next_plan=plan.model_copy(update={'context':ctx.model_copy(update={'run_id':'refresh-two'})})
    with pytest.raises(OperationError,match='limit exhausted'):
        RefreshService(OperationsStore(ops.path),store,fetcher=lambda *a:pytest.fail('budget bypass')).run(next_plan)


def test_mcp_generic_dispatch_reserves_the_exact_effective_tool_and_rejects_mutation():
    from tests.test_mcp_evidence_permissions import manifest
    source=manifest()
    pin=source.tools[0].model_copy(update={'max_cost_usd':Decimal('.25')})
    source=source.model_copy(update={'tools':(pin,),'budget_usd':Decimal('.25')})
    args={'name':'get_orders','arguments':{'status':'open'}}
    value=FetchTask(task_id='orders',provider='mcp',operation='call_tool',
        arguments_json=canonical(args).decode(),mcp_manifest=source)
    assert value.reserved_cost==Decimal('.25')
    args['name']='submit_order'
    with pytest.raises(Exception,match='operation_not_permitted'):
        FetchTask(task_id='bad',provider='mcp',operation='call_tool',
                  arguments_json=canonical(args).decode(),mcp_manifest=source)


def _hung_worker(*args):
    time.sleep(30)


def test_actual_worker_deadline_terminates_hung_fetch(tmp_path,monkeypatch):
    monkeypatch.setattr(refresh,'_worker',_hung_worker)
    started=time.monotonic()
    result=refresh.bounded_fetch(task(deadline_seconds=1),context(),EvidenceStore(tmp_path/'sources.db'))
    assert result['code']=='deadline_cost_unknown' and time.monotonic()-started<10


def test_worker_missing_credentials_returns_redacted_failure(tmp_path,monkeypatch):
    monkeypatch.delenv('FMP_API_KEY',raising=False)
    result=refresh.bounded_fetch(task(deadline_seconds=10),context(),EvidenceStore(tmp_path/'sources.db'))
    assert result['status']=='unavailable' and result['code']=='credentials_missing'


def test_fetch_cannot_reuse_prior_admission_as_new_receipt(tmp_path):
    ops=OperationsStore(tmp_path/'ops.db')
    store=EvidenceStore(tmp_path/'sources.db')
    ctx=context()
    item=observation(store,ctx,instrument=task().instrument)
    plan=RefreshPlan(context=ctx,scheduled_for=ctx.as_of,tasks=(task(),))
    service=RefreshService(ops,store,fetcher=lambda *a:{'status':'available','evidence_ids':[item.record.evidence_id]})
    with pytest.raises(OperationError,match='receipt within'):
        service.run(plan)
