"""Actual budget/gateway authority drives reporting and native release gates."""
import copy
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest

from src.agents.budget import AgentBudgetError
from src.operations.reporting import budget_view
from src.research.native import NativeRequest, NativeSupervisor, _budget_audit_error
from src.research.native_recovery import _budget_receipt
from tests.native_budget_fixtures import project, receipt
from tests.test_agent_budget_usage import MODEL, budget, dispatch, gateway, raw
from tests.test_agent_budget_usage import cfg as budget_config
from tests.test_native_rd_adapter import request

cfg = budget_config
UTC = timezone.utc


def native_request(g):
    payload = request().model_dump(mode='json')
    r = g.receipt()
    payload.update(run_id=g.config.run_id, gateway_config_hash=g.config.content_hash(),
                   chat_model='openai/'+g.config.chat.model,
                   embedding_model='openai/'+g.config.embedding.model,
                   gateway_limits=r['limits'], gateway_day_limits=r['day_limits'], max_seconds=60)
    return NativeRequest.model_validate(payload)


@pytest.mark.parametrize('case', ['known', 'wrong-model', 'partial', 'error', 'overrun'])
def test_actual_gateway_accounting_is_not_reservation_equals_bill(cfg, case):
    calls = []

    def respond(req):
        calls.append(req)
        usage = {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}
        if case in ('partial', 'overrun'):
            usage['prompt_tokens'] = 100001
            usage['total_tokens'] = 100003
        if case == 'partial':
            usage.pop('completion_tokens')
        return httpx.Response(500 if case == 'error' else 200, json={
            'model': 'unknown-private-model' if case == 'wrong-model' else MODEL,
            'choices': [{'message': {'role': 'assistant', 'content': 'fixture only'}, 'finish_reason': 'stop', 'index': 0}],
            'usage': usage,
        })

    g, token = gateway(cfg, respond)
    req = native_request(g)
    NativeSupervisor.verify_receipt(req, g.receipt(), now=datetime.now(UTC))
    dispatch(g, token)
    started = datetime.now(UTC)
    public = g.receipt()
    saved = _budget_receipt(req, public, datetime.now(UTC), not_before=started)
    scope = saved['accounting']['run']
    assert len(calls) == 1
    assert scope['known_input_tokens'] == (100001 if case in ('partial', 'overrun') else 10)
    if case in ('known', 'error'):
        assert _budget_audit_error(saved) is None
        assert scope['conservative_tokens'] > scope['actual_tokens']
        assert Decimal(saved['charged']['cost_usd']) > Decimal(saved['actual']['cost_usd'])
        assert saved['uncertain']['requests'] == 0
    else:
        assert _budget_audit_error(saved) in ('budget_breached', 'uncertain_usage_retained')
        with pytest.raises(ValueError):
            NativeSupervisor.verify_receipt(req, public, now=datetime.now(UTC))
        if case in ('wrong-model', 'partial'):
            assert saved['actual']['cost_usd'] is None
            assert scope['unpriced_requests'] == 1
        assert 'unknown-private-model' not in json.dumps(saved)


def test_another_run_day_breach_blocks_known_native_run(cfg):
    def known(req):
        return httpx.Response(200, json={'model': MODEL, 'choices': [{'message': {'role': 'assistant', 'content': 'x'}}],
                                       'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}})
    g, token = gateway(cfg, known)
    req = native_request(g)
    dispatch(g, token)
    other, other_token = gateway(cfg, lambda req: httpx.Response(200, json={
        'model': 'unknown', 'usage': {'prompt_tokens': 100001, 'completion_tokens': 2, 'total_tokens': 100003}}), run='other')
    dispatch(other, other_token)
    saved = _budget_receipt(req, g.receipt(), datetime.now(UTC))
    assert not saved['accounting']['run']['breached']
    assert saved['accounting']['day']['breached']
    assert _budget_audit_error(saved) == 'budget_breached'
    with pytest.raises(ValueError):
        NativeSupervisor.verify_receipt(req, g.receipt(), now=datetime.now(UTC))


@pytest.mark.parametrize('fault', [
    'missing-v2', 'policy', 'run', 'future', 'other-day', 'stale-final', 'bool-count',
    'nan', 'false-exact', 'old-public', 'status-bool', 'empty-with-cost', 'day-count', 'clock-type',
])
def test_native_refuses_non_authoritative_or_inconsistent_scope(fault):
    req, now = request(), datetime.now(UTC)
    r = receipt(req, now)
    a, scope = r['accounting'], r['accounting']['run']
    if fault == 'missing-v2':
        del r['accounting']
    elif fault == 'policy':
        a['policy'] = 'legacy'
    elif fault == 'run':
        a['run_id'] = 'another'
    elif fault == 'future':
        a['as_of'] = (now+timedelta(seconds=1)).isoformat()
    elif fault == 'clock-type':
        a['as_of'] = False
    elif fault == 'other-day':
        a['day_id'] = (now-timedelta(days=1)).date().isoformat()
    elif fault == 'stale-final':
        a['as_of'] = (now-timedelta(seconds=1)).isoformat()
    elif fault == 'bool-count':
        scope['requests'] = True
    elif fault == 'nan':
        scope['actual_cost_usd'] = 'NaN'
    elif fault == 'false-exact':
        scope.update(usage_unknown_requests=1, unpriced_requests=1)
    elif fault == 'old-public':
        r['actual']['cost_usd'] = '0'
    elif fault == 'status-bool':
        r['ready'] = 'true'
    elif fault == 'empty-with-cost':
        scope.update(requests=0, statuses={}, observed_models=[], observed_model_hashes=[])
        project(r)
    elif fault == 'day-count':
        a['day']['requests'] = False
    with pytest.raises((ValueError, TypeError)):
        _budget_receipt(req, r, now, not_before=now)


def test_reporting_sole_reader_preserves_unknowns_and_detects_corruption(cfg):
    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)
    with pytest.raises(AgentBudgetError):
        b.finish(identity, price, raw=raw(i=10001, model='private-unrecognized-model'), status='ok')
    before = b.path.read_bytes()
    view = budget_view(b.path, run_id='run')
    assert view['status'] == 'available'
    for key in ('day_usage', 'run_usage'):
        scope = view[key]
        assert scope['known_input_tokens'] == 10001
        assert scope['actual_cost_usd'] is None and scope['unpriced_requests'] == 1
        assert scope['breached']
    assert 'private-unrecognized-model' not in json.dumps(view)
    assert b.path.read_bytes() == before
    with sqlite3.connect(b.path) as db:
        db.execute("UPDATE calls SET charged_tokens=0")
    invalid = budget_view(b.path)
    assert invalid == {'status': 'unavailable', 'reason': 'generative_usage_authority_invalid', 'error_type': 'ValueError'}


def test_public_scope_copy_cannot_be_changed_after_verification():
    req, now = request(), datetime.now(UTC)
    r = receipt(req, now)
    saved = _budget_receipt(req, r, now)
    original = copy.deepcopy(saved)
    assert _budget_receipt(req, saved, now) == saved
    r['accounting']['run']['conservative_cost_usd'] = '999'
    assert saved == original
