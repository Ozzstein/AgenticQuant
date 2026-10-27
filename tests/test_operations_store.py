"""Crash/restart and concurrent-operation boundaries use real SQLite/processes."""
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.domain.common import RunContext
from src.operations.store import (
    OperationError,
    OperationsStore,
    RecoveryRequired,
    RunManifest,
    canonical,
    digest,
)


def admitted(tmp_path, stages=('data','orders','report')):
    store = OperationsStore(tmp_path/'ops.db')
    at = datetime(2026,1,1,tzinfo=timezone.utc)
    context = RunContext(run_id='cycle-one',mode='paper',as_of=at,config_hash='1'*64,
                         code_version='test',account_id='paper:test')
    inputs = {'snapshot':'actual-fixture-identity'}
    manifest = RunManifest(context=context,kind='paper_cycle',input_hash=digest(canonical(inputs)),
                           scheduled_for=at,stages=stages)
    store.admit(manifest,inputs)
    return store,manifest


def test_immutable_admission_ordered_stages_and_exact_restart_receipts(tmp_path):
    store,manifest = admitted(tmp_path)
    with pytest.raises(OperationError,match='earlier'):
        store.stage('cycle-one','orders',{},lambda:{'wrong':True})
    store.stage('cycle-one','data',{'request':1},lambda:{'archive':'real'})
    restarted = OperationsStore(store.path)
    assert restarted.stage('cycle-one','data',{'request':1},lambda:pytest.fail('must not rerun')) == {'archive':'real'}
    with pytest.raises(OperationError,match='cannot change'):
        restarted.stage('cycle-one','data',{'request':2},lambda:{})
    with pytest.raises(OperationError,match='different inputs'):
        store.admit(manifest.model_copy(update={'stages':('other',)}),{'snapshot':'actual-fixture-identity'})
    restarted.stage('cycle-one','orders',{},lambda:{'outbox':'prepared'})
    assert restarted.status('cycle-one')['status'] != 'completed'
    restarted.stage('cycle-one','report',{},lambda:{'sent':False})
    assert restarted.status('cycle-one')['status'] == 'completed'
    assert OperationsStore(store.path,read_only=True).recent()[0]['status'] == 'completed'


def test_failure_is_not_success_and_only_recoverable_stage_retries(tmp_path):
    store,_ = admitted(tmp_path,('data',))
    def failure():
        raise RuntimeError('fixture transport failure')
    with pytest.raises(RuntimeError):
        store.stage('cycle-one','data',{},failure)
    assert store.status('cycle-one')['status'] == 'failed'
    assert len(store.pending_alerts('local')) == 1
    with pytest.raises(RecoveryRequired):
        store.stage('cycle-one','data',{},lambda:{'result':1})
    assert store.stage('cycle-one','data',{},lambda:{'result':1},recoverable=True) == {'result':1}
    assert store.status('cycle-one')['stages']['data']['attempt'] == 2


def test_subprocess_death_after_work_never_blindly_resubmits(tmp_path):
    store,_ = admitted(tmp_path,('orders',))
    marker = tmp_path/'external-action.txt'
    code = '''import os,sys
from pathlib import Path
from src.operations.store import OperationsStore
store=OperationsStore(sys.argv[1])
def action():
    Path(sys.argv[2]).write_text('accepted')
    os._exit(23)
store.stage('cycle-one','orders',{},action)
'''
    result = subprocess.run([sys.executable,'-c',code,str(store.path),str(marker)],check=False,
                            env={**os.environ,'PYTHONPATH':str(Path.cwd())},capture_output=True,text=True)
    assert result.returncode == 23 and marker.read_text() == 'accepted'
    restarted = OperationsStore(store.path)
    with pytest.raises(RecoveryRequired):
        restarted.stage('cycle-one','orders',{},lambda:pytest.fail('duplicate submission'))
    # The owned outbox resolver checks external acceptance before returning a
    # receipt. No second action occurs in this recovery callback.
    assert restarted.stage('cycle-one','orders',{},lambda:{'observed':marker.read_text()},recoverable=True) == {'observed':'accepted'}


def test_kernel_lock_rejects_competing_process_then_releases(tmp_path):
    store,_ = admitted(tmp_path)
    code = '''import sys
from src.operations.store import OperationsStore,OperationBusy
try:
    with OperationsStore(sys.argv[1]).exclusive('paper:test'): pass
except OperationBusy:
    sys.exit(17)
'''
    args = [sys.executable,'-c',code,str(store.path)]
    with store.exclusive('paper:test'):
        blocked = subprocess.run(args,capture_output=True,check=False)
    assert blocked.returncode == 17
    assert subprocess.run(args,capture_output=True,check=False).returncode == 0


def test_alerts_dedupe_and_timeout_requires_delivery_reconciliation(tmp_path):
    store,_ = admitted(tmp_path)
    values = dict(run_id='cycle-one',severity='critical',code='drawdown',payload={'drawdown':'.16'})
    key = store.alert(**values)
    assert store.alert(**values) == key
    sent = []
    def sender(value):
        sent.append(value)
        return {'accepted':True,'receipt':'local-fixture'}
    store.deliver(key,'local',sender)
    store.deliver(key,'local',sender)
    assert len(sent) == 1 and not store.pending_alerts('local')
    def timeout(value):
        raise TimeoutError('unknown delivery')
    with pytest.raises(TimeoutError):
        store.deliver(key,'email',timeout)
    assert not store.pending_alerts('email')
    with pytest.raises(RecoveryRequired):
        store.deliver(key,'email',sender)
    assert len(sent) == 1


def test_account_history_is_not_truncated_by_other_accounts(tmp_path):
    store, manifest = admitted(tmp_path, ('data',))
    store.stage('cycle-one', 'data', {}, lambda: {'anchor':'original'})
    payload = {'snapshot':'actual-fixture-identity'}
    for number in range(1002):
        context = manifest.context.model_copy(update={'run_id':f'other-{number}', 'account_id':'paper:other'})
        store.admit(manifest.model_copy(update={'context':context}), payload)
    own = manifest.model_copy(update={'context':manifest.context.model_copy(update={'run_id':'cycle-two'})})
    store.admit(own, payload)
    assert not any(r['run_id']=='cycle-one' for r in store.recent(1000))
    history = OperationsStore(store.path,read_only=True).account_runs('paper:test')
    assert [(r['run_id'],r['status']) for r in history] == [('cycle-two','pending'),('cycle-one','completed')]
    assert history[1]['stages']['data']['output_hash'] == store.status('cycle-one')['stages']['data']['output_hash']
    assert store.account_runs('paper:test',kind='research') == []
    with pytest.raises(ValueError):
        store.account_runs('',kind='paper_cycle')
