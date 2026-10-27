"""Real calendar/clock decisions, frozen inputs and restart receipts."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.operations.scheduler import ScheduleSpec, ScheduleSupervisor
from src.operations.store import OperationError, OperationsStore

T = datetime(2026,1,1,tzinfo=timezone.utc)


def interval(tmp_path):
    return ScheduleSpec(schedule_id='crypto',starts_at=T,kind='interval',interval_seconds=60,
                        max_lag_seconds=120,input_directory=str(tmp_path/'inputs'))


def test_daily_schedule_honors_dst_and_exchange_holidays(tmp_path):
    spec = ScheduleSpec(schedule_id='equities',starts_at=datetime(2026,3,6,tzinfo=timezone.utc),kind='daily',
                        timezone='America/New_York',hour=16,minute=15,calendar='XNYS',input_directory=str(tmp_path))
    friday = spec.next_at()
    assert friday == datetime(2026,3,6,21,15,tzinfo=timezone.utc)
    monday = spec.next_at(friday)
    assert monday == datetime(2026,3,9,20,15,tzinfo=timezone.utc)


def test_waiting_input_never_becomes_success_and_misses_are_recorded(tmp_path):
    store = OperationsStore(tmp_path/'ops.db')
    scheduler = ScheduleSupervisor(store,lambda *_:pytest.fail('missing inputs cannot execute'))
    spec = interval(tmp_path)
    scheduler.register(spec)
    assert scheduler.tick(T)[0]['status'] == 'waiting'
    assert scheduler.tick(T+timedelta(seconds=121))[0]['status'] == 'missed'
    assert store.pending_alerts('local')[0]['code'] == 'scheduled_input_missed'
    scheduler.register(spec)  # restart registration cannot reset progress
    assert scheduler.tick(T+timedelta(seconds=121))[0]['scheduled_for'] == (T+timedelta(seconds=60)).isoformat()


def test_completed_tick_is_not_reexecuted_on_restart(tmp_path):
    store = OperationsStore(tmp_path/'ops.db')
    spec = interval(tmp_path)
    path = spec.input_path(T)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'run_id':'one'}))
    calls=[]
    def handler(file,at):
        calls.append((json.loads(file.read_text()),at))
        path.write_text('{"run_id":"substituted-after-claim"}')
        assert json.loads(file.read_text()) == {'run_id':'one'}
        return {'status':'completed','actual_run':'one'}
    scheduler = ScheduleSupervisor(store,handler)
    scheduler.register(spec)
    assert scheduler.tick(T)[0]['status'] == 'completed'
    restarted = ScheduleSupervisor(OperationsStore(store.path),handler)
    assert restarted.tick(T) == [] and len(calls)==1


def test_failed_tick_needs_exact_explicit_recovery_and_uses_frozen_input(tmp_path):
    store = OperationsStore(tmp_path/'ops.db')
    spec = interval(tmp_path)
    path = spec.input_path(T)
    path.parent.mkdir(parents=True)
    path.write_text('{"run_id":"one"}')
    def failure(*_):
        raise RuntimeError('simulated interruption before receipt')
    scheduler = ScheduleSupervisor(store,failure)
    scheduler.register(spec)
    assert scheduler.tick(T)[0]['status'] == 'failed'
    assert scheduler.tick(T)[0]['status'] == 'recovery_required'
    with store.connection() as db:
        identity = db.execute('SELECT input_hash FROM schedule_ticks').fetchone()[0]
    with pytest.raises(OperationError,match='exact incomplete'):
        scheduler.resume('crypto',expected_input_hash='0'*64)
    scheduler.resume('crypto',expected_input_hash=identity)
    path.write_text('{"run_id":"forged"}')
    def recover(frozen,at):
        assert json.loads(frozen.read_text()) == {'run_id':'one'}
        return {'status':'completed'}
    scheduler.handler=recover
    assert scheduler.tick(T)[0]['status'] == 'completed'
