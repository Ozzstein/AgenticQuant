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


@pytest.mark.parametrize('delay', [1, 121])
def test_completed_receipt_survives_crash_before_schedule_advances(tmp_path, monkeypatch, delay):
    store = OperationsStore(tmp_path/'ops.db')
    spec = interval(tmp_path)
    path = spec.input_path(T)
    path.parent.mkdir(parents=True)
    path.write_text('{"run_id":"original"}')
    calls = []

    def handler(file, at):
        calls.append(json.loads(file.read_text()))
        return {'status':'completed', 'actual_run':'original'}

    scheduler = ScheduleSupervisor(store, handler)
    scheduler.register(spec)
    with monkeypatch.context() as patch:
        def crash(*_):
            raise KeyboardInterrupt('after durable completion, before cursor advance')
        patch.setattr(ScheduleSpec, 'next_at', crash)
        with pytest.raises(KeyboardInterrupt):
            scheduler.tick(T)
    with store.connection() as db:
        before = dict(db.execute('SELECT * FROM schedule_ticks').fetchone())
        assert before['state'] == 'completed'
        assert db.execute('SELECT next_at FROM schedules').fetchone()[0] == T.isoformat()
    path.write_text('{"run_id":"replacement"}')
    restarted = ScheduleSupervisor(OperationsStore(store.path), handler)
    # Even after the lag window, recorded completion wins over redispatch/missed.
    result = restarted.tick(T+timedelta(seconds=delay))[0]
    assert result['status'] == 'completed'
    assert result['receipt_hash'] == before['receipt_hash']
    assert calls == [{'run_id':'original'}]
    with store.connection() as db:
        assert dict(db.execute('SELECT * FROM schedule_ticks').fetchone()) == before
        assert db.execute('SELECT next_at FROM schedules').fetchone()[0] == (T+timedelta(seconds=60)).isoformat()
    assert store.pending_alerts('local') == []


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


@pytest.mark.parametrize('damage', ['input', 'receipt', 'incomplete_receipt'])
def test_completed_cursor_recovery_requires_original_artifacts(tmp_path, monkeypatch, damage):
    store = OperationsStore(tmp_path/'ops.db')
    spec = interval(tmp_path)
    path = spec.input_path(T)
    path.parent.mkdir(parents=True)
    path.write_text('{"run_id":"original"}')
    scheduler = ScheduleSupervisor(store, lambda *_: {'status':'completed'})
    scheduler.register(spec)
    with monkeypatch.context() as patch:
        def crash(*_):
            raise KeyboardInterrupt('after receipt')
        patch.setattr(ScheduleSpec, 'next_at', crash)
        with pytest.raises(KeyboardInterrupt):
            scheduler.tick(T)
    replacement = store.archive({'status':'failed'}) if damage == 'incomplete_receipt' else '0'*64
    with store.connection() as db:
        column = 'input_hash' if damage == 'input' else 'receipt_hash'
        db.execute(f'UPDATE schedule_ticks SET {column}=?', (replacement,))
    restarted = ScheduleSupervisor(OperationsStore(store.path), lambda *_: pytest.fail('no redispatch'))
    with pytest.raises(OperationError, match='artifact absent or corrupt|original completion receipt'):
        restarted.tick(T)
    with store.connection() as db:
        assert db.execute('SELECT next_at FROM schedules').fetchone()[0] == T.isoformat()
