"""Durable clock schedules consuming explicit, archived operating input bundles.

The scheduler cannot invent observations, train prices or completion receipts.
A due job waits for its actual input file; expired inputs produce a missed-run
alert. A failed/uncertain run blocks that schedule until explicit recovery.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal
from zoneinfo import ZoneInfo

from pydantic import Field, model_serializer, model_validator

from src.domain.common import NonEmptyString, UtcTimestamp, VersionedContract, _utc
from src.operations.rolling import RollingRefreshTemplate
from src.operations.store import OperationBusy, OperationError, OperationsStore, canonical, digest


class ScheduleSpec(VersionedContract):
    schedule_id: NonEmptyString
    execution_lane: Literal['operating','research'] = 'operating'
    starts_at: UtcTimestamp
    kind: Literal['daily','interval']
    timezone: NonEmptyString = 'UTC'
    hour: int = Field(default=0,ge=0,le=23)
    minute: int = Field(default=0,ge=0,le=59)
    interval_seconds: int | None = Field(default=None,ge=60,le=86400*7)
    calendar: Literal['XNYS','24/7'] = '24/7'
    max_lag_seconds: int = Field(default=3600,ge=0,le=86400*7)
    input_directory: NonEmptyString | None = None
    refresh_template: RollingRefreshTemplate | None = None

    @model_serializer(mode='wrap')
    def legacy_identity(self, handler):
        value = handler(self)
        if self.execution_lane == 'operating':
            value.pop('execution_lane', None)
        if self.refresh_template is None:
            value.pop('refresh_template', None)
        if self.input_directory is None:
            value.pop('input_directory', None)
        return value

    @model_validator(mode='after')
    def coherent(self):
        ZoneInfo(self.timezone)
        if (self.kind == 'interval') != (self.interval_seconds is not None):
            raise ValueError('interval schedule needs an explicit interval only')
        if any(c in self.schedule_id for c in ('/','\\','..')):
            raise ValueError('schedule ID must be a simple directory identity')
        if (self.input_directory is None) == (self.refresh_template is None):
            raise ValueError('schedule needs exactly one input directory or rolling refresh template')
        if self.refresh_template is not None and self.execution_lane != 'operating':
            raise ValueError('rolling refresh belongs to the operating lane')
        if self.input_directory is not None and not Path(self.input_directory).is_absolute():
            raise ValueError('schedule input directory must be absolute')
        return self

    def next_at(self, after: datetime | None = None) -> datetime:
        """Actual UTC times honor local DST and exchange session dates."""
        after = _utc(after) if after else None
        if self.kind == 'interval':
            value = self.starts_at if after is None else after+timedelta(seconds=self.interval_seconds)
        else:
            from apscheduler.triggers.cron import CronTrigger
            trigger = CronTrigger(hour=self.hour,minute=self.minute,timezone=ZoneInfo(self.timezone),start_date=self.starts_at)
            value = trigger.get_next_fire_time(after,after+timedelta(microseconds=1) if after else self.starts_at).astimezone(timezone.utc)
        if self.calendar == '24/7':
            return value
        from src.data.calendars import sessions
        # A bounded calendar search errors visibly if the configured calendar
        # cannot describe the next observation, instead of guessing weekdays.
        start_date = value.astimezone(ZoneInfo(self.timezone)).date()
        calendar = sessions(start_date.isoformat(),(start_date+timedelta(days=32)).isoformat(),calendar=self.calendar)
        for _ in range(31*24*60):
            date = value.astimezone(ZoneInfo(self.timezone)).date().isoformat()
            if date in calendar:
                return value
            if self.kind == 'interval':
                value += timedelta(seconds=self.interval_seconds)
            else:
                value = trigger.get_next_fire_time(value,value+timedelta(microseconds=1)).astimezone(timezone.utc)
        raise OperationError('no exchange session within bounded schedule search')

    def input_path(self, at: datetime) -> Path:
        if self.input_directory is None:
            raise OperationError('rolling refresh has no mutable per-window input path')
        return Path(self.input_directory)/self.schedule_id/at.astimezone(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ.json')


class ScheduleSupervisor:
    def __init__(self, store: OperationsStore, handler: Callable[[Path,datetime],dict]):
        if store.read_only:
            raise OperationError('scheduler requires its writable operating store')
        self.store,self.handler = store,handler
        with store.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS schedules(id TEXT PRIMARY KEY,hash TEXT NOT NULL,spec TEXT NOT NULL,next_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS schedule_ticks(schedule TEXT NOT NULL,at TEXT NOT NULL,input_hash TEXT,state TEXT NOT NULL,
                    receipt_hash TEXT,error_type TEXT,PRIMARY KEY(schedule,at));
            ''')

    def register(self, spec: ScheduleSpec) -> None:
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT hash FROM schedules WHERE id=?',(spec.schedule_id,)).fetchone()
            if old:
                if old[0] != spec.content_hash():
                    raise OperationError('schedule identity/config is immutable; register a new identity')
                return
            db.execute('INSERT INTO schedules VALUES (?,?,?,?)',(spec.schedule_id,spec.content_hash(),spec.model_dump_json(),spec.next_at().isoformat()))

    def tick(self, now: datetime, *, lane: Literal['operating','research'] | None = None) -> list[dict]:
        now = _utc(now)
        with self.store.connection() as db:
            rows = db.execute('SELECT * FROM schedules ORDER BY next_at').fetchall()
        receipts = []
        for row in rows:
            spec = ScheduleSpec.model_validate_json(row['spec'])
            if spec.content_hash() != row['hash']:
                raise OperationError('schedule content hash differs')
            if lane is not None and spec.execution_lane != lane:
                continue
            at = datetime.fromisoformat(row['next_at'])
            if at > now:
                continue
            try:
                with self.store.exclusive('schedule:'+spec.schedule_id):
                    receipts.append(self._run_due(spec,at,now))
            except OperationBusy:
                receipts.append({'schedule_id':spec.schedule_id,'status':'busy','scheduled_for':at.isoformat()})
        return receipts

    def _run_due(self,spec,at,now):
        with self.store.connection() as db:
            current = db.execute('SELECT next_at FROM schedules WHERE id=?',(spec.schedule_id,)).fetchone()[0]
            if current != at.isoformat():
                return {'schedule_id':spec.schedule_id,'status':'already_advanced'}
            old = db.execute('SELECT * FROM schedule_ticks WHERE schedule=? AND at=?',(spec.schedule_id,at.isoformat())).fetchone()
        if old and old['state'] == 'completed':
            # Completion and cursor advancement can be separated by a process
            # crash. The original immutable receipt remains authoritative even
            # after the lag window or replacement of the mutable input file.
            self.store.artifact(old['input_hash'])
            completed = self.store.artifact(old['receipt_hash'])
            if not isinstance(completed, dict) or completed.get('status') != 'completed' or old['error_type'] is not None:
                raise OperationError('completed schedule lacks its original completion receipt')
            next_at = spec.next_at(at)
            with self.store.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                db.execute('UPDATE schedules SET next_at=? WHERE id=? AND next_at=?',
                           (next_at.isoformat(),spec.schedule_id,at.isoformat()))
            return {'schedule_id':spec.schedule_id,'scheduled_for':at.isoformat(),'status':'completed',
                    'receipt_hash':old['receipt_hash'],'error_type':None}
        if old and old['state'] in ('failed','uncertain','running'):
            return {'schedule_id':spec.schedule_id,'status':'recovery_required','scheduled_for':at.isoformat()}
        path = spec.input_path(at) if spec.input_directory is not None else None
        identity = None
        state,error_type,receipt = 'waiting',None,None
        retry = old is not None and old['state'] == 'retry'
        if not retry and now > at+timedelta(seconds=spec.max_lag_seconds):
            state = 'missed'
            self.store.alert(run_id=None,severity='critical',code='scheduled_input_missed',
                             payload={'schedule':spec.schedule_id,'scheduled_for':at.isoformat()})
        elif retry or spec.refresh_template is not None or path.is_file():
            # Stage source bytes before any dispatch; subsequent file replacement
            # can only fail its identity check in the operating bundle handler.
            if retry:
                payload = canonical(self.store.artifact(old['input_hash']))
            elif spec.refresh_template is not None:
                plan = spec.refresh_template.materialize(spec.schedule_id, scheduled_for=at, invoked_at=now)
                payload = canonical({'kind': 'data_refresh', 'plan': plan.model_dump(mode='json')})
            else:
                payload = path.read_bytes()
            identity = old['input_hash'] if retry else digest(payload)
            with self.store.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                self.store._archive(db,payload)
                db.execute('INSERT INTO schedule_ticks VALUES (?,?,?,?,?,?) ON CONFLICT(schedule,at) DO UPDATE SET input_hash=excluded.input_hash,state=excluded.state',
                           (spec.schedule_id,at.isoformat(),identity,'running',None,None))
                claim = {'scheduled_for':at.isoformat(),'input_hash':identity}
                if spec.refresh_template is not None:
                    claim['refresh_template_hash'] = spec.refresh_template.content_hash()
                    claim['invoked_at'] = now.isoformat()
                self.store._event(db,'schedule:'+spec.schedule_id,None,'claimed',claim)
            try:
                import json
                archived = self.store.archive(json.loads(payload))
                # Execute the archived immutable input copy, not a mutable inbox.
                directory = self.store.path.parent/'scheduled-inputs'
                directory.mkdir(exist_ok=True)
                frozen = directory/(archived+'.json')
                from src.core.backtester import write_immutable
                write_immutable(frozen,canonical(self.store.artifact(archived)).decode())
                result = self.handler(frozen,at)
                if result.get('status') != 'completed':
                    raise OperationError('operating handler did not produce a completed receipt')
                receipt = self.store.archive(result)
                state = 'completed'
            except BaseException as exc:
                state = 'failed' if isinstance(exc,Exception) else 'uncertain'
                error_type = type(exc).__name__
                self.store.alert(run_id=None,severity='critical',code='scheduled_run_'+state,
                                 payload={'schedule':spec.schedule_id,'scheduled_for':at.isoformat(),'error_type':error_type})
                if not isinstance(exc,Exception):
                    raise
            finally:
                with self.store.connection() as db:
                    db.execute('UPDATE schedule_ticks SET state=?,receipt_hash=?,error_type=? WHERE schedule=? AND at=?',
                               (state,receipt,error_type,spec.schedule_id,at.isoformat()))
        if state in ('completed','missed'):
            with self.store.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                db.execute('INSERT OR IGNORE INTO schedule_ticks VALUES (?,?,?,?,?,?)',
                           (spec.schedule_id,at.isoformat(),identity,state,receipt,error_type))
                db.execute('UPDATE schedules SET next_at=? WHERE id=? AND next_at=?',
                           (spec.next_at(at).isoformat(),spec.schedule_id,at.isoformat()))
        return {'schedule_id':spec.schedule_id,'scheduled_for':at.isoformat(),'status':state,'receipt_hash':receipt,'error_type':error_type}

    def resume(self, schedule_id: str, *, expected_input_hash: str) -> None:
        """Explicitly re-enter the same immutable bundle through stage recovery."""
        with self.store.exclusive('schedule:'+schedule_id):
            with self.store.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT next_at FROM schedules WHERE id=?',(schedule_id,)).fetchone()
                if row is None:
                    raise OperationError('unknown schedule')
                tick = db.execute('SELECT * FROM schedule_ticks WHERE schedule=? AND at=?',(schedule_id,row[0])).fetchone()
                if tick is None or tick['state'] not in ('failed','uncertain','running') or tick['input_hash'] != expected_input_hash:
                    raise OperationError('recovery must name the exact incomplete schedule/input')
                db.execute("UPDATE schedule_ticks SET state='retry' WHERE schedule=? AND at=?",(schedule_id,row[0]))
                self.store._event(db,'schedule:'+schedule_id,None,'recovery_requested',{'at':row[0],'input_hash':expected_input_hash})
