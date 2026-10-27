"""Persistent run/stage receipts and process locks for unattended operation.

Economic state belongs to PortfolioLedger. This store coordinates work and
archives its exact inputs/results; it never manufactures portfolio state.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import Field, model_validator

from src.domain.common import NonEmptyString, RunContext, Sha256, UtcTimestamp, VersionedContract


def canonical(value) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class OperationError(ValueError):
    pass


class OperationBusy(OperationError):
    pass


class RecoveryRequired(OperationError):
    pass


class RunManifest(VersionedContract):
    context: RunContext
    kind: Literal['paper_cycle','research','data','evaluation']
    input_hash: Sha256
    scheduled_for: UtcTimestamp
    stages: tuple[NonEmptyString,...] = Field(min_length=1)

    @model_validator(mode='after')
    def valid_steps(self):
        if len(set(self.stages)) != len(self.stages):
            raise ValueError('operation stages must be unique')
        if self.kind == 'paper_cycle' and (self.context.mode != 'paper' or not self.context.account_id):
            raise ValueError('paper cycle needs an explicit persistent paper account')
        if self.scheduled_for > self.context.as_of:
            raise ValueError('operation cannot complete a future schedule')
        return self


class OperationsStore:
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = Path(path).resolve()
        self.read_only = read_only
        self.lock_dir = self.path.with_suffix(self.path.suffix+'.locks')
        if read_only:
            if not self.path.is_file():
                raise OperationError('operating store is not present')
            return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock_dir.mkdir(exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS artifacts(hash TEXT PRIMARY KEY,payload BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,hash TEXT NOT NULL,manifest TEXT NOT NULL,created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS stage_attempts(run_id TEXT NOT NULL,stage TEXT NOT NULL,attempt INTEGER NOT NULL,
                    input_hash TEXT NOT NULL,state TEXT NOT NULL,started_at TEXT NOT NULL,finished_at TEXT,
                    output_hash TEXT,error_type TEXT,PRIMARY KEY(run_id,stage,attempt),FOREIGN KEY(run_id) REFERENCES runs(id));
                CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,
                    stage TEXT,at TEXT NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS alerts(id TEXT PRIMARY KEY,created_at TEXT NOT NULL,run_id TEXT,
                    severity TEXT NOT NULL,code TEXT NOT NULL,payload_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS alert_delivery(id TEXT NOT NULL,channel TEXT NOT NULL,attempt INTEGER NOT NULL,
                    state TEXT NOT NULL,at TEXT NOT NULL,receipt_hash TEXT,PRIMARY KEY(id,channel,attempt));
            ''')
            for table in ('artifacts','runs','events','alerts'):
                for operation in ('UPDATE','DELETE'):
                    db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'immutable operation evidence'); END")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,timeout=30) if self.read_only else sqlite3.connect(self.path,timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def exclusive(self, scope: str):
        """Kernel-owned lock survives process crashes without stale-file bypass."""
        if self.read_only:
            raise OperationError('read-only operations view cannot acquire work')
        path = self.lock_dir/(digest(scope.encode())+'.lock')
        with path.open('a+b') as handle:
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise OperationBusy('another process owns this account/run') from exc
            try:
                yield
            finally:
                fcntl.flock(handle,fcntl.LOCK_UN)

    @staticmethod
    def _archive(db, payload: bytes) -> str:
        key = digest(payload)
        db.execute('INSERT OR IGNORE INTO artifacts VALUES (?,?)',(key,payload))
        return key

    def archive(self, value) -> str:
        with self.connection() as db:
            return self._archive(db,canonical(value))

    def artifact(self, key: str):
        with self.connection() as db:
            row = db.execute('SELECT payload FROM artifacts WHERE hash=?',(key,)).fetchone()
        if row is None or digest(row[0]) != key:
            raise OperationError('operation artifact absent or corrupt')
        return json.loads(row[0])

    def admit(self, manifest: RunManifest, inputs) -> str:
        payload = canonical(inputs)
        if digest(payload) != manifest.input_hash:
            raise OperationError('operation input differs from declared hash')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT hash FROM runs WHERE id=?',(manifest.context.run_id,)).fetchone()
            if old:
                if old[0] != manifest.content_hash():
                    raise OperationError('run identity already binds different inputs/config/schedule')
                return old[0]
            self._archive(db,payload)
            db.execute('INSERT INTO runs VALUES (?,?,?,?)',(manifest.context.run_id,manifest.content_hash(),
                         manifest.model_dump_json(),datetime.now(timezone.utc).isoformat()))
        return manifest.content_hash()

    def manifest(self, run_id: str) -> RunManifest:
        with self.connection() as db:
            row = db.execute('SELECT hash,manifest FROM runs WHERE id=?',(run_id,)).fetchone()
        if row is None:
            raise OperationError('unknown operating run')
        value = RunManifest.model_validate_json(row[1])
        if value.content_hash() != row[0]:
            raise OperationError('operating manifest hash differs')
        return value

    @staticmethod
    def _latest(db,run_id,stage):
        return db.execute('SELECT * FROM stage_attempts WHERE run_id=? AND stage=? ORDER BY attempt DESC LIMIT 1',
                          (run_id,stage)).fetchone()

    @staticmethod
    def _event(db,run_id,stage,kind,payload):
        db.execute('INSERT INTO events(run_id,stage,at,kind,payload) VALUES (?,?,?,?,?)',
                   (run_id,stage,datetime.now(timezone.utc).isoformat(),kind,canonical(payload).decode()))

    def stage(self, run_id: str, stage: str, inputs, operation: Callable[[],object], *,
              recoverable: bool = False):
        """Commit before work. Completed results replay; uncertain I/O needs a resolver.

        A recoverable operation must use its own durable idempotency/reconciliation
        boundary (such as VenueExecution), or be a pure read/derivation. This flag
        never makes raw provider calls safe to repeat.
        """
        manifest = self.manifest(run_id)
        if stage not in manifest.stages:
            raise OperationError('stage was not declared in the operating manifest')
        request_hash = digest(canonical(inputs))
        with self.exclusive('run:'+run_id):
            with self.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                for prior in manifest.stages[:manifest.stages.index(stage)]:
                    value = self._latest(db,run_id,prior)
                    if value is None or value['state'] != 'completed':
                        raise OperationError('an earlier operating stage has not completed')
                old = self._latest(db,run_id,stage)
                if old and old['input_hash'] != request_hash:
                    raise OperationError('stage inputs cannot change during recovery')
                if old and old['state'] == 'completed':
                    key = old['output_hash']
                else:
                    key = None
                    if old and not recoverable:
                        raise RecoveryRequired('prior stage lacks a completed receipt; explicit reconciliation required')
                    attempt = old['attempt']+1 if old else 1
                    self._archive(db,canonical(inputs))
                    db.execute('INSERT INTO stage_attempts VALUES (?,?,?,?,?,?,?,?,?)',
                               (run_id,stage,attempt,request_hash,'running',datetime.now(timezone.utc).isoformat(),None,None,None))
                    self._event(db,run_id,stage,'started',{'attempt':attempt,'input_hash':request_hash})
            if key:
                return self.artifact(key)
            try:
                result = operation()
                payload = canonical(result)
            except BaseException as exc:
                state = 'failed' if isinstance(exc,Exception) else 'uncertain'
                with self.connection() as db:
                    db.execute('UPDATE stage_attempts SET state=?,finished_at=?,error_type=? WHERE run_id=? AND stage=? AND attempt=?',
                               (state,datetime.now(timezone.utc).isoformat(),type(exc).__name__,run_id,stage,attempt))
                    self._event(db,run_id,stage,state,{'attempt':attempt,'error_type':type(exc).__name__})
                self.alert(run_id=run_id,severity='critical',code='stage_'+state,
                           payload={'stage':stage,'attempt':attempt,'error_type':type(exc).__name__})
                raise
            with self.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                key = self._archive(db,payload)
                db.execute('UPDATE stage_attempts SET state=?,finished_at=?,output_hash=? WHERE run_id=? AND stage=? AND attempt=?',
                           ('completed',datetime.now(timezone.utc).isoformat(),key,run_id,stage,attempt))
                self._event(db,run_id,stage,'completed',{'attempt':attempt,'output_hash':key})
            return result

    def status(self, run_id: str) -> dict:
        manifest = self.manifest(run_id)
        with self.connection() as db:
            stages = {stage:dict(row) if (row:=self._latest(db,run_id,stage)) else {'state':'pending'}
                      for stage in manifest.stages}
        states = {row['state'] for row in stages.values()}
        status = 'completed' if states == {'completed'} else ('failed' if 'failed' in states else
                  'uncertain' if 'uncertain' in states else 'in_progress' if 'running' in states else 'pending')
        return {'run_id':run_id,'kind':manifest.kind,'account_id':manifest.context.account_id,
                'as_of':manifest.context.as_of.isoformat(),'manifest_hash':manifest.content_hash(),
                'status':status,'stages':stages}

    def recent(self, limit: int = 100) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError('bounded recent-run query required')
        with self.connection() as db:
            ids = [r[0] for r in db.execute('SELECT id FROM runs ORDER BY created_at DESC LIMIT ?',(limit,))]
        return [self.status(key) for key in ids]

    def alert(self, *, run_id: str | None, severity: str, code: str, payload) -> str:
        if severity not in ('info','warning','critical'):
            raise ValueError('unknown alert severity')
        content = canonical({'run_id':run_id,'severity':severity,'code':code,'payload':payload})
        identity = digest(content)
        with self.connection() as db:
            key = self._archive(db,content)
            db.execute('INSERT OR IGNORE INTO alerts VALUES (?,?,?,?,?,?)',
                       (identity,datetime.now(timezone.utc).isoformat(),run_id,severity,code,key))
        return identity

    def pending_alerts(self, channel: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute('''SELECT a.* FROM alerts a WHERE COALESCE(
                (SELECT d.state FROM alert_delivery d WHERE d.id=a.id AND d.channel=?
                 ORDER BY d.attempt DESC LIMIT 1),'new') IN ('new','retry_authorized')
                ORDER BY created_at''',(channel,)).fetchall()
        return [dict(row) for row in rows]

    def deliver(self, identity: str, channel: str, sender: Callable[[dict],dict]) -> dict:
        """At-most-once automatic sends: a crash after send requires receipt review."""
        with self.exclusive('alert:'+identity+':'+channel):
            with self.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT payload_hash FROM alerts WHERE id=?',(identity,)).fetchone()
                if row is None:
                    raise OperationError('unknown alert')
                previous = db.execute('SELECT * FROM alert_delivery WHERE id=? AND channel=? ORDER BY attempt DESC LIMIT 1',
                                      (identity,channel)).fetchone()
                if previous and previous['state'] != 'retry_authorized':
                    if previous['state'] == 'sent':
                        key = previous['receipt_hash']
                    else:
                        raise RecoveryRequired('alert delivery is uncertain; do not send it again automatically')
                else:
                    key = None
                    attempt = previous['attempt']+1 if previous else 1
                    db.execute('INSERT INTO alert_delivery VALUES (?,?,?,?,?,?)',
                               (identity,channel,attempt,'sending',datetime.now(timezone.utc).isoformat(),None))
            if key:
                return self.artifact(key)
            try:
                receipt = sender(self.artifact(row[0]))
                if not isinstance(receipt,dict) or receipt.get('accepted') is not True:
                    raise OperationError('notification transport did not confirm acceptance')
                key = self.archive(receipt)
            except BaseException:
                with self.connection() as db:
                    db.execute("UPDATE alert_delivery SET state='uncertain' WHERE id=? AND channel=? AND attempt=?",(identity,channel,attempt))
                raise
            with self.connection() as db:
                db.execute("UPDATE alert_delivery SET state='sent',receipt_hash=? WHERE id=? AND channel=? AND attempt=?",(key,identity,channel,attempt))
            return receipt

    def resolve_delivery(self, identity: str, channel: str, *, expected_attempt: int,
                         outcome: Literal['confirmed','abandoned','retry_authorized'], evidence: dict) -> str:
        """Explicit operator resolution, distinct from measured transport acceptance."""
        if outcome not in ('confirmed','abandoned','retry_authorized') or not evidence:
            raise OperationError('explicit resolution and archived review evidence required')
        with self.exclusive('alert:'+identity+':'+channel):
            with self.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                old = db.execute('SELECT * FROM alert_delivery WHERE id=? AND channel=? ORDER BY attempt DESC LIMIT 1',
                                 (identity,channel)).fetchone()
                if old is None or old['attempt'] != expected_attempt or old['state'] not in ('sending','uncertain'):
                    raise OperationError('resolution must identify the current uncertain delivery attempt')
                key = self._archive(db,canonical({'basis':'operator_review','outcome':outcome,'evidence':evidence,
                    'alert_id':identity,'channel':channel,'attempt':expected_attempt,'previous':dict(old)}))
                db.execute('UPDATE alert_delivery SET state=?,receipt_hash=? WHERE id=? AND channel=? AND attempt=?',
                           (outcome,key,identity,channel,expected_attempt))
                self._event(db,'alert:'+identity,None,'delivery_resolved',{'resolution_hash':key})
                return key
