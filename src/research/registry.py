"""Transactional quarantine, trial accounting, holdouts and versioned activation.

Only the coordinator's evaluator writes outcomes. Generative agents may propose
candidate content, but are not given trial completion or promotion authority.
The registry stores exact artifacts and evidence bytes, never unpickles code.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from threading import local
from typing import Literal

from pydantic import Field, model_validator

from src.domain.common import NonEmptyString, Sha256, UtcTimestamp, VersionedContract, _utc


def content_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode()


class Candidate(VersionedContract):
    candidate_id: NonEmptyString
    kind: Literal["factor", "model", "strategy", "regime", "semantic_question"]
    origin: Literal["native_rd_agent", "custom", "copilot", "paper"]
    payload_hash: Sha256
    code_hash: Sha256
    environment_hash: Sha256
    source_evidence_hashes: tuple[Sha256, ...]
    parent_ids: tuple[NonEmptyString, ...] = ()
    created_at: UtcTimestamp
    synthetic: bool = False


class Holdout(VersionedContract):
    holdout_id: NonEmptyString
    dataset_hash: Sha256
    starts_at: UtcTimestamp
    ends_at: UtcTimestamp
    registered_at: UtcTimestamp

    @model_validator(mode="after")
    def interval(self):
        if self.starts_at >= self.ends_at:
            raise ValueError("holdout interval must be positive")
        return self


class Trial(VersionedContract):
    trial_id: NonEmptyString
    candidate_id: NonEmptyString
    family_id: NonEmptyString
    dataset_hash: Sha256
    split_hash: Sha256
    inputs_hash: Sha256
    policy_hash: Sha256
    registered_at: UtcTimestamp
    purpose: Literal["development", "holdout"]
    evaluation_start: UtcTimestamp
    evaluation_end: UtcTimestamp
    holdout_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def holdout_identity(self):
        if (self.purpose == "holdout") != (self.holdout_id is not None):
            raise ValueError("holdout trials require exactly one registered holdout identity")
        if self.evaluation_start >= self.evaluation_end:
            raise ValueError("evaluation interval must be positive")
        return self


class DiscoveryExposure(VersionedContract):
    """All mounted observations exposed before a native search, including failure.

    This is not a trial count. Native internal search multiplicity is unknown
    until an independently verifiable search audit is implemented.
    """
    exposure_id: NonEmptyString
    request_hash: Sha256
    bundle_hash: Sha256
    dataset_hash: Sha256
    starts_at: UtcTimestamp
    ends_at: UtcTimestamp
    admitted_at: UtcTimestamp
    internal_trials: None = None
    document_manifest_hash: Sha256 | None = None
    inherited_exposures: tuple[Sha256, ...] = ()
    observation_scope: Literal['market-snapshot','unbounded'] = 'market-snapshot'


class PromotionPolicy(VersionedContract):
    """Research activation criteria; this does not grant live-capital eligibility."""

    min_sharpe: float = 0.5
    min_ic: float = 0.03
    min_icir: float = 0.5
    max_factor_correlation: float = 0.7
    familywise_alpha: float = Field(default=0.05, gt=0, lt=1)
    min_observations: int = Field(default=252, ge=2)


class Evaluation(VersionedContract):
    trial_id: NonEmptyString
    candidate_hash: Sha256
    inputs_hash: Sha256
    replay_hash: Sha256
    manifest_hash: Sha256
    validation_hash: Sha256
    registry_snapshot_hash: Sha256
    completed_at: UtcTimestamp
    status: Literal["complete", "failed"]
    validation_reliable: bool
    actual_model_hashes: tuple[Sha256, ...]
    actual_decision_hashes: tuple[Sha256, ...] = ()
    regime_evaluation_hash: Sha256 | None = None
    execution_verification_hash: Sha256 | None = None
    paired_evaluation_hash: Sha256 | None = None
    observations: int = Field(ge=0)
    sharpe: float | None = None
    ic: float | None = None
    icir: float | None = None
    max_factor_correlation: float | None = None
    # Must come from the independently computed test, not a generated report.
    unadjusted_p_value: float | None = Field(default=None, ge=0, le=1)
    synthetic: bool = False
    diagnostics: tuple[str, ...] = ()


class Activation(VersionedContract):
    slot: NonEmptyString
    generation: int = Field(ge=1)
    candidate_id: NonEmptyString
    candidate_hash: Sha256
    payload_hash: Sha256
    evaluation_hash: Sha256
    policy_hash: Sha256
    activated_at: UtcTimestamp
    action: Literal["promote", "rollback"]
    previous_generation: int = Field(ge=0)


class RegistryError(ValueError):
    """Invalid lifecycle transition, evidence mismatch or stale activation."""


class MissingDevelopmentEvaluation(RegistryError):
    """A registered development trial has no outcome; distinct from corruption."""


class ResearchRegistry:
    """Append-only artifacts and atomic active generations on one durable path."""

    def __init__(self, path: str | Path, *, read_only: bool = False):
        from src.research.native_mounts import NativeMountAuthority, canonical_path

        self.path = canonical_path(path)
        self.read_only = read_only
        self._reads = local()
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            return
        NativeMountAuthority().protect(self.path,Path(str(self.path)+'-wal'),Path(str(self.path)+'-shm'))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS blobs(hash TEXT PRIMARY KEY, content BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, hash TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS holdouts(id TEXT PRIMARY KEY, dataset TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS trials(id TEXT PRIMARY KEY, candidate TEXT NOT NULL REFERENCES candidates(id),
                    family TEXT NOT NULL, holdout TEXT UNIQUE REFERENCES holdouts(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS outcomes(trial TEXT PRIMARY KEY REFERENCES trials(id), hash TEXT NOT NULL,
                    payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS trial_snapshots(trial TEXT PRIMARY KEY REFERENCES trials(id), hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS activations(slot TEXT NOT NULL, generation INTEGER NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(slot,generation));
                CREATE TABLE IF NOT EXISTS active(slot TEXT PRIMARY KEY, generation INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS discovery_exposures(id TEXT PRIMARY KEY, hash TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS discovery_results(id TEXT PRIMARY KEY, hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS native_launches(id TEXT PRIMARY KEY, hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS native_containers(id TEXT PRIMARY KEY, hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS native_recoveries(id TEXT NOT NULL, sequence INTEGER NOT NULL,
                    hash TEXT NOT NULL, PRIMARY KEY(id,sequence));
            """)
            for table in ("blobs", "candidates", "holdouts", "trials", "outcomes", "trial_snapshots", "activations", "discovery_exposures", "discovery_results", "native_launches", "native_containers", "native_recoveries"):
                for action in ("UPDATE", "DELETE"):
                    db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} BEFORE {action} ON {table} "
                               "BEGIN SELECT RAISE(ABORT, 'research history is immutable'); END")

    @contextmanager
    def _connection(self):
        snapshot = getattr(self._reads, 'snapshot', None)
        if snapshot is not None:
            yield snapshot
            return
        db = (sqlite3.connect(self.path.as_uri()+'?mode=ro', uri=True, timeout=30)
              if self.read_only else sqlite3.connect(self.path, timeout=30))
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA query_only=ON" if self.read_only else "PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def read_snapshot(self):
        """Pin one read transaction for a complete verified monitoring view."""
        if not self.read_only:
            raise RegistryError('read snapshots require a read-only registry')
        if getattr(self._reads, 'snapshot', None) is not None:
            yield self
            return
        with self._connection() as db:
            db.execute('BEGIN')
            self._reads.snapshot = db
            try:
                yield self
            finally:
                del self._reads.snapshot

    def _require_writable(self):
        if self.read_only:
            raise RegistryError('research registry is read-only')

    @staticmethod
    def _blob(db, payload: bytes) -> str:
        key = content_hash(payload)
        db.execute("INSERT OR IGNORE INTO blobs VALUES (?,?)", (key, payload))
        return key

    def blob(self, key: str) -> bytes:
        with self._connection() as db:
            row = db.execute("SELECT content FROM blobs WHERE hash=?", (key,)).fetchone()
        if row is None or content_hash(bytes(row[0])) != key:
            raise RegistryError("missing or corrupt research artifact")
        return bytes(row[0])

    def archive(self, payload: bytes) -> str:
        """Store immutable provenance/artifact bytes for later reference validation."""
        self._require_writable()
        with self._connection() as db:
            return self._blob(db, payload)

    def archive_contract(self, value: VersionedContract) -> str:
        return self.archive(canonical(value.model_dump(mode="json")))

    def quarantine(self, candidate: Candidate, payload: bytes) -> str:
        self._require_writable()
        if content_hash(payload) != candidate.payload_hash:
            raise RegistryError("candidate payload hash mismatch")
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT hash FROM candidates WHERE id=?", (candidate.candidate_id,)).fetchone()
            if existing:
                if existing[0] != candidate.content_hash():
                    raise RegistryError("candidate ID is immutable")
                return existing[0]
            for parent in candidate.parent_ids:
                if not db.execute("SELECT 1 FROM candidates WHERE id=?", (parent,)).fetchone():
                    raise RegistryError("candidate parent is not registered")
            for key in (*candidate.source_evidence_hashes, candidate.code_hash, candidate.environment_hash):
                if not db.execute("SELECT 1 FROM blobs WHERE hash=?", (key,)).fetchone():
                    raise RegistryError("candidate provenance artifact is missing")
            self._blob(db, payload)
            db.execute("INSERT INTO candidates VALUES (?,?,?)", (candidate.candidate_id,
                       candidate.content_hash(), candidate.model_dump_json()))
        return candidate.content_hash()

    def register_holdout(self, holdout: Holdout) -> None:
        self._require_writable()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT payload FROM holdouts WHERE id=?", (holdout.holdout_id,)).fetchone()
            if old:
                if Holdout.model_validate_json(old[0]) != holdout:
                    raise RegistryError("holdout ID is immutable")
                return
            # Renaming a consumed interval cannot manufacture an untouched holdout.
            for row in db.execute("SELECT payload FROM holdouts"):
                prior = Holdout.model_validate_json(row[0])
                if self._datasets_overlap(db,prior.dataset_hash,holdout.dataset_hash) and max(prior.starts_at, holdout.starts_at) < min(prior.ends_at, holdout.ends_at):
                    raise RegistryError("holdout overlaps a previously registered interval")
            for row in db.execute("SELECT payload FROM trials"):
                trial = Trial.model_validate_json(row[0])
                if self._datasets_overlap(db,trial.dataset_hash,holdout.dataset_hash) and (
                    max(trial.evaluation_start, holdout.starts_at) < min(trial.evaluation_end, holdout.ends_at)
                ):
                    raise RegistryError("holdout was already exposed by a research trial")
            for row in db.execute('SELECT payload FROM discovery_exposures'):
                prior=DiscoveryExposure.model_validate_json(row[0])
                if prior.observation_scope=='unbounded' or (self._datasets_overlap(db,prior.dataset_hash,holdout.dataset_hash) and max(prior.starts_at,holdout.starts_at)<min(prior.ends_at,holdout.ends_at)):
                    raise RegistryError('holdout was already exposed to native discovery')
            if not db.execute("SELECT 1 FROM blobs WHERE hash=?", (holdout.dataset_hash,)).fetchone():
                raise RegistryError("holdout dataset manifest is missing")
            db.execute("INSERT INTO holdouts VALUES (?,?,?)", (holdout.holdout_id, holdout.dataset_hash,
                                                              holdout.model_dump_json()))

    def admit_native_discovery(self, request, bundle_manifest: dict, *, at: datetime,
                               document_manifest: dict | None=None,
                               inherited_exposures: tuple[DiscoveryExposure,...]=()) -> DiscoveryExposure:
        """Reserve the ENTIRE verified mount before execution, never just its split.

        The supervisor verifies actual bundle bytes and rejects unlisted files.
        This boundary independently derives scope from the exact archived bundle
        snapshot and enforces existing/future holdout protection transactionally.
        Repeating an admission is not execution recovery permission.
        """
        self._require_writable()
        from src.domain.market import MarketSnapshot
        snapshot=MarketSnapshot.model_validate(bundle_manifest['snapshot'])
        if not snapshot.bars:
            raise RegistryError('native discovery requires actual mounted observations')
        if snapshot.context.as_of > _utc(at):
            raise RegistryError('native discovery snapshot is later than actual admission')
        bundle=canonical(bundle_manifest)
        if content_hash(bundle)!=request.scenario.bundle_hash:
            raise RegistryError('native discovery manifest differs from requested bundle')
        if (request.document is not None)!=(document_manifest is not None):
            raise RegistryError('native source document manifest is required exactly when mounted')
        for prior in inherited_exposures:
            if self.discovery_exposure(prior.exposure_id)!=prior:
                raise RegistryError('native inherited exposure differs from registry authority')
        unbounded=document_manifest is not None or any(p.observation_scope=='unbounded' for p in inherited_exposures)
        request_bytes=canonical(request.model_dump(mode='json'))
        snapshot_bytes=canonical(snapshot.model_dump(mode='json'))
        exposure=DiscoveryExposure(exposure_id='native:'+request.run_id,
            request_hash=content_hash(request_bytes),bundle_hash=content_hash(bundle),
            dataset_hash=content_hash(snapshot_bytes),
            starts_at=min(bar.opened_at for bar in snapshot.bars),
            ends_at=max(bar.closed_at for bar in snapshot.bars)+timedelta(microseconds=1),
            admitted_at=_utc(at),
            document_manifest_hash=content_hash(canonical(document_manifest)) if document_manifest is not None else None,
            inherited_exposures=tuple(sorted(p.content_hash() for p in inherited_exposures)),
            observation_scope='unbounded' if unbounded else 'market-snapshot')
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM discovery_exposures WHERE id=?',(exposure.exposure_id,)).fetchone():
                raise RegistryError('native discovery already admitted; reconcile the original run instead of resubmitting')
            self._blob(db,request_bytes)
            self._blob(db,bundle)
            self._blob(db,snapshot_bytes)
            if document_manifest is not None:
                self._blob(db,canonical(document_manifest))
                for item in document_manifest['files'].values():
                    if not db.execute('SELECT 1 FROM blobs WHERE hash=?',(item['sha256'],)).fetchone():
                        raise RegistryError('native document bytes were not archived')
            for row in db.execute('SELECT payload FROM holdouts'):
                holdout=Holdout.model_validate_json(row[0])
                if unbounded or (self._datasets_overlap(db,exposure.dataset_hash,holdout.dataset_hash) and max(exposure.starts_at,holdout.starts_at)<min(exposure.ends_at,holdout.ends_at)):
                    raise RegistryError('native mounted data touches a reserved holdout')
            key=self._blob(db,canonical(exposure.model_dump(mode='json')))
            db.execute('INSERT INTO discovery_exposures VALUES (?,?,?)',(exposure.exposure_id,key,exposure.model_dump_json()))
        return exposure

    def discovery_exposure(self, exposure_id: str) -> DiscoveryExposure:
        with self._connection() as db:
            row=db.execute('SELECT hash,payload FROM discovery_exposures WHERE id=?',(exposure_id,)).fetchone()
        if row is None:
            raise RegistryError('unknown native discovery exposure')
        value=DiscoveryExposure.model_validate_json(row[1])
        if value.content_hash()!=row[0] or self.blob(row[0])!=canonical(value.model_dump(mode='json')):
            raise RegistryError('native discovery exposure integrity differs')
        for key in (value.request_hash,value.bundle_hash,value.dataset_hash):
            self.blob(key)
        if value.document_manifest_hash:
            document=json.loads(self.blob(value.document_manifest_hash))
            for item in document['files'].values():
                data=self.blob(item['sha256'])
                if len(data)!=item['bytes']:
                    raise RegistryError('native document size differs')
        for key in value.inherited_exposures:
            prior=DiscoveryExposure.model_validate_json(self.blob(key))
            if self.discovery_exposure(prior.exposure_id)!=prior:
                raise RegistryError('native inherited discovery integrity differs')
        return value

    def record_native_result(self, result: dict) -> str:
        """Coordinator-only completion, bound to the original immutable admission."""
        self._require_writable()
        exposure=self.discovery_exposure('native:'+result['run_id'])
        if (result.get('discovery_exposure_hash')!=exposure.content_hash()
                or result.get('request_hash')!=exposure.request_hash
                or result.get('activation')!='quarantined'
                or result.get('internal_search_trials') is not None):
            raise RegistryError('native result differs from its original discovery admission')
        raw=canonical(result)
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            key=self._blob(db,raw)
            old=db.execute('SELECT hash FROM discovery_results WHERE id=?',(exposure.exposure_id,)).fetchone()
            if old and old[0]!=key:
                raise RegistryError('native discovery result is immutable')
            db.execute('INSERT OR IGNORE INTO discovery_results VALUES (?,?)',(exposure.exposure_id,key))
        return key

    def native_result(self, run_id: str) -> dict:
        exposure=self.discovery_exposure('native:'+run_id)
        with self._connection() as db:
            row=db.execute('SELECT hash FROM discovery_results WHERE id=?',(exposure.exposure_id,)).fetchone()
        if row is None:
            raise RegistryError('native run has no authoritative completion record')
        result=json.loads(self.blob(row[0]))
        if (result.get('run_id')!=run_id or result.get('discovery_exposure_hash')!=exposure.content_hash()
                or result.get('request_hash')!=exposure.request_hash):
            raise RegistryError('native result admission integrity differs')
        return result

    def _native_lifecycle_identity(self, value):
        exposure = self.discovery_exposure('native:' + value.run_id)
        if (value.request_hash != exposure.request_hash
                or value.discovery_exposure_hash != exposure.content_hash()):
            raise RegistryError('native lifecycle identity differs from original admission')
        return exposure

    def _validate_native_launch(self, payload):
        from src.domain.market import MarketSnapshot
        from src.research.native import NativeRequest
        from src.research.native_mounts import canonical_path
        from src.research.native_recovery import NativeLaunch
        value = NativeLaunch.model_validate(payload)
        exposure = self._native_lifecycle_identity(value)
        request = NativeRequest.model_validate_json(self.blob(exposure.request_hash))
        snapshot = MarketSnapshot.model_validate_json(self.blob(exposure.dataset_hash))
        if (value.registry_path != str(self.path) or value.account_id != snapshot.context.account_id
                or value.recorded_at < exposure.admitted_at):
            raise RegistryError('native launch registry/account/clock differs from original admission')
        for path in (value.supervisor_root,value.gateway_token_path,value.registry_path):
            if str(canonical_path(path)) != path:
                raise RegistryError('native launch requires fixed canonical absolute paths')
        files = {(item.tree,item.path): self.blob(item.blob_hash) for item in value.files}
        if any(len(files[item.tree,item.path]) != item.size for item in value.files):
            raise RegistryError('native launch file size differs from archived bytes')
        original = json.loads(files['work','request.json'])
        if original.pop('native_args', request.native_args()) != request.native_args() or NativeRequest.model_validate(original) != request:
            raise RegistryError('native launch request differs from original admission')
        if json.loads(files['work','scenario.json']) != {**request.scenario.model_dump(mode='json'),'provider_uri':'/data/qlib'}:
            raise RegistryError('native launch scenario differs from admitted scenario')
        admission = json.loads(files['work','admission.json'])
        if (admission.get('discovery_exposure_hash') != exposure.content_hash()
                or admission.get('discovery_exposure') != exposure.model_dump(mode='json')
                or admission.get('image_id') != value.image_id or admission.get('network_id') != value.network_id):
            raise RegistryError('native launch preflight or exposure differs')
        if ('frozen','qlib/manifest.json') not in files or json.loads(files['frozen','qlib/manifest.json']) != json.loads(self.blob(exposure.bundle_hash)):
            raise RegistryError('native launch frozen bundle differs from original admission')
        bundle = json.loads(self.blob(exposure.bundle_hash))
        expected = {'qlib/' + name: digest for name,digest in bundle['files'].items()}
        if exposure.document_manifest_hash:
            documents = json.loads(self.blob(exposure.document_manifest_hash))
            expected.update({'sources/' + name: item['sha256'] for name,item in documents['files'].items()})
        for source in request.knowledge:
            prior = self.native_result(source.source_run)
            name = 'knowledge/' + source.artifact + '.pkl'
            if prior['files'].get(name) != source.artifact_hash:
                raise RegistryError('native launch inherited knowledge differs from completion')
            expected[name] = source.artifact_hash
        if request.ui_source_run:
            expected.update({name: digest for name,digest in self.native_result(request.ui_source_run)['files'].items()
                             if name.startswith('log/')})
        actual = {name: content_hash(data) for (tree,name),data in files.items() if tree=='frozen'}
        actual.pop('qlib/manifest.json')
        if actual != expected:
            raise RegistryError('native launch frozen inventory differs from authoritative admitted bytes')
        root = Path(value.supervisor_root)
        work, frozen = root/request.run_id, root/'.inputs'/request.run_id
        mounts = {(str(work),'/work',False),(str(frozen/'qlib'),'/data/qlib',True),
                  (value.gateway_token_path,'/run/gateway-token',True)}
        mounts.update((str(work/name),'/work/'+name,True) for name in ('request.json','scenario.json','admission.json'))
        if request.document:
            mounts.add((str(frozen/'sources'),'/sources',True))
        if request.knowledge:
            mounts.add((str(frozen/'knowledge'),'/work/knowledge-input',True))
        if request.ui_source_run:
            mounts.add((str(frozen/'log'),'/work/log',True))
        if {(m.source,m.destination,m.read_only) for m in value.mounts} != mounts:
            raise RegistryError('native launch mounts differ from original admitted paths')
        return value

    def record_native_launch(self, payload: dict) -> str:
        self._require_writable()
        value = self._validate_native_launch(payload)
        return self._record_native_once('native_launches',value)

    def _record_native_once(self, table, value):
        if table not in ('native_launches','native_containers'):
            raise RegistryError('unsupported native lifecycle table')
        raw = canonical(value.model_dump(mode='json'))
        key = content_hash(raw)
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute(f'SELECT hash FROM {table} WHERE id=?',(value.run_id,)).fetchone()
            if old and old[0] != key:
                raise RegistryError('native launch/container identity is immutable')
            self._blob(db,raw)
            db.execute(f'INSERT OR IGNORE INTO {table} VALUES (?,?)',(value.run_id,key))
        return key

    def native_launch(self, run_id: str) -> dict | None:
        self.discovery_exposure('native:' + run_id)
        with self._connection() as db:
            row = db.execute('SELECT hash FROM native_launches WHERE id=?',(run_id,)).fetchone()
        if row is None:
            return None
        value = self._validate_native_launch(json.loads(self.blob(row[0])))
        if value.run_id != run_id or value.content_hash() != row[0]:
            raise RegistryError('native launch identity differs')
        return value.model_dump(mode='json')

    def _validate_native_container(self, payload):
        from src.research.native_recovery import NativeContainerBinding, NativeLaunch
        value = NativeContainerBinding.model_validate(payload)
        self._native_lifecycle_identity(value)
        launch_payload = self.native_launch(value.run_id)
        if launch_payload is None:
            raise RegistryError('native container has no original launch authority')
        launch = NativeLaunch.model_validate(launch_payload)
        if (value.launch_hash != launch.content_hash() or value.created_at < launch.recorded_at
                or value.bound_at < value.created_at):
            raise RegistryError('native container identity/clock differs from original launch')
        identity = json.loads(self.blob(value.identity_hash))
        expected = {'container_id':value.container_id, 'created_at':value.created_at.isoformat(),
                    'launch_hash':launch.content_hash(), 'image_id':launch.image_id,
                    'network_id':launch.network_id, 'name':'/quantagent-rd-'+value.run_id}
        if any(identity.get(key) != actual for key,actual in expected.items()):
            raise RegistryError('native container projection differs from original binding')
        if identity.get('mounts') != [m.model_dump(mode='json') for m in launch.mounts]:
            raise RegistryError('native container mount projection differs')
        return value

    def record_native_container(self, payload: dict) -> str:
        self._require_writable()
        value = self._validate_native_container(payload)
        return self._record_native_once('native_containers',value)

    def native_container(self, run_id: str) -> dict | None:
        self.discovery_exposure('native:' + run_id)
        with self._connection() as db:
            row = db.execute('SELECT hash FROM native_containers WHERE id=?',(run_id,)).fetchone()
        if row is None:
            return None
        value = self._validate_native_container(json.loads(self.blob(row[0])))
        if value.run_id != run_id or value.content_hash() != row[0]:
            raise RegistryError('native container binding identity differs')
        return value.model_dump(mode='json')

    def _validate_native_recovery(self, payload, *, recording=False):
        from src.research.native_recovery import (
            NativeContainerBinding,
            NativeLaunch,
            NativeRecoveryEvent,
        )
        value = NativeRecoveryEvent.model_validate(payload)
        exposure = self._native_lifecycle_identity(value)
        if value.observed_at < exposure.admitted_at:
            raise RegistryError('native recovery precedes original admission')
        for name, reader, model, clock in (
            ('launch_hash',self.native_launch,NativeLaunch,'recorded_at'),
            ('container_binding_hash',self.native_container,NativeContainerBinding,'bound_at'),
        ):
            authority = reader(value.run_id)
            original = model.model_validate(authority) if authority is not None else None
            if not recording and original is not None and getattr(original,clock) > value.observed_at:
                original = None
            if (getattr(value,name) != (original.content_hash() if original else None)
                    or (original is not None and value.observed_at < getattr(original,clock))):
                raise RegistryError('native recovery authority/clock differs')
        for name in ('inspection_hash','artifact_manifest_hash','gateway_receipt_hash','result_hash'):
            key = getattr(value,name)
            if key is not None:
                self.blob(key)
        if value.result_hash is not None and value.result_hash != content_hash(canonical(self.native_result(value.run_id))):
            raise RegistryError('native recovery result is not the authoritative completion')
        return value

    def append_native_recovery(self, payload: dict) -> str:
        self._require_writable()
        from src.research.native_recovery import NativeRecoveryEvent
        value = self._validate_native_recovery(payload, recording=True)
        raw = canonical(value.model_dump(mode='json'))
        key = content_hash(raw)
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT sequence,hash FROM native_recoveries WHERE id=? ORDER BY sequence DESC LIMIT 1',(value.run_id,)).fetchone()
            if old and old[0] == value.sequence and old[1] == key:
                return key
            if (value.sequence != (old[0]+1 if old else 1)
                    or value.previous_hash != (old[1] if old else None)):
                raise RegistryError('native recovery sequence changed concurrently')
            if old:
                previous_raw = db.execute('SELECT content FROM blobs WHERE hash=?',(old[1],)).fetchone()
                if previous_raw is None or content_hash(bytes(previous_raw[0])) != old[1]:
                    raise RegistryError('native recovery predecessor integrity differs')
                previous = NativeRecoveryEvent.model_validate_json(previous_raw[0])
                if previous.observed_at > value.observed_at:
                    raise RegistryError('native recovery observation clock moved backwards')
            self._blob(db,raw)
            db.execute('INSERT INTO native_recoveries VALUES (?,?,?)',(value.run_id,value.sequence,key))
        return key

    def native_recoveries(self, run_id: str) -> tuple[dict, ...]:
        self.discovery_exposure('native:' + run_id)
        with self._connection() as db:
            rows = db.execute('SELECT sequence,hash FROM native_recoveries WHERE id=? ORDER BY sequence',(run_id,)).fetchall()
        values = []
        previous_hash, previous_at = None, None
        for sequence,key in rows:
            value = self._validate_native_recovery(json.loads(self.blob(key)))
            if (value.run_id != run_id or value.sequence != len(values)+1 or sequence != value.sequence
                    or value.previous_hash != previous_hash or value.content_hash() != key
                    or (previous_at is not None and previous_at > value.observed_at)):
                raise RegistryError('native recovery history integrity differs')
            values.append(value.model_dump(mode='json'))
            previous_hash, previous_at = key, value.observed_at
        return tuple(values)

    @staticmethod
    def _datasets_overlap(db, first: str, second: str) -> bool:
        if first == second:
            return True
        scopes = []
        for key in (first,second):
            row = db.execute('SELECT content FROM blobs WHERE hash=?',(key,)).fetchone()
            if row is None:
                raise RegistryError('dataset artifact is missing')
            try:
                data = json.loads(row[0])
                scope = {bar['instrument']['instrument_id'] for bar in data.get('bars',[])}
            except (ValueError,TypeError,AttributeError,KeyError):
                scope = set()
            scopes.append(scope)
        # Changing run context, source revision or feature recipe cannot make
        # the same security/time interval untouched. Opaque non-market inputs
        # still require exact content identity and their own domain verifier.
        return bool(scopes[0] & scopes[1])

    def register_trial(self, trial: Trial) -> str:
        """Register before execution; failed and incomplete trials still count."""
        self._require_writable()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT payload FROM trials WHERE id=?", (trial.trial_id,)).fetchone()
            if old:
                if Trial.model_validate_json(old[0]) != trial:
                    raise RegistryError("trial ID is immutable")
                return db.execute("SELECT hash FROM trial_snapshots WHERE trial=?", (trial.trial_id,)).fetchone()[0]
            row = db.execute("SELECT hash,payload FROM candidates WHERE id=?", (trial.candidate_id,)).fetchone()
            if row is None:
                raise RegistryError("candidate must enter quarantine before an experiment")
            candidate = Candidate.model_validate_json(row[1])
            if trial.registered_at < candidate.created_at:
                raise RegistryError("trial precedes candidate registration")
            for key in (trial.dataset_hash, trial.split_hash, trial.inputs_hash, trial.policy_hash):
                if not db.execute("SELECT 1 FROM blobs WHERE hash=?", (key,)).fetchone():
                    raise RegistryError("trial provenance artifact is missing")
            if trial.holdout_id:
                hrow = db.execute("SELECT payload FROM holdouts WHERE id=?", (trial.holdout_id,)).fetchone()
                if not hrow:
                    raise RegistryError("holdout must be preregistered")
                holdout = Holdout.model_validate_json(hrow[0])
                if holdout.dataset_hash != trial.dataset_hash or holdout.registered_at > trial.registered_at:
                    raise RegistryError("holdout dataset/time mismatch")
                if (trial.evaluation_start, trial.evaluation_end) != (holdout.starts_at, holdout.ends_at):
                    raise RegistryError("evaluation must cover the exact preregistered holdout")
                if db.execute("SELECT 1 FROM trials WHERE holdout=?", (trial.holdout_id,)).fetchone():
                    raise RegistryError("holdout has already been consumed by an attempted trial")
            else:
                for hrow in db.execute("SELECT payload FROM holdouts"):
                    holdout = Holdout.model_validate_json(hrow[0])
                    if self._datasets_overlap(db,trial.dataset_hash,holdout.dataset_hash) and max(trial.evaluation_start, holdout.starts_at) < min(trial.evaluation_end, holdout.ends_at):
                        raise RegistryError("development trial touches a reserved holdout")
            db.execute("INSERT INTO trials VALUES (?,?,?,?,?)", (trial.trial_id, trial.candidate_id,
                                                               trial.family_id, trial.holdout_id, trial.model_dump_json()))
            # Snapshot at admission is immutable. Later trials can only increase
            # the multiplicity penalty checked again at activation.
            count = db.execute("SELECT count(*) FROM trials").fetchone()[0]
            snapshot = canonical({"trial": trial.model_dump(mode="json"), "candidate_hash": row[0],
                                  "family_trials_at_admission": count})
            key = self._blob(db, snapshot)
            db.execute("INSERT INTO trial_snapshots VALUES (?,?)", (trial.trial_id, key))
            return key

    def trial(self, trial_id: str) -> Trial:
        with self._connection() as db:
            row = db.execute("SELECT payload,candidate,family,holdout FROM trials WHERE id=?", (trial_id,)).fetchone()
        if row is None:
            raise RegistryError("unknown trial")
        value = Trial.model_validate_json(row[0])
        snapshot = json.loads(self.blob(self.trial_snapshot(trial_id)))
        if (value.trial_id != trial_id or (value.candidate_id,value.family_id,value.holdout_id) != row[1:]
                or snapshot.get('trial') != value.model_dump(mode='json')
                or snapshot.get('candidate_hash') != self.candidate(value.candidate_id).content_hash()):
            raise RegistryError('trial differs from immutable admission')
        return value

    def candidate(self, candidate_id: str) -> Candidate:
        with self._connection() as db:
            row = db.execute("SELECT payload,hash FROM candidates WHERE id=?", (candidate_id,)).fetchone()
        if row is None:
            raise RegistryError("unknown candidate")
        value = Candidate.model_validate_json(row[0])
        if value.candidate_id != candidate_id or value.content_hash() != row[1]:
            raise RegistryError('candidate identity differs')
        return value

    def trial_snapshot(self, trial_id: str) -> str:
        with self._connection() as db:
            row = db.execute("SELECT hash FROM trial_snapshots WHERE trial=?", (trial_id,)).fetchone()
        if row is None:
            raise RegistryError("unknown trial")
        return row[0]

    def family_count(self, family_id: str) -> int:
        """Conservative registry-wide multiplicity; renaming a family cannot reset it."""
        with self._connection() as db:
            return db.execute("SELECT count(*) FROM trials").fetchone()[0]

    def trial_count_at_admission(self, trial_id: str) -> int:
        return int(json.loads(self.blob(self.trial_snapshot(trial_id)))['family_trials_at_admission'])

    def record_evaluation(self, evaluation: Evaluation) -> None:
        """Persist independently computed outcome, with exact referenced evidence."""
        self._require_writable()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT hash FROM outcomes WHERE trial=?", (evaluation.trial_id,)).fetchone()
            if old:
                if old[0] != evaluation.content_hash():
                    raise RegistryError("trial outcome is immutable")
                return
            row = db.execute("SELECT payload FROM trials WHERE id=?", (evaluation.trial_id,)).fetchone()
            if row is None:
                raise RegistryError("unregistered trial cannot report an outcome")
            trial = Trial.model_validate_json(row[0])
            candidate_hash = db.execute("SELECT hash FROM candidates WHERE id=?", (trial.candidate_id,)).fetchone()[0]
            snapshot_hash = db.execute("SELECT hash FROM trial_snapshots WHERE trial=?", (trial.trial_id,)).fetchone()[0]
            if (evaluation.candidate_hash != candidate_hash or evaluation.inputs_hash != trial.inputs_hash
                    or evaluation.registry_snapshot_hash != snapshot_hash or evaluation.completed_at < trial.registered_at):
                raise RegistryError("evaluation lineage/time mismatch")
            for key in (evaluation.replay_hash, evaluation.manifest_hash, evaluation.validation_hash,
                        *evaluation.actual_model_hashes, *evaluation.actual_decision_hashes,
                        *(key for key in (evaluation.regime_evaluation_hash, evaluation.execution_verification_hash,
                                          evaluation.paired_evaluation_hash) if key)):
                if not db.execute("SELECT 1 FROM blobs WHERE hash=?", (key,)).fetchone():
                    raise RegistryError("evaluation evidence/model artifact is missing")
            db.execute("INSERT INTO outcomes VALUES (?,?,?)", (evaluation.trial_id, evaluation.content_hash(),
                                                             evaluation.model_dump_json()))

    def development_evaluation(self, trial_id: str) -> Evaluation:
        """Read a verified development outcome without exposing held-out results."""
        trial = self.trial(trial_id)
        if trial.purpose != 'development':
            raise RegistryError('research memos cannot read holdout outcomes')
        with self._connection() as db:
            row = db.execute('SELECT hash,payload FROM outcomes WHERE trial=?',(trial_id,)).fetchone()
        if row is None:
            raise MissingDevelopmentEvaluation('development trial has no recorded outcome')
        value = Evaluation.model_validate_json(row[1])
        candidate = self.candidate(trial.candidate_id)
        if (value.content_hash() != row[0] or value.trial_id != trial_id
                or value.candidate_hash != candidate.content_hash()
                or value.inputs_hash != trial.inputs_hash
                or value.registry_snapshot_hash != self.trial_snapshot(trial_id)
                or value.completed_at < trial.registered_at):
            raise RegistryError('development outcome identity or chronology differs')
        for key in (candidate.payload_hash,candidate.code_hash,candidate.environment_hash,
                    value.inputs_hash,value.registry_snapshot_hash,value.replay_hash,
                    value.manifest_hash,value.validation_hash,*value.actual_model_hashes,*value.actual_decision_hashes,
                    *(key for key in (value.regime_evaluation_hash,value.execution_verification_hash,
                                      value.paired_evaluation_hash) if key)):
            self.blob(key)
        return value

    @staticmethod
    def _require_audited_origin(db,candidate):
        pending=[candidate]
        seen=set()
        while pending:
            item=pending.pop()
            if item.candidate_id in seen:
                continue
            seen.add(item.candidate_id)
            if item.origin=='native_rd_agent':
                raise RegistryError('native discovery multiplicity and exact implementation audit are unverified; candidate remains quarantined')
            for parent in item.parent_ids:
                record=db.execute('SELECT payload FROM candidates WHERE id=?',(parent,)).fetchone()
                if record is None:
                    raise RegistryError('candidate lineage is incomplete')
                pending.append(Candidate.model_validate_json(record[0]))

    def _eligible(self, db, trial_id: str, policy: PromotionPolicy):
        trow = db.execute("SELECT payload FROM trials WHERE id=?", (trial_id,)).fetchone()
        erow = db.execute("SELECT payload FROM outcomes WHERE trial=?", (trial_id,)).fetchone()
        if trow is None or erow is None:
            raise RegistryError("completed registered evaluation required")
        trial, evaluation = Trial.model_validate_json(trow[0]), Evaluation.model_validate_json(erow[0])
        crow = db.execute("SELECT payload FROM candidates WHERE id=?", (trial.candidate_id,)).fetchone()
        candidate = Candidate.model_validate_json(crow[0])
        self._require_audited_origin(db,candidate)
        if trial.purpose != "holdout" or trial.policy_hash != policy.content_hash():
            raise RegistryError("activation requires the preregistered holdout policy")
        if candidate.synthetic or evaluation.synthetic or evaluation.status != "complete" or not evaluation.validation_reliable:
            raise RegistryError("synthetic, incomplete or unreliable research cannot activate")
        if not (evaluation.actual_model_hashes or (candidate.kind == 'strategy' and evaluation.actual_decision_hashes)) or evaluation.observations < policy.min_observations:
            raise RegistryError("actual model or verified strategy decision artifacts and sufficient observations required")
        if candidate.kind in ('strategy','regime') and evaluation.execution_verification_hash is None:
            raise RegistryError('strategy/regime activation requires independent recipe reproduction')
        if candidate.kind == 'regime':
            if evaluation.paired_evaluation_hash is None:
                raise RegistryError('regime activation requires same-trial paired baseline evidence')
            from src.research.execution_proof import verified_paired_returns
            if sum(verified_paired_returns(self,evaluation.paired_evaluation_hash)) <= 0:
                raise RegistryError('regime candidate does not improve its preregistered baseline')
        if evaluation.sharpe is None or evaluation.sharpe <= policy.min_sharpe:
            raise RegistryError("research Sharpe gate failed")
        if candidate.kind in ("factor", "semantic_question") and (
            evaluation.ic is None or evaluation.ic <= policy.min_ic
            or evaluation.icir is None or evaluation.icir <= policy.min_icir
            or evaluation.max_factor_correlation is None
            or abs(evaluation.max_factor_correlation) >= policy.max_factor_correlation
        ):
            raise RegistryError("factor IC/ICIR/correlation gate failed")
        trials = db.execute("SELECT count(*) FROM trials").fetchone()[0]
        if evaluation.unadjusted_p_value is None or evaluation.unadjusted_p_value > policy.familywise_alpha / trials:
            raise RegistryError("multiple-trial significance gate failed")
        return candidate, evaluation

    def activate(self, slot: str, trial_id: str, policy: PromotionPolicy, *, expected_generation: int,
                 at: datetime) -> Activation:
        self._require_writable()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            candidate, evaluation = self._eligible(db, trial_id, policy)
            if _utc(at) < evaluation.completed_at:
                raise RegistryError("activation precedes completed evaluation")
            return self._activate(db, slot, candidate, evaluation.content_hash(), policy.content_hash(),
                                  expected_generation, at, "promote")

    def _activate(self, db, slot, candidate, evaluation_hash, policy_hash, expected, at, action):
        row = db.execute("SELECT generation FROM active WHERE slot=?", (slot,)).fetchone()
        generation = row[0] if row else 0
        if generation != expected:
            raise RegistryError("active manifest changed; reload before promotion/rollback")
        at = _utc(at)
        if generation:
            prior = Activation.model_validate_json(db.execute("SELECT payload FROM activations WHERE slot=? AND generation=?",
                                                             (slot, generation)).fetchone()[0])
            if at < prior.activated_at:
                raise RegistryError("activation time cannot move backwards")
        activation = Activation(slot=slot, generation=generation + 1, candidate_id=candidate.candidate_id,
                                candidate_hash=candidate.content_hash(), payload_hash=candidate.payload_hash,
                                evaluation_hash=evaluation_hash, policy_hash=policy_hash, activated_at=at,
                                action=action, previous_generation=generation)
        db.execute("INSERT INTO activations VALUES (?,?,?)", (slot, activation.generation, activation.model_dump_json()))
        db.execute("INSERT INTO active VALUES (?,?) ON CONFLICT(slot) DO UPDATE SET generation=excluded.generation",
                   (slot, activation.generation))
        return activation

    def rollback(self, slot: str, to_generation: int, *, expected_generation: int, at: datetime) -> Activation:
        self._require_writable()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM activations WHERE slot=? AND generation=?", (slot, to_generation)).fetchone()
            if row is None or to_generation >= expected_generation:
                raise RegistryError("rollback must name a prior verified activation")
            prior = Activation.model_validate_json(row[0])
            candidate = Candidate.model_validate_json(db.execute("SELECT payload FROM candidates WHERE id=?",
                                                                  (prior.candidate_id,)).fetchone()[0])
            self._require_audited_origin(db,candidate)
            return self._activate(db, slot, candidate, prior.evaluation_hash, prior.policy_hash,
                                  expected_generation, at, "rollback")

    def active(self, slot: str, *, as_of: datetime | None = None) -> Activation | None:
        """Pin a complete generation at run start, optionally at a historical cutoff."""
        with self._connection() as db:
            rows = db.execute("SELECT payload FROM activations WHERE slot=? ORDER BY generation DESC", (slot,)).fetchall()
        cutoff = _utc(as_of) if as_of is not None else None
        for (payload,) in rows:
            activation = Activation.model_validate_json(payload)
            if cutoff is None or activation.activated_at <= cutoff:
                self.blob(activation.payload_hash)
                with self._connection() as db:
                    self._require_audited_origin(db,self.candidate(activation.candidate_id))
                return activation
        return None

    def resolve(self, slot: str, *, payload_hash: str, as_of: datetime) -> Activation | None:
        """Resolve an exact candidate for operation at a historical cutoff.

        A YAML flag cannot grant admission. Both the pinned artifact and its
        recorded evaluation must still be present and internally consistent.
        """
        activation = self.active(slot, as_of=as_of)
        if activation is None or activation.payload_hash != payload_hash:
            return None
        with self._connection() as db:
            row = db.execute("SELECT payload FROM outcomes WHERE hash=?", (activation.evaluation_hash,)).fetchone()
            if row is None:
                raise RegistryError("active evaluation is absent")
            evaluation = Evaluation.model_validate_json(row[0])
            candidate = Candidate.model_validate_json(db.execute("SELECT payload FROM candidates WHERE id=?",
                                                                (activation.candidate_id,)).fetchone()[0])
        if (evaluation.content_hash() != activation.evaluation_hash or candidate.content_hash() != activation.candidate_hash
                or evaluation.candidate_hash != activation.candidate_hash or candidate.synthetic or evaluation.synthetic
                or evaluation.status != "complete" or not evaluation.validation_reliable
                or evaluation.completed_at > activation.activated_at):
            raise RegistryError("active candidate/evaluation identity or eligibility mismatch")
        for key in (candidate.code_hash, candidate.environment_hash, candidate.payload_hash,
                    activation.policy_hash, evaluation.inputs_hash, evaluation.replay_hash,
                    evaluation.manifest_hash, evaluation.validation_hash, *evaluation.actual_model_hashes,
                    *evaluation.actual_decision_hashes,
                    *(key for key in (evaluation.regime_evaluation_hash,evaluation.execution_verification_hash,
                                      evaluation.paired_evaluation_hash) if key)):
            self.blob(key)
        return activation

    def status(self) -> dict:
        with self._connection() as db:
            counts = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                      for table in ("candidates", "trials", "holdouts", "outcomes")}
            discoveries=db.execute('SELECT count(*) FROM discovery_exposures').fetchone()[0]
            active,blocked=[],[]
            for row in db.execute("SELECT a.payload FROM activations a JOIN active p ON a.slot=p.slot AND a.generation=p.generation"):
                value=Activation.model_validate_json(row[0])
                candidate=Candidate.model_validate_json(db.execute('SELECT payload FROM candidates WHERE id=?',(value.candidate_id,)).fetchone()[0])
                try:
                    self._require_audited_origin(db,candidate)
                except RegistryError as exc:
                    blocked.append({'activation':value.model_dump(mode='json'),'reason':str(exc)})
                else:
                    active.append(value.model_dump(mode='json'))
        return {"counts": counts, "active": active, "live_capital_authorized": False,
                "blocked_activations":blocked,
                "native_discovery": {"admissions":discoveries,"internal_trials":None}}
