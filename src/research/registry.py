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
from datetime import datetime
from pathlib import Path
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


class ResearchRegistry:
    """Append-only artifacts and atomic active generations on one durable path."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
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
            """)
            for table in ("blobs", "candidates", "holdouts", "trials", "outcomes", "trial_snapshots", "activations"):
                for action in ("UPDATE", "DELETE"):
                    db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} BEFORE {action} ON {table} "
                               "BEGIN SELECT RAISE(ABORT, 'research history is immutable'); END")

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

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
        with self._connection() as db:
            return self._blob(db, payload)

    def archive_contract(self, value: VersionedContract) -> str:
        return self.archive(canonical(value.model_dump(mode="json")))

    def quarantine(self, candidate: Candidate, payload: bytes) -> str:
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
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT payload FROM holdouts WHERE id=?", (holdout.holdout_id,)).fetchone()
            if old:
                if Holdout.model_validate_json(old[0]) != holdout:
                    raise RegistryError("holdout ID is immutable")
                return
            # Renaming a consumed interval cannot manufacture an untouched holdout.
            for row in db.execute("SELECT payload FROM holdouts WHERE dataset=?", (holdout.dataset_hash,)):
                prior = Holdout.model_validate_json(row[0])
                if max(prior.starts_at, holdout.starts_at) < min(prior.ends_at, holdout.ends_at):
                    raise RegistryError("holdout overlaps a previously registered interval")
            for row in db.execute("SELECT payload FROM trials"):
                trial = Trial.model_validate_json(row[0])
                if trial.dataset_hash == holdout.dataset_hash and (
                    max(trial.evaluation_start, holdout.starts_at) < min(trial.evaluation_end, holdout.ends_at)
                ):
                    raise RegistryError("holdout was already exposed by a research trial")
            if not db.execute("SELECT 1 FROM blobs WHERE hash=?", (holdout.dataset_hash,)).fetchone():
                raise RegistryError("holdout dataset manifest is missing")
            db.execute("INSERT INTO holdouts VALUES (?,?,?)", (holdout.holdout_id, holdout.dataset_hash,
                                                              holdout.model_dump_json()))

    def register_trial(self, trial: Trial) -> str:
        """Register before execution; failed and incomplete trials still count."""
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
                for hrow in db.execute("SELECT payload FROM holdouts WHERE dataset=?", (trial.dataset_hash,)):
                    holdout = Holdout.model_validate_json(hrow[0])
                    if max(trial.evaluation_start, holdout.starts_at) < min(trial.evaluation_end, holdout.ends_at):
                        raise RegistryError("development trial touches a reserved holdout")
            db.execute("INSERT INTO trials VALUES (?,?,?,?,?)", (trial.trial_id, trial.candidate_id,
                                                               trial.family_id, trial.holdout_id, trial.model_dump_json()))
            # Snapshot at admission is immutable. Later trials can only increase
            # the multiplicity penalty checked again at activation.
            count = db.execute("SELECT count(*) FROM trials WHERE family=?", (trial.family_id,)).fetchone()[0]
            snapshot = canonical({"trial": trial.model_dump(mode="json"), "candidate_hash": row[0],
                                  "family_trials_at_admission": count})
            key = self._blob(db, snapshot)
            db.execute("INSERT INTO trial_snapshots VALUES (?,?)", (trial.trial_id, key))
            return key

    def trial(self, trial_id: str) -> Trial:
        with self._connection() as db:
            row = db.execute("SELECT payload FROM trials WHERE id=?", (trial_id,)).fetchone()
        if row is None:
            raise RegistryError("unknown trial")
        return Trial.model_validate_json(row[0])

    def candidate(self, candidate_id: str) -> Candidate:
        with self._connection() as db:
            row = db.execute("SELECT payload FROM candidates WHERE id=?", (candidate_id,)).fetchone()
        if row is None:
            raise RegistryError("unknown candidate")
        return Candidate.model_validate_json(row[0])

    def trial_snapshot(self, trial_id: str) -> str:
        with self._connection() as db:
            row = db.execute("SELECT hash FROM trial_snapshots WHERE trial=?", (trial_id,)).fetchone()
        if row is None:
            raise RegistryError("unknown trial")
        return row[0]

    def family_count(self, family_id: str) -> int:
        with self._connection() as db:
            return db.execute("SELECT count(*) FROM trials WHERE family=?", (family_id,)).fetchone()[0]

    def record_evaluation(self, evaluation: Evaluation) -> None:
        """Persist independently computed outcome, with exact referenced evidence."""
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
                        *evaluation.actual_model_hashes):
                if not db.execute("SELECT 1 FROM blobs WHERE hash=?", (key,)).fetchone():
                    raise RegistryError("evaluation evidence/model artifact is missing")
            db.execute("INSERT INTO outcomes VALUES (?,?,?)", (evaluation.trial_id, evaluation.content_hash(),
                                                             evaluation.model_dump_json()))

    def _eligible(self, db, trial_id: str, policy: PromotionPolicy):
        trow = db.execute("SELECT payload FROM trials WHERE id=?", (trial_id,)).fetchone()
        erow = db.execute("SELECT payload FROM outcomes WHERE trial=?", (trial_id,)).fetchone()
        if trow is None or erow is None:
            raise RegistryError("completed registered evaluation required")
        trial, evaluation = Trial.model_validate_json(trow[0]), Evaluation.model_validate_json(erow[0])
        crow = db.execute("SELECT payload FROM candidates WHERE id=?", (trial.candidate_id,)).fetchone()
        candidate = Candidate.model_validate_json(crow[0])
        if trial.purpose != "holdout" or trial.policy_hash != policy.content_hash():
            raise RegistryError("activation requires the preregistered holdout policy")
        if candidate.synthetic or evaluation.synthetic or evaluation.status != "complete" or not evaluation.validation_reliable:
            raise RegistryError("synthetic, incomplete or unreliable research cannot activate")
        if not evaluation.actual_model_hashes or evaluation.observations < policy.min_observations:
            raise RegistryError("actual model artifacts and sufficient observations required")
        if evaluation.sharpe is None or evaluation.sharpe <= policy.min_sharpe:
            raise RegistryError("research Sharpe gate failed")
        if candidate.kind in ("factor", "semantic_question") and (
            evaluation.ic is None or evaluation.ic <= policy.min_ic
            or evaluation.icir is None or evaluation.icir <= policy.min_icir
            or evaluation.max_factor_correlation is None
            or abs(evaluation.max_factor_correlation) >= policy.max_factor_correlation
        ):
            raise RegistryError("factor IC/ICIR/correlation gate failed")
        trials = db.execute("SELECT count(*) FROM trials WHERE family=?", (trial.family_id,)).fetchone()[0]
        if evaluation.unadjusted_p_value is None or evaluation.unadjusted_p_value > policy.familywise_alpha / trials:
            raise RegistryError("multiple-trial significance gate failed")
        return candidate, evaluation

    def activate(self, slot: str, trial_id: str, policy: PromotionPolicy, *, expected_generation: int,
                 at: datetime) -> Activation:
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
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM activations WHERE slot=? AND generation=?", (slot, to_generation)).fetchone()
            if row is None or to_generation >= expected_generation:
                raise RegistryError("rollback must name a prior verified activation")
            prior = Activation.model_validate_json(row[0])
            candidate = Candidate.model_validate_json(db.execute("SELECT payload FROM candidates WHERE id=?",
                                                                  (prior.candidate_id,)).fetchone()[0])
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
                    evaluation.manifest_hash, evaluation.validation_hash, *evaluation.actual_model_hashes):
            self.blob(key)
        return activation

    def status(self) -> dict:
        with self._connection() as db:
            counts = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                      for table in ("candidates", "trials", "holdouts", "outcomes")}
            active = [Activation.model_validate_json(row[0]).model_dump(mode="json") for row in db.execute(
                "SELECT a.payload FROM activations a JOIN active p ON a.slot=p.slot AND a.generation=p.generation")]
        return {"counts": counts, "active": active, "live_capital_authorized": False}
