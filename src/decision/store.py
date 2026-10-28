"""Append-only decision observations and transactional operating reservations.

This stores model observations, not source documents (owned by EvidenceStore).
Usage reservations remain charged after uncertain failures: an interrupted HTTP
request may have been processed. Run/day limits therefore survive restarts.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.domain.decisions import DecisionRequest, DecisionResult


def _evaluation_types():
    from src.decision.evaluation import EVALUATION_TYPES
    from src.decision.experiments import EXPERIMENT_TYPES

    return {**EVALUATION_TYPES, **EXPERIMENT_TYPES}


class DecisionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS observations (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, observation_id TEXT UNIQUE NOT NULL,
                    cache_key TEXT NOT NULL, request_json TEXT NOT NULL,
                    result_json TEXT NOT NULL, status TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS observation_key ON observations(cache_key, seq);
                CREATE TRIGGER IF NOT EXISTS observation_immutable_update BEFORE UPDATE
                    ON observations BEGIN SELECT RAISE(ABORT, 'immutable observation'); END;
                CREATE TRIGGER IF NOT EXISTS observation_immutable_delete BEFORE DELETE
                    ON observations BEGIN SELECT RAISE(ABORT, 'immutable observation'); END;
                CREATE TABLE IF NOT EXISTS reservations (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    day TEXT NOT NULL, reserved_tokens INTEGER NOT NULL,
                    reserved_cost REAL, at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, observation_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL, status INTEGER, body TEXT,
                    elapsed_ms REAL NOT NULL, failure TEXT);
                CREATE TRIGGER IF NOT EXISTS attempt_immutable_update BEFORE UPDATE
                    ON attempts BEGIN SELECT RAISE(ABORT, 'immutable attempt'); END;
                CREATE TRIGGER IF NOT EXISTS attempt_immutable_delete BEFORE DELETE
                    ON attempts BEGIN SELECT RAISE(ABORT, 'immutable attempt'); END;
                CREATE TABLE IF NOT EXISTS evaluation_artifacts (
                    identity TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS evaluation_immutable_update BEFORE UPDATE
                    ON evaluation_artifacts BEGIN SELECT RAISE(ABORT, 'immutable evaluation'); END;
                CREATE TRIGGER IF NOT EXISTS evaluation_immutable_delete BEFORE DELETE
                    ON evaluation_artifacts BEGIN SELECT RAISE(ABORT, 'immutable evaluation'); END;
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY, manifest_hash TEXT UNIQUE NOT NULL,
                    frozen_id TEXT UNIQUE NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluation_dispatches (
                    cache_key TEXT PRIMARY KEY, artifact_id TEXT UNIQUE NOT NULL,
                    observation_id TEXT UNIQUE NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluation_dispatch_starts (
                    artifact_id TEXT PRIMARY KEY, started_at TEXT NOT NULL);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(reservations)")}
            for name, kind in (("observation_id", "TEXT"), ("attempt", "INTEGER")):
                if name not in columns:
                    db.execute(f"ALTER TABLE reservations ADD COLUMN {name} {kind}")
            for action in ("UPDATE", "DELETE"):
                db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS reservation_{action.lower()} BEFORE {action} ON reservations BEGIN SELECT RAISE(ABORT, 'immutable reservation'); END"
                )
                for table in (
                    "evaluation_runs",
                    "evaluation_dispatches",
                    "evaluation_dispatch_starts",
                ):
                    db.execute(
                        f"CREATE TRIGGER IF NOT EXISTS {table}_{action.lower()} BEFORE {action} "
                        f"ON {table} BEGIN SELECT RAISE(ABORT, 'immutable evaluation admission'); END"
                    )

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def _append_evaluation(self, value):
        """Owned EvaluationService write boundary; never a supplied pass flag."""
        kind = type(value).__name__
        if _evaluation_types().get(kind) is not type(value):
            raise ValueError("unknown evaluation contract")
        value = type(value).model_validate(value.model_dump())
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO evaluation_artifacts VALUES(?,?,?)",
                (value.content_hash(), kind, value.model_dump_json()),
            )
        return value.content_hash()

    def evaluation_artifact(self, identity, expected_type):
        """Read an immutable typed record; semantic proof is resolved by service."""
        if _evaluation_types().get(expected_type.__name__) is not expected_type:
            raise ValueError("unknown evaluation contract")
        with self.connect() as db:
            row = db.execute(
                "SELECT kind,payload FROM evaluation_artifacts WHERE identity=?", (identity,)
            ).fetchone()
        if row is None or row[0] != expected_type.__name__:
            raise ValueError("missing authoritative evaluation record")
        value = expected_type.model_validate_json(row[1])
        if value.content_hash() != identity:
            raise ValueError("evaluation record hash differs")
        return value

    def experiment_for_manifest(self, manifest_hash):
        with self.connect() as db:
            row = db.execute(
                "SELECT frozen_id FROM evaluation_runs WHERE manifest_hash=?", (manifest_hash,)
            ).fetchone()
        return row[0] if row else None

    def evaluation_cohort_role(self, cohort_id):
        """Admitted experiment role, if this cohort belongs to a manifest."""
        from src.decision.experiments import FrozenExperiment

        with self.connect() as db:
            rows = db.execute(
                "SELECT DISTINCT r.frozen_id FROM evaluation_runs r "
                "JOIN evaluation_artifacts a ON a.identity=r.frozen_id "
                "JOIN json_each(a.payload, '$.cohort_ids') c "
                "WHERE json_extract(c.value, '$[1]')=?",
                (cohort_id,),
            ).fetchall()
        if len(rows) > 1:
            raise ValueError("cohort is bound to multiple experiment admissions")
        if not rows:
            return None
        frozen = self.evaluation_artifact(rows[0][0], FrozenExperiment)
        names = [name for name, identity in frozen.cohort_ids if identity == cohort_id]
        matches = [c.role for c in frozen.manifest.cohorts if c.name in names]
        if len(matches) != 1:
            raise ValueError("cohort role differs from frozen experiment")
        return matches[0]

    def freeze_experiment(self, frozen, cohorts):
        """Atomically admit all frozen cohorts and their single shared budget run."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM reservations WHERE run_id=?", (frozen.manifest.run_id,)
            ).fetchone():
                raise ValueError("experiment run already has usage or uncertain reservations")
            for cohort in cohorts:
                for case in cohort.cases:
                    if (
                        db.execute(
                            "SELECT 1 FROM observations WHERE cache_key=?",
                            (case.request.cache_key(),),
                        ).fetchone()
                        or db.execute(
                            "SELECT 1 FROM evaluation_dispatches WHERE cache_key=?",
                            (case.request.cache_key(),),
                        ).fetchone()
                    ):
                        raise ValueError(
                            "case was already observed or admitted before experiment freeze"
                        )
            try:
                db.execute(
                    "INSERT INTO evaluation_runs VALUES(?,?,?)",
                    (
                        frozen.manifest.run_id,
                        frozen.manifest.content_hash(),
                        frozen.content_hash(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("experiment run or manifest already admitted") from exc
            for value in (*cohorts, frozen):
                db.execute(
                    "INSERT OR IGNORE INTO evaluation_artifacts VALUES(?,?,?)",
                    (
                        value.content_hash(),
                        type(value).__name__,
                        value.model_dump_json(),
                    ),
                )

    def evaluation_dispatch_for(self, key):
        from src.decision.experiments import EvaluationDispatch

        with self.connect() as db:
            row = db.execute(
                "SELECT artifact_id,observation_id FROM evaluation_dispatches WHERE cache_key=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        value = self.evaluation_artifact(row[0], EvaluationDispatch)
        if value.cache_key != key or value.observation_id != row[1]:
            raise ValueError("evaluation dispatch index differs from immutable record")
        return value

    def claim_evaluation_dispatch(self, claim):
        from src.decision.experiments import EvaluationDispatch

        claim = EvaluationDispatch.model_validate(claim.model_dump())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM observations WHERE cache_key=?", (claim.cache_key,)
            ).fetchone():
                raise ValueError("case already observed outside this experiment dispatch")
            try:
                db.execute(
                    "INSERT INTO evaluation_dispatches VALUES(?,?,?)",
                    (
                        claim.cache_key,
                        claim.content_hash(),
                        claim.observation_id,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("case already claimed; ambiguous calls cannot be retried") from exc
            db.execute(
                "INSERT INTO evaluation_artifacts VALUES(?,?,?)",
                (
                    claim.content_hash(),
                    type(claim).__name__,
                    claim.model_dump_json(),
                ),
            )

    def start_evaluation_dispatch(self, identity, request):
        """Consume an existing pre-dispatch intent once, before any network call."""
        from datetime import datetime, timezone

        from src.decision.experiments import EvaluationDispatch, FrozenExperiment

        claim = self.evaluation_artifact(identity, EvaluationDispatch)
        frozen = self.evaluation_artifact(claim.experiment_hash, FrozenExperiment)
        cohort_ids = dict(frozen.cohort_ids)
        matches = [
            case
            for spec in frozen.manifest.cohorts
            if cohort_ids.get(spec.name) == claim.cohort_id
            for case in spec.cases
            if case.case_id == claim.case_id and case.request == request
        ]
        if (
            len(matches) != 1
            or claim.request_hash != request.content_hash()
            or claim.cache_key != request.cache_key()
            or self.evaluation_dispatch_for(claim.cache_key) != claim
            or not frozen.recorded_at
            <= claim.recorded_at
            <= datetime.now(timezone.utc)
            < frozen.deadline_at
        ):
            raise ValueError("invalid or expired authoritative case dispatch")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT INTO evaluation_dispatch_starts VALUES(?,?)",
                    (identity, datetime.now(timezone.utc).isoformat()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "case dispatch already started; cannot repeat an ambiguous call"
                ) from exc
        return claim.observation_id, (
            frozen.deadline_at - datetime.now(timezone.utc)
        ).total_seconds()

    def evaluation_dispatch_usage(self, claim, *, at):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            reservations = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM reservations WHERE observation_id=? AND at<=? ORDER BY attempt",
                    (claim.observation_id, at.timestamp()),
                )
            ]
            # Attempts are append-only and have no wall-clock column. A report
            # of an interrupted call conservatively includes all known attempts;
            # a later write changes the proof and invalidates that old report.
            attempts = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM attempts WHERE observation_id=? ORDER BY attempt",
                    (claim.observation_id,),
                )
            ]
        return attempts, reservations

    def evaluation_run_reservations(self, run_id, *, at):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM reservations WHERE run_id=? AND at<=? ORDER BY seq",
                    (run_id, at.timestamp()),
                )
            ]

    def evaluation_dispatch_started_at(self, identity):
        from datetime import datetime

        with self.connect() as db:
            row = db.execute(
                "SELECT started_at FROM evaluation_dispatch_starts WHERE artifact_id=?", (identity,)
            ).fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def append(self, request: DecisionRequest, result: DecisionResult):
        with self.connect() as db:
            db.execute(
                "INSERT INTO observations(observation_id,cache_key,request_json,"
                "result_json,status) VALUES(?,?,?,?,?)",
                (
                    result.observation_id,
                    result.cache_key,
                    request.model_dump_json(),
                    result.model_dump_json(),
                    result.status.value,
                ),
            )

    def find(
        self, key: str, *, observation_id: str | None = None, successful_only: bool = True
    ) -> DecisionResult | None:
        sql = "SELECT result_json FROM observations WHERE cache_key=?"
        args: list = [key]
        if observation_id:
            sql += " AND observation_id=?"
            args.append(observation_id)
        if successful_only:
            sql += " AND status='ok'"
        sql += " ORDER BY seq ASC LIMIT 1"
        with self.connect() as db:
            row = db.execute(sql, args).fetchone()
        return DecisionResult.model_validate_json(row[0]) if row else None

    def attempt(self, observation_id, attempt, status, body, elapsed_ms, failure=None):
        with self.connect() as db:
            db.execute(
                "INSERT INTO attempts(observation_id,attempt,status,body,elapsed_ms,"
                "failure) VALUES(?,?,?,?,?,?)",
                (observation_id, attempt, status, body, elapsed_ms, failure),
            )

    def reserve(
        self, *, run_id, day, tokens, cost, now, limits, observation_id=None, attempt=None
    ) -> tuple[bool, str, float]:
        """Atomic reservation; rate delay returned before consuming a request."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            admitted = db.execute(
                "SELECT frozen_id FROM evaluation_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if admitted:
                from src.decision.experiments import FrozenExperiment

                frozen = self.evaluation_artifact(admitted[0], FrozenExperiment)
                if (
                    limits != frozen.manifest.budget.limits()
                    or now >= frozen.deadline_at.timestamp()
                ):
                    return False, "frozen experiment limits or deadline differ", 0
                claim_row = db.execute(
                    "SELECT artifact_id FROM evaluation_dispatches WHERE observation_id=?",
                    (observation_id,),
                ).fetchone()
                if (
                    claim_row is None
                    or db.execute(
                        "SELECT 1 FROM evaluation_dispatch_starts WHERE artifact_id=?",
                        (claim_row[0],),
                    ).fetchone()
                    is None
                ):
                    return False, "experiment usage lacks exact one-time dispatch admission", 0
                from src.decision.experiments import EvaluationDispatch

                claim = self.evaluation_artifact(claim_row[0], EvaluationDispatch)
                if claim.experiment_hash != frozen.content_hash():
                    return False, "experiment usage belongs to a different admission", 0
            for column, value, requests, max_tokens, max_cost in (
                ("run_id", run_id, limits.run_requests, limits.run_tokens, limits.run_spend_usd),
                ("day", day, limits.day_requests, limits.day_tokens, limits.day_spend_usd),
            ):
                row = db.execute(
                    f"SELECT count(*),coalesce(sum(reserved_tokens),0),"
                    f"coalesce(sum(reserved_cost),0),count(*)-count(reserved_cost) "
                    f"FROM reservations WHERE {column}=?",
                    (value,),
                ).fetchone()
                if row[0] >= requests or row[1] + tokens > max_tokens:
                    return False, "request/token budget exhausted", 0
                if max_cost is not None and (cost is None or row[3] or row[2] + cost > max_cost):
                    return False, "spend budget exhausted or price unavailable", 0
            recent = db.execute("SELECT max(at) FROM reservations").fetchone()[0]
            wait = max(0, (recent or 0) + limits.min_interval_seconds - now)
            if wait:
                return False, "rate_limit", wait
            db.execute(
                "INSERT INTO reservations(run_id,day,reserved_tokens,reserved_cost,at,observation_id,attempt) "
                "VALUES(?,?,?,?,?,?,?)",
                (run_id, day, tokens, cost, now, observation_id, attempt),
            )
            return True, "", 0

    def export_observation(self, observation_id: str) -> dict:
        """Read the exact authoritative request/result, attempts and reservations.

        Cache/replay dependencies are included recursively; supplied observation
        objects are never accepted as proof. Old unlinked charges fail closed.
        """
        return self._export_observation(observation_id, set())

    def _export_observation(self, observation_id, seen):
        if observation_id in seen or len(seen) >= 128:
            raise ValueError("cyclic or excessive observation reuse")
        seen.add(observation_id)
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM observations WHERE observation_id=?", (observation_id,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown authoritative Jev observation")
            request = DecisionRequest.model_validate_json(row["request_json"])
            result = DecisionResult.model_validate_json(row["result_json"])
            attempts = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM attempts WHERE observation_id=? ORDER BY attempt",
                    (observation_id,),
                )
            ]
            reservations = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM reservations WHERE observation_id=? ORDER BY attempt",
                    (observation_id,),
                )
            ]
        if (
            result.observation_id != observation_id
            or result.request_hash != request.content_hash()
            or result.cache_key != request.cache_key()
            or row["cache_key"] != result.cache_key
            or result.policy_hash != request.policy.content_hash()
            or result.state_hash != request.state_hash
            or result.run_id != request.context.run_id
            or result.source_hashes != tuple(s.content_hash() for s in request.sources)
            or result.completed_at < result.started_at
        ):
            raise ValueError("Jev observation identity differs from request")
        if len(reservations) != result.attempts or any(
            r["run_id"] != result.run_id for r in reservations
        ):
            raise ValueError("missing authoritative Jev admission linkage")
        return {
            "request": request.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "attempts": attempts,
            "reservations": reservations,
            "reused": self._export_observation(result.reused_observation_id, seen)
            if result.reused_observation_id
            else None,
        }
