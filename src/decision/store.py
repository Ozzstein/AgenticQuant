"""Append-only decision observations and transactional operating reservations.

This stores model observations, not source documents (owned by EvidenceStore).
Usage reservations remain charged after uncertain failures: an interrupted HTTP
request may have been processed. Run/day limits therefore survive restarts.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.domain.decisions import DecisionRequest, DecisionResult


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
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(reservations)")}
            for name, kind in (("observation_id", "TEXT"), ("attempt", "INTEGER")):
                if name not in columns:
                    db.execute(f"ALTER TABLE reservations ADD COLUMN {name} {kind}")
            for action in ("UPDATE", "DELETE"):
                db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS reservation_{action.lower()} BEFORE {action} ON reservations BEGIN SELECT RAISE(ABORT, 'immutable reservation'); END"
                )

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def _append_evaluation(self, value):
        """Owned EvaluationService write boundary; never a supplied pass flag."""
        from src.decision.evaluation import EVALUATION_TYPES

        kind = type(value).__name__
        if EVALUATION_TYPES.get(kind) is not type(value):
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
        from src.decision.evaluation import EVALUATION_TYPES

        if EVALUATION_TYPES.get(expected_type.__name__) is not expected_type:
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
