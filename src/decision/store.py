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
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

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

    def reserve(self, *, run_id, day, tokens, cost, now, limits) -> tuple[bool, str, float]:
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
                "INSERT INTO reservations(run_id,day,reserved_tokens,reserved_cost,at) "
                "VALUES(?,?,?,?,?)",
                (run_id, day, tokens, cost, now),
            )
            return True, "", 0
