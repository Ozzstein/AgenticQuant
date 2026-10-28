"""Append-only decision observations and transactional operating reservations.

This stores model observations, not source documents (owned by EvidenceStore).
Usage reservations remain charged after uncertain failures: an interrupted HTTP
request may have been processed. Run/day limits therefore survive restarts.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.domain.common import Sha256, UtcTimestamp, VersionedContract
from src.domain.decisions import DecisionRequest, DecisionResult, canonical_json, digest


class AttemptUsage(VersionedContract):
    """Provider-reported usage, independently of semantic response validity.

    The original admission day owns the charge even if a response crosses UTC
    midnight. This is provider response evidence, not a reconciled invoice.
    """

    reservation_hash: Sha256
    admission_hash: Sha256
    attempt_hash: Sha256
    request_hash: Sha256
    source_hashes: tuple[Sha256, ...]
    recorded_at: UtcTimestamp
    actual_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    actual_cost_usd: float | None
    charged_tokens: int
    charged_cost_usd: float | None
    breaches: tuple[str, ...]


def response_usage(body):
    """Parse accounting independently; invalid answers cannot hide valid usage."""

    class ObjectPairs(list):
        pass

    ambiguous = False

    def field(value, key):
        nonlocal ambiguous
        if not isinstance(value, ObjectPairs):
            return None
        matches = [item for name, item in value if name == key]
        if len(matches) > 1:
            ambiguous = True
            return None
        return matches[0] if matches else None

    try:
        value = json.loads(body, object_pairs_hook=ObjectPairs)
    except (ValueError, TypeError, RecursionError):
        return None, None, None, False
    if not isinstance(value, ObjectPairs):
        return None, None, None, False
    model = field(value, "model")
    model = model if isinstance(model, str) and 0 < len(model) <= 160 else None
    usage = field(value, "usage")
    values = tuple(field(usage, k) for k in ("input_tokens", "output_tokens"))
    # Keep independently known counters. Unknown/invalid counters are never zero.
    return (model, *(v if type(v) is int and 0 <= v < 2**62 else None for v in values), ambiguous)


def _derive_usage(reservation, admission, attempt, recorded_at):
    request = DecisionRequest.model_validate(admission["request"])
    model, tokens_in, tokens_out, ambiguous = response_usage(attempt["body"])
    price = admission["limits"]["input_usd_per_million"]
    known = tokens_in is not None and tokens_out is not None and not ambiguous
    priced = model == request.policy.model and price is not None
    actual_cost = tokens_in * price / 1_000_000 if known and priced else None
    # Even partial usage supplies a lower bound; never refund the reservation.
    token_floor = sum(v for v in (tokens_in, tokens_out) if v is not None)
    cost_floor = tokens_in * price / 1_000_000 if tokens_in is not None and priced else None
    if cost_floor is not None and not math.isfinite(cost_floor):
        raise ValueError("unrepresentable reported provider charge")
    reserved_cost = reservation["reserved_cost"]
    breaches = []
    if ambiguous:
        breaches.append("ambiguous_usage_response")
    if token_floor > reservation["reserved_tokens"]:
        breaches.append("token_reservation_exceeded")
    if cost_floor is not None and reserved_cost is not None and cost_floor > reserved_cost:
        breaches.append("cost_reservation_exceeded")
    if (model is not None or tokens_in is not None or tokens_out is not None) and not priced:
        breaches.append("response_model_or_price_unverifiable")
    return AttemptUsage(
        reservation_hash=digest(canonical_json(reservation)),
        admission_hash=digest(canonical_json(admission)),
        attempt_hash=digest(canonical_json(attempt)),
        request_hash=request.content_hash(),
        source_hashes=tuple(s.content_hash() for s in request.sources),
        recorded_at=recorded_at,
        actual_model=model,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        actual_cost_usd=actual_cost,
        charged_tokens=max(reservation["reserved_tokens"], token_floor),
        charged_cost_usd=max(reserved_cost, cost_floor or 0) if reserved_cost is not None else None,
        breaches=tuple(breaches),
    )


def summarize_usage(charges):
    """Safe counters/identities only; no requests, raw responses or credentials."""
    usage = [v["usage"] for v in charges if v["usage"] is not None]
    known_costs = [v["actual_cost_usd"] for v in usage if v["actual_cost_usd"] is not None]
    unknown = sum(
        v["usage"] is None
        or any(v["usage"][k] is None for k in ("input_tokens", "output_tokens", "actual_cost_usd"))
        for v in charges
    )
    costs = [v["charged_cost_usd"] for v in charges]
    return {
        "accounting_policy": "attempt-conservative-max-original-utc-day-1",
        "requests": len(charges),
        "known_input_tokens": sum(v["input_tokens"] or 0 for v in usage),
        "known_output_tokens": sum(v["output_tokens"] or 0 for v in usage),
        "known_cost_usd": sum(known_costs),
        "total_cost_usd": sum(known_costs) if not unknown else None,
        "unknown_usage_reservations": unknown,
        "conservative_tokens": sum(v["charged_tokens"] for v in charges),
        "conservative_cost_usd": sum(costs) if all(v is not None for v in costs) else None,
        "breached": any(v["breaches"] for v in charges),
        "breaches": [
            {
                "observation_id": v["reservation"]["observation_id"],
                "attempt": v["reservation"]["attempt"],
                "run_id": v["reservation"]["run_id"],
                "day": v["reservation"]["day"],
                "reasons": v["breaches"],
                "usage_hash": digest(canonical_json(v["usage"])) if v["usage"] else None,
            }
            for v in charges
            if v["breaches"]
        ],
    }


def read_usage_summary(path, *, day, run_id=None, as_of):
    """Read-only monitoring projection; absent/old authority never becomes zero usage.

    `day` is the original UTC admission date (YYYY-MM-DD), not response date.
    Corruption raises; callers must expose unavailable rather than assume zero.
    """
    if as_of.tzinfo is None or datetime.strptime(day, "%Y-%m-%d").date().isoformat() != day:
        raise ValueError("aware cutoff and exact UTC budget day required")
    path = Path(path).resolve()
    if not path.is_file():
        return {"status": "unavailable", "reason": "decision_store_missing"}
    store = DecisionStore.__new__(DecisionStore)
    store.path, store._read_only = path, True
    with store.connect() as db:
        db.execute("BEGIN")
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"reservations", "reservation_authorities", "attempt_usage"}.issubset(tables):
            return {"status": "unavailable", "reason": "usage_authority_schema_missing"}
        values = store._charges(db, run_id=run_id, day=day, at=as_of)
        return {
            "status": "available",
            "as_of": as_of.isoformat(),
            "day": day,
            "day_usage": summarize_usage([v for v in values if v["reservation"]["day"] == day]),
            "run_id": run_id,
            "run_usage": summarize_usage(
                [v for v in values if v["reservation"]["run_id"] == run_id]
            )
            if run_id is not None
            else None,
        }


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
                CREATE TABLE IF NOT EXISTS reservation_authorities (
                    reservation_seq INTEGER PRIMARY KEY, hash TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempt_usage (
                    reservation_seq INTEGER PRIMARY KEY, hash TEXT NOT NULL, payload TEXT NOT NULL);
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
                    "reservation_authorities",
                    "attempt_usage",
                ):
                    db.execute(
                        f"CREATE TRIGGER IF NOT EXISTS {table}_{action.lower()} BEFORE {action} "
                        f"ON {table} BEGIN SELECT RAISE(ABORT, 'immutable evaluation admission'); END"
                    )

    def connect(self):
        if getattr(self, "_read_only", False):
            return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=10)
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def _record(db, table, identity, value):
        payload = canonical_json(value)
        db.execute(f"INSERT INTO {table} VALUES(?,?,?)", (identity, digest(payload), payload))

    @staticmethod
    def _read_record(db, table, identity):
        row = db.execute(
            f"SELECT hash,payload FROM {table} WHERE reservation_seq=?", (identity,)
        ).fetchone()
        if row is None:
            return None
        if digest(row[1]) != row[0]:
            raise ValueError("provider usage authority hash differs")
        return json.loads(row[1])

    def _charges(self, db, *, run_id=None, day=None, at=None, observation_id=None):
        """Reverify authority in the caller's admission/read transaction."""
        db.row_factory = sqlite3.Row
        if observation_id is not None:
            where, args = "observation_id=?", (observation_id,)
        elif day is not None:
            where, args = "(run_id=? OR day=?)", (run_id, day)
        else:
            where, args = "run_id=?", (run_id,)
        values = []
        for row in db.execute(
            "SELECT * FROM reservations WHERE " + where + " ORDER BY seq", args
        ).fetchall():
            reservation = dict(row)
            if at is not None and reservation["at"] > at.timestamp():
                continue
            admission = self._read_record(db, "reservation_authorities", row["seq"])
            stored = self._read_record(db, "attempt_usage", row["seq"])
            attempts = [
                dict(a)
                for a in db.execute(
                    "SELECT * FROM attempts WHERE observation_id=? AND attempt=? ORDER BY seq",
                    (row["observation_id"], row["attempt"]),
                )
            ]
            usage = None
            tokens, cost, breaches = row["reserved_tokens"], row["reserved_cost"], ()
            if admission is not None:
                from src.decision.service import OperatingLimits

                request = DecisionRequest.model_validate(admission["request"])
                limits = OperatingLimits.model_validate(admission["limits"])
                expected_tokens = (
                    len(canonical_json(request.wire()).encode())
                    + 1024
                    + len(request.questions) * 256
                )
                price = limits.input_usd_per_million
                if (
                    set(admission) != {"reservation_hash", "request", "limits"}
                    or admission["reservation_hash"] != digest(canonical_json(reservation))
                    or request.context.run_id != row["run_id"]
                    or row["day"]
                    != datetime.fromtimestamp(row["at"], timezone.utc).date().isoformat()
                    or expected_tokens != row["reserved_tokens"]
                    or row["reserved_cost"]
                    != (expected_tokens * price / 1_000_000 if price is not None else None)
                    or len(attempts) > 1
                    or bool(attempts) != (stored is not None)
                ):
                    raise ValueError("provider usage admission/attempt linkage differs")
                if stored is not None:
                    usage = AttemptUsage.model_validate(stored)
                    if (
                        usage
                        != _derive_usage(reservation, admission, attempts[0], usage.recorded_at)
                        or usage.recorded_at.timestamp() < row["at"]
                    ):
                        raise ValueError(
                            "provider usage differs from original raw response/admission"
                        )
                    if at is None or usage.recorded_at <= at:
                        tokens, cost, breaches = (
                            usage.charged_tokens,
                            usage.charged_cost_usd,
                            usage.breaches,
                        )
                    else:
                        usage = None
            elif stored is not None:
                raise ValueError("provider usage lacks original admission authority")
            else:
                # Do not rewrite legacy exports/receipt hashes. Existing raw usage
                # can still fence a historical overrun before the next admission.
                original = db.execute(
                    "SELECT result_json,request_json FROM observations WHERE observation_id=?",
                    (row["observation_id"],),
                ).fetchone()
                result = DecisionResult.model_validate_json(original[0]) if original else None
                if at is None or (result is not None and result.completed_at <= at):
                    expected_model = (
                        DecisionRequest.model_validate_json(original[1]).policy.model
                        if original
                        else None
                    )
                    for attempt in attempts:
                        model, ti, to, ambiguous = response_usage(attempt["body"])
                        tokens = max(tokens, sum(v for v in (ti, to) if v is not None))
                        if ambiguous:
                            breaches += ("legacy_ambiguous_usage_response",)
                        if (
                            model is not None
                            and expected_model is not None
                            and model != expected_model
                        ):
                            breaches += ("legacy_response_model_unverifiable",)
                    if tokens > row["reserved_tokens"]:
                        breaches += ("legacy_token_reservation_exceeded",)
                if original is not None:
                    if result.attempts == row["attempt"] and (
                        at is None or result.completed_at <= at
                    ):
                        if result.cost_usd is not None and cost is not None:
                            cost = max(cost, result.cost_usd)
                            if cost > row["reserved_cost"]:
                                breaches += ("legacy_cost_reservation_exceeded",)
            values.append(
                {
                    "reservation": reservation,
                    "admission": admission,
                    "usage": usage.model_dump(mode="json") if usage else None,
                    "charged_tokens": tokens,
                    "charged_cost_usd": cost,
                    "breaches": list(breaches),
                }
            )
        return values

    def budget_usage(self, run_id, *, at=None):
        with self.connect() as db:
            db.execute("BEGIN")
            return self._charges(db, run_id=run_id, at=at)

    def budget_fences(self, run_id, day, *, at=None):
        with self.connect() as db:
            db.execute("BEGIN")
            return [v for v in self._charges(db, run_id=run_id, day=day, at=at) if v["breaches"]]

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
            db.execute("BEGIN IMMEDIATE")
            db.row_factory = sqlite3.Row
            reservations = db.execute(
                "SELECT * FROM reservations WHERE observation_id=? AND attempt=?",
                (observation_id, attempt),
            ).fetchall()
            if len(reservations) != 1:
                raise ValueError("attempt requires exactly its original reservation")
            reservation = dict(reservations[0])
            admission = self._read_record(db, "reservation_authorities", reservation["seq"])
            if db.execute(
                "SELECT 1 FROM attempts WHERE observation_id=? AND attempt=?",
                (observation_id, attempt),
            ).fetchone():
                raise ValueError("provider attempt already recorded")
            cursor = db.execute(
                "INSERT INTO attempts(observation_id,attempt,status,body,elapsed_ms,"
                "failure) VALUES(?,?,?,?,?,?)",
                (observation_id, attempt, status, body, elapsed_ms, failure),
            )
            if admission is not None:
                raw_attempt = dict(
                    db.execute("SELECT * FROM attempts WHERE seq=?", (cursor.lastrowid,)).fetchone()
                )
                usage = _derive_usage(
                    reservation, admission, raw_attempt, datetime.now(timezone.utc)
                )
                self._record(db, "attempt_usage", reservation["seq"], usage.model_dump(mode="json"))
                # Validate the whole atomic write before exposing the raw response
                # or permitting the next admission in another process.
                self._charges(db, observation_id=observation_id)
                return usage
            return None

    def reserve(
        self,
        *,
        run_id,
        day,
        tokens,
        cost,
        now,
        limits,
        observation_id=None,
        attempt=None,
        request=None,
    ) -> tuple[bool, str, float]:
        """Atomic reservation; rate delay returned before consuming a request."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            charges = self._charges(db, run_id=run_id, day=day)
            if any(v["breaches"] for v in charges):
                return False, "durable provider usage breach fences run or original UTC day", 0
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
                scope = [v for v in charges if v["reservation"][column] == value]
                row = (
                    len(scope),
                    sum(v["charged_tokens"] for v in scope),
                    sum(v["charged_cost_usd"] or 0 for v in scope),
                    sum(v["charged_cost_usd"] is None for v in scope),
                )
                if row[0] >= requests or row[1] + tokens > max_tokens:
                    return False, "request/token budget exhausted", 0
                if max_cost is not None and (cost is None or row[3] or row[2] + cost > max_cost):
                    return False, "spend budget exhausted or price unavailable", 0
            recent = db.execute("SELECT max(at) FROM reservations").fetchone()[0]
            wait = max(0, (recent or 0) + limits.min_interval_seconds - now)
            if wait:
                return False, "rate_limit", wait
            cursor = db.execute(
                "INSERT INTO reservations(run_id,day,reserved_tokens,reserved_cost,at,observation_id,attempt) "
                "VALUES(?,?,?,?,?,?,?)",
                (run_id, day, tokens, cost, now, observation_id, attempt),
            )
            if request is not None:
                request = DecisionRequest.model_validate(request.model_dump())
                if not observation_id or type(attempt) is not int or attempt < 1:
                    raise ValueError("source-bound reservation requires an exact attempt")
                row = dict(
                    db.execute(
                        "SELECT * FROM reservations WHERE seq=?", (cursor.lastrowid,)
                    ).fetchone()
                )
                self._record(
                    db,
                    "reservation_authorities",
                    row["seq"],
                    {
                        "reservation_hash": digest(canonical_json(row)),
                        "request": request.model_dump(mode="json"),
                        "limits": limits.model_dump(mode="json"),
                    },
                )
                self._charges(db, observation_id=observation_id)
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
            db.execute("BEGIN")
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
            charges = self._charges(db, observation_id=observation_id)
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
        extended = {}
        if any(v["admission"] is not None for v in charges):
            if any(
                v["admission"] is None
                or v["admission"]["request"] != request.model_dump(mode="json")
                for v in charges
            ):
                raise ValueError("observation differs from admitted source/question/model request")
            settled = [AttemptUsage.model_validate(v["usage"]) for v in charges if v["usage"]]
            if any(v.recorded_at > result.completed_at for v in settled):
                raise ValueError("usage receipt completed after original observation")
            last = charges[-1]["usage"] if charges else None
            if last and (
                result.actual_model,
                result.input_tokens,
                result.output_tokens,
                result.cost_usd,
            ) != (
                last["actual_model"],
                last["input_tokens"],
                last["output_tokens"],
                last["actual_cost_usd"],
            ):
                raise ValueError("observation accounting differs from original final attempt")
            extended["usage_authority"] = charges
        return {
            "request": request.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "attempts": attempts,
            "reservations": reservations,
            "reused": self._export_observation(result.reused_observation_id, seen)
            if result.reused_observation_id
            else None,
            **extended,
        }
