"""Immutable model admissions, settlements and exact report-generation records.

The mutable accounting row remains the budget authority. These append-only
records bind its admission and terminal state, without recording credentials.
Legacy accounting rows cannot be retroactively certified by this module.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.domain.common import NonEmptyString, Sha256, UtcTimestamp, VersionedContract
from src.domain.decisions import SpecialistReport


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def create_tables(db):
    # execute individually: executescript would commit the caller's transaction.
    for table in ("agent_admissions", "agent_settlements", "agent_generations"):
        db.execute(
            f"CREATE TABLE IF NOT EXISTS {table}(id TEXT PRIMARY KEY, hash TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        for action in ("UPDATE", "DELETE"):
            db.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_{action.lower()} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable agent receipt'); END"
            )


def insert(db, table, identity, value):
    payload = canonical(value)
    db.execute(
        f"INSERT INTO {table} VALUES(?,?,?)",
        (identity, hashlib.sha256(payload.encode()).hexdigest(), payload),
    )


def read(db, table, identity):
    row = db.execute(f"SELECT hash,payload FROM {table} WHERE id=?", (identity,)).fetchone()
    if not row or hashlib.sha256(row[1].encode()).hexdigest() != row[0]:
        raise ValueError("missing or corrupted authoritative agent receipt")
    return json.loads(row[1])


def admission(db, identity, budget, model, price, now, tokens, cost):
    def price_field(name):
        value = getattr(price, name)
        return value.isoformat() if isinstance(value, datetime) else str(value)

    usage = {}
    for column, value in (("run_id", budget.run_id), ("day", now.date().isoformat())):
        rows = db.execute(
            f"SELECT charged_tokens,charged_cost FROM calls WHERE {column}=?", (value,)
        ).fetchall()
        usage[column] = {
            "scope": value,
            "requests": len(rows),
            "tokens": sum(r[0] for r in rows),
            "charged_cost_usd": str(sum((Decimal(r[1]) for r in rows), Decimal(0))),
        }
    insert(
        db,
        "agent_admissions",
        identity,
        {
            "schema_version": "1.0",
            "call_id": identity,
            "run_id": budget.run_id,
            "session_id": budget.session_id,
            "started_at": now.isoformat(),
            "provider": budget.config.llm_provider.lower(),
            "model": model,
            "reserved_tokens": tokens,
            "reserved_cost_usd": str(cost),
            "usage_at_admission": usage,
            "pricing": {
                name: price_field(name)
                for name in (
                    "provider",
                    "model",
                    "price_version",
                    "source_uri",
                    "effective_at",
                    "input_usd_per_million",
                    "output_usd_per_million",
                )
            },
            "caps": {
                name: str(getattr(budget.config, name, default))
                for name, default in (
                    ("max_requests_per_analysis", 30),
                    ("max_requests_per_day", 600),
                    ("max_tokens_per_analysis", 100000),
                    ("max_tokens_per_day", 1000000),
                    ("max_cost_per_analysis", ".5"),
                    ("max_cost_per_day_usd", "10"),
                )
            },
        },
    )


def settlement(db, identity, run_id):
    db.row_factory = sqlite3.Row
    row = db.execute("SELECT * FROM calls WHERE id=? AND run_id=?", (identity, run_id)).fetchone()
    insert(
        db,
        "agent_settlements",
        identity,
        {
            "schema_version": "1.0",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "accounting": dict(row),
        },
    )


class AgentCallReceipt(VersionedContract):
    call_id: NonEmptyString
    run_id: NonEmptyString
    session_id: NonEmptyString
    started_at: UtcTimestamp
    completed_at: UtcTimestamp
    provider: NonEmptyString
    model: NonEmptyString
    actual_model: str | None
    status: NonEmptyString
    admission_json: str
    settlement_json: str


class GenerationReceipt(VersionedContract):
    call_id: NonEmptyString
    role: NonEmptyString
    request_json: str
    source_hashes: tuple[Sha256, ...]
    report: SpecialistReport
    available_at: UtcTimestamp


def read_call(path, call_id) -> AgentCallReceipt:
    # Read-only URI avoids creating an empty proof database on a typo.
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        a = read(db, "agent_admissions", call_id)
        s = read(db, "agent_settlements", call_id)
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
    if not row or dict(row) != s["accounting"]:
        raise ValueError("budget accounting differs from immutable settlement")
    if any(
        a[key] != row[column]
        for key, column in (
            ("call_id", "id"),
            ("run_id", "run_id"),
            ("started_at", "started_at"),
            ("provider", "provider"),
            ("model", "model"),
            ("reserved_tokens", "reserved_tokens"),
            ("reserved_cost_usd", "reserved_cost"),
        )
    ):
        raise ValueError("admission differs from accounting")
    if a["pricing"]["price_version"] != row["pricing_version"]:
        raise ValueError("pricing identity differs from accounting")
    if row["input_tokens"] is not None and row["output_tokens"] is not None:
        price = a["pricing"]
        cost = (
            Decimal(row["input_tokens"]) * Decimal(price["input_usd_per_million"])
            + Decimal(row["output_tokens"]) * Decimal(price["output_usd_per_million"])
        ) / Decimal(1000000)
        if (
            cost != Decimal(row["charged_cost"])
            or row["charged_tokens"] != row["input_tokens"] + row["output_tokens"]
        ):
            raise ValueError("settlement differs from actual usage and admitted pricing")
    receipt = AgentCallReceipt(
        call_id=call_id,
        run_id=a["run_id"],
        session_id=a["session_id"],
        started_at=a["started_at"],
        completed_at=s["completed_at"],
        provider=a["provider"],
        model=a["model"],
        actual_model=row["actual_model"],
        status=row["status"],
        admission_json=canonical(a),
        settlement_json=canonical(s),
    )
    if receipt.completed_at < receipt.started_at:
        raise ValueError("invalid generation clocks")
    return receipt


def record_generation(budget, call_id, *, role, messages, sources, report):
    receipt = read_call(budget.path, call_id)
    if receipt.status != "ok" or receipt.actual_model != receipt.model:
        raise ValueError("unsettled or wrong-model generation cannot support a report")
    generation = GenerationReceipt(
        call_id=call_id,
        role=role,
        request_json=canonical([{"type": m.type, "content": m.content} for m in messages]),
        source_hashes=tuple(s.content_hash() for s in sources),
        report=report,
        available_at=datetime.now(timezone.utc),
    )
    with sqlite3.connect(budget.path) as db:
        insert(db, "agent_generations", call_id, generation.model_dump(mode="json"))


def read_generation(path, call_id):
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        value = read(db, "agent_generations", call_id)
    return GenerationReceipt.model_validate(value)


def session_calls(path, session_id):
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        ids = [
            row[0]
            for row in db.execute("SELECT id FROM agent_admissions")
            if read(db, "agent_admissions", row[0])["session_id"] == session_id
        ]
    return tuple(read_call(path, identity) for identity in ids)
