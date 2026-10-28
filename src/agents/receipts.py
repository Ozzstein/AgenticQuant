"""Immutable model admissions, settlements and exact report-generation records.

Append-only admissions and settlements bind and independently verify the mutable
accounting projection, without recording credentials.
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
    value = json.loads(row[1])
    if canonical(value) != row[1]:
        raise ValueError("noncanonical or ambiguous authoritative agent receipt")
    return value


def admission(
    db,
    identity,
    budget,
    model,
    price,
    now,
    tokens,
    cost,
    *,
    input_bound,
    output_limit,
    request_hash=None,
    usage_before=(),
):
    def price_field(name):
        value = getattr(price, name)
        return value.isoformat() if isinstance(value, datetime) else str(value)

    usage = {}
    for column, value in (("run_id", budget.run_id), ("day", now.date().isoformat())):
        rows = [v["usage"] for v in usage_before if v[column] == value]
        usage[column] = {
            "scope": value,
            "requests": len(rows) + 1,
            "tokens": sum(r["charged_tokens"] for r in rows) + tokens,
            "charged_cost_usd": str(
                sum((Decimal(r["charged_cost_usd"]) for r in rows), Decimal(0)) + cost
            ),
        }
    insert(
        db,
        "agent_admissions",
        identity,
        {
            "schema_version": "2.0",
            "input_bound": input_bound,
            "output_limit": output_limit,
            "request_hash": request_hash,
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


def settlement(db, identity, run_id, *, evidence, usage, reported_status, pricing_hash):
    db.row_factory = sqlite3.Row
    row = db.execute("SELECT * FROM calls WHERE id=? AND run_id=?", (identity, run_id)).fetchone()
    insert(
        db,
        "agent_settlements",
        identity,
        {
            "schema_version": "2.0",
            "admission_hash": hashlib.sha256(
                canonical(read(db, "agent_admissions", identity)).encode()
            ).hexdigest(),
            "evidence": evidence,
            "supplied_pricing_hash": pricing_hash,
            "usage": usage,
            "reported_status": reported_status,
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
    # No migration/write; old receipt bytes and outer export shape remain intact.
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        a, s, row, _ = _verified_call(db, call_id)
    if s is None:
        raise ValueError("missing authoritative agent settlement")
    return AgentCallReceipt(
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
    admitted_hash = json.loads(receipt.admission_json).get("request_hash")
    if (
        admitted_hash is not None
        and hashlib.sha256(generation.request_json.encode()).hexdigest() != admitted_hash
    ):
        raise ValueError("generation request differs from admitted request")
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


ACCOUNTING_POLICY = "reservation-floor-2"


def _counter(value):
    return value if type(value) is int and 0 <= value < 2**62 else None


def _model(value):
    if not isinstance(value, str) or not 0 < len(value) <= 160:
        return None
    try:
        value.encode("utf-8")
    except UnicodeError:
        return None
    return value


def capture_usage(raw):
    """Retain each valid counter independently of response/model acceptance."""
    usage = getattr(raw, "usage_metadata", None)
    meta = getattr(raw, "response_metadata", None)
    usage = usage if isinstance(usage, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    models = [meta[k] for k in ("model_name", "model") if meta.get(k) is not None]
    model = models[0] if models else None
    ambiguous = len(models) > 1 and models[0] != models[1]
    model = _model(model) if not ambiguous else None
    return {
        "response_present": raw is not None,
        "model": model,
        "input_tokens": _counter(usage.get("input_tokens")),
        "output_tokens": _counter(usage.get("output_tokens")),
        "total_tokens": _counter(usage.get("total_tokens")),
        "total_required": "total_tokens" in usage,
        "ambiguous": ambiguous,
        "response_hash": None,
    }


def capture_openai_usage(body, *, embedding=False):
    """Parse accounting before strict semantic parsing, retaining duplicate evidence.

    A duplicated field has no chosen value. Unrelated unique counters survive;
    duplicated usage/model or contradictory totals cannot certify exact usage.
    """

    class Pairs(list):
        pass

    ambiguous = False

    def field(obj, name):
        nonlocal ambiguous
        if not isinstance(obj, Pairs):
            return None
        values = [v for k, v in obj if k == name]
        if len(values) > 1:
            ambiguous = True
            return None
        return values[0] if values else None

    evidence = capture_usage(None)
    evidence.update(
        response_present=True, total_required=True, response_hash=hashlib.sha256(body).hexdigest()
    )
    try:
        obj = json.loads(body, object_pairs_hook=Pairs)
        model, usage = field(obj, "model"), field(obj, "usage")
        tokens_in = _counter(field(usage, "prompt_tokens"))
        tokens_out = _counter(field(usage, "completion_tokens"))
        # The embeddings protocol has no generated tokens. A supplied nonzero
        # value is contradictory evidence, not permission to silently drop it.
        if (
            embedding
            and isinstance(usage, Pairs)
            and not any(k == "completion_tokens" for k, _ in usage)
        ):
            tokens_out = 0
        total = _counter(field(usage, "total_tokens"))
        if embedding and tokens_out not in (None, 0):
            ambiguous = True
        evidence.update(
            model=_model(model),
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            total_tokens=total,
            ambiguous=ambiguous,
        )
    except (ValueError, TypeError, UnicodeError, RecursionError):
        pass
    return evidence


def derive_usage(admitted, evidence, reported_status, *, pricing_verified=True):
    """Sole pricing/charge policy, replayed by admission and public readers."""
    if not isinstance(reported_status, str) or not reported_status:
        raise ValueError("invalid accounting status")
    if set(evidence) != set(capture_usage(None)):
        raise ValueError("invalid accounting evidence shape")
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if evidence[key] is not None and _counter(evidence[key]) is None:
            raise ValueError("invalid accounting counter")
    for key in ("ambiguous", "response_present", "total_required"):
        if type(evidence[key]) is not bool:
            raise ValueError("invalid accounting evidence flag")
    model = evidence["model"]
    if model is not None and _model(model) is None:
        raise ValueError("invalid accounting model identity")
    if evidence["response_hash"] is not None:
        value = evidence["response_hash"]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("invalid response hash")
    i, o, t = (evidence[k] for k in ("input_tokens", "output_tokens", "total_tokens"))
    counters = sum(v for v in (i, o) if v is not None)
    contradictory = t is not None and (
        t < counters or (i is not None and o is not None and t != counters)
    )
    exact = (
        i is not None
        and o is not None
        and not evidence["ambiguous"]
        and not contradictory
        and (not evidence["total_required"] or t is not None)
    )
    priced = model == admitted["model"]
    price = admitted["pricing"]
    cost_floor = (
        sum(
            (
                Decimal(v) * Decimal(price[k])
                for v, k in ((i, "input_usd_per_million"), (o, "output_usd_per_million"))
                if v is not None
            ),
            Decimal(0),
        )
        / Decimal(1_000_000)
        if priced
        else None
    )
    floor = max(counters, t or 0)
    breaches = []
    if not pricing_verified:
        breaches.append("settlement_pricing_identity_mismatch")
    if evidence["ambiguous"]:
        breaches.append("ambiguous_usage_response")
    if contradictory:
        breaches.append("contradictory_usage_response")
    if evidence["response_present"] and not priced:
        breaches.append("response_model_unverifiable")
    if floor > admitted["reserved_tokens"]:
        breaches.append("token_reservation_exceeded")
    if cost_floor is not None and cost_floor > Decimal(admitted["reserved_cost_usd"]):
        breaches.append("cost_reservation_exceeded")
    for value, bound, name in (
        (i, "input_bound", "input_reservation_exceeded"),
        (o, "output_limit", "output_reservation_exceeded"),
    ):
        if bound in admitted and value is not None and value > admitted[bound]:
            breaches.append(name)
    # Retain legacy caller-reported output-limit violations as well.
    if reported_status == "usage_overrun" and not any("exceeded" in v for v in breaches):
        breaches.append("reported_usage_overrun")
    status = reported_status
    if not pricing_verified:
        status = "pricing_mismatch"
    if evidence["response_present"] and not priced:
        status = "model_mismatch"
    elif status == "ok" and not exact:
        status = "usage_unknown"
    if any("exceeded" in v or v == "reported_usage_overrun" for v in breaches):
        status = "usage_overrun"
    return {
        "accounting_policy": ACCOUNTING_POLICY,
        "input_tokens": i,
        "output_tokens": o,
        "token_lower_bound": floor,
        "actual_tokens": counters if exact else None,
        "cost_lower_bound_usd": str(cost_floor) if cost_floor is not None else None,
        "actual_cost_usd": str(cost_floor) if exact and priced else None,
        "charged_tokens": max(admitted["reserved_tokens"], floor),
        "charged_cost_usd": str(
            max(Decimal(admitted["reserved_cost_usd"]), cost_floor or Decimal(0))
        ),
        "breaches": breaches,
        "status": status,
    }


def _aware(value):
    at = datetime.fromisoformat(value)
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("aware accounting clock required")
    return at


def _verified_call(db, identity):
    try:
        return _verified_call_inner(db, identity)
    except (KeyError, TypeError, ArithmeticError) as exc:
        raise ValueError("invalid authoritative agent accounting") from exc


def _verified_call_inner(db, identity):
    a = read(db, "agent_admissions", identity)
    db.row_factory = sqlite3.Row
    row = db.execute("SELECT * FROM calls WHERE id=?", (identity,)).fetchone()
    if row is None:
        raise ValueError("missing admitted accounting row")
    row = dict(row)
    for key, column in (
        ("call_id", "id"),
        ("run_id", "run_id"),
        ("started_at", "started_at"),
        ("provider", "provider"),
        ("model", "model"),
        ("reserved_tokens", "reserved_tokens"),
        ("reserved_cost_usd", "reserved_cost"),
    ):
        if a[key] != row[column]:
            raise ValueError("admission differs from accounting")
    price = a["pricing"]
    if (
        price["price_version"] != row["pricing_version"]
        or price["provider"] != a["provider"]
        or price["model"] != a["model"]
        or _aware(a["started_at"]).astimezone(timezone.utc).date().isoformat() != row["day"]
        or _aware(price["effective_at"]) > _aware(a["started_at"])
    ):
        raise ValueError("pricing/day identity differs from admission")
    if type(a["reserved_tokens"]) is not int or not 0 < a["reserved_tokens"] < 2**63:
        raise ValueError("invalid admitted token reservation")
    for value in (
        a["reserved_cost_usd"],
        price["input_usd_per_million"],
        price["output_usd_per_million"],
    ):
        if not Decimal(value).is_finite() or Decimal(value) < 0:
            raise ValueError("invalid admitted pricing")
    if a["schema_version"] == "2.0":
        request_hash = a["request_hash"]
        if request_hash is not None and (
            not isinstance(request_hash, str)
            or len(request_hash) != 64
            or any(c not in "0123456789abcdef" for c in request_hash)
        ):
            raise ValueError("invalid admitted request hash")
        if (
            type(a["input_bound"]) is not int
            or type(a["output_limit"]) is not int
            or a["input_bound"] <= 0
            or a["output_limit"] < 0
            or a["input_bound"] + a["output_limit"] != a["reserved_tokens"]
            or (
                Decimal(a["input_bound"]) * Decimal(price["input_usd_per_million"])
                + Decimal(a["output_limit"]) * Decimal(price["output_usd_per_million"])
            )
            / Decimal(1_000_000)
            != Decimal(a["reserved_cost_usd"])
        ):
            raise ValueError("reservation differs from immutable input/output bounds")
    elif a["schema_version"] != "1.0":
        raise ValueError("unsupported admission version")
    found = db.execute("SELECT 1 FROM agent_settlements WHERE id=?", (identity,)).fetchone()
    s = read(db, "agent_settlements", identity) if found else None
    if s is None:
        if (
            any(
                row[k] is not None
                for k in ("input_tokens", "output_tokens", "actual_model", "latency_ms")
            )
            or row["status"] != "reserved"
            or row["charged_tokens"] != a["reserved_tokens"]
            or row["charged_cost"] != a["reserved_cost_usd"]
        ):
            raise ValueError("unsettled accounting differs from admission")
        return a, s, row, derive_usage(a, capture_usage(None), "reserved")
    if row != s["accounting"] or _aware(s["completed_at"]) < _aware(a["started_at"]):
        raise ValueError("budget accounting differs from immutable settlement or clocks")
    if s["schema_version"] == "2.0":
        u = derive_usage(
            a,
            s["evidence"],
            s["reported_status"],
            pricing_verified=(
                s["supplied_pricing_hash"]
                == hashlib.sha256(canonical(a["pricing"]).encode()).hexdigest()
            ),
        )
        if (
            s["admission_hash"] != hashlib.sha256(canonical(a).encode()).hexdigest()
            or u != s["usage"]
        ):
            raise ValueError("settlement usage differs from original admission/evidence")
        for key, column in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("charged_tokens", "charged_tokens"),
            ("charged_cost_usd", "charged_cost"),
            ("status", "status"),
        ):
            if u[key] != row[column]:
                raise ValueError("settlement projection differs from verified usage")
        if row["actual_model"] != s["evidence"]["model"]:
            raise ValueError("settlement model differs from evidence")
    elif s["schema_version"] == "1.0":
        # Validate original v1 math without rewriting/refunding any receipt.
        i, o = row["input_tokens"], row["output_tokens"]
        known = i is not None and o is not None
        cost = (
            (
                Decimal(i) * Decimal(price["input_usd_per_million"])
                + Decimal(o) * Decimal(price["output_usd_per_million"])
            )
            / Decimal(1_000_000)
            if known
            else Decimal(a["reserved_cost_usd"])
        )
        if (
            (known and (_counter(i) is None or _counter(o) is None))
            or row["charged_tokens"] != (i + o if known else a["reserved_tokens"])
            or Decimal(row["charged_cost"]) != cost
        ):
            raise ValueError("legacy settlement differs from admitted pricing")
        evidence = capture_usage(None)
        evidence.update(
            response_present=known
            or row["actual_model"] is not None
            or row["status"] == "model_mismatch",
            model=row["actual_model"],
            input_tokens=i,
            output_tokens=o,
        )
        u = derive_usage(a, evidence, row["status"])
    else:
        raise ValueError("unsupported settlement version")
    return a, s, row, u


def verified_charges(db, *, day, run_id=None, as_of):
    """Use one caller-owned read/write transaction; select by immutable identity."""
    values = []
    ids = {r[0] for r in db.execute("SELECT id FROM agent_admissions")}
    if (
        ids != {r[0] for r in db.execute("SELECT id FROM calls")}
        or not {r[0] for r in db.execute("SELECT id FROM agent_settlements")} <= ids
    ):
        raise ValueError("unbound or missing agent accounting authority")
    for identity in sorted(ids):
        a = read(db, "agent_admissions", identity)
        started = _aware(a["started_at"])
        original_day = started.astimezone(timezone.utc).date().isoformat()
        if (as_of is not None and started > as_of) or (
            original_day != day and a["run_id"] != run_id
        ):
            continue
        a, s, row, usage = _verified_call(db, identity)
        if as_of is not None and s is not None and _aware(s["completed_at"]) > as_of:
            s = None
            usage = derive_usage(a, capture_usage(None), "reserved")
        values.append(
            {
                "id": identity,
                "run_id": a["run_id"],
                "day": original_day,
                "pending": s is None,
                "model": row["actual_model"] if s else None,
                "expected_model": a["model"],
                "usage": usage,
            }
        )
    return values


def summarize_usage(values):
    usages = [v["usage"] for v in values]
    actual_tokens = [u["actual_tokens"] for u in usages]
    actual_costs = [u["actual_cost_usd"] for u in usages]
    return {
        "requests": len(values),
        "pending_requests": sum(v["pending"] for v in values),
        "known_input_tokens": sum(u["input_tokens"] or 0 for u in usages),
        "known_output_tokens": sum(u["output_tokens"] or 0 for u in usages),
        "known_total_tokens_lower_bound": sum(u["token_lower_bound"] for u in usages),
        "known_cost_lower_bound_usd": str(
            sum(
                (
                    Decimal(u["cost_lower_bound_usd"])
                    for u in usages
                    if u["cost_lower_bound_usd"] is not None
                ),
                Decimal(0),
            )
        ),
        "actual_tokens": sum(actual_tokens) if all(v is not None for v in actual_tokens) else None,
        "actual_cost_usd": str(sum((Decimal(v) for v in actual_costs), Decimal(0)))
        if all(v is not None for v in actual_costs)
        else None,
        "usage_unknown_requests": sum(v is None for v in actual_tokens),
        "unpriced_requests": sum(v is None for v in actual_costs),
        "uncertain_tokens": sum(
            u["charged_tokens"] for u in usages if u["actual_cost_usd"] is None
        ),
        "uncertain_cost_usd": str(
            sum(
                (Decimal(u["charged_cost_usd"]) for u in usages if u["actual_cost_usd"] is None),
                Decimal(0),
            )
        ),
        "conservative_tokens": sum(u["charged_tokens"] for u in usages),
        "conservative_cost_usd": str(
            sum((Decimal(u["charged_cost_usd"]) for u in usages), Decimal(0))
        ),
        "breached": any(u["breaches"] for u in usages),
        "breach_call_ids": [v["id"] for v in values if v["usage"]["breaches"]],
        "breach_reasons": sorted({reason for u in usages for reason in u["breaches"]}),
        "statuses": {
            s: sum(u["status"] == s for u in usages) for s in sorted({u["status"] for u in usages})
        },
        # Unknown upstream model strings are untrusted response content. Retain
        # exact identities in immutable receipts; public monitoring uses hashes.
        "observed_models": sorted(
            {v["model"] for v in values if v["model"] == v["expected_model"]}
        ),
        "observed_model_hashes": sorted(
            {
                hashlib.sha256(v["model"].encode()).hexdigest()
                for v in values
                if v["model"] is not None
            }
        ),
    }


def read_usage_summary(path, *, day, run_id=None, as_of):
    """Verified, truly read-only view. Null totals mean unknown, never zero cost.

    Day is the original UTC admission day, even for later settlements. All
    known subtotals are reported lower bounds, not invoices. Legacy receipts
    remain byte-identical; new admission uses their conservative projection.
    """
    if (
        as_of.tzinfo is None
        or as_of.utcoffset() is None
        or datetime.strptime(day, "%Y-%m-%d").date().isoformat() != day
    ):
        raise ValueError("aware cutoff and exact UTC admission day required")
    result = {
        "schema_version": "2.0",
        "status": "unavailable",
        "accounting_policy": ACCOUNTING_POLICY,
        "day": day,
        "run_id": run_id,
        "as_of": as_of.isoformat(),
        "day_usage": None,
        "run_usage": None,
    }
    path = Path(path).resolve()
    if not path.is_file():
        return result
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"calls", "agent_admissions", "agent_settlements"} <= tables:
            return result
        values = verified_charges(db, day=day, run_id=run_id, as_of=as_of)
    result.update(
        status="available",
        day_usage=summarize_usage([v for v in values if v["day"] == day]),
        run_usage=summarize_usage([v for v in values if v["run_id"] == run_id])
        if run_id is not None
        else None,
    )
    return result
