"""Durable generative-model admission; unknown prices never imply free calls."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4


class AgentBudgetError(RuntimeError):
    def __init__(self, message, *, status=None):
        super().__init__(message)
        self.status = status


class AgentCallBudget:
    def __init__(self, config, run_id, *, deadline):
        self.config, self.run_id, self.deadline = config, run_id, deadline
        self.session_id = str(uuid4())
        self.started_at = datetime.now(timezone.utc)
        self.path = Path(getattr(config, "budget_store_path", "outputs/agent-budgets.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            from src.agents.receipts import create_tables

            create_tables(db)
            db.execute("""CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY, run_id TEXT, day TEXT, provider TEXT, model TEXT,
                pricing_version TEXT, reserved_tokens INTEGER, charged_tokens INTEGER,
                reserved_cost TEXT, charged_cost TEXT, input_tokens INTEGER, output_tokens INTEGER,
                status TEXT, started_at TEXT, latency_ms REAL, actual_model TEXT)""")

    def reserve(self, model, messages, output_limit):
        # UTF-8 bytes plus schema/envelope allowance is a conservative token
        # reservation, reconciled only when the provider supplies valid usage.
        input_bound = len(json.dumps([str(m.content) for m in messages]).encode()) + 8192
        return self.reserve_bound(
            model,
            input_bound,
            output_limit,
            request_hash=hashlib.sha256(
                json.dumps(
                    [{"type": m.type, "content": m.content} for m in messages],
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        )

    def reserve_bound(self, model, input_bound, output_limit, *, request_hash=None):
        """Admit a caller-computed conservative bound using the same durable caps."""
        if type(input_bound) is not int or input_bound < 1:
            raise AgentBudgetError("positive integer input reservation required")
        if type(output_limit) is not int or output_limit < 0 or input_bound + output_limit >= 2**63:
            raise AgentBudgetError("nonnegative integer output reservation required")
        if request_hash is not None and (
            not isinstance(request_hash, str)
            or len(request_hash) != 64
            or any(c not in "0123456789abcdef" for c in request_hash)
        ):
            raise AgentBudgetError("invalid request identity")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise AgentBudgetError("analysis deadline expired")
        identity = str(uuid4())
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            # Admission clock/day is measured after acquiring the write lock;
            # waiting across midnight must not consume the previous day's cap.
            now = datetime.now(timezone.utc)
            provider = self.config.llm_provider.lower()
            quotes = [
                p
                for p in getattr(self.config, "pricing", ())
                if p.provider == provider and p.model == model and p.effective_at <= now
            ]
            if not quotes:
                raise AgentBudgetError("verified model pricing unavailable")
            price = max(quotes, key=lambda p: p.effective_at)
            tokens = input_bound + output_limit
            cost = (
                Decimal(input_bound) * price.input_usd_per_million
                + Decimal(output_limit) * price.output_usd_per_million
            ) / Decimal(1_000_000)
            from src.agents.receipts import verified_charges

            try:
                charges = verified_charges(
                    db, day=now.date().isoformat(), run_id=self.run_id, as_of=None
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise AgentBudgetError("invalid immutable budget authority") from exc
            if any(v["usage"]["breaches"] for v in charges):
                raise AgentBudgetError(
                    "prior provider usage or model identity violated admission run/day"
                )
            for column, value, requests, token_limit, cost_limit in (
                (
                    "run_id",
                    self.run_id,
                    getattr(self.config, "max_requests_per_analysis", 30),
                    getattr(self.config, "max_tokens_per_analysis", 100000),
                    Decimal(str(self.config.max_cost_per_analysis)),
                ),
                (
                    "day",
                    now.date().isoformat(),
                    getattr(self.config, "max_requests_per_day", 600),
                    getattr(self.config, "max_tokens_per_day", 1000000),
                    Decimal(str(getattr(self.config, "max_cost_per_day_usd", 10))),
                ),
            ):
                rows = [v["usage"] for v in charges if v[column] == value]
                if (
                    len(rows) >= requests
                    or sum(row["charged_tokens"] for row in rows) + tokens > token_limit
                    or sum((Decimal(row["charged_cost_usd"]) for row in rows), Decimal(0)) + cost
                    > cost_limit
                ):
                    raise AgentBudgetError("analysis/day request, token or spend budget exhausted")
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise AgentBudgetError("analysis deadline expired during admission")
            db.execute(
                "INSERT INTO calls VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identity,
                    self.run_id,
                    now.date().isoformat(),
                    provider,
                    model,
                    price.price_version,
                    tokens,
                    tokens,
                    str(cost),
                    str(cost),
                    None,
                    None,
                    "reserved",
                    now.isoformat(),
                    None,
                    None,
                ),
            )
            from src.agents.receipts import admission

            admission(
                db,
                identity,
                self,
                model,
                price,
                now,
                tokens,
                cost,
                input_bound=input_bound,
                output_limit=output_limit,
                request_hash=request_hash,
                usage_before=charges,
            )
        return identity, price, remaining

    def finish(self, identity, price, *, raw=None, status="failed", elapsed_ms=0, evidence=None):
        from src.agents.receipts import (
            _verified_call,
            canonical,
            capture_usage,
            derive_usage,
            settlement,
        )

        if (
            not isinstance(status, str)
            or not status
            or status == "reserved"
            or not math.isfinite(elapsed_ms)
            or elapsed_ms < 0
        ):
            raise AgentBudgetError("invalid terminal accounting status/latency")
        evidence = capture_usage(raw) if evidence is None else evidence
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            reserved = db.execute(
                "SELECT status FROM calls WHERE id=? AND run_id=?", (identity, self.run_id)
            ).fetchone()
            if not reserved or reserved[0] != "reserved":
                raise AgentBudgetError("unknown or already settled reservation")
            a, _, _, _ = _verified_call(db, identity)
            # A bad caller-supplied price cannot hide valid usage. Always price
            # against the immutable admission, and retain/fence the mismatch.
            supplied = {}
            for key in a["pricing"]:
                value = getattr(price, key, None)
                value = value.isoformat() if isinstance(value, datetime) else str(value)
                supplied[key] = value
            usage = derive_usage(a, evidence, status, pricing_verified=supplied == a["pricing"])
            if datetime.now(timezone.utc) < datetime.fromisoformat(a["started_at"]):
                raise AgentBudgetError("settlement precedes original admission")
            db.execute(
                "UPDATE calls SET charged_tokens=?,charged_cost=?,input_tokens=?,output_tokens=?,status=?,latency_ms=?,actual_model=? WHERE id=?",
                (
                    usage["charged_tokens"],
                    usage["charged_cost_usd"],
                    usage["input_tokens"],
                    usage["output_tokens"],
                    usage["status"],
                    elapsed_ms,
                    evidence["model"],
                    identity,
                ),
            )
            settlement(
                db,
                identity,
                self.run_id,
                evidence=evidence,
                usage=usage,
                reported_status=status,
                pricing_hash=hashlib.sha256(canonical(supplied).encode()).hexdigest(),
            )
        # In-flight results always settle, even after another call breaches.
        # Raise only AFTER the immutable charge and original run/day fence commit.
        if usage["status"] == "usage_overrun":
            raise AgentBudgetError(
                "provider usage exceeded conservative reservation", status="usage_overrun"
            )
        if (status == "ok" and usage["status"] != "ok") or any(
            reason in usage["breaches"]
            for reason in (
                "ambiguous_usage_response",
                "contradictory_usage_response",
                "settlement_pricing_identity_mismatch",
            )
        ):
            raise AgentBudgetError(
                "provider accounting cannot certify success", status=usage["status"]
            )
