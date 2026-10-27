"""Durable generative-model admission; unknown prices never imply free calls."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4


class AgentBudgetError(RuntimeError):
    pass


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
        return self.reserve_bound(model, input_bound, output_limit)

    def reserve_bound(self, model, input_bound, output_limit):
        """Admit a caller-computed conservative bound using the same durable caps."""
        if type(input_bound) is not int or input_bound < 1:
            raise AgentBudgetError("positive integer input reservation required")
        if type(output_limit) is not int or output_limit < 0:
            raise AgentBudgetError("nonnegative integer output reservation required")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise AgentBudgetError("analysis deadline expired")
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
        identity = str(uuid4())
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM calls WHERE run_id=? AND status='usage_overrun' LIMIT 1",
                (self.run_id,),
            ).fetchone():
                raise AgentBudgetError("prior provider usage exceeded reservation")
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
                rows = db.execute(
                    f"SELECT charged_tokens,charged_cost FROM calls WHERE {column}=?", (value,)
                ).fetchall()
                if (
                    len(rows) >= requests
                    or sum(row[0] for row in rows) + tokens > token_limit
                    or sum((Decimal(row[1]) for row in rows), Decimal(0)) + cost > cost_limit
                ):
                    raise AgentBudgetError("analysis/day request, token or spend budget exhausted")
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

            admission(db, identity, self, model, price, now, tokens, cost)
        return identity, price, remaining

    def finish(self, identity, price, *, raw=None, status="failed", elapsed_ms=0):
        usage = getattr(raw, "usage_metadata", None) or {}
        metadata = getattr(raw, "response_metadata", None) or {}
        actual = metadata.get("model_name") or metadata.get("model")
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        known = all(type(v) is int and 0 <= v < 2**62 for v in (input_tokens, output_tokens))
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            reserved = db.execute(
                "SELECT reserved_tokens,reserved_cost,status FROM calls WHERE id=? AND run_id=?",
                (identity, self.run_id),
            ).fetchone()
            if not reserved or reserved[2] != "reserved":
                raise AgentBudgetError("unknown or already settled reservation")
            if known:
                tokens = input_tokens + output_tokens
                cost = (
                    Decimal(input_tokens) * price.input_usd_per_million
                    + Decimal(output_tokens) * price.output_usd_per_million
                ) / Decimal(1_000_000)
            else:
                tokens, cost = reserved[0], Decimal(reserved[1])
                input_tokens = output_tokens = None
                if status == "ok":
                    status = "usage_unknown"
            overrun = tokens > reserved[0] or cost > Decimal(reserved[1])
            if overrun:
                status = "usage_overrun"
            db.execute(
                "UPDATE calls SET charged_tokens=?,charged_cost=?,input_tokens=?,output_tokens=?,status=?,latency_ms=?,actual_model=? WHERE id=?",
                (
                    tokens,
                    str(cost),
                    input_tokens,
                    output_tokens,
                    status,
                    elapsed_ms,
                    actual,
                    identity,
                ),
            )
            from src.agents.receipts import settlement

            settlement(db, identity, self.run_id)
        # Raise only after commit: an overrun must never roll back its actual charge.
        if overrun:
            raise AgentBudgetError("provider usage exceeded conservative reservation")
