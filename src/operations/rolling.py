"""Closed clock substitutions for durable, rolling read-only refresh plans.

This chooses query windows, never source availability or economic timestamps.
The scheduler archives the generated plan before dispatch and reuses it on recovery.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, model_validator

from src.domain.common import NonEmptyString, VersionedContract, _utc
from src.operations.refresh import RefreshPlan
from src.operations.store import canonical, digest

# Only date/time arguments of existing closed adapters may vary. Credentials,
# identifiers, source references, pagination and endpoints remain frozen.
DATE_WINDOWS = {
    ("fred", "observations"),
    ("finnhub", "news"),
    ("yfinance", "bars"),
    ("operating", "alpaca_calendar"),
    ("operating", "alpaca_corporate_actions"),
    ("operating", "kraken_calendar"),
    ("operating", "binance_calendar"),
}
UTC_WINDOWS = {
    ("ccxt", "funding"),
    ("ccxt", "open_interest"),
    ("operating", "alpaca_bars"),
    ("operating", "alpaca_trades"),
    ("operating", "kraken_bars"),
    ("operating", "kraken_trades"),
    ("operating", "binance_bars"),
    ("operating", "binance_trades"),
    ("operating", "alpaca_cash_dividend_sources"),
    ("operating", "alpaca_account_dividend_sources"),
}


class RefreshClockBinding(VersionedContract):
    task_id: NonEmptyString
    argument: Literal["start", "end", "since", "started_at", "ended_at"]
    anchor: Literal["scheduled_for", "invoked_at"]
    offset_seconds: int = Field(default=0, ge=-366 * 86400 * 20, le=0, strict=True)


class RollingRefreshTemplate(VersionedContract):
    plan: RefreshPlan
    bindings: tuple[RefreshClockBinding, ...] = Field(default=(), max_length=200)

    @model_validator(mode="after")
    def closed_bindings(self):
        if self.plan.context.data_snapshot_ids:
            raise ValueError("rolling refresh cannot inherit snapshot approvals from a seed run")
        tasks = {task.task_id: task for task in self.plan.tasks}
        seen = set()
        for binding in self.bindings:
            if binding.task_id not in tasks or (binding.task_id, binding.argument) in seen:
                raise ValueError("clock binding must name one unique existing task argument")
            seen.add((binding.task_id, binding.argument))
            task = tasks[binding.task_id]
            key = (task.provider, task.operation)
            if key not in DATE_WINDOWS | UTC_WINDOWS:
                raise ValueError("adapter operation has no approved rolling clock arguments")
            arguments = (
                {"since", "end"}
                if task.provider == "ccxt"
                else {"started_at", "ended_at"}
                if task.operation.endswith("dividend_sources")
                else {"start", "end"}
            )
            if binding.argument not in arguments or binding.argument not in json.loads(
                task.arguments_json
            ):
                raise ValueError(
                    "clock binding must replace an existing permitted temporal argument"
                )
        return self

    def materialize(
        self, schedule_id: str, *, scheduled_for: datetime, invoked_at: datetime
    ) -> RefreshPlan:
        scheduled_for, invoked_at = _utc(scheduled_for), _utc(invoked_at)
        if invoked_at < scheduled_for:
            raise ValueError("rolling refresh cannot execute a future schedule window")
        tasks = []
        for task in self.plan.tasks:
            values = json.loads(task.arguments_json)
            for binding in self.bindings:
                if binding.task_id != task.task_id:
                    continue
                anchor = scheduled_for if binding.anchor == "scheduled_for" else invoked_at
                at = anchor + timedelta(seconds=binding.offset_seconds)
                values[binding.argument] = (
                    at.date().isoformat()
                    if (task.provider, task.operation) in DATE_WINDOWS
                    else at.isoformat()
                )
            # Adapters retain their native inclusive/exclusive window semantics.
            for start, end in (("start", "end"), ("since", "end"), ("started_at", "ended_at")):
                if (
                    start in values
                    and end in values
                    and any(b.task_id == task.task_id for b in self.bindings)
                ):
                    if datetime.fromisoformat(values[start]) > datetime.fromisoformat(values[end]):
                        raise ValueError("rolling query window is reversed")
            tasks.append(task.model_copy(update={"arguments_json": canonical(values).decode()}))
        run_key = digest(
            canonical(
                {
                    "schedule_id": schedule_id,
                    "scheduled_for": scheduled_for.isoformat(),
                    "template_hash": self.content_hash(),
                }
            )
        )
        context = self.plan.context.model_copy(
            update={"run_id": "scheduled-refresh:" + run_key, "as_of": invoked_at}
        )
        return RefreshPlan.model_validate(
            self.plan.model_copy(
                update={
                    "context": context,
                    "scheduled_for": scheduled_for,
                    "tasks": tuple(tasks),
                }
            ).model_dump()
        )
