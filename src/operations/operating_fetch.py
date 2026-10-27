"""Closed operating-source dispatch under one refresh request allowance."""

from __future__ import annotations

import json
import os
from datetime import datetime

from src.data.adapters.base import ProviderError, ReadHTTP
from src.data.adapters.operating import (
    AlpacaOperatingAdapter,
    BinanceOperatingAdapter,
    KrakenOperatingAdapter,
    OperatingCapture,
    economic_sources,
    verify_operating_capture,
)
from src.data.evidence.service import EvidenceService
from src.domain.common import Instrument


def load_operating_sources(operations, store, refresh_run_id):
    """Read the final economic bindings from an authoritative completed refresh."""
    from src.data.adapters.operating import KINDS
    from src.domain.common import RunContext
    from src.operations.refresh import RefreshPlan
    from src.operations.store import OperationError
    from src.research.evidence import SourceBinding

    manifest = operations.manifest(refresh_run_id)
    status = operations.status(refresh_run_id)
    if manifest.kind != "data" or status["status"] != "completed":
        raise OperationError("economic sources require a completed data refresh")
    plan = RefreshPlan.model_validate(operations.artifact(manifest.input_hash))
    if plan.context != manifest.context or plan.content_hash() != manifest.input_hash:
        raise OperationError("refresh plan identity differs")
    task = plan.tasks[-1]
    if task.provider != "operating" or task.operation != "economic_bundle":
        raise OperationError("refresh has no final operating economic bundle")
    report = operations.artifact(status["stages"]["report"]["output_hash"])
    context = RunContext.model_validate(report["context"])
    if (
        report["requested_context"] != plan.context.model_dump(mode="json")
        or report["plan_hash"] != plan.content_hash()
        or context
        != plan.context.model_copy(update={"as_of": datetime.fromisoformat(report["completed_at"])})
    ):
        raise OperationError("refresh completion context differs")
    stage = status["stages"]["fetch:" + task.task_id]
    receipt = operations.artifact(stage["output_hash"])
    if (
        receipt != report["fetches"][task.task_id]
        or receipt["task_hash"] != task.content_hash()
        or receipt["status"] != "available"
        or len(receipt["sources"]) != 1
    ):
        raise OperationError("economic bundle fetch receipt differs")
    source = receipt["sources"][0]
    envelope = store.get(source["evidence_id"], as_of=context.as_of)
    if (
        envelope.content_hash() != source["envelope_hash"]
        or envelope.kind != "operating_bundle"
        or envelope.record.provider != "operating"
        or envelope.record.requested_as_of != plan.context.as_of
        or not datetime.fromisoformat(stage["started_at"])
        <= envelope.record.ingested_at
        <= context.as_of
    ):
        raise OperationError("economic bundle source identity or receipt clock differs")
    value = verify_operating_capture(store, source["evidence_id"], as_of=context.as_of)
    bindings = tuple(
        (kind, tuple(SourceBinding.model_validate(v) for v in sources))
        for kind, sources in value["economic_sources"]
    )
    if tuple(kind for kind, _ in bindings) != KINDS or any(not sources for _, sources in bindings):
        raise OperationError("economic bundle source kinds or coverage differ")
    for _, sources in bindings:
        for source in sources:
            if source.available_at > envelope.record.available_at:
                raise OperationError("economic source was unavailable at bundle capture")
            store.get(source.evidence_id, as_of=source.available_at)
    return bindings


# Every accepted method and argument is fixed here; no caller endpoint/method.
OPERATIONS = {
    "alpaca_calendar": ("start", "end"),
    "alpaca_bars": ("start", "end", "calendar_id", "feed", "symbol_asof", "max_pages"),
    "alpaca_trades": ("start", "end", "calendar_id", "feed", "symbol_asof", "max_pages"),
    "alpaca_reference": ("asset_id", "bar_ids", "calendar_id"),
    "alpaca_corporate_actions": ("start", "end", "calendar_id", "max_pages"),
    "kraken_bars": ("market_id", "start", "end"),
    "kraken_reference": ("market_id", "bar_ids", "fx_evidence_id"),
    "kraken_trades": ("market_id", "start", "end", "calendar_id", "max_pages", "fx_evidence_id"),
    "kraken_calendar": ("start", "end", "reference_id"),
    "binance_bars": ("start", "end", "max_pages"),
    "binance_reference": ("bar_ids", "fx_evidence_id"),
    "binance_trades": ("start", "end", "calendar_id", "fx_evidence_id"),
    "binance_calendar": ("start", "end", "reference_id"),
    "current_universe": ("instruments", "reference_ids", "universe_version"),
    "economic_bundle": (
        "instruments",
        "market_start",
        "reference_ids",
        "universe_ids",
        "calendar_ids",
        "print_ids",
        "action_ids",
    ),
}


class _BudgetHTTP:
    def __init__(self, owner, provider, base):
        self.owner = owner
        self.inner = ReadHTTP(
            provider, base, max_requests=owner.limit, attempts=2, timeout=owner.timeout
        )
        self.base_url = self.inner.base_url

    def get(self, *args, **kwargs):
        remaining = self.owner.limit - sum(t.inner.requests for t in self.owner.transports)
        if remaining <= 0:
            raise ProviderError("operating", "request_budget_exhausted")
        self.inner.max_requests = self.inner.requests + remaining
        return self.inner.get(*args, **kwargs)

    def close(self):
        self.inner.close()


class _Transports:
    def __init__(self, task):
        self.limit, self.timeout = task.max_http_requests, min(20, task.deadline_seconds)
        self.transports = []

    def make(self, provider, base):
        transport = _BudgetHTTP(self, provider, base)
        self.transports.append(transport)
        return transport

    def close(self):
        for transport in self.transports:
            transport.close()


def fetch_operating(task, context, store):
    if task.operation not in OPERATIONS:
        raise ProviderError("operating", "operation_not_permitted")
    arguments = json.loads(task.arguments_json)
    if set(arguments) - set(OPERATIONS[task.operation]):
        raise ProviderError("operating", "unsupported_arguments")
    if "max_pages" in arguments:
        arguments["max_pages"] = min(arguments["max_pages"], task.max_http_requests)
    elif "max_pages" in OPERATIONS[task.operation]:
        arguments["max_pages"] = task.max_http_requests
    transports = _Transports(task)
    try:
        operation = task.operation
        if operation in ("current_universe", "economic_bundle"):
            instruments = tuple(Instrument.model_validate(i) for i in arguments.pop("instruments"))
            capture = OperatingCapture(store)
            if operation == "current_universe":
                result = capture.current_universe(instruments, context, **arguments)
            else:
                cutoff = capture.now()
                snapshot = EvidenceService(store).market_snapshot(
                    instruments,
                    context.model_copy(update={"as_of": cutoff}),
                    start=datetime.fromisoformat(arguments.pop("market_start")),
                    price_basis="raw",
                )
                bindings = economic_sources(store, snapshot, context=context, **arguments)
                ids = tuple(dict.fromkeys(b.evidence_id for _, values in bindings for b in values))
                sources = tuple(store.get(i, as_of=capture.now()) for i in ids)
                for identity in ids:
                    verify_operating_capture(store, identity, as_of=capture.now())
                bundle = capture.derived(
                    context,
                    sources,
                    lambda at: {
                        "economic_sources": [
                            [kind, [binding.model_dump(mode="json") for binding in values]]
                            for kind, values in bindings
                        ],
                        "snapshot_evidence_ids": [r.evidence_id for r in snapshot.evidence],
                        "snapshot_hash": snapshot.content_hash(),
                        "assembly_at": at.isoformat(),
                    },
                    kind="operating_bundle",
                    logical_key=context.run_id,
                )
                result = capture.result((bundle,))
        else:
            family, method = operation.split("_", 1)
            if family == "alpaca":
                adapter = AlpacaOperatingAdapter(
                    store,
                    os.getenv("ALPACA_API_KEY", ""),
                    os.getenv("ALPACA_API_SECRET", ""),
                    fmp_key=os.getenv("FMP_API_KEY", ""),
                    data_http=transports.make("alpaca", "https://data.alpaca.markets"),
                    trading_http=transports.make("alpaca", "https://paper-api.alpaca.markets"),
                    profile_http=transports.make("fmp", "https://financialmodelingprep.com"),
                )
            elif family == "kraken":
                adapter = KrakenOperatingAdapter(
                    store, http=transports.make("kraken", "https://api.kraken.com")
                )
            else:
                adapter = BinanceOperatingAdapter(
                    store, http=transports.make("binance", "https://api.binance.com")
                )
            if method in ("bars", "trades"):
                for key in ("start", "end"):
                    arguments[key] = datetime.fromisoformat(arguments[key])
            if operation == "alpaca_calendar":
                result = adapter.calendar(context, **arguments)
            else:
                if task.instrument is None:
                    raise ProviderError("operating", "explicit_instrument_required")
                result = getattr(adapter, method)(task.instrument, context, **arguments)
        return result, sum(t.inner.requests for t in transports.transports)
    finally:
        transports.close()
