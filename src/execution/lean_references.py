"""Exact per-intent native reference authority shared with the LEAN process."""

from __future__ import annotations

import json
import sqlite3

from src.domain.common import RunContext, _utc
from src.execution.ccxt_transport import canonical, digest
from src.execution.ledger import LedgerError


def _resolver(evidence_store, context, bindings, intent):
    # The data-owned parser is the sole authority. There is no execution-side
    # source discovery, price override, fallback parser or cached success.
    try:
        from src.data.adapters.base import ProviderError
        from src.data.binance_execution import (
            BinanceEvidenceReferenceResolver,
            BinanceReferenceBinding,
        )
    except ImportError:
        raise LedgerError("data-owned certified Binance source parser is unavailable") from None

    context = RunContext.model_validate(context.model_dump())
    if (
        context.account_id != intent.account_id
        or context.mode != "live"
        or context.run_id != intent.run_id
        or context.as_of != intent.decision_at
    ):
        raise LedgerError("LEAN reference context differs from exact admitted intent")
    bindings = tuple(
        BinanceReferenceBinding.model_validate(value.model_dump()) for value in bindings
    )
    if {binding.instrument for binding in bindings} != {intent.instrument}:
        raise LedgerError("LEAN reference bindings must describe exactly the intended instrument")
    try:
        return BinanceEvidenceReferenceResolver(evidence_store, context, bindings)
    except (ProviderError, OSError, sqlite3.Error) as exc:
        code = exc.code if isinstance(exc, ProviderError) else "frozen_source_unavailable"
        raise LedgerError("LEAN certified native reference denied: " + code) from None


def bind_binance_reference(execution, intent_id, *, evidence_store, context, bindings):
    """Certify/archive exact sources for an already prepared LEAN intent.

    Returns the immutable manifest hash. Exact retries remain readable after
    dispatch; a new/changed binding cannot be introduced after a submit attempt.
    """
    state = execution.state(intent_id)
    adapter = execution.adapters.get("lean")
    if (
        adapter is None
        or state["venue"] != "lean"
        or adapter.deployment.brokerage != "BinanceBrokerage"
        or execution.ledger.spec(execution.account_id).mode != "live"
    ):
        raise LedgerError("per-intent Binance sources require the exclusive live LEAN book")
    intent = execution.ledger.order(execution.account_id, intent_id)["intent"]
    resolver = _resolver(evidence_store, context, bindings, intent)
    manifest = {
        "implementation": "lean-intent-binance-reference-1",
        "intent_hash": intent.content_hash(),
        "context": resolver.context.model_dump(mode="json"),
        "bindings": [value.model_dump(mode="json") for value in resolver.bindings],
    }
    payload = {"manifest": manifest, "manifest_hash": digest(manifest)}
    with execution.ledger.atomic():
        key = "lean-binance-reference:" + intent_id
        old = execution.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (execution.account_id, key),
        ).fetchone()
        if old is None and execution.state(intent_id)["state"] != "prepared":
            raise LedgerError("native reference must be bound before the original submit attempt")
        execution.ledger.record_event(execution.account_id, key, canonical(payload))
    return payload["manifest_hash"]


def resolve_binance_reference(adapter, intent, *, as_of):
    """Reconstruct exactly this intent's admitted resolver in the runtime process."""
    try:
        from src.data.binance_execution import BinanceReferenceBinding
    except ImportError:
        raise LedgerError("data-owned certified Binance source parser is unavailable") from None

    if adapter.evidence_store is None:
        raise LedgerError("LEAN Binance needs its configured actual EvidenceStore")
    row = adapter.ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?",
        (adapter.account_id, "lean-binance-reference:" + intent.intent_id),
    ).fetchone()
    if row is None:
        raise LedgerError("LEAN intent has no exact frozen native reference binding")
    payload = json.loads(row[0])
    manifest = payload["manifest"]
    if (
        payload["manifest_hash"] != digest(manifest)
        or manifest["implementation"] != "lean-intent-binance-reference-1"
        or manifest["intent_hash"] != intent.content_hash()
        or _utc(as_of) < intent.decision_at
    ):
        raise LedgerError("LEAN original native reference manifest/intent changed")
    return _resolver(
        adapter.evidence_store,
        RunContext.model_validate(manifest["context"]),
        tuple(BinanceReferenceBinding.model_validate(value) for value in manifest["bindings"]),
        intent,
    )
