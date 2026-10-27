"""Durable stop election from observed opens, shared by replay and local paper.

The first eligible tradable open initializes a trailing anchor. Only subsequent
eligible tradable opens can improve it; bar high/low/close and the decision's
reference price never stand in for observations. Election survives zero capacity,
partial fills and restarts. Native brokers retain their own trigger authority.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Literal

from src.domain.common import NonEmptyString, Sha256, UtcTimestamp, VersionedContract
from src.domain.portfolio import (
    ExecutionBar,
    OpeningEvent,
    OrderIntent,
    Positive,
    exact_decimal_product,
    exact_decimal_sum,
)
from src.execution.ledger import LedgerError

POLICY = "observed-open-stop-election-1"


class StopObservation(VersionedContract):
    policy: Literal["observed-open-stop-election-1"] = POLICY
    intent_id: NonEmptyString
    intent_hash: Sha256
    event_id: NonEmptyString
    event_hash: Sha256
    at: UtcTimestamp
    previous_hash: Sha256 | None
    anchor: Positive | None
    # A fixed offset greater than the initial sell anchor has no positive trigger.
    threshold: Decimal
    elected_event_id: NonEmptyString | None
    elected_at: UtcTimestamp | None


def _prefix(intent):
    return "paper-stop:" + intent.content_hash() + ":"


def _source(ledger, account, event_id):
    row = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?",
        (account, "opening-event:" + event_id),
    ).fetchone()
    if row:
        return OpeningEvent.model_validate_json(row[0])
    # An original completed bar can authenticate a pre-OpeningEvent execution.
    row = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?", (account, "open:" + event_id)
    ).fetchone()
    if row:
        return OpeningEvent.from_bar(ExecutionBar.model_validate_json(row[0]))
    raise LedgerError("stop observation lacks its actual retained opening source")


def _step(intent, event, previous):
    if (
        intent.order_type not in ("stop", "trailing_stop")
        or not event.tradable
        or event.instrument != intent.instrument
        or event.opened_at < intent.eligible_at
        or intent.expires_at is not None
        and event.opened_at > intent.expires_at
        or previous is not None
        and event.opened_at < previous.at
    ):
        raise LedgerError("stop observation differs from eligible chronological order inputs")
    anchor = previous.anchor if previous else None
    elected = previous.elected_event_id if previous else None
    elected_at = previous.elected_at if previous else None
    if elected:
        threshold = previous.threshold
    elif intent.order_type == "stop":
        threshold = intent.stop_price
    else:
        anchor = (
            event.price
            if anchor is None
            else (max if intent.side == "sell" else min)(anchor, event.price)
        )
        offset = (
            intent.trail_price
            if intent.trail_price is not None
            else exact_decimal_product(
                anchor, exact_decimal_product(intent.trail_percent, Decimal(".01"))
            )
        )
        threshold = exact_decimal_sum(
            [anchor, offset.copy_negate() if intent.side == "sell" else offset]
        )
    if elected is None and (
        event.price <= threshold if intent.side == "sell" else event.price >= threshold
    ):
        elected, elected_at = event.event_id, event.opened_at
    return StopObservation(
        intent_id=intent.intent_id,
        intent_hash=intent.content_hash(),
        event_id=event.event_id,
        event_hash=event.content_hash(),
        at=event.opened_at,
        previous_hash=previous.content_hash() if previous else None,
        anchor=anchor,
        threshold=threshold,
        elected_event_id=elected,
        elected_at=elected_at,
    )


def _history(ledger, intent):
    prefix = _prefix(intent)
    policy = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?",
        (intent.account_id, "paper-stop-policy:" + intent.intent_id),
    ).fetchone()
    if policy and json.loads(policy[0]) != {
        "policy": POLICY,
        "intent": intent.model_dump(mode="json"),
    }:
        raise LedgerError("local stop policy differs from original intent")
    previous, history = None, []
    seen = set()
    for row in ledger.db.execute(
        "SELECT id,payload FROM events WHERE account=? AND id>=? AND id<? ORDER BY rowid",
        (intent.account_id, prefix, prefix + "~"),
    ):
        current = StopObservation.model_validate_json(row["payload"])
        event = _source(ledger, intent.account_id, current.event_id)
        expected = _step(intent, event, previous)
        if (
            current != expected
            or row["id"] != prefix + current.content_hash()
            or current.event_id in seen
        ):
            raise LedgerError("stop observation hash/source/transition differs")
        seen.add(current.event_id)
        history.append(current)
        previous = current
    if bool(policy) != bool(history):
        raise LedgerError("local stop policy is missing its observation history or vice versa")
    return history


def read_stop_state(ledger, account_id, intent_id) -> StopObservation | None:
    """Read and verify retained trigger state without changing account history."""
    with ledger._reading():
        intent = ledger.order(account_id, intent_id)["intent"]
        history = _history(ledger, intent)
        return history[-1] if history else None


def observe_stop(ledger, intent: OrderIntent, event: OpeningEvent) -> StopObservation:
    """Called inside the same transaction as the consumed opening event and fills."""
    if ledger.spec(intent.account_id).mode == "live":
        raise LedgerError("live stop election requires actual native venue authority")
    original = ledger.order(intent.account_id, intent.intent_id)["intent"]
    if original != intent or _source(ledger, intent.account_id, event.event_id) != event:
        raise LedgerError("stop input differs from retained intent/opening event")
    history = _history(ledger, intent)
    for previous in history:
        if previous.event_id == event.event_id:
            return previous
    ledger.record_event(
        intent.account_id,
        "paper-stop-policy:" + intent.intent_id,
        json.dumps({"policy": POLICY, "intent": intent.model_dump(mode="json")}, sort_keys=True),
    )
    if not history:
        # A legacy partial stop really executed. Authenticate its old threshold
        # crossing instead of undoing election after an upgrade/restart. No NAV,
        # fill, source or published historical snapshot is rewritten.
        old_fills = [f for f in ledger.fills(intent.account_id) if f.intent_id == intent.intent_id]
        for old_fill in old_fills:
            old_event = _source(ledger, intent.account_id, old_fill.event_id)
            if (
                intent.order_type != "stop"
                or old_fill.fill_id != intent.intent_id + "@" + old_event.event_id
                or old_fill.executed_at != old_event.opened_at
            ):
                raise LedgerError("legacy stop fill lacks canonical observed-open authority")
            elected = _step(intent, old_event, history[-1] if history else None)
            if elected.elected_event_id is None:
                raise LedgerError("legacy stop fill has no authentic threshold crossing")
            ledger.record_event(
                intent.account_id,
                _prefix(intent) + elected.content_hash(),
                elected.model_dump_json(),
            )
            history.append(elected)
    observation = _step(intent, event, history[-1] if history else None)
    ledger.record_event(
        intent.account_id,
        _prefix(intent) + observation.content_hash(),
        observation.model_dump_json(),
    )
    return observation


def verify_stop_history(ledger, account):
    """Verify all retained local election chains and their actual fill bindings."""
    rows = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id>=? AND id<? ORDER BY rowid",
        (account, "paper-stop:", "paper-stop;"),
    ).fetchall()
    identities = {StopObservation.model_validate_json(r[0]).intent_id for r in rows}
    for row in ledger.db.execute(
        "SELECT id,payload FROM events WHERE account=? AND id>=? AND id<?",
        (account, "paper-stop-policy:", "paper-stop-policy;"),
    ):
        payload = json.loads(row["payload"])
        intent = ledger.order(account, payload["intent"]["intent_id"])["intent"]
        if (
            payload != {"policy": POLICY, "intent": intent.model_dump(mode="json")}
            or row["id"] != "paper-stop-policy:" + intent.intent_id
        ):
            raise LedgerError("local stop policy differs from original intent")
        identities.add(intent.intent_id)
    fills = ledger.fills(account) if identities else []
    for identity in identities:
        intent = ledger.order(account, identity)["intent"]
        history = _history(ledger, intent)
        by_event = {item.event_id: item for item in history}
        if not history:
            raise LedgerError("stop observation belongs to a different immutable intent")
        for fill in fills:
            if fill.intent_id != identity:
                continue
            proof = by_event.get(fill.event_id)
            if (
                proof is None
                or proof.elected_event_id is None
                or fill.executed_at != proof.at
                or fill.fill_id != identity + "@" + proof.event_id
            ):
                raise LedgerError("local stop fill lacks its verified election observation")
