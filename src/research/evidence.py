"""Independently verify research references against immutable source contents.

Hash-shaped strings alone are not evidence. Bindings identify the exact JSON
subtree of an archived normalized source and the historical time it was usable.
Adapters may archive a separately versioned normalization before binding it.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from src.data.evidence.store import EvidenceStore
from src.domain.common import NonEmptyString, Sha256, UtcTimestamp, VersionedContract
from src.domain.market import MarketBar, MarketSnapshot
from src.research.datasets import ResearchInputs
from src.research.registry import RegistryError, ResearchRegistry, canonical, content_hash


class SourceBinding(VersionedContract):
    evidence_id: NonEmptyString
    pointer: str = ""
    available_at: UtcTimestamp


class BoundArtifact(VersionedContract):
    kind: Literal["universe_history", "availability_audit", "calendar", "corporate_actions",
                  "execution_bars", "opening_events", "opening_liquidity", "instrument_metadata"]
    inputs_hash: Sha256
    value_json: NonEmptyString
    sources: tuple[SourceBinding, ...] = Field(min_length=1)

    def value(self) -> Any:
        value = json.loads(self.value_json)
        if canonical(value).decode() != self.value_json:
            raise RegistryError("artifact value must use canonical JSON")
        return value


def json_pointer(value: Any, pointer: str) -> Any:
    if not pointer:
        return value
    if not pointer.startswith("/"):
        raise RegistryError("JSON pointer must be empty or begin with /")
    try:
        for part in pointer[1:].split("/"):
            token = part.replace("~1", "/").replace("~0", "~")
            value = value[int(token)] if isinstance(value, list) else value[token]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RegistryError("source JSON pointer does not identify archived contents") from exc
    return value


def execution_evidence_events(inputs):
    """Select one explicit execution evidence convention without fabricating bars."""
    openings=tuple(getattr(inputs,'opening_events',()))
    bars=tuple(inputs.bars)
    if openings:
        from src.domain.portfolio import OpeningEvent
        if bars:
            raise RegistryError('opening-event proofs and completed execution bars must be separate')
        if inputs.context.mode!='paper' or inputs.account.mode!='paper':
            raise RegistryError('opening-event evidence requires an explicit paper operating context')
        events=tuple(OpeningEvent.model_validate(e.model_dump()) for e in openings)
    else:
        events=bars
    if len({e.event_id for e in events})!=len(events):
        raise RegistryError('duplicate execution evidence event identity')
    return events


def expected_artifacts(inputs: ResearchInputs) -> dict:
    """Exact economic assumptions that external source evidence must substantiate."""
    events=execution_evidence_events(inputs)
    calendars = {canonical({k: b.model_dump(mode="json")[k] for k in
                           ("calendar", "calendar_version", "session", "session_open", "session_close")})
                 for b in events}
    kind='opening_events' if getattr(inputs,'opening_events',()) else 'execution_bars'
    return {
        "universe_history": [u.model_dump(mode="json") for u in inputs.universes],
        "availability_audit": [r.model_dump(mode="json") for r in inputs.snapshot.evidence],
        "calendar": [json.loads(v) for v in sorted(calendars)],
        "corporate_actions": [a.model_dump(mode="json") for a in inputs.actions],
        kind: [b.model_dump(mode="json") for b in events],
        "opening_liquidity": [{k: b.model_dump(mode="json")[k] for k in
                               ("event_id", "instrument", "opened_at", "opening_volume", "fx_rate")}
                              for b in events],
        "instrument_metadata": [m.model_dump(mode="json") for m in inputs.metadata],
    }


def verify_market_snapshot(store: EvidenceStore, snapshot: MarketSnapshot) -> None:
    """Reconstruct normalized observations, not just their envelope hashes."""
    for record in snapshot.evidence:
        stored = store.get(record.evidence_id, as_of=snapshot.context.as_of).record
        if stored != record or content_hash(store.raw(record.evidence_id)) != record.raw_hash:
            raise RegistryError("market snapshot source differs from archived evidence")
    for bar in snapshot.bars:
        payload = store.normalized(bar.evidence_id, as_of=snapshot.context.as_of)
        archived = MarketBar.model_validate({**payload, "evidence_id": bar.evidence_id})
        if archived != bar:
            raise RegistryError("market bar differs from archived normalized source")


def verify_provenance(registry: ResearchRegistry, store: EvidenceStore, inputs: ResearchInputs,
                      references: dict[str, str]) -> dict[str, str]:
    return _verify_provenance(registry,store,inputs,references)


def _verify_provenance(registry,store,inputs,references,*,paper_execution_receipts=None):
    """Shared content checks; public historical verification never enables late receipts."""
    names = {"universe_history": "universe_history_id", "availability_audit": "availability_audit_id",
             "calendar": "calendar_artifact_id", "corporate_actions": "corporate_actions_id",
             "execution_bars": "execution_bars_id", "opening_events": "opening_events_id", "opening_liquidity": "opening_liquidity_id",
             "instrument_metadata": "instrument_metadata_id"}
    checked = {}
    for kind, expected in expected_artifacts(inputs).items():
        key = references.get(names[kind])
        if not key:
            raise RegistryError(f"missing {kind} source artifact")
        artifact = BoundArtifact.model_validate_json(registry.blob(key))
        if artifact.kind != kind or artifact.inputs_hash != inputs.content_hash() or artifact.value() != expected:
            raise RegistryError(f"{kind} contents do not match the frozen research inputs")
        source_values = []
        for source in artifact.sources:
            if paper_execution_receipts is not None and source.available_at > inputs.context.as_of:
                raise RegistryError('paper source binding is after the actual cycle cutoff')
            record = store.get(source.evidence_id, as_of=source.available_at).record
            raw = store.raw(source.evidence_id)
            if content_hash(raw) != record.raw_hash:
                raise RegistryError("source raw hash mismatch")
            normalized = store.normalized(source.evidence_id, as_of=source.available_at)
            value = json_pointer(normalized, source.pointer)
            source_values.extend(value if isinstance(value, list) else [value])
        if source_values != expected:
            raise RegistryError(f"{kind} referenced source contents do not substantiate the artifact")
        # Historical membership and calendars must be available by their effective dates.
        for source in artifact.sources:
            values = json_pointer(store.normalized(source.evidence_id, as_of=source.available_at), source.pointer)
            for value in values if isinstance(values, list) else [values]:
                if kind == "universe_history" and source.available_at > datetime.fromisoformat(value["available_at"].replace("Z", "+00:00")):
                    raise RegistryError("universe source became available after declared membership")
                if kind == "calendar" and source.available_at > datetime.fromisoformat(value["session_open"].replace("Z", "+00:00")):
                    raise RegistryError("session calendar source was unavailable at the session open")
                if kind == "opening_liquidity" and source.available_at > datetime.fromisoformat(value["opened_at"].replace("Z", "+00:00")):
                    receipt=(paper_execution_receipts or {}).get(value['event_id'])
                    if receipt is None or receipt.evidence_id!=source.evidence_id or receipt.available_at!=source.available_at:
                        raise RegistryError("opening liquidity or FX was unavailable at the observed open")
                if kind == "instrument_metadata" and source.available_at > datetime.fromisoformat(value["as_of"].replace("Z", "+00:00")):
                    raise RegistryError("instrument metadata was unavailable at its declared cutoff")
        checked[names[kind]] = key
    verify_market_snapshot(store, inputs.snapshot)
    return checked
