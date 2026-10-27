"""Evidence-backed symbol aliases and historical membership; no current-list fallback."""
from __future__ import annotations

from datetime import datetime

from src.data.evidence.store import EvidenceStore
from src.domain.common import Instrument, UtcTimestamp, VersionedContract, _utc


class SymbolAlias(VersionedContract):
    instrument: Instrument
    symbol: str
    venue: str
    valid_from: UtcTimestamp
    valid_to: UtcTimestamp | None = None


class UniverseMembership(VersionedContract):
    universe: str
    instrument_id: str
    valid_from: UtcTimestamp
    valid_to: UtcTimestamp | None = None


class SecurityMaster:
    def __init__(self, store: EvidenceStore):
        self.store = store

    def resolve(self, symbol: str, venue: str, *, at: datetime, as_of: datetime) -> Instrument:
        at = _utc(at)
        if at > _utc(as_of):
            raise ValueError("cannot resolve future symbol identity")
        result = self.store.query(kind="symbol_alias", as_of=as_of, current_revisions_only=True)
        matches = {}
        for envelope in result.records:
            alias = SymbolAlias.model_validate(self.store.normalized(envelope.record.evidence_id, as_of=as_of))
            if alias.symbol == symbol and alias.venue == venue and alias.valid_from <= at and (alias.valid_to is None or at < alias.valid_to):
                matches[alias.instrument.content_hash()] = alias.instrument
        if len(matches) != 1:
            raise ValueError(f"symbol identity coverage is {'missing' if not matches else 'ambiguous'}: {symbol}@{venue}")
        return next(iter(matches.values()))

    def members(self, universe: str, *, at: datetime, as_of: datetime) -> tuple[str, ...]:
        at = _utc(at)
        if at > _utc(as_of):
            raise ValueError("cannot request future universe membership")
        result = self.store.query(kind="universe_membership", as_of=as_of, current_revisions_only=True)
        members = set()
        for envelope in result.records:
            item = UniverseMembership.model_validate(self.store.normalized(envelope.record.evidence_id, as_of=as_of))
            if item.universe == universe and item.valid_from <= at and (item.valid_to is None or at < item.valid_to):
                members.add(item.instrument_id)
        if not members:
            raise ValueError(f"historical universe coverage is missing: {universe}")
        return tuple(sorted(members))
