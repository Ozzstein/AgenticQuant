"""Paper-only verification of actual post-event execution receipts.

This verifies evidence. PaperVenue separately proves that an order was durably
accepted before the actual print; no source receipt alone authorizes a fill.
Historical research continues to use its unchanged pre-open availability gate.
"""
from __future__ import annotations

from src.data.evidence.store import EvidenceStore
from src.research.evidence import (
    _verify_provenance,
    execution_evidence_events,
    expected_artifacts,
    json_pointer,
    verify_provenance,
)
from src.research.registry import RegistryError, content_hash


def prepare_opening_source(store, venue, event, receipt, *, as_of):
    """Authenticate native print identity and migrate only proven old receipts.

    Every source and economic payload is checked before committing any mapping.
    The venue retains sole authority over consumption, acceptance and aliases.
    """
    from src.data.opening_identity import verified_opening_source_identity
    from src.domain.common import _utc
    from src.domain.portfolio import ExecutionBar, OpeningEvent

    as_of = _utc(as_of)
    event = OpeningEvent.from_bar(event) if isinstance(event, ExecutionBar) else OpeningEvent.model_validate(event.model_dump())

    def authenticated(document, *, expected=None):
        if document.available_at > as_of:
            raise RegistryError('opening source receipt exceeds migration cutoff')
        envelope = store.get(document.evidence_id, as_of=as_of)
        if (document.evidence_hash != envelope.content_hash()
                or document.observed_at != envelope.record.ingested_at
                or document.available_at != envelope.record.available_at):
            raise RegistryError('opening source migration receipt differs from original archive')
        identity = verified_opening_source_identity(store, document.evidence_id,
                                                     document.event_id, as_of=as_of)
        payload = store.normalized(document.evidence_id, as_of=as_of)
        legacy = envelope.record.provider_version == 'operating-wire-1'
        rows = payload['execution_bars' if legacy else 'opening_events']
        rows = [row for row in rows if row['event_id'] == document.event_id]
        if len(rows) != 1:
            raise RegistryError('opening receipt lacks its exact archived event')
        actual = (ExecutionBar if legacy else OpeningEvent).model_validate(rows[0])
        opening = OpeningEvent.from_bar(actual) if legacy else actual
        if expected is not None and opening != expected:
            raise RegistryError('opening event differs from authenticated native print economics')
        if hasattr(document, 'event_hash') and (
                document.event_hash not in (actual.content_hash(), opening.content_hash())
                or document.occurred_at != opening.opened_at):
            raise RegistryError('committed opening receipt differs from original source economics')
        if opening.instrument != event.instrument:
            raise RegistryError('opening migration refers to another instrument')
        return identity

    if receipt.event_id != event.event_id:
        raise RegistryError('current opening source receipt identifies another event')
    identity = authenticated(receipt, expected=event)
    previous = venue.unbound_opening_receipts(event.instrument)
    proven = [(old, authenticated(old)) for old in previous]
    with venue.ledger.atomic():
        if venue.unbound_opening_receipts(event.instrument) != previous:
            raise RegistryError('opening receipts changed during source migration')
        for old, native in proven:
            venue.bind_opening_source(native, event_id=old.event_id,
                                      evidence_id=old.evidence_id, evidence_hash=old.evidence_hash)
    return identity


def _source(store,source):
    envelope=store.get(source.evidence_id,as_of=source.available_at)
    if content_hash(store.raw(source.evidence_id))!=envelope.record.raw_hash:
        raise RegistryError('paper source raw content differs')
    for key in (envelope.normalized_hash,envelope.document_hash):
        if key:
            store.blob(key)
    value=json_pointer(store.normalized(source.evidence_id,as_of=source.available_at),source.pointer)
    return envelope,value if isinstance(value,list) else [value]


def verify_paper_provenance(registry,store:EvidenceStore,inputs,references,*,receipts,rule_sources=()):
    if inputs.context.mode!='paper' or inputs.account.mode!='paper' or inputs.context.account_id!=inputs.account.account_id:
        raise RegistryError('observed receipt settlement requires the matching explicit paper account')
    if inputs.snapshot.context.as_of>inputs.context.as_of:
        raise RegistryError('paper feature snapshot exceeds the actual cycle cutoff')
    if tuple(receipts)!=tuple(inputs.receipts):
        raise RegistryError('paper execution receipts differ from frozen economic inputs')
    receipt_by_event={r.event_id:r for r in receipts}
    if len(receipt_by_event)!=len(receipts):
        raise RegistryError('duplicate paper execution receipt event')
    if receipts:
        bars={b.event_id:b for b in execution_evidence_events(inputs)}
        if set(receipt_by_event)!=set(bars):
            raise RegistryError('paper execution receipts must exactly cover observed events')
        liquidity={v['event_id']:v for v in expected_artifacts(inputs)['opening_liquidity']}
        # Authenticate each late receipt against the exact bound source pointer.
        from src.research.evidence import BoundArtifact
        bound=BoundArtifact.model_validate_json(registry.blob(references['opening_liquidity_id']))
        matched=set()
        for source in bound.sources:
            envelope,values=_source(store,source)
            for value in values:
                if not isinstance(value,dict) or value.get('event_id') not in receipt_by_event:
                    raise RegistryError('opening source contains an unrelated event')
                identity=value['event_id']
                receipt=receipt_by_event[identity]
                record=envelope.record
                if (value!=liquidity[identity] or receipt.evidence_id!=source.evidence_id
                        or receipt.evidence_hash!=envelope.content_hash()
                        or receipt.observed_at!=record.ingested_at
                        or receipt.available_at!=record.available_at
                        or source.available_at!=receipt.available_at):
                    raise RegistryError('paper receipt clocks/source identity or opening print contents differ')
                if not bars[identity].opened_at<=receipt.observed_at<=receipt.available_at<=inputs.context.as_of:
                    raise RegistryError('paper print/receipt/availability/cycle chronology differs')
                if identity in matched:
                    raise RegistryError('paper opening receipt occurs more than once')
                matched.add(identity)
        if matched!=set(bars):
            raise RegistryError('missing exact opening print source coverage')
        if getattr(inputs,'opening_events',()):
            # Price and liquidity must originate in the SAME observed print
            # envelope. An unrelated array with the same event ID is insufficient.
            opening=BoundArtifact.model_validate_json(registry.blob(references['opening_events_id']))
            matched=set()
            for source in opening.sources:
                envelope,values=_source(store,source)
                for value in values:
                    if not isinstance(value,dict) or value.get('event_id') not in receipt_by_event:
                        raise RegistryError('opening price source contains an unrelated event')
                    identity=value['event_id']
                    receipt=receipt_by_event[identity]
                    if (value!=bars[identity].model_dump(mode='json')
                            or source.evidence_id!=receipt.evidence_id
                            or source.available_at!=receipt.available_at
                            or envelope.content_hash()!=receipt.evidence_hash
                            or identity in matched):
                        raise RegistryError('opening event price and liquidity must share the exact receipt envelope')
                    matched.add(identity)
            if matched!=set(bars):
                raise RegistryError('missing exact opening event price source coverage')
        checked=_verify_provenance(registry,store,inputs,references,paper_execution_receipts=receipt_by_event)
    else:
        if getattr(inputs,'opening_events',()):
            raise RegistryError('opening events require exact observed source receipts')
        checked=verify_provenance(registry,store,inputs,references)
    values=[]
    for source in rule_sources:
        _,rows=_source(store,source)
        for value in rows:
            from src.execution.venue import VenueRules
            rule=VenueRules.model_validate(value)
            if source.available_at>rule.available_at or rule.available_at>inputs.context.as_of or rule.venue!='paper':
                raise RegistryError('paper venue rule source/clock differs')
        values.extend(rows)
    if not rule_sources or values!=[r.model_dump(mode='json') for r in inputs.rules]:
        raise RegistryError('paper venue rules do not match exact source bindings')
    return checked
