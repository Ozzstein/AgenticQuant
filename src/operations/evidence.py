"""Paper-only verification of actual post-event execution receipts.

This verifies evidence. PaperVenue separately proves that an order was durably
accepted before the actual print; no source receipt alone authorizes a fill.
Historical research continues to use its unchanged pre-open availability gate.
"""
from __future__ import annotations

from src.data.evidence.store import EvidenceStore
from src.research.evidence import (
    _verify_provenance,
    expected_artifacts,
    json_pointer,
    verify_provenance,
)
from src.research.registry import RegistryError, content_hash


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
    if tuple(receipts)!=tuple(inputs.receipts):
        raise RegistryError('paper execution receipts differ from frozen economic inputs')
    receipt_by_event={r.event_id:r for r in receipts}
    if len(receipt_by_event)!=len(receipts):
        raise RegistryError('duplicate paper execution receipt event')
    if receipts:
        bars={b.event_id:b for b in inputs.bars}
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
        checked=_verify_provenance(registry,store,inputs,references,paper_execution_receipts=receipt_by_event)
    else:
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
