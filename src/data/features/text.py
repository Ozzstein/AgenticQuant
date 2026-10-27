"""Persist actual Jev text-feature outputs for the same temporal numeric flow.

Historical source availability and generation time remain distinct. A feature
computed today from an old transcript cannot enter an old model fit. This does
not remove pretrained-model knowledge or confer research promotion.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import EvidenceRecord, Instrument, RunContext, _utc

if TYPE_CHECKING:
    from src.decision.features import TextFeature
    from src.domain.decisions import DecisionSource


def store_text_feature(store: EvidenceStore, feature: TextFeature, instrument: Instrument,
                       context: RunContext, *, sources: tuple[DecisionSource, ...], ingested_at):
    ingested = _utc(ingested_at)
    if feature.generated_at > ingested:
        raise ValueError("text feature cannot be ingested before actual generation")
    if tuple(source.content_hash() for source in sources) != feature.source_hashes:
        raise ValueError("text feature source identities do not match")
    if feature.evidence_available_at != max((source.record.available_at for source in sources if source.record.available_at), default=None):
        raise ValueError("text feature evidence availability does not match its sources")
    for source in sources:
        source.record.require_available(context.as_of)
        source.record.require_available(feature.generated_at)
        stored = store.get(source.record.evidence_id, as_of=context.as_of)
        if stored.record != source.record:
            raise ValueError("text feature references a different immutable record")
        document = store.document(source.record.evidence_id, as_of=context.as_of)
        if source.document_hash != document.document_hash or document.text[source.start:source.end] != source.text:
            raise ValueError("text feature source span does not match immutable document")
    raw = canonical_bytes(feature.model_dump(mode="json"))
    identity = digest(canonical_bytes({"feature": digest(raw), "instrument": instrument.instrument_id, "cutoff": context.as_of.isoformat()}))
    available = max(feature.generated_at, feature.evidence_available_at) if feature.evidence_available_at else None
    usable = feature.status == "ok" and feature.value is not None and available is not None and bool(sources)
    record = EvidenceRecord(evidence_id="text:"+identity, provider="jev-text-feature",
                            provider_version=feature.transformation_version, source_uri="decision-feature:"+identity,
                            raw_hash=digest(raw), ingested_at=ingested, available_at=available,
                            requested_as_of=context.as_of, observed_at=context.as_of,
                            availability_basis="documented" if available else "unknown", quality="usable" if usable else "missing",
                            instrument_ids=(instrument.instrument_id,), tool_name=feature.name, tool_version=feature.model)
    normalized = feature.model_dump(mode="json")
    normalized["source_evidence_ids"] = [source.record.evidence_id for source in sources]
    return store.put(record, raw, kind="text_feature", logical_key=f"{instrument.instrument_id}:{feature.name}:{context.as_of.isoformat()}",
                     normalized=normalized, document=raw.decode(), published_at=feature.generated_at)
