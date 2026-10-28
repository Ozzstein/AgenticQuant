"""Narrow adapter to the data lane's immutable document/passage APIs."""

from typing import Literal

from src.domain.common import UtcTimestamp, VersionedContract
from src.domain.decisions import DecisionSource, digest


def source_from_passage(store, passage, *, as_of, kind) -> DecisionSource:
    """Revalidate full-document identity and exact Unicode span before inference."""
    document = store.document(passage.evidence.evidence_id, as_of=as_of)
    if (
        document.evidence != passage.evidence
        or document.document_hash != passage.document_hash
        or digest(document.text) != document.document_hash
        or document.text[passage.start : passage.end] != passage.text
    ):
        raise ValueError("passage does not match immutable stored document")
    return DecisionSource.from_document(
        document.evidence, document.text, start=passage.start, end=passage.end, kind=kind
    )


def retrieve_sources(
    store, evidence_ids, *, as_of, kind="other", query="", max_chars=6000, max_passages=8
):
    return tuple(
        source_from_passage(store, passage, as_of=as_of, kind=kind)
        for passage in store.retrieve(
            evidence_ids, as_of=as_of, query=query, max_chars=max_chars, max_passages=max_passages
        )
    )


class DeliveredCoverage(VersionedContract):
    """Desk-input delivery, not a claim that available documents are true/complete."""

    status: Literal["available", "partial", "missing", "ineligible", "conflicting"]
    store_status: str
    as_of: UtcTimestamp
    allocation_policy: Literal["role-document-round-robin-1"] = "role-document-round-robin-1"
    evidence_ids: tuple[str, ...]
    eligible_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    partial_ids: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...]
    delivered_spans: tuple[dict, ...]
    delivery_complete: bool


def allocate_desk_sources(
    store,
    instrument,
    context,
    *,
    allowed_evidence_ids=None,
    max_chars=6000,
    max_passages=32,
    passage_chars=1200,
):
    """Fair bounded source allocation over actual documents, with explicit losses.

    Read each kind once. Preserve all revisions/conflicts and the supplied whitelist;
    selection never fetches ambient sources. Exact passages come from EvidenceStore.
    """
    from src.data.evidence.service import ROLE_KINDS

    if any(type(v) is not int for v in (max_chars, max_passages, passage_chars)) or not (
        0 < max_chars <= 100000 and 0 < max_passages <= 100 and 0 < passage_chars <= 10000
    ):
        raise ValueError("invalid aggregate source delivery budget")
    allowed = None if allowed_evidence_ids is None else set(allowed_evidence_ids)
    queries, records, excluded, by_id, kind_by_id = {}, {}, {}, {}, {}
    source_kind = {
        "news": "news",
        "analyst_estimates": "news",
        "crypto_global": "news",
        "transcript": "transcript",
        "technical": "technical",
        "funding": "technical",
        "open_interest": "technical",
        "liquidity": "technical",
    }
    for kind in sorted(set().union(*ROLE_KINDS.values())):
        query = store.query(
            as_of=context.as_of,
            kind=kind,
            instrument_id=instrument.instrument_id,
            current_revisions_only=False,
        )
        queries[kind] = query
        records[kind] = tuple(
            e for e in query.records if allowed is None or e.record.evidence_id in allowed
        )
        excluded[kind] = list(query.excluded)
        for envelope in query.records:
            identity = envelope.record.evidence_id
            if allowed is not None and identity not in allowed:
                excluded[kind].append((identity, "outside_frozen_whitelist"))
            elif not envelope.document_hash:
                excluded[kind].append((identity, "complete_document_unavailable"))
            else:
                by_id[identity] = envelope
                kind_by_id[identity] = source_kind.get(kind, "fundamental")
    # Give each role a document in turn; duplicate role references consume no
    # extra context. Only after every selected document has a span can any get
    # another. Order is independent of insertion order and provider responses.
    queues = {
        role: sorted(
            {
                e.record.evidence_id
                for kind in kinds
                for e in records[kind]
                if e.record.evidence_id in by_id
            }
        )
        for role, kinds in ROLE_KINDS.items()
    }
    ordered, seen = [], set()
    while any(queues.values()):
        for queue in queues.values():
            while queue and queue[0] in seen:
                queue.pop(0)
            if queue:
                identity = queue.pop(0)
                ordered.append(identity)
                seen.add(identity)
    selected = ordered[: min(max_passages, max_chars)]
    documents = {identity: store.document(identity, as_of=context.as_of) for identity in selected}
    for identity, document in documents.items():
        if (
            document.evidence != by_id[identity].record
            or document.document_hash != by_id[identity].document_hash
        ):
            raise ValueError("selected document differs from queried immutable envelope")
    offsets, sources, remaining = {identity: 0 for identity in selected}, [], max_chars
    active = [identity for identity in selected if documents[identity].text]
    while active and remaining and len(sources) < max_passages:
        for index, identity in enumerate(active):
            if not remaining or len(sources) == max_passages:
                break
            start = offsets[identity]
            length = min(
                passage_chars,
                max(1, remaining // (len(active) - index)),
                len(documents[identity].text) - start,
            )
            end = start + length
            passage = store.passage(identity, start, end, as_of=context.as_of)
            sources.append(
                source_from_passage(store, passage, as_of=context.as_of, kind=kind_by_id[identity])
            )
            offsets[identity] = end
            remaining -= length
        active = [
            identity for identity in active if offsets[identity] < len(documents[identity].text)
        ]
    delivered = {s.record.evidence_id for s in sources}
    coverage = {}
    for role, kinds in ROLE_KINDS.items():
        coverage[role] = {}
        for kind in kinds:
            ids = {e.record.evidence_id for e in records[kind]}
            present = ids & delivered
            partial = {
                identity
                for identity in present
                if offsets[identity] < len(documents[identity].text)
            }
            omitted = ids - delivered
            complete = bool(ids) and not partial and not omitted
            status = (
                "conflicting"
                if ids and queries[kind].status == "conflicting"
                else "available"
                if complete
                else "partial"
                if present
                else "ineligible"
                if not ids and excluded[kind]
                else "missing"
            )
            spans = tuple(
                {
                    "evidence_id": s.record.evidence_id,
                    "document_hash": s.document_hash,
                    "source_hash": s.content_hash(),
                    "passage_hash": s.passage_hash,
                    "start": s.start,
                    "end": s.end,
                }
                for s in sources
                if s.record.evidence_id in ids
            )
            coverage[role][kind] = DeliveredCoverage(
                status=status,
                store_status=queries[kind].status,
                as_of=context.as_of,
                evidence_ids=tuple(sorted(present)),
                eligible_ids=tuple(sorted(ids)),
                omitted_ids=tuple(sorted(omitted)),
                partial_ids=tuple(sorted(partial)),
                excluded=tuple(sorted(set(excluded[kind]))),
                delivered_spans=spans,
                delivery_complete=complete,
            )
    return tuple(sources), coverage


def load_desk_sources(service, instrument, context, **bounds):
    """Canonical CLI and H share one allocation/coverage policy; no ambient fetch."""
    return allocate_desk_sources(service.store, instrument, context, **bounds)


def validate_delivered_coverage(coverage, sources, *, store, context):
    """Check delivery claims against exact model input spans and full documents.

    The eligible set is the declared frozen selection scope, not a claim about
    documents outside that scope or source truth. Missing fields cannot imply ALL.
    """
    if not coverage:
        return
    if store is None:
        raise ValueError("delivered coverage requires source authority")
    for kinds in coverage.values():
        for kind, raw in kinds.items():
            value = DeliveredCoverage.model_validate(raw)
            if value.as_of > context.as_of or len(set(value.eligible_ids)) != len(
                value.eligible_ids
            ):
                raise ValueError("invalid delivery scope/cutoff")
            selected = [s for s in sources if s.record.evidence_id in value.eligible_ids]
            spans = tuple(
                {
                    "evidence_id": s.record.evidence_id,
                    "document_hash": s.document_hash,
                    "source_hash": s.content_hash(),
                    "passage_hash": s.passage_hash,
                    "start": s.start,
                    "end": s.end,
                }
                for s in selected
            )
            present = {s.record.evidence_id for s in selected}
            partial = set()
            for identity in value.eligible_ids:
                envelope = store.get(identity, as_of=value.as_of)
                if envelope.kind != kind:
                    raise ValueError("delivery scope source kind differs")
                if identity not in present:
                    continue
                doc = store.document(identity, as_of=value.as_of)
                cursor = 0
                full = True
                for source in sorted(
                    (s for s in selected if s.record.evidence_id == identity), key=lambda s: s.start
                ):
                    if (
                        source.record != doc.evidence
                        or source.document_hash != doc.document_hash
                        or doc.text[source.start : source.end] != source.text
                    ):
                        raise ValueError("delivered source differs from original document")
                    full &= source.start == cursor
                    cursor = source.end
                if not full or cursor != len(doc.text):
                    partial.add(identity)
            omitted = set(value.eligible_ids) - present
            complete = bool(value.eligible_ids) and not partial and not omitted
            expected_status = (
                "conflicting"
                if value.eligible_ids and value.store_status == "conflicting"
                else "available"
                if complete
                else "partial"
                if present
                else "ineligible"
                if not value.eligible_ids and value.excluded
                else "missing"
            )
            if (
                value.delivered_spans != spans
                or value.evidence_ids != tuple(sorted(present))
                or value.partial_ids != tuple(sorted(partial))
                or value.omitted_ids != tuple(sorted(omitted))
                or value.delivery_complete != complete
                or value.status != expected_status
            ):
                raise ValueError("delivered coverage differs from actual input spans")
