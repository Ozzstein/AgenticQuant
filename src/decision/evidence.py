"""Narrow adapter to the data lane's immutable document/passage APIs."""

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


def load_desk_sources(service, instrument, context):
    """Connect EvidenceService's real role bundles to the complete desk.

    Deduplicate exact spans fetched for multiple roles; preserve all contrary
    candidates. Return coverage alongside sources for operator diagnostics.
    """
    kinds = {
        "news": "news",
        "analyst_estimates": "news",
        "crypto_global": "news",
        "transcript": "transcript",
        "technical": "technical",
        "funding": "technical",
        "open_interest": "technical",
        "liquidity": "technical",
    }
    sources, coverage = {}, {}
    for role in ("fundamental", "sentiment", "technical", "risk", "strategist"):
        bundle = service.source_bundle(role, instrument, context)
        coverage[role] = bundle["coverage"]
        by_id = {}
        for kind, query in bundle["coverage"].items():
            for envelope in query.records:
                by_id[envelope.record.evidence_id] = kinds.get(kind, "fundamental")
        for passage in bundle["passages"]:
            source = source_from_passage(
                service.store,
                passage,
                as_of=context.as_of,
                kind=by_id.get(passage.evidence.evidence_id, "other"),
            )
            sources[(source.record.evidence_id, source.start, source.end)] = source
    return tuple(sources.values()), coverage
