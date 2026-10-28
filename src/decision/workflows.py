"""Shared J1–J6 workflows. Storage/retrieval eligibility and exact matches stay in code."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from src.decision import questions as q
from src.decision.service import DecisionService
from src.domain.decisions import (
    CitedClaim,
    DecisionPolicy,
    DecisionRequest,
    DecisionResult,
    DecisionSource,
    DecisionStatus,
    canonical_json,
)


@dataclass(frozen=True)
class CitationCheck:
    claim: CitedClaim
    status: str
    reason: str
    result: DecisionResult | None = None


PROJECTION_VERSION = "source-bound-evaluated-only-1"


def exact_citation(claim, sources, context):
    """Deterministic diagnostics only; safe in shadow and evaluated prompts."""
    matches = [
        s
        for s in sources
        if s.record.evidence_id == claim.evidence_id and s.passage_hash == claim.passage_hash
    ]
    if len(matches) != 1:
        return CitationCheck(claim, "invalid_reference", "unknown or ambiguous source/hash")
    source = matches[0]
    if claim.quote not in source.text:
        return CitationCheck(claim, "quote_missing", "exact quote absent from referenced passage")
    try:
        source.record.require_available(context.as_of)
    except ValueError:
        return CitationCheck(claim, "ineligible", "citation was unavailable at as_of")
    return CitationCheck(
        claim, "quote_verified", "exact quote/reference only; semantic support not established"
    )


def semantic_projection(store, evidence_store, result, *, context, instrument, sources, as_of):
    """Re-export actual evaluated authority before a semantic value reaches a model.

    Source eligibility and publication use the original decision cutoff. Provider
    completion uses the actual prompt creation clock, never a backdated as-of.
    """
    from src.decision.evaluation import EvaluationService
    from src.decision.service import parse_response

    if as_of.tzinfo is None:
        raise ValueError("semantic projection requires an actual aware prompt cutoff")
    proof = store.export_observation(result.observation_id)
    recorded = DecisionResult.model_validate(proof["result"])
    request = DecisionRequest.model_validate(proof["request"])
    if (
        result != recorded
        or result.status != DecisionStatus.OK
        or request.policy.activation != "evaluated"
        or request.context != context
        or request.instrument != instrument
        or result.completed_at > as_of
        or not set(result.source_hashes).issubset({s.content_hash() for s in sources})
    ):
        raise ValueError("semantic projection differs from original eligible observation")
    authority = EvaluationService(store, evidence_store=evidence_store)
    publication = authority.resolve(
        request.policy,
        question_hashes=tuple(q.content_hash() for q in request.questions),
        as_of=context.as_of,
    )
    for source in request.sources:
        authority._source(source, context.as_of)
    model, parsed, _, _ = parse_response(request, result.raw_response)
    normalized = tuple(
        j.model_copy(update={"disposition": "observed"}) if j.disposition == "accepted" else j
        for j in result.judgments
    )
    if parsed != normalized or model != result.actual_model:
        raise ValueError("semantic projection differs from raw response")
    # Cache/replay keeps the original raw observation's identity and chronology.
    original = proof
    while original["reused"] is not None:
        parent = DecisionResult.model_validate(original["result"])
        original = original["reused"]
        reused = DecisionResult.model_validate(original["result"])
        if (
            reused.cache_key != result.cache_key
            or reused.completed_at > as_of
            or reused.raw_response != result.raw_response
            or reused.completed_at > parent.started_at
        ):
            raise ValueError("semantic projection reuse differs from original observation")
    leaf = DecisionResult.model_validate(original["result"])
    attempts, reservations = original["attempts"], original["reservations"]
    if (
        leaf.status != DecisionStatus.OK
        or not attempts
        or len(attempts) != leaf.attempts
        or len(reservations) != leaf.attempts
        or attempts[-1]["status"] != 200
        or attempts[-1]["body"] != result.raw_response
        or [a["attempt"] for a in attempts] != list(range(1, leaf.attempts + 1))
        or [r["attempt"] for r in reservations] != list(range(1, leaf.attempts + 1))
        or any(
            not leaf.started_at.timestamp() <= r["at"] <= leaf.completed_at.timestamp()
            for r in reservations
        )
    ):
        raise ValueError("semantic projection lacks original provider/admission proof")
    source_refs = [
        {
            "source_hash": s.content_hash(),
            "evidence_id": s.record.evidence_id,
            "raw_hash": s.record.raw_hash,
            "document_hash": s.document_hash,
            "passage_hash": s.passage_hash,
            "start": s.start,
            "end": s.end,
            "available_at": s.record.available_at.isoformat(),
        }
        for s in request.sources
    ]
    return [
        {
            "projection_version": PROJECTION_VERSION,
            "observation_id": result.observation_id,
            "request_hash": result.request_hash,
            "original_observation_id": original["result"]["observation_id"],
            "completed_at": result.completed_at.isoformat(),
            "decision_as_of": context.as_of.isoformat(),
            "model": result.actual_model,
            "policy_hash": request.policy.content_hash(),
            "evaluation_id": publication.content_hash(),
            "publication_at": publication.published_at.isoformat(),
            "sources": source_refs,
            "state": json.loads(request.state_json),
            "judgment": j.model_dump(mode="json"),
        }
        for j in result.judgments
        if j.disposition == "accepted"
    ]


class EvidenceJudgments:
    def __init__(
        self,
        service: DecisionService,
        context,
        instrument,
        *,
        policy=None,
        timeout_seconds=30,
        mode="cache",
        deadline=None,
    ):
        self.service, self.context, self.instrument = service, context, instrument
        self.policy = policy or DecisionPolicy()
        self.timeout_seconds, self.mode = timeout_seconds, mode
        self.deadline = deadline

    def ask(self, sources, questions, state=None):
        return self.service.evaluate(
            DecisionRequest(
                context=self.context,
                instrument=self.instrument,
                policy=self.policy,
                sources=tuple(sources),
                questions=tuple(questions),
                state_json=canonical_json(state or {}),
            ),
            timeout_seconds=min(self.timeout_seconds, max(0, self.deadline - time.monotonic()))
            if self.deadline
            else self.timeout_seconds,
            mode=self.mode,
        )

    def news(self, source: DecisionSource):
        return self.ask((source,), (q.NEWS_TONE, q.RELEVANCE, q.NEWS_EVENT))

    def transcript(self, source: DecisionSource):
        return self.ask((source,), (q.MANAGEMENT_TONE, q.GUIDANCE, q.HEDGING, q.RESPONSIVENESS))

    def verify(self, claim: CitedClaim, sources) -> CitationCheck:
        checked = exact_citation(claim, sources, self.context)
        if checked.status != "quote_verified":
            return checked
        source = next(
            s
            for s in sources
            if s.record.evidence_id == claim.evidence_id and s.passage_hash == claim.passage_hash
        )
        result = self.ask(
            (source,), (q.CLAIM_SUPPORT,), {"claim": claim.claim, "quote": claim.quote}
        )
        label = (
            result.judgments[0].choice
            if result.status == DecisionStatus.OK
            else result.status.value
        )
        return CitationCheck(claim, label, "semantic source support, not source truth", result)

    def retrieve(self, sources, *, query: str, reference: str):
        """Score all candidates; preserve conflicts AND unknown/unavailable passages.

        Returns scored candidates, never a falsely clean corpus. The caller may
        order them but must retain contradictory evidence inside its context bound.
        """
        return tuple(
            (
                source,
                self.ask(
                    (source,), (q.RELEVANCE, q.CONFLICT), {"query": query, "reference": reference}
                ),
            )
            for source in sources
        )

    def debate(self, sources, reports):
        return self.ask(
            sources,
            (q.DEBATE_DISCREPANCY,),
            {
                "reports": [
                    {
                        "role": r.role,
                        "action": r.action.value,
                        "reasoning": r.reasoning,
                        "claims": [c.model_dump(mode="json") for c in r.claims],
                    }
                    for r in reports
                ]
            },
        )
